# =============================================================================
# pipeline/run_pipeline.py – main orchestrator of the OSCAR+ pipeline
# =============================================================================
#
# Orchestrates all 8 pipeline steps in the correct order:
#
#   Step 1: Object localization        (GroundingDINO + SAM)
#   Step 2: Point cloud generation     (Open3D)
#   Step 3: Semantic candidates        (CLIP)
#   Step 4: Image-based re-ranking     (DINOv2)
#   Step 5: Shape matching             (ULIP-2)
#   Step 6: Score fusion               (Weighted Sum / RRF / Intersection)
#   Step 7: Geometric check            (dGeDi re-ranking of the shortlist)
#   Step 8: Pose estimation            (FoundationPose / ICP)
#
# THE NORMAL ENTRY POINT IS `run_pipeline.py` IN THE REPOSITORY ROOT — it takes
# --rgb/--depth/--intrinsics/--prompt/--gallery/--top-k/--out, wraps itself into
# the oscar-plus container and writes ranking.csv.  Add the Stage-5 grasp trial on
# the rank-1 model with `run_pipeline_sim.py`.  This module is the ORCHESTRATOR
# behind both (class OSCARPlusPipeline); its own CLI below stays for developers
# who need the full set of internal flags (encoder swaps, per-channel top-k,
# --skip_steps, --until-step, ablation switches):
#
#   python3 -m pipeline.run_pipeline \
#       --rgb path/to/image.png \
#       --depth path/to/depth.png \
#       --prompt "pick up the mustard bottle" \
#       --gallery ycbv
#
# =============================================================================

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from PIL import Image

# --- Pipeline Module ---
from .config import PipelineConfig
from .step1_localization import ObjectLocalizer
from .step2_pointcloud import PointCloudGenerator
from .step3_clip_retrieval import CLIPRetriever
from .step4_dino_reranking import DINOReRanker
from .step5_shape_matching import ShapeMatcher
from .step6_fusion import ScoreFusion
from .step8_pose_estimation import PoseEstimator
from .utils import load_depth_image, ensure_dir

# =============================================================================
# Logging konfigurieren
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# =============================================================================
# Datenstruktur für extrahierte Prompt-Elemente
# =============================================================================

@dataclass
class PromptElements:
    """Search and detection elements extracted from the user prompt.

    Attributes:
        object_name:      Core object name (noun).  e.g. "mustard bottle"
        color:            Colour, if mentioned.     e.g. "yellow"
        shape:            Shape descriptor.         e.g. "cylindrical"
        material:         Material.                 e.g. "plastic"
        detection_phrase: Full adjective+noun for GroundingDINO.
                          e.g. "yellow mustard bottle"
        visual_query:     Enriched text for CLIP text retrieval.
                          e.g. "yellow plastic mustard bottle"
    """
    object_name: str
    color: str = ""
    shape: str = ""
    material: str = ""
    detection_phrase: str = ""   # wird in __post_init__ gesetzt wenn leer
    visual_query: str = ""       # wird in __post_init__ gesetzt wenn leer

    def __post_init__(self):
        attrs = " ".join(x for x in [self.color, self.shape, self.material,
                                      self.object_name] if x)
        if not self.detection_phrase:
            # Kurze Phrase für GroundingDINO: Farbe + Objektname
            self.detection_phrase = " ".join(
                x for x in [self.color, self.object_name] if x
            )
        if not self.visual_query:
            self.visual_query = attrs


# =============================================================================
# Pipeline-Orchestrator
# =============================================================================

class OSCARPlusPipeline:
    """Main pipeline: from the language prompt to the 6D pose.

    Connects the 8 pipeline steps into one coherent flow.
    Each step is a self-contained, testable module.

    Architecture overview:
    ┌──────────────────────────────────────────────────────────────────┐
    │                    OSCAR+ Pipeline                               │
    │                                                                  │
    │  RGB-D Image + Prompt                                            │
    │       │                                                          │
    │       ▼                                                          │
    │  ┌─────────────────────┐                                         │
    │  │ 1. Localization     │ GroundingDINO + SAM                     │
    │  │    → Mask, ROI      │                                         │
    │  └────────┬────────────┘                                         │
    │           │                                                      │
    │           ├──────────────────────────────┐                       │
    │           ▼                              ▼                       │
    │  ┌────────────────────┐     ┌────────────────────┐               │
    │  │ 3. CLIP Retrieval  │     │ 2. Point cloud      │ only if      │
    │  │    → Top-20        │     │    (lazy, only if   │ Step 5/7/8   │
    │  └────────┬───────────┘     │    5/7/8 active)    │ active       │
    │           ▼                 └──────────┬──────────┘              │
    │  ┌────────────────────┐                │                         │
    │  │ 4. DINOv2 Re-Rank  │                │                         │
    │  │    → Top-5         │                │                         │
    │  └────────┬───────────┘                │                         │
    │           │              ┌─────────────┘                         │
    │           ▼              ▼                                       │
    │      ┌──────────────────────┐                                    │
    │      │ 5. ULIP-2 Shape Match│ re-ranks CLIP candidates           │
    │      └──────────┬───────────┘                                    │
    │                 │                                                │
    │                 └──┐                                             │
    │                    | (+ CLIP + DINO scores)                      │
    │                    ▼                                             │
    │              ┌────────────┐                                      │
    │              │6. Fusion   │ Weighted sum / RRF / Intersection    │
    │              └─────┬──────┘                                      │
    │                    ▼                                             │
    │              ┌────────────┐                                      │
    │              │7. Geo-Check│ dGeDi-Re-Ranking                     │
    │              └─────┬──────┘                                      │
    │                    ▼                                             │
    │              ┌────────────┐                                      │
    │              │8. Pose     │ FoundationPose / ICP                 │
    │              └────────────┘                                      │
    │                    │                                             │
    │                    ▼                                             │
    │              6D Pose [R|t] + scaled CAD model                    │
    └──────────────────────────────────────────────────────────────────┘

    Usage:
        >>> config = PipelineConfig(
        ...     description_file="object_database/ycbv/descriptions.json",
        ...     reference_images_dir="object_images/ycbv/",
        ...     cad_models_dir="object_database/ycbv/",
        ... )
        >>> pipeline = OSCARPlusPipeline(config)
        >>> result = pipeline.run(rgb_image, depth_image, "pick up the mustard bottle")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = ensure_dir(config.output_dir)

        # --- Module initialisieren (Lazy Loading, Modelle werden bei Bedarf geladen) ---
        self.localizer = ObjectLocalizer(config)
        self.pc_generator = PointCloudGenerator(config)
        self.clip_retriever = CLIPRetriever(config)
        self.dino_reranker = DINOReRanker(config)
        self.shape_matcher = ShapeMatcher(config)
        self.fusion = ScoreFusion(config)
        self.pose_estimator = PoseEstimator(config)

        self._initialized = False

    def initialize(self):
        """Loads all models and data up front (optional, otherwise lazy loading).

        Useful when the pipeline is run several times, so the models
        are not reloaded on every run.
        """
        logger.info("=" * 60)
        logger.info("OSCAR+ Pipeline – initialization")
        logger.info("=" * 60)

        t0 = time.time()

        # CLIP Beschreibungen laden
        if self.config.description_file:
            logger.info("Loading CLIP descriptions...")
            self.clip_retriever.load_descriptions()

        # DINOv2 Referenzbilder laden
        if self.config.reference_images_dir:
            logger.info("Loading DINOv2 reference images...")
            self.dino_reranker.load_reference_images()

        # ULIP-2 CAD-Modelle laden
        if self.config.cad_models_dir:
            logger.info("Loading ULIP-2 CAD models...")
            self.shape_matcher.load_cad_models()

        self._initialized = True
        logger.info(f"Initialization finished in {time.time()-t0:.1f}s")

    def run(
        self,
        rgb_image: Image.Image,
        depth_image: np.ndarray,
        prompt: str,
        camera_intrinsics: dict = None,
        skip_steps: list = None,
    ) -> dict:
        """Runs the whole pipeline.

        Args:
            rgb_image: RGB input image (PIL).
            depth_image: Depth image as a numpy array (H, W), in mm or m.
            prompt: Natural-language prompt, e.g. "pick up the mustard bottle".
            camera_intrinsics: Dict with 'fx', 'fy', 'cx', 'cy' (optional).
            skip_steps: List of step numbers to skip.

        Returns:
            Dict with the results of all steps:
            {
                "localization": LocalizationResult,
                "point_cloud": PointCloudResult,
                "clip_retrieval": CLIPRetrievalResult,
                "dino_reranking": DINOReRankingResult,
                "shape_matching": ShapeMatchingResult,
                "fusion": FusionResult,
                "geometry_reranking": GeometryReRankingResult,
                "pose_estimation": PoseEstimationResult,
                "timing": {...},
                "summary": {...},
            }
        """
        skip_steps = skip_steps or []
        results = {}
        timings = {}
        cam = camera_intrinsics or {}
        cam["gt_bbox_center_compensation"] = self.config.gt_bbox_center_compensation

        logger.info("=" * 60)
        logger.info(f"OSCAR+ Pipeline – Start")
        logger.info(f"Prompt: \"{prompt}\"")
        logger.info("=" * 60)
        t_start = time.time()

        # =================================================================
        # Schritt 1: Objektlokalisierung (GroundingDINO + SAM)
        # =================================================================
        if 1 not in skip_steps:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Step 1: Object localization")

            # Prompt analysieren: Objekt + visuelle Attribute extrahieren
            prompt_elements = self._extract_prompt_elements(prompt)
            results["prompt_elements"] = prompt_elements
            logger.info(f"  Object:   '{prompt_elements.object_name}'")
            logger.info(f"  Color:    '{prompt_elements.color}'")
            logger.info(f"  Shape:    '{prompt_elements.shape}'")
            logger.info(f"  Material: '{prompt_elements.material}'")
            logger.info(f"  Detection phrase:  '{prompt_elements.detection_phrase}'")
            logger.info(f"  CLIP query:        '{prompt_elements.visual_query}'")

            loc_result = self.localizer.localize(rgb_image, prompt_elements.visual_query)
            results["localization"] = loc_result
            timings["step1_localization"] = time.time() - t0

            if loc_result is None:
                logger.error("Object not found – pipeline aborted.")
                return {"error": "Object not found", "prompt": prompt}

            logger.info(
                f"  ✓ Object found (confidence: {loc_result.confidence:.3f})"
            )

        # =================================================================
        # Schritt 3: CLIP Retrieval
        # =================================================================
        if 3 not in skip_steps and "localization" in results:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Step 3: CLIP retrieval (semantic candidates)")

            if not self._initialized and self.config.description_file:
                self.clip_retriever.load_descriptions()

            loc = results["localization"]
            elements = results.get("prompt_elements")
            visual_query = elements.visual_query if elements else None
            clip_result = self.clip_retriever.retrieve(
                loc.roi_image
            )
            results["clip_retrieval"] = clip_result
            timings["step3_clip"] = time.time() - t0

            logger.info(f"  ✓ {len(clip_result.candidates)} CLIP candidates (S_text, full database)")
            for i, c in enumerate(clip_result.candidates[:5]):
                logger.info(f"    {i+1}. {c.object_id} (S_text={c.score:.4f})")

        # =================================================================
        # Schritt 4: DINOv2 Re-Ranking
        # =================================================================
        if 4 not in skip_steps and "localization" in results:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Step 4: DINOv2 re-ranking")

            if not self._initialized and self.config.reference_images_dir:
                self.dino_reranker.load_reference_images()

            loc = results["localization"]
            clip_res = results.get("clip_retrieval")  # None if Step 3 was skipped
            dino_result = self.dino_reranker.rerank(loc.roi_image, clip_res)
            results["dino_reranking"] = dino_result
            timings["step4_dino"] = time.time() - t0

            encoder_name = "SigLIP" if self.config.appearance_encoder == "siglip" else "DINOv2"
            logger.info(f"  ✓ {len(dino_result.candidates)} {encoder_name} candidates "
                        f"(S_view, top-{self.config.dino_view_topk} softmax, full database)")
            for i, c in enumerate(dino_result.candidates[:5]):
                logger.info(
                    f"    {i+1}. {c.object_id} "
                    f"(S_view={c.dino_score:.4f}, S_text={c.clip_score:.4f})"
                )

        # =================================================================
        # Schritt 2: Punktwolke erzeugen (lazy — nur wenn Step 5/7/8 aktiv)
        # =================================================================
        _needs_pc = any(s not in skip_steps for s in [5, 7, 8])
        if 2 not in skip_steps and _needs_pc and "localization" in results:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Step 2: Generate point cloud")

            loc = results["localization"]
            rgb_np = np.array(rgb_image)

            pc_result = self.pc_generator.generate(
                rgb_np, depth_image, loc.mask,
                fx=cam.get("fx"), fy=cam.get("fy"),
                cx=cam.get("cx"), cy=cam.get("cy"),
            )
            results["point_cloud"] = pc_result
            timings["step2_pointcloud"] = time.time() - t0

            if pc_result:
                logger.info(
                    f"  ✓ Point cloud: {pc_result.num_points} points, "
                    f"size: {pc_result.bbox_size}"
                )

        # =================================================================
        # Schritt 5: ULIP-2 Shape Matching
        # =================================================================
        if 5 not in skip_steps and "point_cloud" in results:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Step 5: ULIP-2 shape matching")

            if not self._initialized and self.config.cad_models_dir:
                self.shape_matcher.load_cad_models()

            pc = results["point_cloud"]
            if pc:
                # Full-database scoring (thesis Sec. 3.5.2): all three channels
                # score every CAD model; no early pruning by any single channel.
                candidate_ids = None

                query_img = results.get("localization", None)
                shape_result = self.shape_matcher.match(
                    pc,
                    candidate_ids=candidate_ids,
                    query_image=query_img.roi_image if query_img else None,
                )
                results["shape_matching"] = shape_result
                timings["step5_ulip"] = time.time() - t0

                encoder_name = "Uni3D" if self.config.shape_encoder == "uni3d" else "ULIP-2"
                logger.info(f"  ✓ {len(shape_result.candidates)} {encoder_name} candidates "
                            f"(S_shape, mode={self.config.ulip2_mode}, full database)")
                for i, c in enumerate(shape_result.candidates[:5]):
                    logger.info(
                        f"    {i+1}. {c.object_id} (S_shape={c.shape_score:.4f})"
                    )

        # =================================================================
        # Schritt 6: Score-Fusion
        # =================================================================
        if 6 not in skip_steps:
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Step 6: Score fusion")

            fusion_result = self.fusion.fuse(
                clip_result=results.get("clip_retrieval"),
                dino_result=results.get("dino_reranking"),
                shape_result=results.get("shape_matching"),
            )
            results["fusion"] = fusion_result
            timings["step6_fusion"] = time.time() - t0

            if fusion_result.best_match:
                bm = fusion_result.best_match
                logger.info(
                    f"  ✓ Best match: {bm.object_id} "
                    f"(fused={bm.fused_score:.4f} | "
                    f"S_text={bm.clip_score:.4f}, S_view={bm.dino_score:.4f}, "
                    f"S_shape={bm.ulip_score:.4f})"
                )
                logger.info(
                    f"  Method: {fusion_result.method} | "
                    f"Weights: clip={self.config.weight_clip}, dino={self.config.weight_dino}, "
                    f"ulip={self.config.weight_ulip}"
                )

        # =================================================================
        # Schritt 7: Geometrischer Check (dGeDi-Re-Ranking, vormals B2)
        # =================================================================
        if (
            getattr(self.config, "geometry_reranking_enabled", False)
            and "fusion" in results
            and results["fusion"].best_match
            and "point_cloud" in results
            and 7 not in skip_steps
        ):
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Step 7: Geometric check (dGeDi re-ranking)")
            logger.info("  Signal: %s | Top-K: %d",
                        self.config.geometry_reranking_signal,
                        self.config.geometry_reranking_top_k)

            from .step7_geometry_reranking import GeometryReRanker
            reranker = GeometryReRanker(self.config)
            pc = results["point_cloud"]

            b2_result = reranker.rerank(
                fused_candidates=results["fusion"].candidates,
                observed_pcd=pc.point_cloud,
            )
            results["geometry_reranking"] = b2_result
            timings["step7_geometry"] = time.time() - t0

            # Detailed per-candidate log
            logger.info("  ┌─────────────────────────────────────────────────────────────────┐")
            logger.info("  │  Step 7 · dGeDi Geometry Re-ranking                             │")
            logger.info("  ├─────┬──────────────────────────┬────────┬────────────┬──────────┤")
            logger.info("  │ Rank│ Object ID                │ GeDi   │ Chamfer    │ Geo Score│")
            logger.info("  ├─────┼──────────────────────────┼────────┼────────────┼──────────┤")
            for i, gc in enumerate(b2_result.candidates, 1):
                chamfer_str = f"{gc.chamfer_score:.6f}" if gc.chamfer_score < float("inf") else "    N/A   "
                marker = " ◄" if i == 1 else ""
                logger.info(
                    "  │ %3d │ %-24s │ %6.0f │ %10s │ %8.4f │%s",
                    i, gc.object_id[:24], gc.gedi_score,
                    chamfer_str, gc.geometry_score, marker,
                )
            logger.info("  └─────┴──────────────────────────┴────────┴────────────┴──────────┘")

            if b2_result.best_candidate:
                best = b2_result.best_candidate
                logger.info(
                    "  ✓ Step 7 winner: %s (fitness=%.4f)",
                    best.object_id, best.ransac_fitness,
                )

        # Für Schritte 7+8 werden diese Variablen geteilt
        resolved_mesh = None   # aufgelöster Mesh-Pfad (kein PNG-Fallback)
        effective_best_model = None  # may differ from fusion best_match after Schritt 7

        # Ergebnis von Schritt 7 anwenden: override effective_best_model
        if "geometry_reranking" in results and results["geometry_reranking"].best_candidate:
            b2_best = results["geometry_reranking"].best_candidate
            # Wrap as FusedCandidate-like object for downstream compatibility
            from .step6_fusion import FusedCandidate
            effective_best_model = FusedCandidate(
                object_id=b2_best.object_id,
                fused_score=b2_best.geometry_score,
                clip_score=b2_best.clip_score,
                dino_score=b2_best.dino_score,
                ulip_score=b2_best.ulip_score,
                cad_model_path=b2_best.cad_model_path,
                best_view_path=b2_best.best_view_path,
            )



        # =================================================================
        # Schritt 8: Pose Estimation
        # =================================================================
        if (
            8 not in skip_steps
            and "fusion" in results
            and results["fusion"].best_match
        ):
            t0 = time.time()
            logger.info("─" * 40)
            logger.info("Step 8: Pose estimation")

            best_model = effective_best_model or results["fusion"].best_match
            # Skalenschaetzung entfernt (2026-09-18): Meshes werden in ihren
            # nativen Einheiten verwendet; die Einheiten-Zuordnung liegt beim
            # Aufrufer (wie in den Stage-3/4/5-Treibern).
            scale_factor = 1.0
            loc = results.get("localization")

            # Mesh-Pfad auflösen falls Schritt 7 übersprungen wurde
            if resolved_mesh is None:
                resolved_mesh = self._resolve_mesh_path_for_candidate(best_model)
                if not resolved_mesh:
                    logger.warning("No valid mesh path found.")

            mesh_to_use = resolved_mesh
            if mesh_to_use:
                pose_result = self.pose_estimator.estimate(
                    rgb_image=np.array(rgb_image),
                    depth_image=depth_image,
                    mask=loc.mask if loc else np.ones_like(depth_image, dtype=bool),
                    cad_model_path=mesh_to_use,
                    scale_factor=scale_factor,
                    observed_pc=results.get("point_cloud"),
                    fx=cam.get("fx"), fy=cam.get("fy"),
                    cx=cam.get("cx"), cy=cam.get("cy"),
                    initial_pose=None,
                )
                results["pose_estimation"] = pose_result
                timings["step8_pose"] = time.time() - t0

                logger.info(
                    f"  ✓ Pose estimated (method={pose_result.method}, "
                    f"confidence={pose_result.confidence:.4f})"
                )
                logger.info(
                    f"  Translation: [{pose_result.translation[0]:.4f}, "
                    f"{pose_result.translation[1]:.4f}, "
                    f"{pose_result.translation[2]:.4f}] m"
                )

        # =================================================================
        # Zusammenfassung
        # =================================================================
        total_time = time.time() - t_start
        timings["total"] = total_time

        results["timing"] = timings
        results["summary"] = self._create_summary(results)

        logger.info("=" * 60)
        logger.info(f"Pipeline finished in {total_time:.2f}s")
        logger.info("=" * 60)

        # Ergebnisse speichern
        self._save_results(results)

        return results

    # ------------------------------------------------------------------
    # Mesh-Pfad-Auflösung (Schritt 8)
    # ------------------------------------------------------------------

    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def _resolve_mesh_path_for_candidate(self, candidate):
        """Resolve the real CAD mesh path for a fused candidate.

        Falls back to recursive mesh search when cad_model_path points to an
        image file (which can happen when ULIP did not have the object in
        its Top-K).
        """
        resolved = getattr(candidate, "cad_model_path", "")
        # Der Pfad kommt aus dem Embedding-Cache und wurde dort beim Bauen der
        # Galerie abgelegt. Er zeigt auf das Standard-Layout; wer cad_root
        # verlegt, findet die Datei dort nicht mehr. Deshalb: existiert sie
        # nicht, unter cad_models_dir nachsehen statt einen toten Pfad an die
        # Pose weiterzureichen.
        if (not resolved
                or os.path.splitext(resolved)[1].lower() in self._IMG_EXTS
                or not os.path.isfile(resolved)):
            resolved = self._find_cad_mesh(
                candidate.object_id, self.config.cad_models_dir
            ) or resolved
        return resolved or None

    @staticmethod
    def _find_cad_mesh(object_id: str, cad_root: str) -> str:
        """Recursively finds a CAD mesh for an object in the CAD directory.

        Prefers typical GSO/YCBV file names in `meshes/`.
        """
        if not object_id or not cad_root:
            return ""

        obj_dir = os.path.join(cad_root, object_id)
        if not os.path.isdir(obj_dir):
            return ""

        allowed = {".obj", ".ply", ".glb", ".gltf"}
        preferred_names = ("textured_simple.obj", "model.obj", "mesh.obj")
        candidates = []

        for root, _, files in os.walk(obj_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in allowed:
                    candidates.append(os.path.join(root, fname))

        if not candidates:
            return ""

        def sort_key(path: str):
            base = os.path.basename(path).lower()
            in_meshes = 0 if os.path.basename(os.path.dirname(path)).lower() == "meshes" else 1
            try:
                pref_idx = preferred_names.index(base)
            except ValueError:
                pref_idx = len(preferred_names)
            return (in_meshes, pref_idx, path)

        return sorted(candidates, key=sort_key)[0]


    def _extract_prompt_elements(self, prompt: str) -> "PromptElements":
        """Extracts object + visual attributes (colour, shape, material) from the prompt.

        Rule-based extraction (DE + EN).

        Examples:
            "pick up the yellow mustard bottle"
                → object: mustard bottle | color: yellow
            "greife die zylindrische Plastikflasche"
                → object: Flasche | shape: zylindrisch | material: Plastik

        Returns:
            PromptElements (detection_phrase and visual_query are built automatically).
        """
        return self._extract_prompt_elements_heuristic(prompt)

    @staticmethod
    def _extract_prompt_elements_heuristic(prompt: str) -> "PromptElements":
        """Rule-based prompt decomposition for _extract_prompt_elements.

        Recognises common colour, shape and material words (DE + EN) and
        extracts the remaining noun part as the object name.
        """
        # --- Bekannte Attribut-Wörter ---
        _COLORS = {
            "red", "green", "blue", "yellow", "orange", "purple", "pink",
            "white", "black", "gray", "grey", "brown", "cyan", "magenta",
            "rot", "grün", "blau", "gelb", "orange", "lila", "rosa",
            "weiß", "schwarz", "grau", "braun",
        }
        _SHAPES = {
            "round", "square", "rectangular", "cylindrical", "flat", "spherical",
            "cubic", "triangular", "oval", "elongated",
            "rund", "eckig", "rechteckig", "zylindrisch", "flach", "kugelig",
        }
        _MATERIALS = {
            "plastic", "metal", "wooden", "glass", "rubber", "cardboard",
            "paper", "fabric", "ceramic", "foam",
            "plastik", "metall", "holz", "glas", "gummi", "pappe", "stoff",
        }
        _VERBS    = {
            # Deutsch
            "greife", "nehme", "hole", "bringe", "hol", "gib", "brauch",
            "brauche", "braucht", "möchte", "möchten", "bitte", "geben",
            # Englisch
            "pick", "grab", "get", "take", "fetch", "bring", "need", "needs",
            "want", "wants", "give", "hand", "pass", "find", "bring", "please",
            "could", "would", "should", "like",
        }
        _PREPS    = {
            # Deutsch
            "nach", "auf", "mit", "vor", "für", "von", "zu", "beim", "bitte",
            "der", "die", "das", "dem", "den", "einer", "einem", "einen", "mir",
            "ich", "du", "er", "sie", "wir", "ihr",
            # Englisch
            "up", "at", "with", "from", "to", "for", "the", "a", "an", "in",
            "on", "of", "me", "i", "you", "we", "us", "my", "your", "our",
        }

        words = prompt.strip().split()
        color = shape = material = ""
        remaining = []

        for w in words:
            wl = w.lower()
            if wl in _VERBS or wl in _PREPS:
                continue
            elif wl in _COLORS and not color:
                color = wl
            elif wl in _SHAPES and not shape:
                shape = wl
            elif wl in _MATERIALS and not material:
                material = wl
            else:
                remaining.append(w)

        object_name = " ".join(remaining).strip().lower() or prompt.lower()
        return PromptElements(
            object_name=object_name,
            color=color,
            shape=shape,
            material=material,
        )

    @staticmethod
    def _extract_object_name_heuristic(prompt: str) -> str:
        """Compatibility wrapper for external callers."""
        return OSCARPlusPipeline._extract_prompt_elements_heuristic(prompt).object_name

    def _create_summary(self, results: dict) -> dict:
        """Builds a compact summary of the pipeline results."""
        summary = {
            "timestamp": datetime.now().isoformat(),
        }

        if "localization" in results and results["localization"]:
            loc = results["localization"]
            summary["object_detected"] = True
            summary["detection_confidence"] = loc.confidence
            summary["prompt"] = loc.prompt

        if "fusion" in results and results["fusion"].best_match:
            best = results["fusion"].best_match
            summary["best_model"] = best.object_id
            summary["fusion_score"] = best.fused_score
            summary["fusion_method"] = results["fusion"].method

        if "pose_estimation" in results:
            pose = results["pose_estimation"]
            summary["pose_confidence"] = pose.confidence
            summary["pose_method"] = pose.method
            summary["translation"] = pose.translation.tolist()

        if "timing" in results:
            summary["total_time_s"] = results["timing"].get("total", 0)

        return summary

    def _write_ranking_csvs(self, results: dict) -> None:
        """Write per-step ranking CSVs to output_dir for post-hoc analysis."""

        def _write(filename, fieldnames, rows):
            path = os.path.join(self.output_dir, filename)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            logger.info("  [CSV] %s (%d rows)", filename, len(rows))

        # --- CLIP ---
        if "clip_retrieval" in results:
            cands = results["clip_retrieval"].candidates
            _write("rankings_clip.csv",
                   ["rank", "object_id", "score", "description"],
                   [{"rank": i + 1, "object_id": c.object_id,
                     "score": round(float(c.score), 6),
                     "description": getattr(c, "description", "")}
                    for i, c in enumerate(cands)])

        # --- DINOv2 ---
        if "dino_reranking" in results:
            cands = results["dino_reranking"].candidates
            _write("rankings_dino.csv",
                   ["rank", "object_id", "dino_score", "clip_score", "best_view_path"],
                   [{"rank": i + 1, "object_id": c.object_id,
                     "dino_score": round(float(c.dino_score), 6),
                     "clip_score": round(float(getattr(c, "clip_score", 0.0)), 6),
                     "best_view_path": getattr(c, "best_view_path", "")}
                    for i, c in enumerate(cands)])

        # --- ULIP-2 ---
        if "shape_matching" in results:
            cands = results["shape_matching"].candidates
            _write("rankings_ulip.csv",
                   ["rank", "object_id", "shape_score", "best_view_idx",
                    "registration_fitness", "registration_rmse", "cad_model_path"],
                   [{"rank": i + 1, "object_id": c.object_id,
                     "shape_score": round(float(c.shape_score), 6),
                     "best_view_idx": getattr(c, "best_view_idx", -1),
                     "registration_fitness": round(float(getattr(c, "registration_fitness", 0.0)), 6),
                     "registration_rmse": round(float(getattr(c, "registration_rmse", 0.0)), 8),
                     "cad_model_path": getattr(c, "cad_model_path", "")}
                    for i, c in enumerate(cands)])

        # --- Fusion ---
        if "fusion" in results:
            cands = results["fusion"].candidates
            method = getattr(results["fusion"], "method", "")
            _write("rankings_fusion.csv",
                   ["rank", "object_id", "fused_score", "clip_score",
                    "dino_score", "ulip_score", "fusion_method", "cad_model_path"],
                   [{"rank": i + 1, "object_id": c.object_id,
                     "fused_score": round(float(c.fused_score), 6),
                     "clip_score": round(float(getattr(c, "clip_score", 0.0)), 6),
                     "dino_score": round(float(getattr(c, "dino_score", 0.0)), 6),
                     "ulip_score": round(float(getattr(c, "ulip_score", 0.0)), 6),
                     "fusion_method": method,
                     "cad_model_path": getattr(c, "cad_model_path", "")}
                    for i, c in enumerate(cands)])

        # --- B2 Geometry Re-ranking ---
        if "geometry_reranking" in results:
            cands = results["geometry_reranking"].candidates
            _write("rankings_b2_geometry.csv",
                   ["rank", "object_id", "gedi_score", "chamfer_score",
                    "geometry_score", "ransac_fitness", "fused_score", "cad_model_path"],
                   [{"rank": i + 1, "object_id": c.object_id,
                     "gedi_score": round(float(c.gedi_score), 1),
                     "chamfer_score": round(float(c.chamfer_score), 8)
                         if c.chamfer_score < float("inf") else "inf",
                     "geometry_score": round(float(c.geometry_score), 4),
                     "ransac_fitness": round(float(c.ransac_fitness), 6),
                     "fused_score": round(float(c.fused_score), 6),
                     "cad_model_path": getattr(c, "cad_model_path", "")}
                    for i, c in enumerate(cands)])



    def _save_results(self, results: dict) -> None:
        """Saves the pipeline summary as JSON."""
        summary = results.get("summary", {})
        out_path = os.path.join(self.output_dir, "pipeline_result.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Results saved: {out_path}")
        self._write_ranking_csvs(results)


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OSCAR+ pipeline: from the language prompt to the 6D pose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m pipeline.run_pipeline \\
      --rgb scene/rgb/000001.png \\
      --depth scene/depth/000001.png \\
      --prompt "pick up the mustard bottle" \\
      --descriptions object_database/ycbv/descriptions_attributes.json \\
      --reference_images object_images/ycbv/ \\
      --cad_models object_database/ycbv/
        """,
    )
    parser.add_argument("--rgb", required=True, help="path to the RGB image")
    parser.add_argument("--depth", required=True, help="path to the depth image")
    parser.add_argument("--prompt", required=True, help="language prompt, e.g. 'the yellow mustard bottle'")
    parser.add_argument("--gallery", default="",
                        help="gallery name following the standard layout: replaces "
                             "--cad_models/--reference_images/--descriptions with "
                             "<cad_root>/<name>/, <gallery_root>/<name>/ and "
                             "<cad_root>/<name>/descriptions_attributes.json "
                             "(roots from config/paths.yaml). "
                             "To prepare a new gallery: "
                             "python3 preprocess_gallery.py --dataset <name> --step all")
    parser.add_argument("--descriptions", default="", help="path to the descriptions JSON")
    parser.add_argument("--reference_images", default="", help="path to the rendered-views directory")
    parser.add_argument("--cad_models", default="", help="path to the CAD model directory")
    parser.add_argument("--camera", default=None, help="path to scene_camera.json (BOP format)")
    parser.add_argument("--output", default="pipeline_output", help="output directory")
    parser.add_argument("--fusion_method", default="weighted_sum", choices=["weighted_sum", "intersection", "rank_fusion", "majority_voting"])
    parser.add_argument("--pose_method", default="foundationpose",
                        choices=["foundationpose"],
                        help="pose backend (FoundationPose is the only one; a failed call counts as a failure)")
    parser.add_argument("--foundationpose_url", default="http://foundationpose:5050", help="URL of the FoundationPose HTTP service")
    parser.add_argument("--foundationpose_refine_iter", type=int, default=5, help="refinement iterations for FoundationPose register()")
    parser.add_argument("--foundationpose_debug", type=int, default=0, help="FoundationPose debug level (0 = headless)")
    parser.add_argument("--appearance-encoder", choices=["dinov2", "siglip"], default="dinov2",
                        dest="appearance_encoder",
                        help="Appearance encoder for Step 4 re-ranking (ablation E4)")
    parser.add_argument("--shape-encoder", choices=["ulip2", "uni3d"], default="ulip2",
                        dest="shape_encoder",
                        help="Shape encoder for Step 5 matching (ablation E7)")
    parser.add_argument("--num-views", type=int, default=42, dest="num_views",
                        help="Number of rendered views per object to use (ablation O4: 8/16/42)")
    parser.add_argument("--clip_top_k", type=int, default=20)
    parser.add_argument("--dino_top_k", type=int, default=5)
    parser.add_argument("--ulip_top_k", type=int, default=5)
    parser.add_argument("--ulip_repo", default="", help="path to the cloned ULIP repository")
    parser.add_argument("--ulip_checkpoint", default="", help="path to the ULIP-2 checkpoint (.pt)")
    parser.add_argument(
        "--ulip_mode", default="cross", choices=["pc", "cross", "both"],
        help="ULIP-2 retrieval mode: 'pc' (PC->PC), 'cross' (image->PC, default), 'both' (weighted mix)"
    )
    parser.add_argument(
        "--ulip_image_weight", type=float, default=0.5,
        help="weight of the image embedding in mode 'both' (PC weight = 1 - w)."
    )
    parser.add_argument(
        "--ulip-partial-views", action="store_true", dest="ulip_partial_views",
        help="Use precomputed partial point clouds per view instead of full mesh sampling"
    )
    # Geometrischer Check (Schritt 7, dGeDi)
    parser.add_argument("--geometry-reranking", action="store_true", dest="geometry_reranking_enabled",
                        help="enable step 7: dGeDi re-ranking of the fused shortlist")
    parser.add_argument("--geometry-reranking-signal",
                        choices=["fitness", "chamfer_unaligned",
                                 "chamfer_ransac", "chamfer_icp",
                                 # legacy aliases (historisch; Abbildung in step7_geometry_reranking)
                                 "gedi", "chamfer", "both"],
                        default="chamfer_ransac", dest="geometry_reranking_signal",
                        help="Geometry signal for B2 re-ranking. "
                             "'chamfer_unaligned' is a diagnostic control and "
                             "should not be used in production.")
    parser.add_argument("--geometry-reranking-top-k", type=int, default=5,
                        dest="geometry_reranking_top_k",
                        help="Number of fused candidates to re-rank in B2")

    # Rotation evaluation
    parser.add_argument("--ulip-rotation-eval", action="store_true", dest="ulip_rotation_eval",
                        help="Run ICP rotation evaluation for ULIP Top-K candidates")
    parser.add_argument("--ulip-rotation-eval-top-k", type=int, default=5, dest="ulip_rotation_eval_top_k")
    parser.add_argument("--ulip-rotation-eval-weight", type=float, default=0.0, dest="ulip_rotation_eval_weight",
                        help="Rerank weight for ICP fitness (0.0 = debug-only)")

    parser.add_argument("--skip_steps", type=int, nargs="*", default=[], help="skip steps (e.g. --skip_steps 5 8)")
    parser.add_argument("--until-step", type=int, default=8, dest="until_step",
                        help="run the pipeline up to and including step N (1-8, default: 8)")
    parser.add_argument("--gt-bbox-compensation", action="store_true", dest="gt_bbox_compensation",
                        help="Enable bbox-center compensation for GT wireframe overlay (default: off)")
    return parser.parse_args()


def main():
    """Main entry point for CLI execution."""
    args = parse_args()

    # --gallery <name>: Kurzform fuer das Standard-Layout einer Gallery. Die
    # Wurzeln kommen aus config/paths.yaml, die Pfade sind damit absolut — kein
    # bestimmtes Arbeitsverzeichnis noetig.
    if args.gallery:
        from .paths import resolve as _resolve_paths
        _p = _resolve_paths()
        g = args.gallery
        args.cad_models = args.cad_models or os.path.join(_p["cad_root"], g)
        args.reference_images = args.reference_images or os.path.join(
            _p["gallery_root"], g)
        args.descriptions = args.descriptions or os.path.join(
            _p["cad_root"], g, "descriptions_attributes.json")
    missing = [f for f, v in [("--cad_models", args.cad_models),
                              ("--reference_images", args.reference_images),
                              ("--descriptions", args.descriptions)] if not v]
    if missing:
        raise SystemExit(
            f"Missing {', '.join(missing)} — either pass them directly or "
            f"derive them from the standard layout via --gallery <name> "
            f"(<cad_root>/<name>/ + <gallery_root>/<name>/).")
    for label, p in [("--cad_models", args.cad_models),
                     ("--reference_images", args.reference_images),
                     ("--descriptions", args.descriptions)]:
        if not os.path.exists(p):
            raise SystemExit(
                f"{label}: {p} does not exist. Prepare a new gallery:\n"
                f"  python3 preprocess_gallery.py --dataset <name> "
                f"[--cad-dir <dir>] --step all")
    # ULIP-Standardpfade des Containers, wenn nicht gesetzt.
    if not args.ulip_repo and os.path.isdir("/ulip"):
        args.ulip_repo = "/ulip"
        args.ulip_checkpoint = (args.ulip_checkpoint
                                or "/ulip/checkpoints/ulip2_pointbert_10k.pt")

    # --- Config aufbauen ---
    config = PipelineConfig(
        appearance_encoder=args.appearance_encoder,
        shape_encoder=args.shape_encoder,
        description_file=args.descriptions,
        reference_images_dir=args.reference_images,
        cad_models_dir=args.cad_models,
        output_dir=args.output,
        fusion_method=args.fusion_method,
        pose_method=args.pose_method,
        foundationpose_url=args.foundationpose_url,
        foundationpose_est_refine_iter=args.foundationpose_refine_iter,
        foundationpose_debug=args.foundationpose_debug,
        num_views=args.num_views,
        clip_top_k=args.clip_top_k,
        dino_top_k=args.dino_top_k,
        ulip2_top_k=args.ulip_top_k,
        ulip_repo_path=args.ulip_repo,
        ulip2_checkpoint=args.ulip_checkpoint,
        ulip2_mode=args.ulip_mode,
        ulip2_image_weight=args.ulip_image_weight,
        ulip2_use_partial_views=args.ulip_partial_views,
        ulip2_rotation_eval=args.ulip_rotation_eval,
        ulip2_rotation_eval_top_k=args.ulip_rotation_eval_top_k,
        ulip2_rotation_eval_weight=args.ulip_rotation_eval_weight,
        geometry_reranking_enabled=args.geometry_reranking_enabled,
        geometry_reranking_signal=args.geometry_reranking_signal,
        geometry_reranking_top_k=args.geometry_reranking_top_k,
        gt_bbox_center_compensation=args.gt_bbox_compensation,
    )

    # --- Bilder laden ---
    logger.info(f"Loading RGB: {args.rgb}")
    rgb_image = Image.open(args.rgb).convert("RGB")

    # --- Kameraintrinsics ---
    camera_intrinsics = None
    if args.camera:
        from .utils import load_camera_intrinsics
        # Image-ID aus dem RGB-Dateinamen ableiten (z.B. "000001.png" → 1)
        image_id = int(os.path.splitext(os.path.basename(args.rgb))[0])
        camera_intrinsics = load_camera_intrinsics(args.camera, image_id=image_id)

    logger.info(f"Loading depth: {args.depth}")
    depth_image = np.array(Image.open(args.depth)).astype(np.float32)

    # Determine depth_scale: prefer BOP scene_camera.json, fall back to config
    # BOP convention: raw * depth_scale = mm → raw * depth_scale / 1000 = meters
    # Config convention: raw / config.depth_scale = meters
    if camera_intrinsics and camera_intrinsics.get("depth_scale", 0) > 0:
        bop_ds = camera_intrinsics["depth_scale"]
        depth_image = depth_image * bop_ds / 1000.0
        logger.info("Depth: BOP depth_scale=%.4f → raw * %.4f / 1000 = meters", bop_ds, bop_ds)
    else:
        depth_image = depth_image / config.depth_scale
        logger.info("Depth: config depth_scale=%.1f → raw / %.1f = meters", config.depth_scale, config.depth_scale)

    # --- until_step → skip_steps ---
    skip_steps = list(args.skip_steps)
    if args.until_step < 8:
        skip_steps = sorted(set(skip_steps) | set(range(args.until_step + 1, 9)))

    # --- Pipeline ausführen ---
    pipeline = OSCARPlusPipeline(config)
    pipeline.initialize()
    result = pipeline.run(
        rgb_image=rgb_image,
        depth_image=depth_image,
        prompt=args.prompt,
        camera_intrinsics=camera_intrinsics,
        skip_steps=skip_steps,
    )

    # --- Zusammenfassung ausgeben ---
    summary = result.get("summary", {})
    print("\n" + "=" * 60)
    print("PIPELINE RESULT")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
