# =============================================================================
# pipeline/step5_shape_matching.py – Thesis Step B1: Shape Channel S_shape
# =============================================================================
#
# Computes the shape score S_shape (thesis Sec. 3.3, Step B1).
#
# Default encoder: ULIP-2 (Xue et al., CVPR 2024) — frozen multimodal
# point-cloud encoder. Pre-trained on complete 10k-point clouds; partial
# query clouds are upsampled with replacement (Gaussian jitter follows
# standard point-cloud augmentation practice, Qi et al., 2017 — PointNet).
#
# The frozen PointBERT backbone (Yu et al., 2022) gives competitive 3D
# retrieval descriptors out of the box (van den Herrewegen et al., 2024).
# ULIP-2 achieves 50.6% top-1 on Objaverse-LVIS and 84.7% on ModelNet40
# without task-specific training (Xue et al., 2024).
#
# Partial reference point clouds per CAD view enable partial-to-partial
# comparison, mitigating the ambiguity of partial-to-full matching
# noted by U-RED (Di et al., 2023).
#
# Alternative encoder (ablation E7):
#   • Uni3D (Zhou et al., ICLR 2024) — unified 3D encoder aligned with
#     EVA-CLIP embedding space. PC-only mode (no cross-modal branch).
#
# Architecture (Colored PointBERT, 10k points):
#   PointTransformer_Colored(xyzrgb) → 768-dim → pc_projection → 1280-dim
#   Final embedding shares the OpenCLIP ViT-bigG-14 space (cross-modal).
#
# Efficient loading: only point_encoder + pc_projection loaded from
# checkpoint (~400 MB), not the full OpenCLIP ViT-bigG-14 (~5 GB).
#
# Known limitation: point-cloud rotation sensitivity is a general property
# of networks trained on roughly aligned shapes (Yu et al., 2020).
#
# Refs:
#   ULIP-2: https://github.com/salesforce/ULIP (Xue et al., 2024)
#   Point-BERT: (Yu et al., 2022)
#   Uni3D: https://github.com/baaivision/Uni3D (Zhou et al., 2024)
# =============================================================================

import logging
import os
import sys
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PipelineConfig
from .step2_pointcloud import PointCloudResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-view aggregation (inspired by OPEN, Chu et al. TCSVT 2024)
# ---------------------------------------------------------------------------

def _aggregate_view_scores(
    scores: torch.Tensor,
    method: str = "topk_softmax",
    top_k: int = 8,
    temperature: float = 1.0,
) -> Tuple[float, int]:
    """Aggregate per-view similarity scores into a single object score.

    See step4_dino_reranking._aggregate_view_scores for full docstring.

    Returns:
        (aggregated_score, best_view_index)
    """
    best_idx = scores.argmax().item()

    if len(scores) <= 1 or method == "max":
        return scores[best_idx].item(), best_idx

    if method == "mean":
        return scores.mean().item(), best_idx

    if method == "topk_softmax":
        k = min(top_k, len(scores))
        topk_vals, _ = scores.topk(k)
        weights = torch.softmax(topk_vals / temperature, dim=0)
        return (weights * topk_vals).sum().item(), best_idx

    if method == "softmax":
        weights = torch.softmax(scores / temperature, dim=0)
        return (weights * scores).sum().item(), best_idx

    return scores[best_idx].item(), best_idx


# ---------------------------------------------------------------------------
# Datenstruktur für Shape-Matching-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class ShapeCandidate:
    """A single shape-match candidate.

    Attributes:
        object_id: Identifier of the CAD model.
        shape_score: ULIP-2 cosine similarity.
        cad_model_path: Path to the CAD model.
    """
    object_id: str
    shape_score: float
    cad_model_path: str = ""
    best_view_idx: int = -1   # index of best matching partial view (-1 = full mesh)
    best_partial_pc_path: str = ""  # path to best matching partial PC .npz
    registration_fitness: float = 0.0  # ICP fitness (0..1), populated by rotation eval
    registration_rmse: float = 0.0     # ICP inlier RMSE, populated by rotation eval


@dataclass
class ShapeMatchingResult:
    """Result of the shape matching (step 5).

    Attributes:
        candidates: List of the top-K shape matches, sorted by score.
        query_embedding: ULIP-2 embedding of the observed point cloud.
    """
    candidates: List[ShapeCandidate]
    query_embedding: np.ndarray


# ---------------------------------------------------------------------------
# Hilfsfunktionen für Punktwolkenverarbeitung
# ---------------------------------------------------------------------------

def normalize_pointcloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    num_points: int = 10000,
    jitter_std: float = 0.001,
) -> np.ndarray:
    """Normalize and resample a point cloud for ULIP-2 / PointBERT input.

    Preprocessing pipeline (thesis Sec. 3.3, Step B1):
      1. Center on centroid (zero-mean) — XYZ only.
      2. Scale to unit sphere (max distance = 1) — XYZ only.
      3. Resample to fixed point count (10,000 for ULIP-2 PointBERT).
         When upsampling (N < num_points), sampling with replacement is
         used and duplicated points receive per-point Gaussian jitter
         (thesis Sec. 3.5.2) to prevent coincident duplicates that would
         collapse PointBERT's FPS + kNN local-neighbourhood groupings.
         Gaussian jitter is a standard point-cloud augmentation
         (Qi et al., 2017 — PointNet) also applied during Point-BERT
         and ULIP-2 pretraining, keeping the input within the encoder's
         training distribution.
      4. Optionally append RGB colors → (N, 6).

    ULIP-2 PointBERT Colored expects 10,000 points with 6 channels (xyzrgb).

    Args:
        points: Point cloud (N, 3) — XYZ coordinates.
        colors: Optional colors (N, 3) — RGB in [0, 1].
                If None and colored encoder is used, zeros are substituted.
        num_points: Target point count.
        jitter_std: Standard deviation of per-point Gaussian noise applied
                    to duplicated points during upsampling (thesis Sec. 3.5.2).
                    Default 0.001 (0.1% of unit-sphere radius). Set to 0 to
                    disable jitter.

    Returns:
        Normalized point cloud as (num_points, 3) or (num_points, 6).
    """
    if len(points) == 0:
        dim = 6 if colors is not None else 3
        return np.zeros((num_points, dim), dtype=np.float32)

    # --- Centering ---
    centroid = points.mean(axis=0)
    points_centered = points - centroid

    # --- Unit-sphere scaling ---
    max_dist = np.linalg.norm(points_centered, axis=1).max()
    if max_dist > 0:
        points_centered = points_centered / max_dist

    # --- Resampling to fixed point count ---
    # Deterministic seed derived from point cloud content (thesis Sec. 3.7:
    # global seed = 42 for reproducibility; content-hash achieves the same
    # per-cloud determinism without requiring call-site seed management).
    content_hash = hash(points_centered.tobytes()) & 0xFFFFFFFF
    rng = np.random.RandomState(content_hash)
    n = len(points_centered)
    if n >= num_points:
        # Downsample: uniform random subset (no replacement)
        indices = rng.choice(n, num_points, replace=False)
        result_xyz = points_centered[indices].astype(np.float32)
    else:
        # Upsample: sample with replacement + Gaussian jitter on duplicates
        # (thesis Sec. 3.5.2; Qi et al., 2017 — PointNet augmentation).
        indices = rng.choice(n, num_points, replace=True)
        result_xyz = points_centered[indices].astype(np.float32)
        if jitter_std > 0:
            # Identify duplicated points: any index that appears more than
            # once contributes at least one duplicate copy.  We jitter ALL
            # upsampled points (not just the extra copies) because with heavy
            # upsampling (e.g. 800 → 10k) almost every sample is a duplicate
            # and selective masking would be fragile.  The jitter magnitude
            # (0.1% of unit sphere) is small enough that the single original
            # occurrence of each point is negligibly perturbed.
            noise = rng.normal(0.0, jitter_std, size=result_xyz.shape).astype(np.float32)
            result_xyz += noise

    if colors is not None:
        if colors.max() > 1.0:
            colors = colors / 255.0
        result_rgb = colors[indices].astype(np.float32)
        return np.concatenate([result_xyz, result_rgb], axis=1)  # (N, 6)

    return result_xyz  # (N, 3)


def _sample_face_colors(mesh, face_indices) -> Optional[np.ndarray]:
    """Colour per sampled point, from the best available source.

    Three cases, in this order:

    1. ``face_colors`` directly — meshes with face colours (BOP PLYs).
    2. **Textured meshes** (SHREC'18, GSO, YCB-V): the colour lives in an
       image file plus UV coordinates, and ``face_colors`` is ``None`` there.
       ``to_color()`` samples the texture at the UVs but only returns
       ``vertex_colors`` — the three corner colours of each hit triangle are
       averaged here. Without this path the coloured ULIP-2 backbone would get
       zeros instead of colour for exactly these datasets.
    3. No colour carrier at all (T-LESS, ITODD, MI3DOR — uniform grey):
       ``None``, and the caller substitutes zeros. That is a property of the
       files, not of the sampler.
    """
    fc = getattr(mesh.visual, "face_colors", None)
    if fc is not None:
        return (np.asarray(fc)[face_indices][:, :3] / 255.0).astype(np.float32)

    if type(mesh.visual).__name__ == "TextureVisuals":
        try:
            vc = getattr(mesh.visual.to_color(), "vertex_colors", None)
            if vc is not None:
                vcol = np.asarray(vc)[:, :3]
                tri = mesh.faces[face_indices]           # (n, 3) Vertexindizes
                return (vcol[tri].mean(axis=1) / 255.0).astype(np.float32)
        except Exception:
            pass
    return None


def sample_pointcloud_from_mesh(
    mesh_path: str,
    num_points: int = 10000,
    with_colors: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Samples a point cloud from the surface of a 3D mesh.

    Supports OBJ, PLY, GLB and other formats supported by
    Open3D/Trimesh.

    Args:
        mesh_path: Path to the 3D model.
        num_points: Number of points to sample.
        with_colors: Extract vertex colours (if present).

    Returns:
        Tuple (points, colors):
            points: (num_points, 3) numpy array.
            colors: (num_points, 3) numpy array in [0,1], or None.
    """
    colors = None
    try:
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
        points = points.astype(np.float32)

        if with_colors:
            colors = _sample_face_colors(mesh, face_indices)

        return points, colors

    except ImportError:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_triangles():
            logger.warning(f"Mesh {mesh_path} has no triangles.")
            return np.zeros((num_points, 3), dtype=np.float32), None
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(pcd.points, dtype=np.float32)

        if with_colors and pcd.has_colors():
            raw = np.asarray(pcd.colors)
            colors = np.clip(raw, 0.0, 1.0).astype(np.float32)

        return points, colors


# ---------------------------------------------------------------------------
# Leichtgewichtiger ULIP-2 Point-Cloud-Encoder Wrapper
# ---------------------------------------------------------------------------

class ULIP2PointEncoder(nn.Module):
    """Lightweight wrapper around the ULIP-2 point-cloud encoder.

    Loads ONLY the PointBERT/PointNeXt encoder + pc_projection from the
    ULIP-2 checkpoint, without the whole OpenCLIP ViT-bigG-14 model.

    Memory footprint: ~400 MB instead of ~5.5 GB.

    Ref: "ULIP-2: Towards Scalable Multimodal Pre-training for 3D
          Understanding" (Xue et al., CVPR 2024)
    """

    def __init__(self, point_encoder: nn.Module, pc_projection: nn.Parameter):
        super().__init__()
        self.point_encoder = point_encoder
        self.pc_projection = pc_projection

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        """Encodes a point cloud into a ULIP-2 embedding.

        Args:
            pc: Point cloud (B, N, C) with C=3 (xyz) or C=6 (xyzrgb).

        Returns:
            Normalised embedding (B, embed_dim).
        """
        pc_feat = self.point_encoder(pc)             # (B, 768)
        pc_embed = pc_feat @ self.pc_projection      # (B, embed_dim)
        pc_embed = F.normalize(pc_embed, p=2, dim=1)
        return pc_embed


# ---------------------------------------------------------------------------
# ULIP-2 Image Encoder (OpenCLIP ViT-bigG-14) für Cross-Modal Retrieval
# ---------------------------------------------------------------------------

class ULIP2ImageEncoder:
    """OpenCLIP ViT-bigG-14 image encoder for ULIP-2 cross-modal retrieval.

    ULIP-2 freezes the image encoder during training, so vanilla OpenCLIP
    ViT-bigG-14 weights (laion2b_s39b_b160k) are identical to the ULIP-2
    image branch — no separate download needed.

    Cross-modal: image_embedding and pc_embedding live in the same 1280-dim
    space → cosine similarities are directly comparable.

    Ref: "ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding"
         Xue et al., CVPR 2024
    """

    def __init__(self, device: str = "cpu", checkpoint_path: Optional[str] = None):
        self.device = device
        self._checkpoint_path = checkpoint_path
        self.model: Optional[nn.Module] = None
        self.preprocess = None

    def load(self) -> None:
        """Loads OpenCLIP ViT-bigG-14 (lazy init)."""
        if self.model is not None:
            return

        try:
            import open_clip  # type: ignore
        except ImportError:
            raise ImportError(
                "open_clip is not installed.\n"
                "  pip install open-clip-torch"
            )

        logger.info(
            "Loading OpenCLIP ViT-bigG-14 image encoder "
            "(for ULIP-2 cross-modal retrieval)..."
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
        )

        # Optional: visuelle Gewichte aus ULIP-2 Checkpoint laden.
        # Da ULIP-2 den Image-Encoder einfriert, sind vanilla Gewichte
        # identisch — dieser Block ist nur zur Vollständigkeit.
        if self._checkpoint_path and os.path.isfile(self._checkpoint_path):
            try:
                ckpt = torch.load(
                    self._checkpoint_path, map_location="cpu", weights_only=False
                )
                sd = ckpt.get("state_dict", ckpt)
                sd = OrderedDict({k.replace("module.", ""): v for k, v in sd.items()})
                visual_sd = {k[len("visual."):]: v for k, v in sd.items()
                             if k.startswith("visual.")}
                if visual_sd:
                    res = model.visual.load_state_dict(visual_sd, strict=False)
                    logger.info(
                        f"Visual weights loaded from checkpoint "
                        f"(missing={len(res.missing_keys)}, "
                        f"unexpected={len(res.unexpected_keys)})"
                    )
                else:
                    logger.info(
                        "No visual.* keys in the checkpoint — "
                        "using vanilla OpenCLIP (expected, ULIP-2 freezes the image encoder)."
                    )
            except Exception as exc:
                logger.warning(
                    f"Could not load visual weights: {exc}. "
                    "Using vanilla OpenCLIP."
                )

        self.model = model.visual.to(self.device)
        self.model.eval()
        self.preprocess = preprocess

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"OpenCLIP ViT-bigG-14 image encoder loaded: "
            f"{n_params / 1e6:.1f}M parameters, device: {self.device}"
        )

    def encode(self, image) -> torch.Tensor:
        """Encodes a PIL image or numpy array into a ULIP-2-compatible embedding.

        Args:
            image: PIL.Image or numpy array (H, W, 3) uint8.

        Returns:
            L2-normalised tensor (1, 1280).
        """
        self.load()
        from PIL import Image as PILImage  # type: ignore

        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image.astype(np.uint8))
        elif not isinstance(image, PILImage.Image):
            raise TypeError(
                f"Expected PIL.Image or numpy.ndarray, got {type(image)}"
            )

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)  # (1, 1280)
            embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding


# ---------------------------------------------------------------------------
# Uni3D Point-Cloud Encoder (ablation E7)
# ---------------------------------------------------------------------------

class Uni3DEncoder(nn.Module):
    """Uni3D point cloud encoder (Zhou et al., ICLR 2024).

    Encodes point clouds into EVA-CLIP-aligned embeddings. Loaded from
    HuggingFace (BAAI/Uni3D) or a local checkpoint.

    Ref: https://github.com/baaivision/Uni3D
         https://huggingface.co/BAAI/Uni3D

    Unlike ULIP-2, Uni3D does NOT require cloning a separate repo or
    compiling CUDA extensions — it loads entirely via torch.hub / HF.
    """

    def __init__(self, config: "PipelineConfig"):
        super().__init__()
        self.config = config
        self.device = config.device
        self._model = None
        self._embed_dim = getattr(config, "uni3d_embed_dim", 1024)

    def _load(self):
        if self._model is not None:
            return
        import sys
        import types

        repo = getattr(self.config, "uni3d_repo_path", "/uni3d")
        ckpt = getattr(self.config, "uni3d_checkpoint",
                       "/uni3d/modelzoo/uni3d-g/model.pt")
        if not os.path.isfile(ckpt):
            raise RuntimeError(
                f"Uni3D checkpoint not found: {ckpt}. Download it from "
                f"https://huggingface.co/BAAI/Uni3D (modelzoo/uni3d-g/model.pt) "
                f"and place it under {os.path.dirname(ckpt)}.")

        logger.info("Building Uni3D (%s) from %s ...",
                    getattr(self.config, "uni3d_pc_model", "?"), repo)

        # The Uni3D repo and the ULIP repo both expose a top-level ``models``
        # package. In the precompute both encoders run in one process, so the
        # first-imported ``models`` would shadow the other. Isolate this import:
        # evict any cached ``models*`` (e.g. ULIP's), import Uni3D's, then
        # restore — the built model keeps its class references either way.
        saved = {k: v for k, v in sys.modules.items()
                 if k == "models" or k.startswith("models.")}
        for k in saved:
            del sys.modules[k]
        sys.path.insert(0, repo)
        try:
            from models.uni3d import create_uni3d
            args = types.SimpleNamespace(
                pc_model=getattr(self.config, "uni3d_pc_model",
                                 "eva_giant_patch14_560"),
                pretrained_pc="",          # EVA weights come from the full ckpt
                drop_path_rate=0.0,
                pc_feat_dim=getattr(self.config, "uni3d_pc_feat_dim", 1408),
                pc_encoder_dim=getattr(self.config, "uni3d_pc_encoder_dim", 512),
                embed_dim=self._embed_dim,
                group_size=getattr(self.config, "uni3d_group_size", 64),
                num_group=getattr(self.config, "uni3d_num_group", 512),
                patch_dropout=0.0,
            )
            model = create_uni3d(args)
        finally:
            try:
                sys.path.remove(repo)
            except ValueError:
                pass
            for k in [k for k in sys.modules
                      if k == "models" or k.startswith("models.")]:
                del sys.modules[k]
            sys.modules.update(saved)

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        for key in ("module", "state_dict", "model"):
            if isinstance(state, dict) and key in state \
                    and isinstance(state[key], dict):
                state = state[key]
                break
        sd = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
        res = model.load_state_dict(sd, strict=False)
        # Only the point-cloud branch is used at inference; missing keys for the
        # unused image head are expected. A missing point_encoder.* key is not.
        pe_missing = [k for k in res.missing_keys if k.startswith("point_encoder")]
        if pe_missing:
            raise RuntimeError(
                f"Uni3D load: {len(pe_missing)} point_encoder weights missing "
                f"(e.g. {pe_missing[:3]}) — checkpoint/arch mismatch.")
        logger.info("Uni3D checkpoint loaded (missing=%d, unexpected=%d)",
                    len(res.missing_keys), len(res.unexpected_keys))

        self._model = model.to(self.device).eval()
        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info("Uni3D loaded: %.1fM params, embed_dim=%d, device=%s",
                    n_params / 1e6, self._embed_dim, self.device)

    def encode(self, points: np.ndarray, colors: Optional[np.ndarray] = None) -> torch.Tensor:
        """Encode a point cloud into a Uni3D embedding.

        Uni3D consumes xyz+rgb (6 channels): its patch embedder concatenates
        per-point color to the grouped geometry, so colors are required —
        zeros are substituted when the source cloud has none (matching the
        Uni3D data pipeline, which fills missing color with a constant).

        Returns an L2-normalized tensor (1, embed_dim).
        """
        self._load()

        num_points = getattr(self.config, "uni3d_num_points", 10000)
        if colors is None:
            colors = np.zeros((len(points), 3), dtype=np.float32)
        pts_norm = normalize_pointcloud(points, colors=colors,
                                        num_points=num_points)  # (num_points, 6)
        pts_tensor = torch.from_numpy(pts_norm).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            embedding = self._model.encode_pc(pts_tensor)  # (1, embed_dim)
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding


# ---------------------------------------------------------------------------
# ULIP-2 Shape Matching Modul
# ---------------------------------------------------------------------------

class ShapeMatcher:
    """Shape matching via ULIP-2 point-cloud embeddings.

    Compares the partial point cloud of the observed object with the
    complete CAD-model point clouds in the ULIP-2 embedding space.

    ULIP-2 maps point clouds into the same embedding space as OpenCLIP
    images and texts (1280-dim for ViT-bigG-14). This also enables
    cross-modal comparisons.

    Model architecture:
        PointTransformer_Colored (PointBERT with 6D input):
        - Input: (B, 10000, 6) xyzrgb
        - Backbone: 18-layer Transformer, 384-dim, 6 heads
        - Grouping: 512 groups × 32 points (FPS + kNN)
        - Output: 768-dim → pc_projection → 1280-dim

    Usage:
        >>> config = PipelineConfig(
        ...     ulip_repo_path="/path/to/ULIP",
        ...     ulip2_checkpoint="/path/to/checkpoint.pt",
        ... )
        >>> matcher = ShapeMatcher(config)
        >>> matcher.load_cad_models("object_database/ycbv/")
        >>> result = matcher.match(point_cloud_result, top_k=5)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self._encoder_type = getattr(config, "shape_encoder", "ulip2")
        self.model: Optional[ULIP2PointEncoder] = None
        self.image_encoder: Optional[ULIP2ImageEncoder] = None
        self._uni3d: Optional[Uni3DEncoder] = None

        # Gecachte CAD-Modell-Embeddings
        self._cad_embeddings: Dict[str, torch.Tensor] = {}
        self._cad_paths: Dict[str, str] = {}
        self._partial_mode: bool = False  # True when partial view embeddings are loaded
        # Partial-view .npz paths per object: {obj_id: [(view_idx, npz_path), ...]}
        self._partial_view_paths: Dict[str, List[Tuple[int, str]]] = {}

    def _collect_mesh_items(
        self,
        cad_dir: str,
        allowed_extensions: Tuple[str, ...],
    ) -> List[Tuple[str, str]]:
        """Collects (object_id, mesh_path) pairs from the CAD directory."""
        items: List[Tuple[str, str]] = []

        for entry in sorted(os.listdir(cad_dir)):
            entry_path = os.path.join(cad_dir, entry)

            if os.path.isfile(entry_path):
                ext = os.path.splitext(entry)[1].lower()
                if ext in allowed_extensions:
                    items.append((os.path.splitext(entry)[0], entry_path))
                continue

            if os.path.isdir(entry_path):
                mesh_file = self._find_mesh_in_dir(entry_path, allowed_extensions)
                if mesh_file:
                    items.append((entry, mesh_file))

        return items

    def _get_cache_path(
        self,
        cad_dir: str,
        mesh_items: List[Tuple[str, str]],
    ) -> str:
        """Builds a stable cache path for CAD embeddings."""
        if self._encoder_type == "uni3d":
            ckpt_tag = "uni3d"
            meta_parts = [
                f"encoder=uni3d",
                f"model={self.config.uni3d_model_name}",
                f"npts={self.config.uni3d_num_points}",
                f"edim={self.config.uni3d_embed_dim}",
            ]
        else:
            ckpt_tag = os.path.basename(self.config.ulip2_checkpoint or "no_ckpt")
            meta_parts = [
                f"backbone={self.config.ulip2_backbone}",
                f"npts={self.config.ulip2_num_points}",
                f"colors={int(self.config.ulip2_use_colors)}",
                f"edim={self.config.ulip2_embed_dim}",
                f"ckpt={ckpt_tag}",
            ]

        inv = []
        for obj_id, path in mesh_items:
            try:
                st = os.stat(path)
                # size only (no mtime) → fingerprint is stable across
                # machines so a cache precomputed elsewhere is reused here.
                inv.append(
                    f"{obj_id}|{os.path.relpath(path, cad_dir)}|{st.st_size}"
                )
            except OSError:
                inv.append(f"{obj_id}|{os.path.relpath(path, cad_dir)}|missing")

        raw = "\n".join(meta_parts + sorted(inv))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        fname = f".ulip_cache_{digest}.pt"
        return os.path.join(cad_dir, fname)

    def _try_load_cache(self, cache_path: str) -> bool:
        """Loads CAD embeddings from cache, if present and valid."""
        if not os.path.isfile(cache_path):
            return False

        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            emb = payload.get("embeddings", {})
            paths = payload.get("paths", {})
            if not isinstance(emb, dict) or not isinstance(paths, dict):
                return False

            loaded_emb: Dict[str, torch.Tensor] = {}
            loaded_paths: Dict[str, str] = {}
            for obj_id, tensor in emb.items():
                if torch.is_tensor(tensor):
                    loaded_emb[obj_id] = tensor.detach().cpu()
            for obj_id, path in paths.items():
                if isinstance(path, str):
                    loaded_paths[obj_id] = path

            if not loaded_emb:
                return False

            self._cad_embeddings = loaded_emb
            self._cad_paths = loaded_paths
            logger.info(
                "CAD embeddings loaded from cache: %d models (%s)",
                len(self._cad_embeddings),
                cache_path,
            )
            return True
        except Exception as exc:
            logger.warning("Could not load ULIP CAD cache (%s): %s", cache_path, exc)
            return False

    def _save_cache(self, cache_path: str) -> None:
        """Saves CAD embeddings to disk."""
        try:
            payload = {
                "embeddings": {k: v.detach().cpu() for k, v in self._cad_embeddings.items()},
                "paths": dict(self._cad_paths),
            }
            torch.save(payload, cache_path)
            logger.info("CAD embeddings cache saved: %s", cache_path)
        except Exception as exc:
            logger.warning("Could not save ULIP CAD cache (%s): %s", cache_path, exc)

    def _load_model(self):
        """Loads the ULIP-2 point-cloud encoder.

        Steps:
        1. Add the ULIP repo to sys.path
        2. Import PointTransformer_Colored from the ULIP repo
        3. Build the model architecture from the YAML config
        4. Load only the point_encoder + pc_projection weights from the checkpoint
        5. Wrap in a ULIP2PointEncoder

        Requires:
        - ULIP repo at config.ulip_repo_path
        - Checkpoint at config.ulip2_checkpoint
        - knn_cuda installed
        """
        if self.model is not None:
            return

        checkpoint_path = self.config.ulip2_checkpoint
        ulip_repo_path = self.config.ulip_repo_path

        # --- Validierung ---
        if not ulip_repo_path or not os.path.isdir(ulip_repo_path):
            raise FileNotFoundError(
                f"ULIP repo not found: '{ulip_repo_path}'\n"
                "Set ulip_repo_path in the config to the path of the cloned ULIP repo.\n"
                "  git clone https://github.com/salesforce/ULIP.git"
            )

        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"ULIP-2 checkpoint not found: '{checkpoint_path}'\n"
                "Download from HuggingFace:\n"
                "  https://huggingface.co/datasets/SFXX/ulip/tree/main/ULIP-2/pretrained_models\n"
                "Recommended: ULIP-2-PointBERT-10k-xyzrgb (402 MB)"
            )

        # --- ULIP-Repo in Python-Path einfügen ---
        if ulip_repo_path not in sys.path:
            sys.path.insert(0, ulip_repo_path)
            logger.info(f"ULIP repo added to sys.path: {ulip_repo_path}")

        logger.info("Loading ULIP-2 point-cloud encoder (PointBERT + projection only)...")

        # --- 1. Modell-Architektur erstellen ---
        point_encoder, pc_feat_dims = self._create_point_encoder(ulip_repo_path)

        # --- 2. Checkpoint laden und Gewichte extrahieren ---
        embed_dim = self.config.ulip2_embed_dim  # 1280 für ViT-bigG-14
        pc_projection = nn.Parameter(torch.empty(pc_feat_dims, embed_dim))

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Checkpoint-Format: {'state_dict': {...}, 'args': ..., 'epoch': ...}
        if "state_dict" in ckpt:
            state_dict = OrderedDict()
            for k, v in ckpt["state_dict"].items():
                # Entferne 'module.' Prefix (von DistributedDataParallel)
                state_dict[k.replace("module.", "")] = v
        else:
            state_dict = ckpt

        # --- 3. point_encoder Gewichte extrahieren ---
        pe_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("point_encoder."):
                pe_state_dict[k[len("point_encoder."):]] = v

        if not pe_state_dict:
            raise RuntimeError(
                "No point_encoder.* keys found in the checkpoint.\n"
                f"Available top-level keys: {list(state_dict.keys())[:20]}"
            )

        result = point_encoder.load_state_dict(pe_state_dict, strict=False)
        if result.missing_keys:
            logger.warning(f"Missing keys in point_encoder: {result.missing_keys}")
        if result.unexpected_keys:
            logger.warning(f"Unexpected keys in point_encoder: {result.unexpected_keys}")

        # --- 4. pc_projection Gewichte laden ---
        if "pc_projection" in state_dict:
            pc_projection.data.copy_(state_dict["pc_projection"])
            logger.info(
                f"pc_projection loaded: {pc_projection.shape} "
                f"({pc_feat_dims} → {embed_dim})"
            )
        else:
            logger.warning(
                "pc_projection not found in the checkpoint. "
                "Using random initialisation."
            )
            nn.init.normal_(pc_projection, std=embed_dim ** -0.5)

        # --- 5. Wrapper erstellen ---
        self.model = ULIP2PointEncoder(point_encoder, pc_projection)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Parameter zählen
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"ULIP-2 point-cloud encoder loaded: "
            f"{n_params / 1e6:.1f}M parameters, "
            f"embedding dim: {embed_dim}, "
            f"device: {self.device}"
        )

    def _create_point_encoder(
        self, ulip_repo_path: str
    ) -> Tuple[nn.Module, int]:
        """Builds the point-cloud encoder based on config.ulip2_backbone.

        Supported backbones:
        - "pointbert_colored": PointTransformer_Colored (6D input, ULIP-2 default)
        - "pointbert": PointTransformer (3D input)
        - "pointnext": PointNEXT

        Args:
            ulip_repo_path: Path to the cloned ULIP repo.

        Returns:
            Tuple (point_encoder, pc_feat_dims):
                point_encoder: The PyTorch module.
                pc_feat_dims: Output feature dimension of the encoder.
        """
        from easydict import EasyDict
        from utils.config import cfg_from_yaml_file  # type: ignore  # ULIP utils

        backbone = self.config.ulip2_backbone

        if backbone == "pointbert_colored":
            from models.pointbert.point_encoder import PointTransformer_Colored  # type: ignore

            yaml_path = os.path.join(
                ulip_repo_path,
                "models", "pointbert",
                "ULIP_2_PointBERT_10k_colored_pointclouds.yaml",
            )
            if not os.path.isfile(yaml_path):
                raise FileNotFoundError(
                    f"PointBERT YAML config not found: {yaml_path}"
                )
            config = cfg_from_yaml_file(yaml_path)

            # evaluate_3d=True verhindert, dass PointTransformer versucht,
            # den PointBERT-Pretrained-Checkpoint zu laden (der nicht nötig
            # ist, da wir den vollen ULIP-2 Checkpoint laden)
            dummy_args = EasyDict({"evaluate_3d": True})
            point_encoder = PointTransformer_Colored(config.model, args=dummy_args)
            pc_feat_dims = 768  # concat_f: cls_token (384) + max_pool (384)

        elif backbone == "pointbert":
            from models.pointbert.point_encoder import PointTransformer  # type: ignore

            yaml_path = os.path.join(
                ulip_repo_path,
                "models", "pointbert",
                "PointTransformer_8192point.yaml",
            )
            config = cfg_from_yaml_file(yaml_path)
            dummy_args = EasyDict({"evaluate_3d": True})
            point_encoder = PointTransformer(config.model, args=dummy_args)
            pc_feat_dims = 768

        elif backbone == "pointnext":
            from models.pointnext.pointnext import PointNEXT  # type: ignore

            point_encoder = PointNEXT()
            pc_feat_dims = 256

        else:
            raise ValueError(
                f"Unknown ULIP-2 backbone: '{backbone}'. "
                "Allowed: 'pointbert_colored', 'pointbert', 'pointnext'"
            )

        logger.info(
            f"Point encoder built: {backbone}, "
            f"output dim: {pc_feat_dims}"
        )
        return point_encoder, pc_feat_dims

    def encode_pointcloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Encodes a point cloud into a shape embedding (ULIP-2 or Uni3D).

        Args:
            points: Point cloud (N, 3) – XYZ coordinates.
            colors: Optional colours (N, 3) – RGB in [0, 1].
                    For 'pointbert_colored': used, or padded with 0.
                    For Uni3D: used as RGB input; if not present, the Uni3D
                    wrapper substitutes replacement colours.
                    For other backbones: ignored.

        Returns:
            Normalised tensor (1, embed_dim).
        """
        if self._encoder_type == "uni3d":
            if self._uni3d is None:
                self._uni3d = Uni3DEncoder(self.config)
            return self._uni3d.encode(points, colors)

        self._load_model()

        use_colors = (
            self.config.ulip2_use_colors
            and self.config.ulip2_backbone == "pointbert_colored"
        )

        jitter = getattr(self.config, "ulip2_jitter_std", 0.001)

        if use_colors:
            # xyzrgb → (N, 6)
            if colors is None:
                colors = np.zeros_like(points)
            pts_norm = normalize_pointcloud(
                points,
                colors=colors,
                num_points=self.config.ulip2_num_points,
                jitter_std=jitter,
            )  # (N, 6)
        else:
            # xyz only → (N, 3)
            pts_norm = normalize_pointcloud(
                points,
                colors=None,
                num_points=self.config.ulip2_num_points,
                jitter_std=jitter,
            )  # (N, 3)

        # (1, N, C) Tensor
        pts_tensor = torch.from_numpy(pts_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(pts_tensor)  # (1, embed_dim)

        return embedding

    def load_cad_models(
        self,
        cad_dir: Optional[str] = None,
        partial_pc_dir: Optional[str] = None,
        allowed_extensions: Tuple[str, ...] = (".obj", ".ply", ".glb", ".gltf"),
        mesh_items: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Loads CAD models and encodes them as point clouds.

        Dual path:
          - partial_pc_dir given + config.ulip2_use_partial_views: load precomputed
            partial PCs per view, encode per-view embeddings (best-of-N scoring).
          - Otherwise: full mesh sampling (legacy).

        Expected directory layout (full mesh):
            cad_dir/
                object_label_1.obj | object_label_1/ ...

        Expected directory layout (partial views):
            partial_pc_dir/{obj_id}/{obj_id}_view{N}_partial.npz

        Args:
            cad_dir: Path to the CAD model directory.
            partial_pc_dir: Path to the directory with pre-rendered images + partial PCs
                            (same as reference_images_dir). If None,
                            config.reference_images_dir is used.
            allowed_extensions: Allowed file extensions.
            mesh_items: Optional, already resolved (object_id, mesh_path) list.
                Bypasses ``_collect_mesh_items``, whose ID = directory name. That
                only holds as long as every object has its own directory; for
                datasets grouped by category (HouseCat6D:
                ``<category>/<object>.obj``) the WHOLE category would otherwise
                collapse onto ONE ID. Callers that know their IDs pass them
                explicitly here.
        """
        if self._encoder_type == "uni3d":
            if self._uni3d is None:
                self._uni3d = Uni3DEncoder(self.config)
        else:
            self._load_model()

        cad_dir = cad_dir or self.config.cad_models_dir
        if not cad_dir:
            raise ValueError("No cad_models_dir configured.")

        partial_pc_dir = partial_pc_dir or self.config.reference_images_dir

        # --- Partial views path ---
        if self.config.ulip2_use_partial_views and partial_pc_dir:
            self._load_cad_models_partial(cad_dir, partial_pc_dir, allowed_extensions)
            return

        # --- Full mesh path (legacy) ---
        logger.info(f"Loading CAD models from: {cad_dir}")
        if mesh_items is None:
            mesh_items = self._collect_mesh_items(cad_dir, allowed_extensions)
        if not mesh_items:
            logger.warning("No CAD meshes found in directory: %s", cad_dir)
            return

        cache_path = self._get_cache_path(cad_dir, mesh_items)
        if self._try_load_cache(cache_path):
            return

        self._cad_embeddings = {}
        self._cad_paths = {}
        count = 0
        for obj_id, mesh_path in mesh_items:
            if self._encode_and_cache(obj_id, mesh_path):
                count += 1

        logger.info(f"CAD model embeddings computed: {count} models.")
        if count > 0:
            self._save_cache(cache_path)

    # ------------------------------------------------------------------
    # Partial-view loading and encoding
    # ------------------------------------------------------------------

    def _load_cad_models_partial(
        self,
        cad_dir: str,
        partial_pc_dir: str,
        allowed_extensions: Tuple[str, ...],
    ) -> None:
        """Load precomputed partial PCs per view, encode per-view embeddings."""
        logger.info(
            "Loading CAD models (partial views) from: %s + %s",
            cad_dir, partial_pc_dir,
        )

        # Discover objects and their partial .npz files (also stored for rotation eval)
        partial_items = self._collect_partial_items(partial_pc_dir)
        self._partial_view_paths = dict(partial_items)
        if not partial_items:
            if self._load_partial_cache_by_dim(partial_pc_dir):
                self._apply_partial_view_limit()
                return
            raise self._partial_cache_error(partial_pc_dir)

        # Also need mesh paths for cad_model_path in results
        mesh_items_dict = {
            obj_id: path
            for obj_id, path in self._collect_mesh_items(cad_dir, allowed_extensions)
        }

        # Cache
        cache_path = self._get_partial_cache_path(partial_pc_dir, partial_items)
        if self._try_load_partial_cache(cache_path):
            # Restore mesh paths
            for obj_id in self._cad_embeddings:
                if obj_id not in self._cad_paths and obj_id in mesh_items_dict:
                    self._cad_paths[obj_id] = mesh_items_dict[obj_id]
            self._apply_partial_view_limit()
            return

        self._cad_embeddings = {}
        self._cad_paths = {}
        count = 0

        for obj_id, view_files in partial_items.items():
            view_embeddings = []
            for view_idx, npz_path in sorted(view_files):
                try:
                    data = np.load(npz_path)
                    points = data["points"]
                    colors = data.get("colors", None)
                    emb = self.encode_pointcloud(points, colors=colors)  # (1, embed_dim)
                    view_embeddings.append(emb.squeeze(0).detach().cpu())
                except Exception as e:
                    logger.warning(
                        "Error on partial PC %s view %d: %s", obj_id, view_idx, e
                    )

            if not view_embeddings:
                # Fallback: try full mesh for this object
                mesh_path = mesh_items_dict.get(obj_id)
                if mesh_path:
                    logger.warning(
                        "No partial view embeddings for %s, falling back to full mesh.",
                        obj_id,
                    )
                    if self._encode_and_cache(obj_id, mesh_path):
                        count += 1
                continue

            # Stack to (num_views, embed_dim)
            self._cad_embeddings[obj_id] = torch.stack(view_embeddings, dim=0)
            self._cad_paths[obj_id] = mesh_items_dict.get(obj_id, "")
            count += 1

        self._partial_mode = True
        logger.info(
            "Partial-view CAD embeddings computed: %d models.", count
        )
        if count > 0:
            self._save_partial_cache(cache_path)

        # Apply view limit after caching (cache stores all views)
        self._apply_partial_view_limit()

    def _apply_partial_view_limit(self) -> None:
        """Trim per-object embeddings to config.num_views.

        The cache stores ALL views.  This trims the stacked tensor
        so ablation O4 (V in {8, 16, 42}) works without cache rebuild.
        """
        max_views = getattr(self.config, "num_views", None)
        if max_views is None or not self._cad_embeddings:
            return

        trimmed = 0
        for obj_id, emb in self._cad_embeddings.items():
            if emb.ndim == 2 and emb.shape[0] > max_views:
                self._cad_embeddings[obj_id] = emb[:max_views]
                trimmed += 1

        if trimmed:
            logger.info(
                "Partial view limit applied: %d objects trimmed to %d views",
                trimmed, max_views,
            )

    def _collect_partial_items(
        self, partial_pc_dir: str
    ) -> Dict[str, List[Tuple[int, str]]]:
        """Discover partial .npz files grouped by object ID.

        Always collects ALL available views so the cache is reusable
        across num_views ablations (O4: V in {8, 16, 42}).
        View filtering is applied after cache load/build.

        Returns:
            {obj_id: [(view_idx, npz_path), ...]}
        """
        import re
        result: Dict[str, List[Tuple[int, str]]] = {}

        if not os.path.isdir(partial_pc_dir):
            return result

        for entry in sorted(os.listdir(partial_pc_dir)):
            obj_dir = os.path.join(partial_pc_dir, entry)
            if not os.path.isdir(obj_dir):
                continue

            views = []
            for fname in os.listdir(obj_dir):
                m = re.match(r".+_view(\d+)_partial\.npz$", fname)
                if m:
                    view_idx = int(m.group(1))
                    views.append((view_idx, os.path.join(obj_dir, fname)))

            if views:
                views.sort()
                result[entry] = views

        return result

    def _get_partial_cache_path(
        self,
        partial_pc_dir: str,
        partial_items: Dict[str, List[Tuple[int, str]]],
    ) -> str:
        """Generate cache path for partial-view embeddings."""
        ckpt_tag = os.path.basename(self.config.ulip2_checkpoint or "no_ckpt")
        # Encoder type must be part of the fingerprint — otherwise a Uni3D
        # run (ablation E7) would collide with the ULIP-2 cache, since the
        # remaining meta fields (backbone/npts/edim/ckpt) are ULIP-2 values
        # in both cases (mirrors _get_cache_path for full meshes).  The tag
        # is only added for non-default encoders so existing ULIP-2 caches
        # keep their fingerprint and are not re-encoded.
        encoder_tag = ([f"encoder={self._encoder_type}"]
                       if self._encoder_type != "ulip2" else [])
        meta_parts = encoder_tag + [
            f"backbone={self.config.ulip2_backbone}",
            f"npts={self.config.ulip2_num_points}",
            f"colors={int(self.config.ulip2_use_colors)}",
            f"edim={self.config.ulip2_embed_dim}",
            f"ckpt={ckpt_tag}",
            "partial=1",
        ]

        inv = []
        for obj_id, views in sorted(partial_items.items()):
            for view_idx, path in sorted(views):
                try:
                    st = os.stat(path)
                    # size only (no mtime) → cross-machine-stable fingerprint
                    inv.append(f"{obj_id}|v{view_idx}|{st.st_size}")
                except OSError:
                    inv.append(f"{obj_id}|v{view_idx}|missing")

        raw = "\n".join(meta_parts + sorted(inv))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        fname = f".ulip_partial_cache_{digest}.pt"
        return os.path.join(partial_pc_dir, fname)

    def _expected_partial_dim(self) -> int:
        """Embedding dimension of the active shape config — uniquely
        distinguishes the three partial caches (coloured 1280 / xyz 512 / uni3d 1024)."""
        if getattr(self.config, "shape_encoder", "ulip2") == "uni3d":
            return int(getattr(self.config, "uni3d_embed_dim", 1024))
        return int(self.config.ulip2_embed_dim)

    def _load_partial_cache_by_dim(self, partial_pc_dir: str) -> bool:
        """The raw *_partial.npz files are dispensable once their embedding
        cache exists. Picks the cache under ``partial_pc_dir`` whose embedding
        dimension matches the active config."""
        if not os.path.isdir(partial_pc_dir):
            return False
        want = self._expected_partial_dim()
        for fn in sorted(os.listdir(partial_pc_dir)):
            if fn.startswith(".ulip_partial_cache_") and fn.endswith(".pt"):
                if self._try_load_partial_cache(
                        os.path.join(partial_pc_dir, fn), expected_dim=want):
                    return True
        return False

    def _partial_cache_error(self, partial_pc_dir: str) -> FileNotFoundError:
        """Single error message: where the caches must live and how to
        generate them. No silent fallback — that has demonstrably
        corrupted runs."""
        want = self._expected_partial_dim()
        pass_name = ("ulip_pc_xyz" if want == 512
                     else "uni3d" if want == 1024 else "base")
        return FileNotFoundError(
            f"Partial reference requested, but {partial_pc_dir} contains "
            f"neither *_partial.npz nor an embedding cache with dim={want} "
            f"(.ulip_partial_cache_*.pt).\n"
            f"Generate (repo root):\n"
            f"  python3 preprocess_gallery.py --dataset <name> --step partial\n"
            f"  python3 preprocess_gallery.py --dataset <name> --step embed "
            f"--passes {pass_name}")

    def _try_load_partial_cache(self, cache_path: str,
                                expected_dim: Optional[int] = None) -> bool:
        """Load partial-view cache if it exists (optionally only when the
        embedding dimension matches)."""
        if not os.path.isfile(cache_path):
            return False

        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if not payload.get("partial", False):
                return False

            emb = payload.get("embeddings", {})
            paths = payload.get("paths", {})
            if not isinstance(emb, dict):
                return False

            loaded_emb: Dict[str, torch.Tensor] = {}
            for obj_id, tensor in emb.items():
                if torch.is_tensor(tensor):
                    loaded_emb[obj_id] = tensor.detach().cpu()

            if not loaded_emb:
                return False

            if expected_dim:
                got = int(next(iter(loaded_emb.values())).shape[-1])
                if got != int(expected_dim):
                    logger.info(
                        "Partial cache %s does not match the config "
                        "(dim %d != %d) — skipped.",
                        cache_path, got, expected_dim)
                    return False

            self._cad_embeddings = loaded_emb
            self._cad_paths = {k: v for k, v in paths.items() if isinstance(v, str)}
            self._partial_mode = True
            logger.info(
                "Partial-view CAD embeddings loaded from cache: %d models (%s)",
                len(self._cad_embeddings), cache_path,
            )
            return True
        except Exception as exc:
            logger.warning(
                "Could not load partial ULIP cache (%s): %s", cache_path, exc
            )
            return False

    def _save_partial_cache(self, cache_path: str) -> None:
        """Save partial-view embeddings cache."""
        try:
            payload = {
                "embeddings": {k: v.detach().cpu() for k, v in self._cad_embeddings.items()},
                "paths": dict(self._cad_paths),
                "partial": True,
            }
            torch.save(payload, cache_path)
            logger.info("Partial-view CAD embeddings cache saved: %s", cache_path)
        except Exception as exc:
            logger.warning(
                "Could not save partial ULIP cache (%s): %s", cache_path, exc
            )

    def _find_mesh_in_dir(
        self, dir_path: str, extensions: Tuple[str, ...]
    ) -> Optional[str]:
        """Recursively finds a suitable mesh file in an object directory.

        Prefers common file names in the `meshes/` subdirectory (GSO/YCBV-GSO),
        otherwise falls back to the first mesh file found.
        """
        preferred_names = ("textured_simple.obj", "model.obj", "mesh.obj")
        candidates: List[str] = []

        for root, _, files in os.walk(dir_path):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in extensions:
                    candidates.append(os.path.join(root, fname))

        if not candidates:
            return None

        def sort_key(path: str) -> Tuple[int, int, str]:
            base = os.path.basename(path).lower()
            in_meshes = 0 if os.path.basename(os.path.dirname(path)).lower() == "meshes" else 1
            try:
                pref_idx = preferred_names.index(base)
            except ValueError:
                pref_idx = len(preferred_names)
            return (in_meshes, pref_idx, path)

        return sorted(candidates, key=sort_key)[0]

    def _encode_and_cache(self, obj_id: str, mesh_path: str) -> bool:
        """Samples points from a mesh and computes the ULIP-2 embedding."""
        try:
            use_colors = (
                self.config.ulip2_use_colors
                and self.config.ulip2_backbone == "pointbert_colored"
            )
            points, colors = sample_pointcloud_from_mesh(
                mesh_path,
                self.config.ulip2_num_points,
                with_colors=use_colors,
            )
            embedding = self.encode_pointcloud(points, colors=colors)
            self._cad_embeddings[obj_id] = embedding.squeeze(0).detach().cpu()
            self._cad_paths[obj_id] = mesh_path
            return True
        except Exception as e:
            logger.warning(f"Error on CAD model {obj_id}: {e}")
            return False

    def _load_image_encoder(self) -> None:
        """Loads the ULIP2ImageEncoder (lazy init)."""
        if self.image_encoder is None:
            self.image_encoder = ULIP2ImageEncoder(
                device=self.device,
                checkpoint_path=self.config.ulip2_checkpoint or None,
            )
        self.image_encoder.load()

    def encode_image(self, image) -> torch.Tensor:
        """Encodes a query image into the ULIP-2 embedding (cross-modal).

        Args:
            image: PIL.Image or numpy array (H, W, 3) uint8.

        Returns:
            L2-normalised tensor (1, embed_dim).
        """
        self._load_image_encoder()
        return self.image_encoder.encode(image)

    def match(
        self,
        observed_pc: PointCloudResult,
        top_k: Optional[int] = None,
        candidate_ids: Optional[List[str]] = None,
        query_image=None,
    ) -> ShapeMatchingResult:
        """Finds the geometrically most similar CAD models.

        The retrieval mode is controlled via config.ulip2_mode:
          - "pc"    : Observed point cloud → PC encoder → compared with CAD PC.
          - "cross" : Cropped query image → OpenCLIP image encoder → compared with CAD PC.
                      Uses the shared ULIP-2 embedding space cross-modally.
          - "both"  : Weighted mean of PC and image embedding
                      (ulip2_image_weight from the config).

        Args:
            observed_pc: Point cloud of the observed object (step 2).
            top_k: Number of results (overrides config).
            candidate_ids: Optional – compare only these objects.
            query_image: PIL.Image or numpy array (H, W, 3) – query image for
                         cross-modal retrieval. Ignored when mode="pc".

        Returns:
            ShapeMatchingResult with sorted candidates.
        """
        if not self._cad_embeddings:
            raise RuntimeError(
                "CAD models not loaded. Call load_cad_models()."
            )

        top_k = top_k or self.config.ulip2_top_k
        mode = getattr(self.config, "ulip2_mode", "pc")

        # Uni3D only supports PC mode (no image encoder)
        if self._encoder_type == "uni3d" and mode != "pc":
            logger.warning(
                "Uni3D only supports mode='pc', ignoring ulip2_mode='%s'.", mode
            )
            mode = "pc"

        # --- Query Embedding aufbauen ---
        query_emb: Optional[torch.Tensor] = None

        # PC-Embedding (immer nötig bei mode="pc" oder "both")
        pc_emb: Optional[torch.Tensor] = None
        if mode in ("pc", "both"):
            n_query = len(observed_pc.points)
            target_n = self.config.ulip2_num_points
            if n_query < target_n:
                logger.info(
                    "  Query PC has %d points → upsampled to %d (%.1f× duplication)",
                    n_query, target_n, target_n / max(n_query, 1),
                )
            colors = getattr(observed_pc, "colors", None)
            pc_emb = self.encode_pointcloud(
                observed_pc.points, colors=colors
            )  # (1, embed_dim)

        # Image-Embedding (nötig bei mode="cross" oder "both")
        img_emb: Optional[torch.Tensor] = None
        if mode in ("cross", "both"):
            if query_image is None:
                if mode == "cross":
                    logger.warning(
                        "ulip2_mode='cross' but no query_image passed — "
                        "falling back to PC mode."
                    )
                    mode = "pc"
                    pc_emb = self.encode_pointcloud(
                        observed_pc.points,
                        colors=getattr(observed_pc, "colors", None),
                    )
                # mode="both" ohne Bild → nur PC nutzen
            else:
                img_emb = self.encode_image(query_image)  # (1, embed_dim)

        # Finales Query-Embedding zusammensetzen
        if mode == "pc" or (mode == "both" and img_emb is None):
            query_emb = pc_emb
            logger.info("ULIP-2: retrieval mode = PC→PC (shape matching)")
        elif mode == "cross":
            query_emb = img_emb
            logger.info("ULIP-2: retrieval mode = Image→PC (cross-modal)")
        else:  # "both" mit beiden Embeddings
            w_img = getattr(self.config, "ulip2_image_weight", 0.5)
            w_pc = 1.0 - w_img
            query_emb = F.normalize(
                w_img * img_emb + w_pc * pc_emb, p=2, dim=-1
            )
            logger.info(
                f"ULIP-2: retrieval mode = both "
                f"(image_weight={w_img:.2f}, pc_weight={w_pc:.2f})"
            )

        # --- Kandidaten zusammenstellen ---
        if candidate_ids:
            obj_ids = [oid for oid in candidate_ids if oid in self._cad_embeddings]
        else:
            obj_ids = list(self._cad_embeddings.keys())

        if not obj_ids:
            logger.warning("No matching CAD model embeddings found.")
            return ShapeMatchingResult(
                candidates=[], query_embedding=query_emb.cpu().numpy()
            )

        # --- Cosine Similarity ---
        agg_method = self.config.ulip_view_aggregation
        agg_topk = self.config.ulip_view_topk
        agg_tau = self.config.ulip_view_temperature

        if self._partial_mode:
            # Multi-view scoring for partial view embeddings
            sims_list = []
            best_view_indices = []
            for oid in obj_ids:
                emb = self._cad_embeddings[oid].to(self.device)
                if emb.dim() == 2:
                    # (num_views, embed_dim)
                    view_sims = (query_emb @ emb.T).squeeze(0)  # (num_views,)
                    agg_score, best_idx = _aggregate_view_scores(
                        view_sims, method=agg_method, top_k=agg_topk,
                        temperature=agg_tau,
                    )
                    sims_list.append(torch.tensor(agg_score, device=self.device))
                    best_view_indices.append(best_idx)
                else:
                    # Fallback: single embedding (1D)
                    sim = (query_emb @ emb.unsqueeze(0).T).squeeze()
                    sims_list.append(sim)
                    best_view_indices.append(-1)
            sims = torch.stack(sims_list)  # (K,)
            if agg_method != "max":
                logger.info(
                    "  ULIP partial-view aggregation: %s (k=%d, τ=%.2f)",
                    agg_method, agg_topk, agg_tau,
                )
        else:
            cad_embs = torch.stack(
                [self._cad_embeddings[oid] for oid in obj_ids]
            ).to(self.device)  # (K, embed_dim)
            sims = (query_emb @ cad_embs.T).squeeze(0)  # (K,)
            best_view_indices = [-1] * len(obj_ids)

        # --- NaN-Scores filtern ---
        nan_mask = torch.isnan(sims)
        if nan_mask.any():
            logger.warning(
                "%d of %d CAD embeddings have NaN similarity (they are ignored).",
                nan_mask.sum().item(), len(sims),
            )
            sims = torch.where(nan_mask, torch.tensor(-1.0, device=sims.device), sims)

        # --- Score distribution diagnostics ---
        valid_sims = sims[~nan_mask] if nan_mask.any() else sims
        if len(valid_sims) > 0:
            logger.info(
                "  S_shape score distribution: min=%.4f, max=%.4f, "
                "mean=%.4f, std=%.4f, spread=%.4f (%d objects)",
                valid_sims.min().item(), valid_sims.max().item(),
                valid_sims.mean().item(), valid_sims.std().item(),
                (valid_sims.max() - valid_sims.min()).item(),
                len(valid_sims),
            )

        # --- Top-K ---
        k = min(top_k, len(obj_ids))
        topk_scores, topk_indices = sims.topk(k)

        candidates = []
        for score, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
            obj_id = obj_ids[idx]
            bvi = best_view_indices[idx]
            # Resolve best partial PC path if available
            best_pc_path = ""
            if bvi >= 0 and obj_id in self._partial_view_paths:
                for vi, vp in self._partial_view_paths[obj_id]:
                    if vi == bvi:
                        best_pc_path = vp
                        break
            candidates.append(ShapeCandidate(
                object_id=obj_id,
                shape_score=score,
                cad_model_path=self._cad_paths.get(obj_id, ""),
                best_view_idx=bvi,
                best_partial_pc_path=best_pc_path,
            ))

        logger.info(
            f"Shape matching: {len(candidates)} candidates "
            f"(Top: {candidates[0].object_id}, "
            f"Score={candidates[0].shape_score:.4f})"
        )

        # --- Optional: rotation sensitivity evaluation via ICP ---
        if (self.config.ulip2_rotation_eval
                and self._partial_mode
                and observed_pc is not None):
            self._run_rotation_eval(candidates, observed_pc)

        return ShapeMatchingResult(
            candidates=candidates,
            query_embedding=query_emb.cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Rotation evaluation helpers
    # ------------------------------------------------------------------

    def _run_rotation_eval(
        self,
        candidates: List[ShapeCandidate],
        observed_pc: "PointCloudResult",
    ) -> None:
        """Run ICP registration between observed PC and each candidate's
        best partial reference PC.  Populates registration_fitness /
        registration_rmse on each candidate in-place.

        If ulip2_rotation_eval_weight > 0, also adjusts shape_score.
        """
        top_k = min(self.config.ulip2_rotation_eval_top_k, len(candidates))
        weight = self.config.ulip2_rotation_eval_weight
        logger.info(
            "  Rotation eval: running ICP for top-%d candidates (weight=%.2f)",
            top_k, weight,
        )

        for cand in candidates[:top_k]:
            if not cand.best_partial_pc_path:
                logger.debug("    %s: no partial PC path, skipping", cand.object_id)
                continue

            try:
                fitness, rmse, _ = _register_partial_pointclouds_icp(
                    query_points=observed_pc.points,
                    query_colors=getattr(observed_pc, "colors", None),
                    ref_npz_path=cand.best_partial_pc_path,
                )
                cand.registration_fitness = fitness
                cand.registration_rmse = rmse
                logger.info(
                    "    %s: fitness=%.4f, rmse=%.6f",
                    cand.object_id, fitness, rmse,
                )

                if weight > 0:
                    cand.shape_score = (
                        cand.shape_score + weight * fitness
                    )
            except Exception as exc:
                logger.warning(
                    "    %s: ICP registration failed: %s", cand.object_id, exc,
                )

        # Re-sort if weight > 0
        if weight > 0:
            candidates.sort(key=lambda c: c.shape_score, reverse=True)


# ---------------------------------------------------------------------------
# ICP registration utility
# ---------------------------------------------------------------------------

def _register_partial_pointclouds_icp(
    query_points: np.ndarray,
    query_colors: Optional[np.ndarray],
    ref_npz_path: str,
    voxel_size: float = 0.005,
) -> Tuple[float, float, np.ndarray]:
    """Lightweight ICP alignment of observed partial PC to reference partial PC.

    Both point clouds are downsampled and centered before alignment.

    Args:
        query_points: Observed partial point cloud (N, 3).
        query_colors: Optional observed colors (N, 3).
        ref_npz_path: Path to reference partial PC .npz file.
        voxel_size: Voxel size for downsampling (in normalized units).

    Returns:
        (fitness, inlier_rmse, 4x4_transformation)
    """
    import open3d as o3d

    # Load reference
    data = np.load(ref_npz_path)
    ref_pts = data["points"].astype(np.float64)

    # Normalize both to unit sphere (same as ULIP preprocessing)
    def _to_unit_sphere(pts):
        centroid = pts.mean(axis=0)
        pts = pts - centroid
        max_d = np.linalg.norm(pts, axis=1).max()
        if max_d > 0:
            pts = pts / max_d
        return pts

    q_pts = _to_unit_sphere(query_points.astype(np.float64))
    r_pts = _to_unit_sphere(ref_pts)

    # Build Open3D point clouds
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(q_pts)

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(r_pts)

    # Downsample
    src = src.voxel_down_sample(voxel_size)
    tgt = tgt.voxel_down_sample(voxel_size)

    # Estimate normals for point-to-plane ICP
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30))

    # ICP (point-to-plane)
    threshold = voxel_size * 3
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )

    return reg.fitness, reg.inlier_rmse, np.asarray(reg.transformation)
