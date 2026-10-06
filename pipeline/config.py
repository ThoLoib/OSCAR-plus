# =============================================================================
# pipeline/config.py – Central configuration for the whole pipeline
# =============================================================================
#
# All hyperparameters, paths and model names in one place.
# Can be overridden via CLI or YAML.
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class PipelineConfig:
    """Central configuration class for the OSCAR+ pipeline.

    Attributes are grouped by pipeline step.
    """

    # -------------------------------------------------------------------------
    # Allgemein
    # -------------------------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # -------------------------------------------------------------------------
    # Schritt 1 – Objektlokalisierung (GroundingDINO + SAM2.1)
    # -------------------------------------------------------------------------
    # GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
    # SAM2.1: https://github.com/facebookresearch/sam2
    #   HF: https://huggingface.co/facebook/sam2.1-hiera-large
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    sam_model: str = "facebook/sam2.1-hiera-large"  # SAM2.1 (Ravi et al., 2024)
    detection_confidence: float = 0.3   # Mindest-Konfidenz für Bounding Boxes

    # Mask post-processing (thesis Step A: largest connected component + dilation)
    mask_largest_cc: bool = True         # Retain only largest connected component
    mask_dilation_kernel: int = 5        # Dilation kernel size (pixels)
    mask_dilation_iterations: int = 1    # Number of dilation iterations (0 = disabled)

    # -------------------------------------------------------------------------
    # Schritt 2 – Punktwolkenerzeugung
    # -------------------------------------------------------------------------
    # Open3D: http://www.open3d.org/docs/release/
    # Standardmäßige Kameraintrinsics (überschreibbar pro Datensatz)
    camera_fx: float = 591.0    # Fokuslänge x (Pixel)
    camera_fy: float = 591.0    # Fokuslänge y (Pixel)
    camera_cx: float = 320.0    # Hauptpunkt x
    camera_cy: float = 240.0    # Hauptpunkt y
    depth_scale: float = 10000.0  # Konversion: depth_raw / depth_scale = Meter
    depth_trunc: float = 2.0      # Maximale Tiefe in Metern (2m für Tabletop-Szenen)
    voxel_size: float = 0.002    # Voxel-Downsampling-Größe (Meter)

    # Depth Gating (2D-Vorfilter auf maskierter Tiefe, vor Rückprojektion)
    depth_gate_enabled: bool = True
    depth_gate_tolerance: float = 0.3   # ±30% um den Median

    # Statistical Outlier Removal (3D-Nachfilter)
    sor_nb_neighbors: int = 10
    sor_std_ratio: float = 1.0

    # Radius Outlier Removal (3D-Nachfilter, optional)
    ror_enabled: bool = False
    ror_nb_points: int = 10       # Min. Nachbarn innerhalb des Radius
    ror_radius: float = 0.01     # Suchradius in Metern

    # -------------------------------------------------------------------------
    # Schritt 3 – Semantische Kandidatensuche (CLIP)
    # -------------------------------------------------------------------------
    # CLIP: https://github.com/openai/CLIP
    clip_model_name: str = "ViT-B/32"
    clip_top_k: int = 20         # Anzahl der CLIP-Kandidaten
    clip_threshold: float = 0.25  # Alternativ: Schwellen-basierte Filterung - noch nicht implementiert

    # Pfad zu den Objektbeschreibungen (JSON, erzeugt via description_generator)
    description_file: str = ""    # z.B. "object_database/ycbv/descriptions_attributes.json"

    # -------------------------------------------------------------------------
    # Schritt 4 – Bildbasiertes Re-Ranking (DINOv2 / SigLIP)
    # -------------------------------------------------------------------------
    # DINOv2: https://github.com/facebookresearch/dinov2
    # SigLIP: https://github.com/google-research/big_vision (Zhai et al., 2023)
    #   HF: https://huggingface.co/google/siglip-base-patch16-224
    appearance_encoder: str = "dinov2"  # "dinov2" (default) | "siglip" (ablation E4)
    dino_model_name: str = "facebook/dinov2-base"
    siglip_model_name: str = "google/siglip-base-patch16-224"  # SigLIP ViT-B/16
    dino_pooling: str = "mean"   # "mean" = Patch-Mittel (Konfiguration ALLER Experimente) | "cls" = CLS-Token
    dino_top_k: int = 5          # Anzahl der DINOv2-Kandidaten nach Re-Ranking

    # Multi-view aggregation for DINOv2 re-ranking (inspired by OPEN, Chu et al. 2024)
    # "max" = hard best-view (legacy), "mean" = average all views,
    # "softmax" = softmax-weighted over all views, "topk_softmax" = softmax over top-k views
    dino_view_aggregation: str = "topk_softmax"
    dino_view_topk: int = 5       # Number of top views for topk_softmax (CNOS default, thesis Table 4.1)
    dino_view_temperature: float = 0.5  # Softmax temperature (lower = sharper)

    # Number of rendered views per object to use at inference (thesis ablation O4).
    # Views are FPS-ordered from a 42-vertex icosphere (CNOS, Nguyen et al. 2023),
    # so the first N views always give optimal angular coverage for that N.
    # Must be <= the number of views actually rendered/available on disk.
    # Set to None to use all available views.  Default 42 (full icosphere).
    num_views: Optional[int] = 42  # O4: {8, 16, 42}

    # Pfad zu vorgerenderten Referenzbildern
    reference_images_dir: str = ""  # z.B. "object_images/ycbv/"

    # -------------------------------------------------------------------------
    # Schritt 5 – Shape Matching (ULIP-2 / Uni3D)
    # -------------------------------------------------------------------------
    # ULIP-2: https://github.com/salesforce/ULIP
    # HuggingFace: https://huggingface.co/datasets/SFXX/ulip
    # Uni3D: https://github.com/baaivision/Uni3D (Zhou et al., ICLR 2024)
    #   HF: https://huggingface.co/BAAI/Uni3D
    shape_encoder: str = "ulip2"   # "ulip2" (default) | "uni3d" (ablation E7)
    # Uni3D-giant (E7 baseline): EVA-giant point transformer, aligned to the
    # EVA02-E-14-plus CLIP space (embed_dim 1024). Loaded from the cloned repo
    # (uni3d_repo_path) + modelzoo checkpoint — see rendering/README / step5.
    uni3d_model_name: str = "uni3d-g"      # variant tag (part of cache key)
    uni3d_repo_path: str = "/uni3d"        # cloned baaivision/Uni3D repo
    uni3d_checkpoint: str = "/uni3d/modelzoo/uni3d-g/model.pt"
    uni3d_pc_model: str = "eva_giant_patch14_560"  # timm point-transformer
    uni3d_pc_feat_dim: int = 1408          # EVA-giant hidden dim
    uni3d_pc_encoder_dim: int = 512        # patch-encoder channel (inference.sh)
    uni3d_num_group: int = 512             # FPS group centers
    uni3d_group_size: int = 64             # neighbours per group
    uni3d_num_points: int = 10000          # Points per cloud (same as ULIP-2)
    uni3d_embed_dim: int = 1024            # Uni3D-g embed dim (EVA02-E-14-plus)
    ulip_repo_path: str = ""       # Pfad zum geklonten ULIP-Repo (für Model-Imports)
    ulip2_checkpoint: str = ""     # Pfad zum ULIP-2-Modell-Checkpoint (.pt)
    ulip2_backbone: str = "pointbert_colored"  # "pointbert_colored" | "pointbert" | "pointnext"
    ulip2_top_k: int = 5          # Anzahl der Shape-Kandidaten
    ulip2_num_points: int = 10000  # Punktanzahl für normalisierte Point Clouds
    ulip2_jitter_std: float = 0.001  # Gaussian jitter σ on upsampled points (thesis Sec. 3.5.2; 0 = off)
    ulip2_use_colors: bool = True  # xyzrgb (True) oder nur xyz (False)
    ulip2_embed_dim: int = 1280    # Embedding-Dimension (1280 für ViT-bigG-14, 512 für ViT-B)

    # Modus für ULIP-2 Query-Embedding:
    #   "pc"    – Beobachtete Punktwolke → ULIP-PC-Encoder → mit CAD-PC-Embeddings vergleichen (wie bisher)
    #   "cross" – Cropped Query-Bild → OpenCLIP ViT-bigG-14 (ULIP Image-Branch) → mit CAD-PC-Embeddings
    #             vergleichen (cross-modal; nutzt den gemeinsamen Embedding-Raum von ULIP-2 voll aus).
    #             ULIP-2 friert den Image-Encoder während Training ein → vanilla OpenCLIP Gewichte sind korrekt.
    #   "both"  – Gewichteter Durchschnitt aus "pc" und "cross" Embeddings.
    ulip2_mode: str = "cross"         # "pc" | "cross" | "both"
    ulip2_image_weight: float = 0.5   # Gewicht für Image-Embedding in Modus "both" (PC-Gewicht = 1 - image_weight)
    ulip2_use_partial_views: bool = False  # True = precomputed partial PCs per view; False = full mesh (legacy)

    # Multi-view aggregation for ULIP-2 partial views (inspired by OPEN, Chu et al. 2024)
    # Same modes as dino_view_aggregation. Only applies when ulip2_use_partial_views=True.
    ulip_view_aggregation: str = "topk_softmax"
    ulip_view_topk: int = 5   # equalised to DINOv2's k_v=5 (CNOS, thesis Table 4.1);
                              # was 8 — back-ported to Stage 1 on 2026-08-26 to match
                              # Stage 2 and the dino_view_topk spec.
    ulip_view_temperature: float = 0.5

    # Rotation sensitivity evaluation for ULIP Top-K candidates
    ulip2_rotation_eval: bool = False
    ulip2_rotation_eval_top_k: int = 5
    ulip2_rotation_eval_weight: float = 0.0  # 0.0 = debug-only, >0 = optional rerank contribution

    # Pfad zu den CAD-Modellen (OBJ/PLY/GLB)
    cad_models_dir: str = ""       # z.B. "object_database/ycbv/"

    # -------------------------------------------------------------------------
    # Schritt 6 – Fusion / Konsens
    # -------------------------------------------------------------------------
    # "weighted_sum" | "intersection" | "rank_fusion" | "majority_voting"
    # "rank_fusion" = Reciprocal Rank Fusion (Cormack et al., SIGIR 2009);
    # it is the rank-based fusion evaluated as thesis ablation E6.
    fusion_method: str = "weighted_sum"
    weight_clip: float = 0.3
    weight_dino: float = 0.4
    weight_ulip: float = 0.3
    fusion_top_k: int = 1               # Finale Anzahl Kandidaten

    # -------------------------------------------------------------------------
    # Sub-step B2 – Geometry Re-ranking (GeDi + trimmed Chamfer)
    # -------------------------------------------------------------------------
    # GeDi: https://github.com/fabiopoiesi/gedi
    # Paper: "Learning General and Distinctive 3D Local Deep Descriptors" (Poiesi & Boscaini, 2022)
    geometry_reranking_enabled: bool = True
    # "fitness" | "chamfer_unaligned" | "chamfer_ransac" | "chamfer_icp"
    # (legacy aliases "gedi"/"chamfer"/"both" are still accepted).
    # Default is the aligned distance: the thesis defines D_trim on the
    # transformed observation, so an unaligned default would silently
    # contradict Eq. eq:methods_trimmed_surface_distance.
    geometry_reranking_signal: str = "chamfer_ransac"
    geometry_reranking_top_k: int = 5        # Shortlist size from fusion for B2 re-ranking

    # GeDi descriptor settings
    # gedi_url/gedi_repo_path/gedi_checkpoint/gedi_num_keypoints entfernt
    # (2026-09-18): der alte In-Process-GeDi-Pfad (Dienst gedi:5060) existiert
    # nicht mehr; der geometrische Check laeuft als Schritt 7 ueber den
    # dGeDi-Dienst (evaluation/dgedi_bridge.py, Port 5061).
    # Directory for the on-disk descriptor cache. None = disabled (compute on
    # every call, the historical behaviour). Descriptors are a per-CLOUD cost
    # (~3.4 s) while the RANSAC fit that consumes them is ~0.43 s per PAIR, so
    # enabling this is what makes a deep re-ranking shortlist affordable:
    # O(clouds) instead of O(queries x K). Entries self-validate against a
    # fingerprint of the input cloud, so a preprocessing change invalidates
    # them rather than silently returning descriptors for a different cloud.
    gedi_cache_dir: Optional[str] = None
    # Skip point-to-plane ICP refinement in B2. ICP costs ~1.1 s of a ~2.9 s
    # per-pair budget and was measured to move nDCG by 0.0001 at K=5, so a
    # deep-shortlist run can skip it and cite the shallow measurement. The
    # "chamfer_icp" signal ignores this flag — that arm is the measurement.
    geometry_skip_icp: bool = False

    # Trimmed Chamfer settings (B2)
    chamfer_trim_ratio: float = 0.1          # Discard top 10% of distances

    # -------------------------------------------------------------------------
    # Schritt 8 – Pose Estimation
    # -------------------------------------------------------------------------
    # FoundationPose: https://github.com/NVlabs/FoundationPose
    # Einziges Backend. Ein fehlgeschlagener Aufruf zaehlt als Fehlschlag
    # (Identitaetspose, Konfidenz 0) — so zaehlt es auch die Arbeit.
    pose_method: str = "foundationpose"  # einziges Backend; Fehlschlag zählt als Fehlschlag
    foundationpose_url: str = "http://foundationpose:5050"  # FoundationPose service URL (docker-compose service name)
    foundationpose_est_refine_iter: int = 5             # Register-Refinement-Iterationen
    foundationpose_debug: int = 0                       # 0=keine GUI/Debug-Ausgaben

    # -------------------------------------------------------------------------
    # Debug-Visualisierung
    # -------------------------------------------------------------------------
    gt_bbox_center_compensation: bool = False  # Compensate GT wireframe for mesh bbox-center offset

    # -------------------------------------------------------------------------
    # Ein-/Ausgabepfade
    # -------------------------------------------------------------------------
    output_dir: str = "pipeline_output"  # Ordner für alle Pipeline-Ergebnisse

    def to_dict(self) -> dict:
        """Convert the configuration into a dictionary (e.g. for JSON export)."""
        return {k: str(v) if isinstance(v, torch.device) else v
                for k, v in self.__dict__.items()}
