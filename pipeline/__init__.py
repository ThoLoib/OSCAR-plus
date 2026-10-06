# =============================================================================
# OSCAR+ Pipeline – Shape-Aware Object Retrieval & Pose Estimation
# =============================================================================
#
# Modular pipeline for:
#   1. Object localization (GroundingDINO + SAM)
#   2. Point cloud generation (Open3D)
#   3. Semantic candidate search (CLIP)
#   4. Image-based re-ranking (DINOv2)
#   5. Shape matching (ULIP-2)
#   6. Score fusion
#   7. Geometry check (dGeDi re-ranking)
#   8. Pose estimation (FoundationPose / ICP)
#
# Based on the OSCAR framework (https://github.com/pullover00/OSCAR)
# Extended with shape-aware retrieval via ULIP-2.
# =============================================================================

__version__ = "0.1.0"
