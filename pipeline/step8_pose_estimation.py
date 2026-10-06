# =============================================================================
# pipeline/step8_pose_estimation.py – Thesis Step C (part 2): 6D Pose
# =============================================================================
#
# Thesis reference: Section 3.4, Step C — Pose Estimation
#
# Estimates the 6D pose (3D rotation + 3D translation) of the detected
# object in camera coordinates, using the CAD model selected by retrieval.
#
# Backend: FoundationPose (Wen et al., CVPR 2024) — model-based 6D pose
# estimation via render-and-compare with a neural object field.  Runs in a
# separate Docker container, called via HTTP (same isolation pattern as the
# dGeDi service).  A failed call counts as a failure: the estimator returns
# an identity pose with confidence 0.
# Ref: https://github.com/NVlabs/FoundationPose
#
# Inputs:
#   - CAD model (retrieval top-1)
#   - RGB image (original)
#   - Depth image (required)
#   - Segmentation mask (Step 1)
#   - Camera intrinsics
#
# Outputs:
#   - 4×4 transformation matrix [R|t] (camera ← object)
#   - Confidence estimate
# =============================================================================

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .step2_pointcloud import PointCloudResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur für Pose-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class PoseEstimationResult:
    """Result of the 6D pose estimation (Step 8).

    The pose describes the transformation from the object coordinate frame
    into the camera coordinate frame: p_cam = R @ p_obj + t

    Attributes:
        pose_matrix: 4×4 homogeneous transformation matrix [R|t; 0 0 0 1].
        rotation: 3×3 rotation matrix R.
        translation: 3D translation vector t (in metres).
        confidence: Estimate of the pose quality (0–1).
        method: Method used.
        cad_model_path: Path to the CAD model used.
        scale_factor: Applied scale factor.
    """
    pose_matrix: np.ndarray      # (4, 4)
    rotation: np.ndarray         # (3, 3)
    translation: np.ndarray      # (3,)
    confidence: float
    method: str
    cad_model_path: str = ""
    scale_factor: float = 1.0


# ---------------------------------------------------------------------------
# Pose Estimation Modul
# ---------------------------------------------------------------------------

class PoseEstimator:
    """Estimates the 6D pose of an object relative to the camera.

    Backend: foundationpose (NVIDIA FoundationPose, via HTTP in its own
    container). A failed call counts as a failure — an identity pose with
    confidence 0 is returned in that case.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device

    def estimate(
        self,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        mask: np.ndarray,
        cad_model_path: str,
        scale_factor: float = 1.0,
        observed_pc: Optional[PointCloudResult] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        method: Optional[str] = None,
        initial_pose: Optional[np.ndarray] = None,
    ) -> PoseEstimationResult:
        """Estimates the 6D pose of the object."""
        method = method or self.config.pose_method

        logger.info(f"Pose estimation with method: {method}")

        if method == "foundationpose":
            return self._estimate_foundationpose(
                rgb_image, depth_image, mask, cad_model_path,
                scale_factor, fx, fy, cx, cy,
            )
        else:
            raise ValueError(f"Unknown pose method: {method}")

    # -----------------------------------------------------------------------
    # Methode 1: FoundationPose (via HTTP)
    # -----------------------------------------------------------------------

    def _estimate_foundationpose(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        mask: np.ndarray,
        cad_path: str,
        scale: float,
        fx: Optional[float],
        fy: Optional[float],
        cx: Optional[float],
        cy: Optional[float],
    ) -> PoseEstimationResult:
        """6D Pose via FoundationPose HTTP service.

        Calls the FoundationPose container over the Docker network.
        A failed call counts as a failure (identity pose, confidence 0).
        """
        try:
            logger.info("Calling FoundationPose service...")
            if depth is None:
                raise RuntimeError("FoundationPose requires a depth image, but got None.")

            K = self._camera_matrix(fx, fy, cx, cy)
            url = self.config.foundationpose_url.rstrip("/") + "/estimate_pose"

            from .foundationpose_bridge import call_foundationpose

            pose_matrix, fp_conf = call_foundationpose(
                url=url,
                rgb=rgb,
                depth=depth,
                mask=mask,
                K=K,
                cad_path=cad_path,
                scale=scale,
                refine_iter=int(self.config.foundationpose_est_refine_iter),
                debug=int(self.config.foundationpose_debug),
                debug_dir=os.path.join(self.config.output_dir, "foundationpose_debug"),
            )

            return PoseEstimationResult(
                pose_matrix=pose_matrix,
                rotation=np.array(pose_matrix[:3, :3]),
                translation=np.array(pose_matrix[:3, 3]),
                confidence=float(fp_conf),
                method="foundationpose",
                cad_model_path=cad_path,
                scale_factor=scale,
            )

        except Exception as e:
            logger.warning("FoundationPose unavailable or failed: %s", e)
            return self._identity_pose(cad_path, scale, "foundationpose_failed")

    def _camera_matrix(
        self,
        fx: Optional[float],
        fy: Optional[float],
        cx: Optional[float],
        cy: Optional[float],
    ) -> np.ndarray:
        """Builds camera intrinsic matrix from args or config defaults."""
        k_fx = float(fx if fx is not None else self.config.camera_fx)
        k_fy = float(fy if fy is not None else self.config.camera_fy)
        k_cx = float(cx if cx is not None else self.config.camera_cx)
        k_cy = float(cy if cy is not None else self.config.camera_cy)

        return np.array(
            [[k_fx, 0.0, k_cx], [0.0, k_fy, k_cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    @staticmethod
    def _identity_pose(
        cad_path: str, scale: float, method: str
    ) -> PoseEstimationResult:
        """Returns an identity pose (no result)."""
        return PoseEstimationResult(
            pose_matrix=np.eye(4),
            rotation=np.eye(3),
            translation=np.zeros(3),
            confidence=0.0,
            method=method,
            cad_model_path=cad_path,
            scale_factor=scale,
        )
