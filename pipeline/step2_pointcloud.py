# =============================================================================
# pipeline/step2_pointcloud.py – Thesis Step A (cont.): Point Cloud Extraction
# =============================================================================
#
# Generates a partial 3D point cloud of the detected object from the
# RGB-D frame and segmentation mask produced in Step 1 (thesis Sec. 3.2).
#
# Pipeline:
#   RGB-D + camera intrinsics + mask → pinhole backprojection → partial PC
#
# Post-processing:
#   • Voxel downsampling — reduces point density for tractable descriptor
#     computation in Steps B2 and C.
#   • Statistical Outlier Removal (SOR) — removes isolated noise points
#     that degrade registration quality (Zhou et al., 2018 — Open3D).
#   • Radius Outlier Removal (ROR) — optional second-pass filter for
#     sensor-specific artefacts.
#   • Depth gating — median-relative gate rejects depth outliers within
#     the mask before backprojection, mitigating flying-pixel artefacts
#     from ToF sensors (Chugunov et al., 2021) and structured-light
#     shadow regions (Shen et al., 2013).
#
# Library:
#   • Open3D (Zhou, Park & Koltun, 2018)
#     Ref: http://www.open3d.org/docs/release/
#
# Outputs:
#   - Partial point cloud of the segmented object (Open3D PointCloud)
#   - 3D bounding box dimensions for scale estimation (Step 7)
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur für Punktwolken-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class PointCloudResult:
    """Result of point cloud generation (Step 2).

    Attributes:
        point_cloud: Open3D PointCloud object (with colors).
        points: Numpy array of the 3D points (N, 3).
        colors: Numpy array of the colors (N, 3), normalized to [0, 1].
        num_points: Number of points.
        bbox_min: Minimum corner of the 3D bounding box.
        bbox_max: Maximum corner of the 3D bounding box.
        bbox_size: Size of the 3D bounding box (width, height, depth).
    """
    point_cloud: object  # open3d.geometry.PointCloud (vermeidet Import-Pflicht)
    points: np.ndarray
    colors: np.ndarray
    num_points: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bbox_size: np.ndarray


# ---------------------------------------------------------------------------
# Punktwolken-Generator
# ---------------------------------------------------------------------------

class PointCloudGenerator:
    """Generates 3D point clouds from RGB-D data and segmentation masks.

    Uses Open3D to backproject depth pixels into 3D space, accounting
    for the camera intrinsics.

    The caller is responsible for converting the depth image to float32
    meters before passing it to generate(). No internal heuristic is applied.

    Usage:
        >>> gen = PointCloudGenerator(config)
        >>> result = gen.generate(rgb, depth, mask)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._check_open3d()

    @staticmethod
    def _check_open3d():
        """Checks whether Open3D is installed."""
        try:
            import open3d as o3d  # noqa: F401
        except ImportError:
            raise ImportError(
                "Open3D is not installed. Install with:\n"
                "  pip install open3d\n"
                "Ref: http://www.open3d.org/docs/release/"
            )

    def _gate_depth(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove depth outliers within the mask using median-relative gating.

        Operates on the 2D depth image before backprojection.
        """
        valid = mask & (depth > 0)
        n_valid = valid.sum()
        if n_valid == 0:
            return depth

        valid_depths = depth[valid]
        median_z = np.median(valid_depths)
        tol = self.config.depth_gate_tolerance
        z_min = median_z * (1.0 - tol)
        z_max = median_z * (1.0 + tol)

        gated = depth.copy()
        out_of_range = mask & ((depth < z_min) | (depth > z_max))
        gated[out_of_range] = 0.0

        n_after = (mask & (gated > 0)).sum()
        n_removed = n_valid - n_after
        logger.info(
            "  Depth gating: median=%.4fm, window=[%.4f, %.4f]m, "
            "%d → %d valid pixels (removed %d, %.1f%%)",
            median_z, z_min, z_max, n_valid, n_after, n_removed,
            100.0 * n_removed / max(n_valid, 1),
        )
        return gated

    def generate(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        mask: np.ndarray,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        depth_trunc: Optional[float] = None,
    ) -> Optional[PointCloudResult]:
        """Generates a point cloud from RGB-D data for the masked region.

        The depth pixels are backprojected into 3D space using the pinhole
        camera model:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth[v, u]

        Args:
            rgb_image: RGB image as a numpy array (H, W, 3), uint8.
            depth_image: Depth image as a numpy array (H, W), float32, in meters.
                         The caller must convert to meters before calling.
            mask: Binary segmentation mask (H, W), bool.
            fx, fy: Focal lengths (override the config values).
            cx, cy: Principal point (overrides the config values).
            depth_trunc: Max. depth in meters (points beyond it are ignored).

        Returns:
            PointCloudResult, or None on failure.
        """
        import open3d as o3d

        # Parameter mit Config-Defaults auffüllen
        fx = fx or self.config.camera_fx
        fy = fy or self.config.camera_fy
        cx = cx or self.config.camera_cx
        cy = cy or self.config.camera_cy
        depth_trunc = depth_trunc or self.config.depth_trunc

        # --- Tiefenbild vorbereiten (already in meters, no heuristic) ---
        depth = depth_image.astype(np.float32)

        # --- Maske anwenden ---
        mask_bool = np.asarray(mask, dtype=bool)
        n_mask_pixels = mask_bool.sum()
        n_valid_depth = (mask_bool & (depth > 0) & (depth < depth_trunc)).sum()
        logger.info(
            "  Mask: %d pixels, %d with valid depth (before gating)",
            n_mask_pixels, n_valid_depth,
        )

        # --- Depth gating (2D, before backprojection) ---
        if self.config.depth_gate_enabled:
            depth = self._gate_depth(depth, mask_bool)

        # --- Maske anwenden: nur Objektpixel behalten ---
        segmented_depth = np.where(mask_bool, depth, 0.0)

        # --- Rückprojektion ---
        points, colors = self._backproject_manual(
            rgb_image, segmented_depth, fx, fy, cx, cy, depth_trunc
        )
        logger.info("  Backprojected: %d raw 3D points", len(points))

        if len(points) == 0:
            logger.warning("No valid depth points in the masked region.")
            return None

        # --- Open3D PointCloud erstellen ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # --- Optional: Voxel-Downsampling ---
        if self.config.voxel_size > 0:
            n_before = len(pcd.points)
            pcd = pcd.voxel_down_sample(voxel_size=self.config.voxel_size)
            logger.info(
                "  Downsampling: %d → %d points (voxel: %.3fm)",
                n_before, len(pcd.points), self.config.voxel_size,
            )

        # --- Statistical Outlier Removal (configurable) ---
        if self.config.sor_nb_neighbors > 0:
            n_before = len(pcd.points)
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.config.sor_nb_neighbors,
                std_ratio=self.config.sor_std_ratio,
            )
            logger.info(
                "  SOR: %d → %d points (removed %d)",
                n_before, len(pcd.points), n_before - len(pcd.points),
            )

        # --- Radius Outlier Removal (optional) ---
        if self.config.ror_enabled:
            n_before = len(pcd.points)
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=self.config.ror_nb_points,
                radius=self.config.ror_radius,
            )
            logger.info(
                "  ROR: %d → %d points (removed %d)",
                n_before, len(pcd.points), n_before - len(pcd.points),
            )

        # --- Bounding Box berechnen ---
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)

        if len(pts) == 0:
            logger.warning("No points remaining after filtering.")
            return None

        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        bbox_size = bbox_max - bbox_min

        logger.info(
            "  Final: %d points, BBox=[%.4f, %.4f, %.4f]m",
            len(pts), bbox_size[0], bbox_size[1], bbox_size[2],
        )

        return PointCloudResult(
            point_cloud=pcd,
            points=pts,
            colors=cols,
            num_points=len(pts),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            bbox_size=bbox_size,
        )

    @staticmethod
    def _backproject_manual(
        rgb: np.ndarray,
        depth: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_trunc: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backprojects depth pixels into 3D via the pinhole model.

        Vectorized implementation for high performance.

        Pinhole model:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth[v, u]

        Args:
            rgb: (H, W, 3), uint8.
            depth: (H, W), float32, in meters, 0 = invalid.
            fx, fy, cx, cy: Camera parameters.
            depth_trunc: Max. depth.

        Returns:
            (points, colors): Arrays of shape (N, 3).
        """
        h, w = depth.shape

        # Gültige Pixel: Tiefe > 0 und < max
        valid = (depth > 0) & (depth < depth_trunc)
        vs, us = np.where(valid)
        zs = depth[valid]

        # Rückprojektion
        xs = (us.astype(np.float32) - cx) * zs / fx
        ys = (vs.astype(np.float32) - cy) * zs / fy

        points = np.stack([xs, ys, zs], axis=-1)  # (N, 3)

        # Farben normalisieren
        colors = rgb[vs, us].astype(np.float32) / 255.0  # (N, 3)

        return points, colors

    def save_pointcloud(self, result: PointCloudResult, path: str) -> None:
        """Saves a point cloud as a PLY file.

        Args:
            result: PointCloudResult from generate().
            path: Target path (e.g. "output/object.ply").
        """
        import open3d as o3d
        o3d.io.write_point_cloud(path, result.point_cloud)
        logger.info(f"Point cloud saved: {path}")

    @staticmethod
    def load_pointcloud(path: str) -> 'PointCloudResult':
        """Loads a point cloud from a PLY file.

        Args:
            path: Path to the PLY file.

        Returns:
            PointCloudResult.
        """
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(pts)
        bbox_min = pts.min(axis=0) if len(pts) > 0 else np.zeros(3)
        bbox_max = pts.max(axis=0) if len(pts) > 0 else np.zeros(3)

        return PointCloudResult(
            point_cloud=pcd,
            points=pts,
            colors=cols,
            num_points=len(pts),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            bbox_size=bbox_max - bbox_min,
        )
