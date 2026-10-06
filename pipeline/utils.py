# =============================================================================
# pipeline/utils.py – Shared utilities for all pipeline steps
# =============================================================================

import os
import json
import logging
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datei-/Ordner-Hilfsfunktionen
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Create a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The same path (for method chaining).
    """
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Bild-Hilfsfunktionen
# ---------------------------------------------------------------------------

def crop_with_mask(
    image: Image.Image,
    mask: np.ndarray,
    background_color: Tuple[int, int, int] = (205, 205, 205),
) -> Optional[Image.Image]:
    """Extract the segmented region from an image.

    The object is placed on a uniform background and cropped to its
    minimal bounding box.

    Ref: Adapted from OSCAR – object_retrieval/i2i_seg_clip.py

    Args:
        image: RGB image (PIL).
        mask: Binary mask (H, W), True/1 = foreground.
        background_color: Background colour for masked-out regions.

    Returns:
        Cropped PIL image, or None if the mask is empty.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.sum() == 0:
        logger.warning("Empty mask – no object found.")
        return None

    img_array = np.array(image)
    canvas = np.full_like(img_array, background_color, dtype=np.uint8)
    canvas[mask_bool] = img_array[mask_bool]

    # Bounding Box des Vordergrunds bestimmen
    coords = np.argwhere(mask_bool)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = canvas[y0:y1, x0:x1]
    return Image.fromarray(cropped)


def load_depth_image(depth_path: str, depth_scale: float = 1000.0) -> np.ndarray:
    """Load a depth image and convert it to metres.

    Args:
        depth_path: Path to the depth image (16-bit PNG is typical for BOP datasets).
        depth_scale: Divisor for the conversion to metres (e.g. 1000 for mm → m).

    Returns:
        Depth image as a float32 array in metres (H, W).
    """
    depth_raw = np.array(Image.open(depth_path))
    return depth_raw.astype(np.float32) / depth_scale


# ---------------------------------------------------------------------------
# Geometrische Distanzmetriken
# ---------------------------------------------------------------------------

def trimmed_chamfer_distance(
    source: np.ndarray,
    target: np.ndarray,
    trim_ratio: float = 0.1,
) -> float:
    """Trimmed one-sided Chamfer distance (source → target).

    For each point in *source*, find the nearest neighbour in *target*.
    Discard the top *trim_ratio* fraction of distances (the largest ones)
    and return the mean of the remaining distances.  This is robust to
    partial overlap — the trimmed tail absorbs query regions that have
    no corresponding CAD surface (e.g. back faces, occluded areas).

    Inspired by the trimmed Chamfer variant in U-RED (Di et al., 2023,
    "U-RED: Unsupervised 3D Shape Retrieval and Deformation for Partial
    Point Clouds") which uses trimmed distances for partial-to-complete
    shape comparison.

    Thesis reference: Sec. 3.3 (Sub-step B2), Equation for S_chamfer.

    Args:
        source: (N, 3) query point cloud (observed partial PC).
        target: (M, 3) reference point cloud (CAD partial view or full mesh).
        trim_ratio: Fraction of largest distances to discard (default 0.1 = 10 %).

    Returns:
        Mean of the trimmed nearest-neighbour distances (lower = better fit).
        Returns ``float('inf')`` if either cloud is empty.
    """
    if len(source) == 0 or len(target) == 0:
        return float("inf")

    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    dists, _ = tree.query(source, k=1)

    # Trim the largest `trim_ratio` fraction
    n_keep = max(1, int(len(dists) * (1.0 - trim_ratio)))
    dists_sorted = np.sort(dists)[:n_keep]
    return float(dists_sorted.mean())


def load_camera_intrinsics(json_path: str, image_id: int = 0) -> Dict:
    """Load camera intrinsics from a BOP-compatible JSON file.

    BOP format: scene_camera.json → {image_id: {"cam_K": [9 floats], "depth_scale": float}}

    Args:
        json_path: Path to scene_camera.json.
        image_id: Image ID (key in the JSON).

    Returns:
        Dict with 'fx', 'fy', 'cx', 'cy', 'depth_scale'.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Exakten Key suchen; Fallback auf ersten verfügbaren Key (Intrinsics sind
    # in BOP-Szenen meist für alle Frames identisch)
    key = str(image_id)
    if key not in data:
        key = next(iter(data))
    entry = data[key]
    K = entry["cam_K"]  # 3x3 Matrix als flache Liste [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    return {
        "fx": K[0],
        "fy": K[4],
        "cx": K[2],
        "cy": K[5],
        "depth_scale": entry.get("depth_scale", 1.0),
    }
