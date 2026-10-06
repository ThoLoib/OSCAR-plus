"""HTTP bridge to call FoundationPose running in a separate container.

OSCAR (this container) sends RGB, depth, mask, camera matrix and CAD model
path to the FoundationPose service via HTTP POST.  The FP service returns
a 4x4 pose matrix and a confidence score.

The FP container mounts the OSCAR repo at /oscar (read-only), so CAD paths
that are /app/... inside OSCAR become /oscar/... inside FP.  This module
handles the path translation automatically.
"""

import base64
import io
import logging
import os
from typing import Tuple

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# Path prefix mapping: OSCAR container -> FP container
_OSCAR_PREFIX = "/app/"
_FP_PREFIX = "/oscar/"
_untranslated_warned: set = set()

# Timeout for the HTTP request (FoundationPose inference can take 10-30s on
# first call due to model loading, subsequent calls are faster)
_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


def _encode_array(arr: np.ndarray) -> str:
    """Encode a numpy array to base64 via .npy format."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _translate_cad_path(oscar_path: str) -> str:
    """Translate a CAD model path from OSCAR's context to FP's /oscar/... mount.

    Handles three cases:
    - Absolute /app/... paths -> /oscar/...
    - Relative paths (e.g. object_database/...) -> /oscar/<path>
    - Already absolute non-/app paths -> returned unchanged
    """
    if oscar_path.startswith(_OSCAR_PREFIX):
        return _FP_PREFIX + oscar_path[len(_OSCAR_PREFIX):]
    if not os.path.isabs(oscar_path):
        return _FP_PREFIX + oscar_path
    # Absolute path outside /app: no translation possible. The FP container
    # only mounts this repository (at /oscar), so unless the user added a
    # matching volume to the foundationpose service, every pose attempt on
    # this mesh fails and is silently counted as a failure — warn loudly,
    # once per path.
    if oscar_path not in _untranslated_warned:
        _untranslated_warned.add(oscar_path)
        logger.warning(
            "CAD path %s lies outside the repository mount; the "
            "FoundationPose container must see it at exactly this path "
            "(add a volume to the foundationpose service in "
            "docker-compose.yml), otherwise every pose estimate on this "
            "mesh will fail.", oscar_path)
    return oscar_path


def call_foundationpose(
    url: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    cad_path: str,
    scale: float = 1.0,
    refine_iter: int = 5,
    debug: int = 0,
    debug_dir: str = "/tmp/fp_debug",
) -> Tuple[np.ndarray, float]:
    """Call the FoundationPose HTTP service and return (pose_matrix, confidence).

    Args:
        url: Full URL of the /estimate_pose endpoint,
             e.g. "http://foundationpose:5050/estimate_pose".
        rgb: uint8 RGB image (H, W, 3).
        depth: float32 depth image (H, W) in meters.
        mask: uint8 binary mask (H, W).
        K: 3x3 camera intrinsic matrix.
        cad_path: Absolute path to CAD mesh (OSCAR container path).
        scale: Scale factor to apply to the mesh.
        refine_iter: Number of refinement iterations.
        debug: Debug level (0 = headless).
        debug_dir: Debug output directory (inside FP container).

    Returns:
        Tuple of (pose_matrix [4x4 ndarray], confidence [float]).

    Raises:
        RuntimeError: If the service returns an error or is unreachable.
    """
    fp_cad_path = _translate_cad_path(cad_path)

    payload = {
        "rgb_b64": _encode_array(np.asarray(rgb, dtype=np.uint8)),
        "depth_b64": _encode_array(np.asarray(depth, dtype=np.float32)),
        "mask_b64": _encode_array(np.asarray(mask, dtype=np.uint8)),
        "K": np.asarray(K, dtype=np.float32).tolist(),
        "cad_path": fp_cad_path,
        "scale": float(scale),
        "refine_iter": int(refine_iter),
        "debug": int(debug),
        "debug_dir": debug_dir,
    }

    logger.info("POST %s  cad=%s  scale=%.3f  iter=%d", url, fp_cad_path, scale, refine_iter)

    try:
        resp = httpx.post(url, json=payload, timeout=_TIMEOUT)
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot connect to FoundationPose service at {url}. "
            f"Is the foundationpose container running? ({exc})"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"FoundationPose request timed out ({exc})"
        ) from exc

    if resp.status_code != 200:
        error_detail = resp.text[:2000] if resp.text else "no response body"
        raise RuntimeError(
            f"FoundationPose service returned HTTP {resp.status_code}: {error_detail}"
        )

    result = resp.json()
    if "error" in result:
        raise RuntimeError(f"FoundationPose error: {result['error'][:2000]}")

    pose_matrix = np.array(result["pose_matrix"], dtype=np.float64).reshape(4, 4)
    if not np.isfinite(pose_matrix).all():
        raise RuntimeError("FoundationPose returned non-finite pose values (NaN/Inf)")

    confidence = float(result.get("confidence", 1.0))
    return pose_matrix, confidence
