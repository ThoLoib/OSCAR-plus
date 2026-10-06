"""
dgedi_bridge.py
===============
OSCAR-side HTTP client for the dGeDi geometry re-rank service (compose service
``dgedi``, port 5061). Same pattern as ``pipeline/foundationpose_bridge.py``.

``dgedi_rerank`` sends the query partial cloud + the fused shortlist ids and
gets back, per candidate, the two E2_both signals ``ransac_fitness`` (higher =
better) and ``d_ransac`` (trimmed Chamfer, lower = better) — combined by the
driver via Borda mean-rank. It **degrades to ``None``** on any transport error
so the Stage-3 driver can fall back to the pre-geometry fused ranking instead of
crashing an overnight run (matching the "degrade, do not halt" rule).
"""

import os

import httpx
import numpy as np

DGEDI_URL = os.environ.get("DGEDI_URL", "http://dgedi:5061")
_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


def dgedi_health():
    """Return the service health dict, or None if unreachable."""
    try:
        r = httpx.get(f"{DGEDI_URL}/health", timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def dgedi_rerank(query_points, candidate_ids,
                 ransac_threshold=0.03, trim_ratio=0.1,
                 ransac_keypoints=None, ransac_max_iter=None,
                 use_icp=False, icp_threshold=0.05):
    """Per-candidate geometry signals for the fused shortlist.

    Args:
        query_points: (N,3) array-like, the query partial cloud (any scale;
            the service self-normalizes).
        candidate_ids: namespaced gallery ids (the fused shortlist).
        ransac_keypoints / ransac_max_iter / use_icp / icp_threshold: override
            the service defaults to match the dGeDi *repo* config (6000 kp /
            10k iter / + ICP); left None -> the fast service defaults.
    Returns:
        ``{id: {"ok": bool, "ransac_fitness": float, "d_ransac": float}}`` for
        every requested id (missing/failed candidates -> ``{"ok": False}``), or
        ``None`` if the service is unreachable.
    """
    pts = np.asarray(query_points, dtype=np.float32)
    payload = {
        "query_points": pts.tolist(),
        "candidate_ids": list(candidate_ids),
        "ransac_threshold": float(ransac_threshold),
        "trim_ratio": float(trim_ratio),
    }
    if ransac_keypoints is not None:
        payload["ransac_keypoints"] = int(ransac_keypoints)
    if ransac_max_iter is not None:
        payload["ransac_max_iter"] = int(ransac_max_iter)
    if use_icp:
        payload["use_icp"] = True
        payload["icp_threshold"] = float(icp_threshold)
    try:
        resp = httpx.post(f"{DGEDI_URL}/rerank", json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return None
    return resp.json().get("results", {})
