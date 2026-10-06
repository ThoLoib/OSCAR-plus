"""
query_cloud.py
==============
Back-project a Stage-3 query (depth + GT visible mask + camera K) into a
**partial** point cloud in the camera frame.

This one cloud feeds BOTH new Stage-3 shape signals:

  * the Uni3D shape arm  — Uni3D encodes the partial cloud (pc-query, replacing
    ULIP-2's image-cross-modal query), matched vs the gallery Uni3D embeddings;
  * the dGeDi geometry re-rank — the same cloud is the RANSAC "query" registered
    against each candidate's cached descriptors.

Both consumers self-normalize (Uni3D recenters+rescales in
``normalize_pointcloud``; dGeDi divides by the cloud diameter), so the absolute
units here are irrelevant — we keep the depth's native scale (metres, as the
Stage-3 driver already produces ``depth_m``). The math mirrors
``pipeline/step2_pointcloud.py::_backproject_manual`` (masked pinhole
back-projection) but without the PipelineConfig / Open3D coupling, so it is
cheap to call per query and testable on CPU.
"""

import numpy as np


def backproject_masked(depth, mask, K, rgb=None, depth_trunc=None):
    """Masked pinhole back-projection → (points, colors).

    Args:
        depth: HxW float array, object depth in *metres* (0 = invalid).
        mask:  HxW mask; non-zero marks the object (GT visible mask).
        K:     3x3 camera intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]].
        rgb:   optional HxWx3 uint8 frame; when given, per-point colors in
            [0,1] are returned so the Uni3D pc-query is XYZ+RGB like the gallery
            partial-view encodings (symmetric modality). ``None`` -> colors None.
        depth_trunc: optional far cutoff (same units as ``depth``); points
            beyond it are dropped. ``None`` = no far gating.

    Returns:
        (points (N,3) float32, colors (N,3) float32 in [0,1] or None). Empty
        (0,3) points array when nothing valid remains.
    """
    depth = np.asarray(depth, dtype=np.float32)
    mask_bool = np.asarray(mask).astype(bool)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    valid = mask_bool & (depth > 0)
    if depth_trunc is not None:
        valid &= depth < float(depth_trunc)

    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32), None

    z = depth[ys, xs]
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    colors = None
    if rgb is not None:
        colors = (np.asarray(rgb)[ys, xs, :3].astype(np.float32) / 255.0)
    return points, colors
