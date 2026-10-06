#!/usr/bin/env python3
"""
dGeDi geometry-descriptor HTTP service (Stage-3 geometry re-rank arm).
=====================================================================
Runs INSIDE the ``oscar-dgedi`` container (CUDA 11.8 / torch 2.1), isolated
from the OSCAR image exactly like the FoundationPose service (two-container
HTTP architecture). The OSCAR-side driver talks to it over HTTP via
``evaluation/dgedi_bridge.py``.

What it does
------------
Mirrors the Stage-1 SHREC'18 winner **E2_both** (``best_config.json``:
geometry="both_borda", skip_icp=true): match a query partial cloud against
candidate gallery clouds with dGeDi local descriptors + **RANSAC (no ICP)**, and
return per candidate BOTH geometry signals —

  * ``ransac_fitness``  (RANSAC inlier fraction, higher = better), and
  * ``d_ransac``  (trimmed one-sided Chamfer query->CAD AFTER the RANSAC
    transform, lower = better; trim 10% for partial overlap, per
    ``pipeline/utils.trimmed_chamfer_distance``).

The caller combines them by **Borda mean-rank** (``stage3_bop._geo_rerank``),
exactly like Stage-1's both_borda. ICP is intentionally skipped (it moved nDCG
by ~1e-4 in Stage-1).

Normalization
-------------
Each cloud is **self-normalized** (centered, divided by its OWN diameter) →
unit scale. This is the documented unit-sphere / scale-invariant geometry
setting: gallery descriptors are precomputed ONCE (independent of any query),
and retrieval is scale-invariant (the accepted cost: it will not separate two
objects of identical shape but different size — the appearance arms handle
that). The dGeDi demo instead co-scales target by the query diameter for metric
registration; that would forbid precompute, so we deviate deliberately here.

Endpoints
---------
  GET  /health               -> {"status","n_gallery","device"}
  POST /features  {points}   -> {"points": norm Nx3, "feats": NxD}
        (used by preprocessing/precompute_dgedi.py to build the cache)
  POST /rerank    {query_points, candidate_ids, [ransac_threshold],
                   [trim_ratio]}
        -> {"results": {id: {ransac_fitness, d_ransac, ok}}}
        Candidates whose descriptors are not cached (or that fail registration)
        get ok=false — never dropped, so the caller keeps their fused order.

Gallery cache layout:  <cache_dir>/<namespaced_id>.npz  with arrays
``points`` (M,3 float32, already self-normalized) and ``feats`` (M,D float32).
The namespaced id uses ``__`` for the dataset separator on disk (``gso/obj_1``
-> ``gso__obj_1.npz``) so the id survives a filename round-trip.
"""

import argparse
import glob
import os
import sys

import numpy as np
import open3d as o3d
import torch
from flask import Flask, jsonify, request
from scipy.spatial import cKDTree

# dGeDi repo (mounted at /dgedi) provides core/ + utils.py.
sys.path.insert(0, os.environ.get("DGEDI_REPO", "/dgedi"))
from core.dgedi_distilled import dgedi          # noqa: E402
from utils import extract_features, load_yaml_config  # noqa: E402

FPS_POINTS = 6000
# RANSAC runs on a sparse KEYPOINT subset (like Stage-1's GeDi keypoints), NOT
# the dense 6000-pt cloud: dense RANSAC was ~1.6 s/pair (K=20 -> 32 s/query,
# infeasible for the robot loop), sparse is ~16 ms/pair. The trimmed Chamfer
# still uses the dense clouds (cheap cKDTree, more accurate distance).
RANSAC_KEYPOINTS = 512
RANSAC_MAXIT = 5000       # confidence stops early on real (converging) features
RANSAC_CONF = 0.99

app = Flask(__name__)
_STATE = {"model": None, "device": "cuda", "gallery": {}, "cache_dir": "",
          "diam": {}}   # nsid -> candidate diameter (metres) for query co-scaling


# ---------------------------------------------------------------------------
# id <-> filename (the '/' in namespaced ids cannot be a path component)
# ---------------------------------------------------------------------------
def id_to_fname(nsid: str) -> str:
    return nsid.replace("/", "__") + ".npz"


def fname_to_id(fname: str) -> str:
    return os.path.basename(fname)[:-4].replace("__", "/")


# ---------------------------------------------------------------------------
# Cloud prep — self-normalize (center + divide by own diameter) then FPS
# ---------------------------------------------------------------------------
def _diameter(pts: np.ndarray) -> float:
    # Max pairwise distance; O(N^2) is fine after FPS (<=6000 pts).
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1).max()
    return float(d) if d > 0 else 1.0


def prep_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Raw (N,3) -> FPS(6000) -> self-normalized Open3D cloud (unit diameter)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    n = len(pcd.points)
    if n > FPS_POINTS:
        pcd = pcd.farthest_point_down_sample(FPS_POINTS)
    pts = np.asarray(pcd.points)
    pts = pts - pts.mean(axis=0)
    pts = pts / _diameter(pts)
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def compute_feats(points: np.ndarray):
    """(N,3) raw -> (self-normalized norm_pcd, feats NxD). Used for the GALLERY
    (each candidate self-normalized by its own diameter) and /features."""
    pcd = prep_cloud(points)
    feats = extract_features(pcd, _STATE["model"], _STATE["device"])
    return pcd, feats


def fps_center(points: np.ndarray) -> np.ndarray:
    """Raw (N,3) -> FPS(6000) -> centered points (ORIGINAL units, not scaled).
    The query is FPS'd + centered ONCE, then divided by each candidate's
    diameter at match time (co-scaling per the dGeDi reference)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if len(pcd.points) > FPS_POINTS:
        pcd = pcd.farthest_point_down_sample(FPS_POINTS)
    pts = np.asarray(pcd.points)
    return pts - pts.mean(axis=0)


def _cloud(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return pcd


# ---------------------------------------------------------------------------
# RANSAC (no ICP) + trimmed one-sided Chamfer — the two E2_both signals
# ---------------------------------------------------------------------------
def _feature(feats: np.ndarray) -> o3d.pipelines.registration.Feature:
    f = o3d.pipelines.registration.Feature()
    f.resize(feats.shape[1], feats.shape[0])
    f.data = feats.T
    return f


def _keypoints(pcd, feats, n):
    """Random keypoint subset of a (dense) cloud + its features, for RANSAC."""
    m = len(pcd.points)
    if m <= n:
        return pcd, feats
    idx = np.random.choice(m, n, replace=False)
    kp = o3d.geometry.PointCloud()
    kp.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[idx])
    return kp, feats[idx]


def ransac_only(pcd_q, feats_q, pcd_t, feats_t, thr,
                max_iter=RANSAC_MAXIT, conf=RANSAC_CONF):
    """Feature-matching RANSAC (mutual filter), NO ICP, on sparse keypoints.
    Mirrors ``utils.register_one``'s RANSAC stage / step_b2 ``_gedi_ransac``."""
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=pcd_q, target=pcd_t,
        source_feature=_feature(feats_q), target_feature=_feature(feats_t),
        mutual_filter=True, max_correspondence_distance=thr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(thr),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, conf),
    )


def trimmed_chamfer(src, tgt, trim_ratio=0.1):
    """Trimmed one-sided Chamfer src->tgt (drop the largest ``trim_ratio``),
    identical to ``pipeline/utils.trimmed_chamfer_distance``."""
    d, _ = cKDTree(tgt).query(src, k=1)
    d.sort()
    keep = max(1, int(round(len(d) * (1.0 - trim_ratio))))
    return float(d[:keep].mean())


# ---------------------------------------------------------------------------
# Model + gallery loading
# ---------------------------------------------------------------------------
def load_model(config_path: str, mode: str, device: str):
    cfg = load_yaml_config(config_path)
    mode_cfg = cfg[mode]
    model_cfg = dict(mode_cfg["model_config"])
    model_cfg["weights_path"] = os.path.join(
        os.path.dirname(os.path.abspath(config_path)), mode_cfg["weights_path"])
    # flash-attn is NOT installed in the image (optional per Dockerfile.dgedi);
    # force it off so the model runs with vanilla attention.
    model_cfg["enable_flash"] = False
    model = dgedi({"query": model_cfg, "target": model_cfg, "device": device})
    return model


def load_gallery(cache_dir: str):
    gallery = {}
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        try:
            d = np.load(f)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(d["points"].astype(np.float64))
            gallery[fname_to_id(f)] = (pcd, d["feats"].astype(np.float32))
        except Exception as exc:  # pragma: no cover - corrupt cache entry
            print(f"[dgedi] skip {f}: {exc}", flush=True)
    return gallery


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify(status="ok", n_gallery=len(_STATE["gallery"]),
                   device=_STATE["device"])


@app.post("/features")
def features():
    pts = np.asarray(request.get_json(force=True)["points"], dtype=np.float32)
    if pts.shape[0] < 4:
        return jsonify(error="too few points"), 400
    pcd, feats = compute_feats(pts)
    return jsonify(points=np.asarray(pcd.points, dtype=np.float32).tolist(),
                   feats=feats.astype(np.float32).tolist())


@app.post("/rerank")
def rerank():
    req = request.get_json(force=True)
    pts = np.asarray(req["query_points"], dtype=np.float32)
    cand_ids = list(req["candidate_ids"])
    rth = float(req.get("ransac_threshold", 0.03))
    trim = float(req.get("trim_ratio", 0.1))
    max_iter = int(req.get("ransac_max_iter", RANSAC_MAXIT))
    nkp = int(req.get("ransac_keypoints", RANSAC_KEYPOINTS))
    use_icp = bool(req.get("use_icp", False))          # dGeDi repo config = RANSAC->ICP
    icp_thr = float(req.get("icp_threshold", 0.05))

    results = {}
    if pts.shape[0] < 4:
        # Degenerate query cloud: no geometry signal, leave fused order intact.
        for cid in cand_ids:
            results[cid] = {"ok": False}
        return jsonify(results=results, query_points=int(pts.shape[0]))

    # Query FPS'd + centered ONCE (original units = metres from the driver).
    # For each candidate it is divided by THAT candidate's diameter so the two
    # clouds share one physical scale (dGeDi reference co-scaling), matching the
    # candidate's cached features (self-normalized by its own diameter).
    q_center = fps_center(pts)
    for cid in cand_ids:
        entry = _STATE["gallery"].get(cid)
        diam = _STATE["diam"].get(cid)
        if entry is None or not diam:
            results[cid] = {"ok": False}
            continue
        pcd_t, feats_t = entry
        try:
            q_norm = q_center / float(diam)            # co-scale by candidate diameter
            pcd_q = _cloud(q_norm)
            feats_q = extract_features(pcd_q, _STATE["model"], _STATE["device"])
            kp_q, kf_q = _keypoints(pcd_q, feats_q, nkp)
            kp_t, kf_t = _keypoints(pcd_t, feats_t, nkp)
            res = ransac_only(kp_q, kf_q, kp_t, kf_t, rth, max_iter=max_iter)
            if len(res.correspondence_set) == 0:      # registration failed
                results[cid] = {"ok": False}
                continue
            T = np.asarray(res.transformation)
            if use_icp:
                # dGeDi reference (utils.register_one): refine RANSAC with ICP on
                # the full (co-scaled, normalized) clouds. Point-to-point (no
                # normals needed); repo criteria (max_iter 2000).
                icp = o3d.pipelines.registration.registration_icp(
                    pcd_q, pcd_t, icp_thr, T,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=2000))
                T = np.asarray(icp.transformation)
            q_aln = q_norm @ T[:3, :3].T + T[:3, 3]    # obs -> CAD frame (dense)
            d_ransac = trimmed_chamfer(q_aln, np.asarray(pcd_t.points), trim)
            results[cid] = {"ok": True,
                            "ransac_fitness": float(res.fitness),
                            "d_ransac": d_ransac}
        except Exception as exc:  # pragma: no cover - registration failure
            results[cid] = {"ok": False, "error": str(exc)}
    return jsonify(results=results, query_points=int(pts.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/dgedi/config_dgedi.yaml")
    ap.add_argument("--mode", default="multi_scale",
                    choices=["single_scale", "multi_scale"])
    ap.add_argument("--cache-dir", default="/cache",
                    help="gallery descriptor cache (<id>.npz) — read at startup")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=5060)
    args = ap.parse_args()

    _STATE["device"] = args.device
    print(f"[dgedi] loading model ({args.mode}) ...", flush=True)
    _STATE["model"] = load_model(args.config, args.mode, args.device)
    _STATE["cache_dir"] = args.cache_dir
    if os.path.isdir(args.cache_dir):
        _STATE["gallery"] = load_gallery(args.cache_dir)
        diam_path = os.path.join(args.cache_dir, "diameters.json")
        if os.path.isfile(diam_path):
            import json as _json
            _STATE["diam"] = _json.load(open(diam_path))
        elif _STATE["gallery"]:
            # Without diameters every request answers ok=False per candidate —
            # say so at startup instead of failing one query at a time.
            print(f"[dgedi] WARNING: {diam_path} is missing — every candidate "
                  f"will be rejected. Generate it with "
                  f"services/dgedi/compute_diameters.py.", flush=True)
    print(f"[dgedi] ready: {len(_STATE['gallery'])} gallery clouds cached, "
          f"{len(_STATE['diam'])} diameters, device={args.device}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
