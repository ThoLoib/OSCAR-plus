"""
stage3_metrics.py
=================
Instance-level retrieval metrics for Stage-3 (3a).

Unlike the category-level MI3DOR/SHREC metrics (recall@|C| over a whole relevant
class), Stage-3a has **exactly one** relevant gallery item per query — the exact
target CAD `d/obj_0000NN`. So the retrieval metrics are the standard
single-relevant ones from the concept doc:

    Recall@1, Recall@5, Recall@10   — is the exact target in the top-k?
    MRR                              — 1 / rank of the exact target (0 if absent)

Pose metrics: BOP-AR (VSD/MSSD/MSPD → AR) is implemented below (Phase B), on
top of vendored bop_toolkit. It is pose-estimator-agnostic — it scores an
estimated pose against GT, wherever the pose came from. The 3b D_sym surface
metric is Phase C.
"""

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def rank_of_target(ranking: Sequence[Tuple[str, float]],
                   target_id: str) -> Optional[int]:
    """1-indexed rank of ``target_id`` in a descending (id, score) ranking.

    Returns None if the target is not present (e.g. it was pruned, or — in 3b —
    deliberately removed from the gallery)."""
    for i, (oid, _score) in enumerate(ranking):
        if oid == target_id:
            return i + 1
    return None


def summarize_retrieval(ranks: List[Optional[int]],
                        ks: Sequence[int] = (1, 5, 10)) -> dict:
    """Aggregate per-query target ranks into Recall@k + MRR.

    ``ranks`` holds one entry per evaluated query: the 1-indexed rank of that
    query's exact target, or None if it never appeared in the ranking. Missing
    targets count as a miss for every Recall@k and contribute 0 to MRR."""
    n = len(ranks)
    out = {"n_queries": n}
    if n == 0:
        for k in ks:
            out[f"recall@{k}"] = 0.0
        out["mrr"] = 0.0
        out["n_target_found"] = 0
        return out
    found = [r for r in ranks if r is not None]
    for k in ks:
        hits = sum(1 for r in found if r <= k)
        out[f"recall@{k}"] = hits / n
    out["mrr"] = sum(1.0 / r for r in found) / n
    out["n_target_found"] = len(found)
    return out


# ============================================================================
# BOP-AR pose metric (Phase B)
# ============================================================================
#
# bop_toolkit is vendored (not pip-installable into the ephemeral container),
# so we put the vendored trees on sys.path here rather than relying on the
# caller's PYTHONPATH. Paths resolve to /app/third_party/... inside the oscar
# container (repo root is bind-mounted at /app).
_TP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party")
for _p in ("bop_toolkit", "pylibs"):
    _ap = os.path.abspath(os.path.join(_TP, _p))
    if _ap not in sys.path:
        sys.path.insert(0, _ap)

# BOP-19 recall threshold grids (http://bop.felk.cvut.cz/challenges/).
#   MSSD: e < θ·diameter,     θ ∈ {0.05,0.10,…,0.50}
#   MSPD: e < θ·(w/640) px,   θ ∈ {5,10,…,50}       (r = image_width/640)
#   VSD:  e < θ (unitless),   θ ∈ {0.05,…,0.50}, averaged over τ ∈ {0.05,…,0.50}·d
# AR_X = mean recall over its grid; BOP-AR = mean(AR_VSD, AR_MSSD, AR_MSPD).
_TH_MSSD = np.arange(0.05, 0.51, 0.05)          # ×diameter
_TH_MSPD = np.arange(5.0, 50.1, 5.0)            # px, ×(w/640)
_TH_VSD = np.arange(0.05, 0.51, 0.05)           # unitless
_VSD_TAUS = np.arange(0.05, 0.51, 0.05)         # ×diameter
_VSD_DELTA = 15.0                               # mm, BOP default


def load_bop_model_points(model_path: str) -> np.ndarray:
    """Nx3 vertices of a BOP model_eval PLY (mm, BOP object frame)."""
    from bop_toolkit_lib import inout
    return np.asarray(inout.load_ply(model_path)["pts"], dtype=np.float64)


def get_symmetries(model_info: dict, max_sym_disc_step: float = 0.01) -> list:
    """Symmetry transforms for one object from its models_info entry
    (continuous axes discretised at ``max_sym_disc_step``·diameter)."""
    from bop_toolkit_lib import misc
    return misc.get_symmetry_transformations(model_info, max_sym_disc_step)


def pose_errors(R_est, t_est, R_gt, t_gt, K, pts, syms,
                depth_test=None, renderer=None, obj_id=None,
                diameter=None) -> Dict[str, object]:
    """MSSD (mm), MSPD (px) and — iff a renderer+depth are supplied — VSD
    (a list, one entry per τ). t_* are 3×1 mm; K is 3×3; pts are Nx3 mm.

    VSD needs a depth renderer (headless GL); when absent it is omitted and
    only MSSD/MSPD contribute to AR (reported explicitly by the aggregator)."""
    from bop_toolkit_lib import pose_error
    R_est = np.asarray(R_est, float); t_est = np.asarray(t_est, float).reshape(3, 1)
    R_gt = np.asarray(R_gt, float);   t_gt = np.asarray(t_gt, float).reshape(3, 1)
    out: Dict[str, object] = {
        "mssd": float(pose_error.mssd(R_est, t_est, R_gt, t_gt, pts, syms)),
        "mspd": float(pose_error.mspd(R_est, t_est, R_gt, t_gt, K, pts, syms)),
    }
    if depth_test is not None and renderer is not None and diameter is not None:
        # BOP-19: taus are DIMENSIONLESS fractions of the diameter and the
        # function divides the surface distance by the diameter internally
        # (normalized_by_diameter=True). Passing `taus * diameter` here would
        # apply the diameter twice -> VSD far too lenient (audit P0.2).
        out["vsd"] = list(pose_error.vsd(
            R_est, t_est, R_gt, t_gt, depth_test, K, _VSD_DELTA,
            _VSD_TAUS, normalized_by_diameter=True,
            diameter=diameter, renderer=renderer, obj_id=obj_id))
    return out


def bop_ar(records: List[dict]) -> dict:
    """Aggregate per-instance pose errors into BOP-AR.

    Each record is a dict with ``mssd`` (mm), ``mspd`` (px), ``diameter`` (mm),
    ``img_w`` (px) and optionally ``vsd`` (list of per-τ errors). Recall at a
    threshold = fraction of instances below it (dataset-level pooling — BOP
    averages per object first; equal when each object has one instance, which
    we note in the summary). A missing pose (no record) is a miss upstream, so
    only estimated instances are passed here alongside ``n_targets`` total.
    """
    n = len(records)
    out = {"n_estimated": n}
    if n == 0:
        return {**out, "ar": 0.0, "ar_mssd": 0.0, "ar_mspd": 0.0, "ar_vsd": None}

    mssd = np.array([r["mssd"] for r in records])
    mspd = np.array([r["mspd"] for r in records])
    diam = np.array([r["diameter"] for r in records])
    imgw = np.array([r["img_w"] for r in records])

    ar_mssd = np.mean([(mssd < th * diam).mean() for th in _TH_MSSD])
    ar_mspd = np.mean([(mspd < th * (imgw / 640.0)).mean() for th in _TH_MSPD])
    out["ar_mssd"] = float(ar_mssd)
    out["ar_mspd"] = float(ar_mspd)

    have_vsd = [("vsd" in r) for r in records]
    if any(have_vsd):
        # VSD was active. Recall averaged over the (τ × θ) grid, pooled over all
        # instances; a failed pose (no "vsd") is a miss at every threshold so it
        # stays in the denominator (inf error).
        n_tau = len(next(r["vsd"] for r in records if "vsd" in r))
        per_tau = []
        for i in range(n_tau):
            errs = np.array([r["vsd"][i] if "vsd" in r else np.inf
                             for r in records])
            per_tau.append(np.mean([(errs < th).mean() for th in _TH_VSD]))
        ar_vsd = float(np.mean(per_tau))
        out["ar_vsd"] = ar_vsd
        out["ar"] = float((ar_vsd + ar_mssd + ar_mspd) / 3.0)
    else:
        # renderer unavailable: report the 2-metric mean and flag it
        out["ar_vsd"] = None
        out["ar"] = float((ar_mssd + ar_mspd) / 2.0)
        out["ar_note"] = "VSD omitted (no renderer); AR = mean(MSSD,MSPD) only"
    return out


# ============================================================================
# D_sym surface discrepancy + F-score  (the Stage-3 pose metric)
# ============================================================================
# Per STAGE3_EVALUATION_CONCEPT.md, D_sym is the single pose-quality scale for
# BOTH the exact-CAD FoundationPose benchmark (D_posed_gt, proxy == target so
# P_P == P_T) and 3b proxy substitution (D_posed):
#
#     D_sym = 0.5 * ( mean_{p in target} min_{q in est} ||p-q||
#                   + mean_{q in est}    min_{p in target} ||q-p|| )
#
# the symmetric point-to-surface (Chamfer-style) distance of DiffCD
# (Härenstam-Nielsen et al., ECCV 2024), reported in mm and /target-diameter.
#
# F-score at a distance threshold τ (Knapitsch et al., "Tanks and Temples",
# ACM TOG 2017) is reported ALONGSIDE the mean distance, because a mean can be
# dominated by outlier geometry while the F-score at a fixed tolerance stays
# stable (Tatarchenko et al., CVPR 2019). With the estimated-posed proxy as the
# "reconstruction" and the GT-posed target as "ground truth":
#
#     precision(τ) = fraction of PROXY  points within τ of the target surface
#     recall(τ)    = fraction of TARGET points within τ of the proxy  surface
#     F(τ)         = 2·precision·recall / (precision + recall)
#
# τ is taken at 1% and 5% of the target diameter (concept doc).
#
# BOP-AR (VSD/MSSD/MSPD) above is descoped from the headline (user decision
# 2026-08-17): it is NOT computed in the driver. Raw estimated poses are still
# stored per instance so official BOP-AR can be derived later if needed.
DSYM_N = 10000               # surface samples per mesh
DSYM_SEED = 0                # fixed so every run samples a mesh identically
FSCORE_FRACS = (0.01, 0.05)  # F-score thresholds, as fractions of diameter


def sample_surface_mm(mesh_path: str, units_m: bool,
                      n: int = DSYM_N, seed: int = DSYM_SEED) -> np.ndarray:
    """N points sampled uniformly on a mesh surface, in mm, deterministically.

    trimesh.sample draws from the GLOBAL numpy RNG, so we seed it immediately
    before sampling. The seed is fixed per call (not per mesh), which is enough
    for reproducibility: the same mesh always yields the same points because the
    RNG state at the sample() call is identical. Determinism note: this reseeds
    the global RNG as a side effect — acceptable here since all randomness in
    the eval is explicitly seeded (see stage3_bop._seed_everything)."""
    import trimesh
    m = trimesh.load(mesh_path, force="mesh")
    if units_m:
        m.apply_scale(1000.0)             # metres -> mm
    np.random.seed(seed)
    return np.asarray(m.sample(n), dtype=np.float64)


def _fscores(d_t2p: np.ndarray, d_p2t: np.ndarray, diameter: float,
             fracs=FSCORE_FRACS) -> dict:
    """precision/recall/F at each τ = frac·diameter, from the per-point NN
    distance arrays (target->proxy = recall side, proxy->target = precision)."""
    out = {}
    for frac in fracs:
        tau = frac * diameter
        recall = float((d_t2p < tau).mean())      # target covered by proxy
        precision = float((d_p2t < tau).mean())   # proxy near target
        denom = precision + recall
        f = (2.0 * precision * recall / denom) if denom > 0 else 0.0
        out[f"{frac:g}"] = {"tau_mm": float(tau), "precision": precision,
                            "recall": recall, "f": f}
    return out


def d_sym(tgt_pts_mm, R_t, t_t, prx_pts_mm, R_p, t_p, diameter,
          fscore_fracs=FSCORE_FRACS) -> dict:
    """Symmetric surface discrepancy (mm) + F-score between the GT-posed target
    points and the estimated-posed proxy points. Both poses map model->camera in
    mm. For the exact-CAD benchmark, pass prx_pts == tgt_pts (P_P == P_T)."""
    from scipy.spatial import cKDTree
    T = (np.asarray(R_t, float) @ tgt_pts_mm.T).T + np.asarray(t_t, float).reshape(3)
    P = (np.asarray(R_p, float) @ prx_pts_mm.T).T + np.asarray(t_p, float).reshape(3)
    d_t2p = cKDTree(P).query(T)[0]   # per target point -> nearest proxy point
    d_p2t = cKDTree(T).query(P)[0]   # per proxy  point -> nearest target point
    m_t2p, m_p2t = float(d_t2p.mean()), float(d_p2t.mean())
    d = 0.5 * (m_t2p + m_p2t)
    return {"d_t2p": m_t2p, "d_p2t": m_p2t, "d_sym": d,
            "d_sym_norm": (d / diameter) if diameter else None,
            "fscore": _fscores(d_t2p, d_p2t, diameter, fscore_fracs)}


def summarize_dsym(records: List[dict], n_attempted: int = None) -> dict:
    """Mean/median D_sym (mm), D_sym/diameter, and mean F-score at each τ over
    the estimated instances.

    ``n_attempted`` is the number of instances the D_sym block was entered for
    (a top-1 with a mesh). When given, the summary reports pose-success
    ``coverage`` and ``n_failed`` so the conditional mean is not read as if it
    covered every instance — a method must not look better by failing on its
    hardest cases (audit P0.7)."""
    n = len(records)
    if n == 0:
        return {"n_estimated": 0, "d_sym_mean": None, "d_sym_norm_mean": None}
    ds = np.array([r["d_sym"] for r in records], float)
    dn = np.array([r["d_sym_norm"] for r in records], float)
    out = {"n_estimated": n,
           "d_sym_mean": float(ds.mean()), "d_sym_median": float(np.median(ds)),
           "d_sym_norm_mean": float(dn.mean()),
           "d_sym_norm_median": float(np.median(dn))}
    # mean F-score / precision / recall at each threshold, over instances that
    # carry an fscore block (all of them, but guard defensively).
    fr = [r.get("fscore") for r in records if r.get("fscore")]
    if fr:
        keys = list(fr[0].keys())
        out["fscore"] = {}
        for k in keys:
            out["fscore"][k] = {
                "f": float(np.mean([r[k]["f"] for r in fr])),
                "precision": float(np.mean([r[k]["precision"] for r in fr])),
                "recall": float(np.mean([r[k]["recall"] for r in fr])),
            }
    if n_attempted:
        out["n_attempted"] = int(n_attempted)
        out["n_failed"] = int(n_attempted - n)
        out["coverage"] = float(n / n_attempted)
    return out


def instance_key(rec: dict) -> tuple:
    """Stable per-instance identity used to pair a 3b record with its exact-CAD
    benchmark record (same physical detection)."""
    return (rec.get("dataset") or rec.get("_dataset"), rec["scene_id"],
            rec["im_id"], rec["obj_id"], rec["gt_idx"])


def summarize_delta(dsym_recs_3b: List[dict], gt_by_key: Dict[tuple, float]) -> dict:
    """Paired substitution cost Delta = D_posed - D_posed_gt.

    ``dsym_recs_3b`` are the 3b per-instance records (each carries ``d_sym`` and
    an instance key); ``gt_by_key`` maps instance_key -> D_posed_gt (mm) from the
    exact-CAD benchmark. Only instances posed in BOTH runs contribute (paired),
    so Delta isolates the proxy-substitution cost from FoundationPose's own
    placement error on that instance."""
    deltas = []
    for r in dsym_recs_3b:
        k = r.get("_key")
        if k is not None and k in gt_by_key:
            deltas.append(r["d_sym"] - gt_by_key[k])
    if not deltas:
        return {"n_paired": 0, "delta_mean": None}
    a = np.array(deltas, float)
    return {"n_paired": int(a.size), "delta_mean": float(a.mean()),
            "delta_median": float(np.median(a))}
