"""
stage3_bop.py
=============
OSCAR+ Stage-3 BOP evaluation (retrieval + 6D pose), per
``STAGE3_EVALUATION_CONCEPT.md`` (revised 2026-08-17). Four settings over the
same RGB-D BOP queries (YCB-V, T-LESS, LM-O; GT visible bbox + mask + 6D pose,
so retrieval and pose are isolated from segmentation). The readable ``--mode``
names map onto the internal 3a/gt/3b/3c settings:

    --mode retrieval  (3a)  exact CAD in the gallery -> RETRIEVAL ONLY
                Recall@1/5/10, MRR, geometry coverage, mean #registered.
                FoundationPose is NOT run in this mode.

    --mode gt   exact-CAD FoundationPose benchmark (the "GT run"):
                FoundationPose with the GROUND-TRUTH target CAD ->
                D_posed_gt = D_sym(T_gt·P_T, T_hat·P_T)  (P_P == P_T).
                No retrieval; the reference D_sym for the Delta pairing.

    --mode pose  (3b)  proxy-only gallery -> retrieve top-1 proxy,
                FoundationPose it -> D_posed = D_sym(T_gt·P_T, T_hat·P_P), and
                (against --gt-records) the paired substitution cost
                Delta = D_posed - D_posed_gt.

    --mode decompose  (3c)  next-best-non-GT diagnostic (decomposes the pose-mode
                substitution cost). REUSES the stored retrieval ranking
                (--from-retrieval; gallery = G_proxy ∪ ALL target CADs, so the
                exact target IS present). Per query it poses the highest-ranked
                candidate that is NOT the exact target — the best available
                stand-in from the *richer* retrieval gallery — and scores
                D_posed + Delta exactly like pose mode. Provenance (real target
                CAD of another object vs a G_proxy item) is recorded so the
                error can be split into "gallery too sparse/different" vs
                "substitution is inherently lossy". No retrieval or encoders are
                run — only FoundationPose + D_sym on the reused shortlist
                (retrieval is free).

Pose quality is reported as D_sym (mm + /diameter) and F-score at 1% and 5% of
the target diameter, for BOTH the gt benchmark and pose mode — the two are
directly comparable on one scale. Official BOP-AR (VSD/MSSD/MSPD) is descoped
from the headline (user decision 2026-08-17); raw estimated poses are stored
per instance so it can be derived later.

Determinism: all explicit RNGs are seeded (see ``_seed_everything``). Two
sources are NOT bit-reproducible and are documented, not silenced:
  * FoundationPose's pose-hypothesis sampling / refinement is stochastic on GPU;
    we fix refine_iter and store the returned pose, but repeated calls can
    differ slightly.
  * open3d RANSAC in the dGeDi service is seeded server-side where the open3d
    build supports it; older builds ignore the seed (documented in DETERMINISM).

Paths come from ``config/paths.yaml`` (overridable per flag, see ``--help``).
Outside the container the script wraps itself into ``docker compose run``
automatically; ``--geometry`` first checks that the dgedi service is up with
the BOP gallery loaded.

How to run
----------
    python3 experiments/stage3_bop.py --mode retrieval --query cross --gallery partial
        # R@1 0.4818 (results/stage3_bop/retrieval_cross_partial.json)
    python3 experiments/stage3_bop.py --mode gt
        # exact-CAD FP benchmark: D_sym median 1.72 mm
    python3 experiments/stage3_bop.py --mode pose --query cross --gallery partial
        # proxy pose: D_sym median 18.37 mm (+ paired Delta vs the gt run)
    python3 experiments/stage3_bop.py --mode decompose \\
        --from-retrieval runs/stage3_bop_retrieval_cross_partial

Outputs land in ``<runs_root>/<--out>/``; a ``run_config.json`` with argv, the
derived internal settings, and git revision is written next to them.  For every
reported configuration the run additionally leaves the two artefacts
``results/stage3_bop/`` is built from, under the RELEASE name of the
configuration: ``<config>.json`` (copy of the combined summary, e.g.
``retrieval_cross_partial.json``) and ``per_query/<config>/<ds>.json`` (the
per-dataset records, hard-linked).  See ``evaluation/configuration_names.py``.
"""

import json
import logging
import os
import random
import sys

# ---------------------------------------------------------------------------
# CLI prelude — before the heavy imports (numpy/PIL/eval_common -> torch,
# open3d) so that `--help` works on the host without the container
# dependencies.
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(1, os.path.join(_REPO, "evaluation"))

from configuration_names import to_release  # noqa: E402

# readable CLI mode -> internal setting name (3a/gt/3b/3c) used by run_stage3
_MODE_WORDS = {"retrieval": "3a", "gt": "gt", "pose": "3b", "decompose": "3c"}

# (mode word, --query, --gallery, --geometry) -> internal run key of the
# reported configuration.  These are the keys results/stage3_bop was produced
# under; to_release() (evaluation/configuration_names.py) turns them into the
# archive file names.  The --oscar-baseline cascade has no query/gallery and is
# handled separately, --mode gt has neither.  A combination that is NOT listed
# was never reported and gets no archive-named artefact rather than a guessed
# name.  Note on the geometry rows: the 3a keys carry the signal in their name
# (..._geo_distance / ..._geo_fitness) while the single reported pose-mode
# geometry run does not (3b_cross_geo) — it is mapped to the module default
# signal, STAGE3_GEO_SIGNAL="distance" = --geometry trimmed-distance.
_ARM_FROM_FLAGS = {
    ("retrieval", "cross", "partial", None): "3a_cross_v2",
    ("retrieval", "cross", "fullmesh", None): "3a_cross_fullmesh_v2",
    ("retrieval", "pc", "partial", None): "3a_pc_v2",
    ("retrieval", "pc", "fullmesh", None): "3a_pc_fullmesh_v2",
    ("retrieval", "cross", "partial", "trimmed-distance"):
        "3a_cross_geo_distance",
    ("retrieval", "cross", "partial", "fitness"): "3a_cross_geo_fitness",
    ("retrieval", "pc", "partial", "trimmed-distance"):
        "3a_pc_geo_distance",
    ("retrieval", "pc", "partial", "fitness"): "3a_pc_geo_fitness",
    ("pose", "cross", "partial", None): "3b_cross",
    ("pose", "cross", "fullmesh", None): "3b_cross_fullmesh",
    ("pose", "cross", "partial", "trimmed-distance"): "3b_cross_geo",
    ("decompose", "cross", "partial", None): "3c_cross",
    ("decompose", "cross", "fullmesh", None): "3c_cross_fullmesh",
}
_OSCAR_ARM = {"retrieval": "3a_oscar", "pose": "3b_oscar"}

_ARGS = None
_PATHS = None


def release_configuration(args):
    """Release name of this run's configuration, or ``None`` if unreported.

    ``--uni3d`` swaps the shape encoder and was never reported as a Stage-3
    configuration, so it never gets an archive name (the headline check excludes
    it from comparability for the same reason).  The gt benchmark and the OSCAR
    cascade have no shape channel at all: they ignore --query/--gallery, but a
    geometry re-rank on top of them is a different run than the reported one and
    is left unnamed rather than mislabelled.
    """
    if args.uni3d:
        arm = None
    elif args.mode_word == "gt":
        arm = "gt" if not args.geometry else None
    elif args.oscar_baseline:
        arm = _OSCAR_ARM.get(args.mode_word) if not args.geometry else None
    else:
        arm = _ARM_FROM_FLAGS.get((args.mode_word, args.query, args.gallery,
                                   args.geometry))
    return to_release(arm, "stage3") if arm else None


def _check_dgedi(expected_n, cache_hint):
    """The dGeDi service must be up AND have the right gallery loaded
    (behaviour modelled on repro_experiment.check_dgedi)."""
    import urllib.request
    start_cmd = (f"DGEDI_CACHE_DIR={cache_hint} docker compose up -d "
                 "--force-recreate dgedi")
    for url in ("http://localhost:5061/health", "http://dgedi:5061/health"):
        try:
            h = json.load(urllib.request.urlopen(url, timeout=5))
        except Exception:
            continue
        n = h.get("n_gallery", -1)
        if n == expected_n:
            print(f"[stage3] dGeDi ok: n_gallery={n}", flush=True)
            return
        sys.exit(f"[stage3] dGeDi is up but has n_gallery={n} instead of "
                 f"{expected_n}. Switch with: {start_cmd}   — then start "
                 "again.")
    sys.exit(f"[stage3] dGeDi service unreachable. Start: {start_cmd}")


if __name__ == "__main__":
    import argparse

    from pipeline import paths as _paths_mod

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode",
                        choices=["retrieval", "gt", "pose", "decompose"],
                        default="retrieval",
                        help="retrieval = exact CAD in the gallery, retrieval "
                             "only (3a); gt = FoundationPose with the GT CAD "
                             "(D_posed_gt reference); pose = proxy-only gallery, "
                             "pose the retrieved top-1 (3b); decompose = "
                             "next-best-non-GT diagnostic from a stored "
                             "retrieval run (3c, needs --from-retrieval). "
                             "(default: retrieval)")
    parser.add_argument("--query", choices=["cross", "pc"], default="cross",
                        help="shape-channel query: cross = ULIP-2 image cross "
                             "query, pc = back-projected partial point cloud "
                             "(default: cross)")
    parser.add_argument("--gallery", choices=["partial", "fullmesh"],
                        default="partial",
                        help="shape gallery representation for the ULIP "
                             "channel (default: partial)")
    parser.add_argument("--geometry",
                        choices=["trimmed-distance", "fitness"], default=None,
                        help="add the dGeDi geometry re-rank of the fused "
                             "top-5 with this signal (requires the dgedi "
                             "service with the BOP gallery loaded)")
    parser.add_argument("--oscar-baseline", action="store_true",
                        help="E5: OSCAR's actual mechanism — CLIP-text "
                             "threshold prune (tau=0.37, top-20 fallback) then "
                             "DINOv2 best-view cascade, NO shape (ranked by "
                             "the oscar_maxview arm).")
    parser.add_argument("--datasets", default="all",
                        help="comma-separated query datasets (ycbv,tless,lmo) "
                             "or 'all' (default: all)")
    parser.add_argument("--max-targets", type=int, default=0,
                        help="limit targets PER dataset (0 = all; smoke runs)")
    parser.add_argument("--refine-iter", type=int, default=5,
                        help="FoundationPose refinement iterations (default 5)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uni3d", action="store_true",
                        help="swap the shape arm ULIP-2 -> Uni3D (pc-query).")
    parser.add_argument("--from-retrieval", default=None,
                        help="decompose only: output folder of a previous "
                             "--mode retrieval run whose stored per-dataset "
                             "rankings are reused (relative paths resolve "
                             "against the repo root, e.g. "
                             "runs/stage3_bop_retrieval_cross_partial)")
    parser.add_argument("--gt-records", default=None,
                        help="pose only: combined_gt.json of the --mode gt run "
                             "for the paired Delta (default: "
                             "<runs_root>/stage3_bop_gt/combined_gt.json)")
    parser.add_argument("--out", default="",
                        help="result folder name below runs_root (default "
                             "derived: stage3_bop_<mode>_<query>_<gallery>"
                             "[_geo_<signal>]; stage3_bop_gt for --mode gt)")
    parser.add_argument("--no-docker", action="store_true",
                        help="do not auto-wrap into the oscar-plus container")
    _paths_mod.add_path_args(parser)
    args = parser.parse_args()

    # _seed_everything() only setdefault()s PYTHONHASHSEED with --seed; setting
    # it here first keeps the hash seed pinned to 0 exactly like the
    # repro_experiment driver did, independent of --seed.
    os.environ["PYTHONHASHSEED"] = "0"
    _PATHS = _paths_mod.from_args(args)

    # --- translate the readable flags onto the internal argparse variables ---
    args.mode_word = args.mode
    args.mode = _MODE_WORDS[args.mode_word]
    args.pc_query = (args.query == "pc")
    args.fullmesh = (args.gallery == "fullmesh")
    args.dgedi = args.dgedi_repo = bool(args.geometry)
    args.dgedi_top_k = 5
    if args.geometry:
        os.environ["STAGE3_GEO_SIGNAL"] = (
            "distance" if args.geometry == "trimmed-distance" else "fitness")

    args.from_3a = None
    if args.mode == "3c":
        if not args.from_retrieval:
            sys.exit("[stage3] --mode decompose needs --from-retrieval "
                     "<output folder of a --mode retrieval run> — first run "
                     "e.g. `python3 experiments/stage3_bop.py --mode "
                     "retrieval`.")
        args.from_3a = (args.from_retrieval
                        if os.path.isabs(args.from_retrieval)
                        else os.path.join(_REPO, args.from_retrieval))

    if args.mode == "3b":
        args.gt_records = (args.gt_records or os.path.join(
            _PATHS["runs_root"], "stage3_bop_gt", "combined_gt.json"))
        if not os.path.isabs(args.gt_records):
            args.gt_records = os.path.join(_REPO, args.gt_records)
        if not os.path.isfile(args.gt_records):
            sys.exit(f"[stage3] {args.gt_records} is missing — first run "
                     "`python3 experiments/stage3_bop.py --mode gt` "
                     "(or set --gt-records).")
    elif args.gt_records:
        # explicit --gt-records on other modes (e.g. Delta in decompose)
        if not os.path.isabs(args.gt_records):
            args.gt_records = os.path.join(_REPO, args.gt_records)

    if args.out:
        _out_name = args.out
    elif args.mode_word == "gt":
        _out_name = "stage3_bop_gt"
    else:
        # schema: stage3_bop_<mode>_<query>_<gallery>[_geo_<signal>]; the
        # oscar-baseline cascade ignores query/gallery (no shape channel) and
        # gets its own name so it never overwrites the fused run.
        if args.oscar_baseline:
            _out_name = f"stage3_bop_{args.mode_word}_oscar_baseline"
        else:
            _out_name = f"stage3_bop_{args.mode_word}_{args.query}_{args.gallery}"
        if args.geometry:
            _out_name += "_geo_" + args.geometry.replace("-", "_")
    args.output = os.path.join(_PATHS["runs_root"], _out_name)

    # --geometry: fail fast if the dgedi service is down or holds the wrong
    # gallery (runs on the host before wrapping AND again inside the container;
    # the union gallery G_proxy ∪ G_target has 1316 entries).
    if args.geometry:
        _check_dgedi(1316, "caches/dgedi/bop")

    if not os.path.exists("/.dockerenv") and not args.no_docker:
        if not os.path.isfile(os.path.join(_REPO, "docker-compose.yml")):
            sys.exit(f"[stage3] {_REPO}/docker-compose.yml is missing — "
                     "incomplete repo?")
        print("[stage3] runs in the oscar-plus container — wrapping automatically.",
              flush=True)
        os.chdir(_REPO)  # docker compose needs the repo as CWD
        os.execvp("docker", ["docker", "compose", "run", "--rm", "oscar-plus",
                             "python3", "/app/experiments/stage3_bop.py"]
                  + sys.argv[1:])

    _ARGS = args

if _PATHS is None:  # imported as a module (no CLI): plain paths.yaml defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()

import numpy as np
from PIL import Image
from tqdm import tqdm

from eval_common import run_query, fusion_ranking, crop_by_bbox, _arm_rankings
from query_cloud import backproject_masked
from dgedi_bridge import dgedi_rerank, dgedi_health
from stage3_gallery import (assemble_gallery, TARGET_DATASETS, UNI3D_OVERRIDES,
                            _pose_mesh_path, split_id)
from stage3_metrics import (rank_of_target, summarize_retrieval,
                            sample_surface_mm, d_sym, summarize_dsym,
                            summarize_delta, instance_key)
from pipeline.foundationpose_bridge import call_foundationpose

logger = logging.getLogger(__name__)

# FoundationPose runs in its own container on the compose network. It works in
# METRES; BOP is millimetres — so depth px*depth_scale/1000 -> m, BOP meshes
# (models_eval, mm) pass scale=0.001, and the returned translation *1000 -> mm.
FP_URL = "http://foundationpose:5050/estimate_pose"
_M_TO_MM = 1000.0

# Minimum points in the query partial cloud for the shape/geometry arms; below
# this, skip pc-query encode + dGeDi (degrade to the appearance arms only).
MIN_CLOUD_PTS = 64


def _seed_everything(seed: int = 0):
    """Seed every explicit RNG the eval touches. FoundationPose (separate
    container, GPU) and — on older open3d builds — RANSAC remain stochastic;
    that residual is documented, not hidden."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ============================================================================
# Per-dataset BOP query layout (test scenes + targets)
# ============================================================================
DATASET_TEST = {
    "ycbv":  dict(test_root=os.path.join(_PATHS["datasets_root"], "ycbv", "test"),
                  targets=os.path.join(_PATHS["datasets_root"], "ycbv",
                                       "test_targets_bop19.json")),
    "tless": dict(test_root=os.path.join(_PATHS["datasets_root"], "tless",
                                         "test_primesense"),
                  targets=os.path.join(_PATHS["datasets_root"], "tless",
                                       "test_targets_bop19.json")),
    "lmo":   dict(test_root=os.path.join(_PATHS["datasets_root"], "lmo", "test"),
                  targets=os.path.join(_PATHS["datasets_root"], "lmo",
                                       "test_targets_bop19.json")),
}


# ============================================================================
# BOP loaders
# ============================================================================

def load_bop_targets(path):
    with open(path) as f:
        return json.load(f)


def _load_scene_json(scene_dir, name, im_id):
    p = os.path.join(scene_dir, name)
    if not os.path.isfile(p):
        return []
    with open(p) as f:
        return json.load(f).get(str(im_id), [])


def _matching_instances(scene_dir, im_id, obj_id):
    """All (gt_idx, gt, gt_info) instances of obj_id in an image (inst_count>1
    handled). gt_idx indexes scene_gt — it names the mask_visib file."""
    gts = _load_scene_json(scene_dir, "scene_gt.json", im_id)
    infos = _load_scene_json(scene_dir, "scene_gt_info.json", im_id)
    out = []
    for i, g in enumerate(gts):
        if g.get("obj_id") == obj_id:
            info = infos[i] if i < len(infos) else {}
            out.append((i, g, info))
    return out


def _bbox_of(info):
    b = info.get("bbox_visib") or info.get("bbox_obj")
    if not b or b[2] <= 0 or b[3] <= 0:   # w,h must be positive
        return None
    return b


def _pad_bbox(bbox, img_w, img_h, min_size=16):
    """Grow a tiny (heavily-occluded) bbox to a minimum size, centred and
    clamped to the image (a 1px-thin crop crashes the HF image processor)."""
    x, y, w, h = (float(v) for v in bbox)
    cx, cy = x + w / 2.0, y + h / 2.0
    w, h = max(w, min_size), max(h, min_size)
    x = min(max(0.0, cx - w / 2.0), max(0.0, img_w - w))
    y = min(max(0.0, cy - h / 2.0), max(0.0, img_h - h))
    return [x, y, min(w, img_w), min(h, img_h)]


# ============================================================================
# Pose inputs + FoundationPose call
# ============================================================================

def _cam_entry(scene_dir, im_id):
    p = os.path.join(scene_dir, "scene_camera.json")
    with open(p) as f:
        return json.load(f)[str(im_id)]


def _gt_pose(gt):
    """(R 3x3, t 3) from a scene_gt entry — BOP camera frame, mm."""
    R = np.array(gt["cam_R_m2c"], float).reshape(3, 3)
    t = np.array(gt["cam_t_m2c"], float).reshape(3)
    return R, t


def _pose_inputs(scene_dir, im_id, gt_idx, cam):
    """Full-frame depth (metres for FP), mask, K for one instance."""
    im6 = f"{im_id:06d}"
    depth_raw = np.array(Image.open(os.path.join(scene_dir, "depth", f"{im6}.png")))
    depth_m = depth_raw.astype(np.float32) * float(cam["depth_scale"]) / _M_TO_MM
    mask_p = os.path.join(scene_dir, "mask_visib", f"{im6}_{gt_idx:06d}.png")
    mask = (np.array(Image.open(mask_p)) > 0).astype(np.uint8)
    K = np.array(cam["cam_K"], float).reshape(3, 3)
    return depth_m, mask, K


def estimate_pose(cad_path, rgb_np, depth_m, mask, K, mesh_units_m, refine_iter):
    """FoundationPose register() -> (R 3x3, t 3 in mm, conf). ``mesh_units_m``
    True if the mesh is already in metres (scale 1.0); False for BOP-mm meshes."""
    scale = 1.0 if mesh_units_m else (1.0 / _M_TO_MM)
    pose, conf = call_foundationpose(FP_URL, rgb=rgb_np, depth=depth_m, mask=mask,
                                     K=K, cad_path=cad_path, scale=scale,
                                     refine_iter=refine_iter)
    return pose[:3, :3], pose[:3, 3] * _M_TO_MM, float(conf)


def _models_eval_dir(dataset):
    return os.path.join(_PATHS["datasets_root"], dataset, "models_eval")


class _ModelCache:
    """Lazily loads BOP models_eval diameter + surface sample per obj_id (mm)."""
    def __init__(self, dataset):
        self.dir = _models_eval_dir(dataset)
        self.info = json.load(open(os.path.join(self.dir, "models_info.json")))
        self._c = {}

    def get(self, obj_id):
        if obj_id not in self._c:
            mp = os.path.join(self.dir, f"obj_{obj_id:06d}.ply")
            mi = self.info[str(obj_id)]
            self._c[obj_id] = dict(path=mp, diameter=float(mi["diameter"]),
                                   pts=sample_surface_mm(mp, units_m=False))
        return self._c[obj_id]


# ============================================================================
# Geometry re-rank (dGeDi, Stage-1 E2_both — Borda mean-rank of RANSAC fitness
# and trimmed Chamfer)
# ============================================================================

GEO_SIGNAL = os.environ.get("STAGE3_GEO_SIGNAL", "distance")   # distance | borda | fitness


# Umsortierungsregel seit 2026-09-18 in der Pipeline (Schritt 7) beheimatet —
# von dort importiert, damit Treiber und interaktive Pipeline dieselbe
# Implementierung teilen. Rangfolgen unveraendert (Beleg: Vergleich gegen
# gespeicherte records.json, siehe AI_LOG 2026-09-18).
from pipeline.step7_geometry_reranking import geo_rerank as _geo_rerank  # noqa: E402


# ============================================================================
# Per-object aggregation (concept doc requires per-object tables)
# ============================================================================

def _per_object(dsym_recs, value_key="d_sym"):
    """Group per-instance D_sym records by obj_id -> mean/median + n."""
    by = {}
    for r in dsym_recs:
        by.setdefault(r["obj_id"], []).append(r[value_key])
    out = {}
    for oid, vals in sorted(by.items()):
        a = np.array(vals, float)
        out[str(oid)] = {"n": int(a.size), "mean": float(a.mean()),
                         "median": float(np.median(a))}
    return out


# ============================================================================
# Mode `gt`: exact-CAD FoundationPose benchmark (D_posed_gt, P_P == P_T)
# ============================================================================

def _eval_gt_dataset(dataset, refine_iter, max_targets):
    models = _ModelCache(dataset)
    ds_test = DATASET_TEST[dataset]
    test_root = ds_test["test_root"]
    targets = load_bop_targets(ds_test["targets"])
    if max_targets > 0:
        targets = targets[:max_targets]
    print(f"[stage3-gt] {dataset}: {len(targets)} BOP targets (FP with GT CAD)")

    records = []
    dsym_recs = []
    n_att = 0
    for t in tqdm(targets, desc=f"{dataset} gt"):
        scene_id, im_id, obj_id = t["scene_id"], t["im_id"], t["obj_id"]
        scene_dir = os.path.join(test_root, f"{scene_id:06d}")
        rgb_path = os.path.join(scene_dir, "rgb", f"{im_id:06d}.png")
        if not os.path.isfile(rgb_path):
            alt = rgb_path[:-4] + ".jpg"
            rgb_path = alt if os.path.isfile(alt) else rgb_path
        if not os.path.isfile(rgb_path):
            continue
        rgb_np = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        cam = _cam_entry(scene_dir, im_id)
        for gt_idx, gt, info in _matching_instances(scene_dir, im_id, obj_id):
            if _bbox_of(info) is None:
                continue
            depth_m, mask, K = _pose_inputs(scene_dir, im_id, gt_idx, cam)
            m = models.get(obj_id)
            R_gt, t_gt = _gt_pose(gt)
            n_att += 1
            rec = {"dataset": dataset, "scene_id": scene_id, "im_id": im_id,
                   "obj_id": obj_id, "gt_idx": gt_idx, "diameter": m["diameter"]}
            try:
                R_e, t_e, conf = estimate_pose(m["path"], rgb_np, depth_m, mask,
                                               K, mesh_units_m=False,
                                               refine_iter=refine_iter)
                ds = d_sym(m["pts"], R_gt, t_gt, m["pts"], R_e, t_e, m["diameter"])
                rec.update({"d_posed_gt": round(ds["d_sym"], 3),
                            "d_sym_norm": round(ds["d_sym_norm"], 4),
                            "fscore": ds["fscore"], "pose_conf": round(conf, 4),
                            "R": np.asarray(R_e).reshape(9).tolist(),
                            "t": np.asarray(t_e).reshape(3).tolist()})
                dsym_recs.append({**ds, "obj_id": obj_id,
                                  "_key": instance_key(rec)})
            except Exception as exc:
                logger.warning("FP-gt failed (%s im %s obj %s): %s",
                               scene_id, im_id, obj_id, exc)
                rec["failed"] = True
            records.append(rec)

    summary = {"dataset": dataset, "mode": "gt",
               "n_queries_evaluated": len(records),
               "dsym": summarize_dsym(dsym_recs, n_attempted=n_att),
               "per_object": _per_object(dsym_recs)}
    return {"summary": summary, "records": records, "dsym_recs": dsym_recs}


# ============================================================================
# Mode `3c`: next-best-non-GT diagnostic (reuse the stored 3a ranking, pose the
# best available stand-in that is NOT the exact target, D_sym + Delta)
# ============================================================================

def _load_3a_records(from_3a, dataset):
    """Load the per-dataset 3a records (retrieval rankings) to reuse in 3c.

    ``from_3a`` is a Stage-3 output dir written in --mode 3a; per dataset the
    driver stored ``<ds>_stage3a/records.json`` (each record carries target_id,
    target_rank and the top-10 namespaced shortlist)."""
    p = os.path.join(from_3a, f"{dataset}_stage3a", "records.json")
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"decompose needs the stored retrieval ranking but {p} is missing. "
            f"Point --from-retrieval at a --mode retrieval output dir "
            f"(e.g. runs/stage3_bop_retrieval_cross_partial).")
    with open(p) as f:
        return json.load(f)


def _next_best_non_gt(top10, target_id):
    """Highest-ranked shortlist id that is NOT the exact target (id, score).

    The next-best-non-GT is always within the top-2 (rank 1 if it isn't the
    target, else rank 2), so the stored top-10 always contains it."""
    for e in (top10 or []):
        if e["id"] != target_id:
            return e["id"], e.get("score")
    return None, None


def _provenance_breakdown(records, dsym_recs):
    """Split the next-best D_sym by substitute provenance — a real target CAD of
    another object vs a G_proxy item — the core 3c decomposition."""
    by = {}
    for r in dsym_recs:
        by.setdefault(r.get("provenance", "proxy"), []).append(r["d_sym"])
    out = {}
    for prov, vals in by.items():
        a = np.array(vals, float)
        out[prov] = {"n": int(a.size),
                     "d_sym_mean": float(a.mean()) if a.size else None,
                     "d_sym_median": float(np.median(a)) if a.size else None}
    out["counts"] = {
        "n_target_cad": sum(1 for r in records
                            if r.get("nb_provenance") == "target_cad"),
        "n_proxy": sum(1 for r in records if r.get("nb_provenance") == "proxy"),
        "n_same_dataset": sum(1 for r in records if r.get("nb_same_dataset")),
        "n_target_was_top1": sum(1 for r in records if r.get("target_was_top1")),
    }
    return out


def _eval_3c_dataset(dataset, records_3a, refine_iter, max_targets, gt_by_key,
                     nb_samples):
    """Pose the next-best-non-GT candidate of each reused 3a record and score
    D_sym vs the GT-posed target (+ Delta against the exact-CAD benchmark)."""
    models = _ModelCache(dataset)
    ds_test = DATASET_TEST[dataset]
    test_root = ds_test["test_root"]
    recs_in = records_3a[:max_targets] if max_targets > 0 else records_3a
    print(f"[stage3-3c] {dataset}: {len(recs_in)} reused 3a records "
          f"(pose next-best-non-GT)")

    records, dsym_recs, tgt_samples = [], [], {}
    n_att = n_no_candidate = 0
    for t in tqdm(recs_in, desc=f"{dataset} 3c"):
        scene_id, im_id = t["scene_id"], t["im_id"]
        obj_id, gt_idx = t["obj_id"], t["gt_idx"]
        target_id = t.get("target_id") or f"{dataset}/obj_{obj_id:06d}"
        nb_id, nb_score = _next_best_non_gt(t.get("top10", []), target_id)
        if nb_id is None:
            n_no_candidate += 1
            continue

        scene_dir = os.path.join(test_root, f"{scene_id:06d}")
        rgb_path = os.path.join(scene_dir, "rgb", f"{im_id:06d}.png")
        if not os.path.isfile(rgb_path):
            alt = rgb_path[:-4] + ".jpg"
            rgb_path = alt if os.path.isfile(alt) else rgb_path
        if not os.path.isfile(rgb_path):
            continue
        rgb_np = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        cam = _cam_entry(scene_dir, im_id)
        gts = _load_scene_json(scene_dir, "scene_gt.json", im_id)
        if gt_idx >= len(gts):
            continue
        R_gt, t_gt = _gt_pose(gts[gt_idx])
        depth_m, mask, K = _pose_inputs(scene_dir, im_id, gt_idx, cam)
        m = models.get(obj_id)

        nb_ds, nb_obj = split_id(nb_id)
        provenance = "target_cad" if nb_ds in TARGET_DATASETS else "proxy"
        rec = {"dataset": dataset, "scene_id": scene_id, "im_id": im_id,
               "obj_id": obj_id, "gt_idx": gt_idx, "target_id": target_id,
               "target_rank": t.get("target_rank"),
               "target_was_top1": (t.get("target_rank") == 1),
               "nb_id": nb_id, "nb_score": nb_score,
               "nb_provenance": provenance,
               "nb_same_dataset": (nb_ds == dataset), "diameter": m["diameter"]}

        nb_path, nb_units = _pose_mesh_path(nb_ds, nb_obj)
        if not (nb_path and os.path.isfile(nb_path)):
            logger.warning("3c next-best mesh missing for %s -> %s", nb_id, nb_path)
            rec["failed"] = True
            records.append(rec)
            continue
        n_att += 1
        try:
            if obj_id not in tgt_samples:
                tgt_samples[obj_id] = m["pts"]
            R_e, t_e, conf = estimate_pose(nb_path, rgb_np, depth_m, mask, K,
                                           mesh_units_m=nb_units,
                                           refine_iter=refine_iter)
            if nb_id not in nb_samples:
                nb_samples[nb_id] = sample_surface_mm(nb_path, units_m=nb_units)
            ds = d_sym(tgt_samples[obj_id], R_gt, t_gt,
                       nb_samples[nb_id], R_e, t_e, m["diameter"])
            rec.update({"d_posed": round(ds["d_sym"], 3),
                        "d_sym_norm": round(ds["d_sym_norm"], 4),
                        "fscore": ds["fscore"], "nb_pose_conf": round(conf, 4),
                        "nb_R": np.asarray(R_e).reshape(9).tolist(),
                        "nb_t": np.asarray(t_e).reshape(3).tolist()})
            key = instance_key(rec)
            drec = {**ds, "obj_id": obj_id, "_key": key, "provenance": provenance}
            if gt_by_key and key in gt_by_key:
                rec["delta"] = round(ds["d_sym"] - gt_by_key[key], 3)
            dsym_recs.append(drec)
        except Exception as exc:
            logger.warning("3c FP failed (%s): %s", nb_id, exc)
            rec["failed"] = True
        records.append(rec)

    summary = {"dataset": dataset, "mode": "3c",
               "n_queries_evaluated": len(records),
               "n_no_candidate": n_no_candidate,
               "dsym": summarize_dsym(dsym_recs, n_attempted=n_att),
               "provenance": _provenance_breakdown(records, dsym_recs),
               "per_object": _per_object(dsym_recs)}
    if gt_by_key:
        summary["delta"] = summarize_delta(dsym_recs, gt_by_key)
    return {"summary": summary, "records": records, "dsym_recs": dsym_recs}


# ============================================================================
# Modes `3a` / `3b`: retrieval (+ 3b proxy pose + D_sym + Delta)
# ============================================================================

def _eval_retrieval_dataset(dataset, gallery, components, mode, max_targets,
                            refine_iter, prx_samples, gt_by_key,
                            use_uni3d=False, use_dgedi=False, dgedi_top_k=10,
                            use_pc_query=False, dgedi_repo=False,
                            oscar_baseline=False):
    """3a: retrieval only. 3b: retrieval (proxy gallery) + FP top-1 + D_sym."""
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = components
    cfg = gallery.eval_cfg
    include_target = (mode == "3a")
    do_pose = (mode == "3b")
    G = len(gallery.gallery_ids)
    top_k = G + 5
    clip_rows = len(clip_retr._desc_labels)

    models = _ModelCache(dataset) if do_pose else None
    need_cloud = use_uni3d or use_pc_query or use_dgedi or do_pose

    ds_test = DATASET_TEST[dataset]
    test_root = ds_test["test_root"]
    targets = load_bop_targets(ds_test["targets"])
    if max_targets > 0:
        targets = targets[:max_targets]
    print(f"[stage3] {dataset} {mode}: {len(targets)} BOP targets vs |gallery|={G}")

    ranks, fused_ranks, records, dsym_recs = [], [], [], []
    tgt_samples = {}
    n_missing_rgb = 0

    for t in tqdm(targets, desc=f"{dataset} {mode}"):
        scene_id, im_id, obj_id = t["scene_id"], t["im_id"], t["obj_id"]
        scene_dir = os.path.join(test_root, f"{scene_id:06d}")
        rgb_path = os.path.join(scene_dir, "rgb", f"{im_id:06d}.png")
        if not os.path.isfile(rgb_path):
            alt = rgb_path[:-4] + ".jpg"
            rgb_path = alt if os.path.isfile(alt) else rgb_path
        if not os.path.isfile(rgb_path):
            n_missing_rgb += 1
            continue
        rgb = Image.open(rgb_path).convert("RGB")
        rgb_np = np.asarray(rgb, dtype=np.uint8)
        cam = _cam_entry(scene_dir, im_id) if need_cloud else None
        target_nsid = f"{dataset}/obj_{obj_id:06d}"

        for gt_idx, gt, info in _matching_instances(scene_dir, im_id, obj_id):
            bbox = _bbox_of(info)
            if bbox is None:
                continue
            roi = crop_by_bbox(rgb, _pad_bbox(bbox, rgb.width, rgb.height))

            depth_m = mask = K = None
            if need_cloud:
                depth_m, mask, K = _pose_inputs(scene_dir, im_id, gt_idx, cam)

            q_cloud = q_colors = None
            if need_cloud and mask is not None:
                q_cloud, q_colors = backproject_masked(depth_m, mask, K, rgb=rgb_np)
                if len(q_cloud) < MIN_CLOUD_PTS:
                    q_cloud = q_colors = None

            ulip_q_emb = None
            pc_query_fallback = False
            if use_uni3d or use_pc_query:
                if q_cloud is not None:
                    try:
                        ulip_q_emb = shape_m.encode_pointcloud(q_cloud, colors=q_colors)
                    except Exception as exc:
                        logger.warning("pc-query encode failed (%s im %s obj %s): %s",
                                       scene_id, im_id, obj_id, exc)
                if ulip_q_emb is None:
                    pc_query_fallback = True

            out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m,
                            roi, cfg, ulip_query_emb=ulip_q_emb,
                            dino_full_top_k=top_k, ulip_full_top_k=top_k,
                            clip_full_top_k=clip_rows)
            # E5 OSCAR baseline: rank by the faithful cascade (CLIP-τ shortlist
            # -> DINOv2 best-view, no shape). Else the OSCAR+ full 3-way fusion.
            if oscar_baseline:
                fused_ranking = _arm_rankings(out)["oscar_maxview"]
            else:
                fused_ranking = fusion_ranking(out["fused_full"])

            ranking = fused_ranking
            geo_applied = False
            dgedi_n_req = dgedi_n_ok = 0
            if use_dgedi and q_cloud is not None:
                cand_ids = [oid for oid, _ in fused_ranking[:dgedi_top_k]]
                _dg = ({"ransac_keypoints": 6000, "ransac_max_iter": 10000,
                        "use_icp": True} if dgedi_repo else {})
                geo = dgedi_rerank(q_cloud, cand_ids, **_dg)
                if geo:
                    dgedi_n_req = len(cand_ids)
                    dgedi_n_ok = sum(1 for v in geo.values() if v.get("ok"))
                    if dgedi_n_ok > 0:
                        ranking = _geo_rerank(fused_ranking, geo, dgedi_top_k)
                        geo_applied = True
                        # Rohwerte mitschreiben: damit ist das Rangkriterium
                        # spaeter offline ableitbar (Tier-2, wie in Stage 1) und
                        # ein Wechsel kostet keine neuen Registrierungen.
                        geo_raw = {oid: {"fitness": float(g.get("ransac_fitness", 0.0)),
                                         "d_ransac": (float(g["d_ransac"])
                                                      if g.get("d_ransac") is not None else None),
                                         "ok": bool(g.get("ok"))}
                                   for oid, g in geo.items()}

            r = rank_of_target(ranking, target_nsid) if include_target else None
            if include_target:
                ranks.append(r)

            rec = {"dataset": dataset, "scene_id": scene_id, "im_id": im_id,
                   "obj_id": obj_id, "gt_idx": gt_idx, "target_id": target_nsid,
                   "target_rank": r,
                   # ranked shortlist with fused/geo scores (top-10, matching the
                   # deepest reported Recall@k). In 3b these are the proxies that
                   # displaced the removed exact target.
                   "top10": [{"id": oid, "score": round(s, 5)}
                             for oid, s in ranking[:10]]}
            # Rang des Ziels je EINZELKANAL. run_query berechnet diese Arme
            # ohnehin in demselben Durchlauf (_arm_rankings); sie kosten hier
            # nur die Rangsuche. Damit liefert jeder Lauf den isolierten
            # Shape-Kanal (`ulip_only_full`) gratis mit — sonst braeuchte man
            # dafuer eigene Laeufe mit Gewichten (0,0,1), und der
            # partial-vs-full-mesh-Effekt bliebe hinter der Fusion verborgen,
            # die ihn zu 70 % abfedert.
            if include_target:
                try:
                    rec["arm_ranks"] = {
                        arm: rank_of_target(rk, target_nsid)
                        for arm, rk in _arm_rankings(out).items()}
                except Exception as exc:            # nie den Lauf abbrechen
                    rec["arm_ranks_error"] = str(exc)[:120]
            if use_uni3d or use_pc_query:
                rec["pc_query_fallback"] = pc_query_fallback
            if geo_applied:
                rec["geo_raw"] = geo_raw
                rec["geo_signal"] = GEO_SIGNAL
            if use_dgedi:
                rec["fused_rank"] = (rank_of_target(fused_ranking, target_nsid)
                                     if include_target else None)
                rec["geo_applied"] = geo_applied
                rec["dgedi_n_requested"] = dgedi_n_req
                rec["dgedi_n_ok"] = dgedi_n_ok
                if include_target:
                    fused_ranks.append(rec["fused_rank"])

            # --- 3b: pose the RETRIEVED top-1 proxy, D_sym vs GT-posed target ---
            if do_pose and ranking:
                m = models.get(obj_id)
                R_gt, t_gt = _gt_pose(gt)
                top1 = ranking[0][0]
                rec["top1"] = top1
                rec["top1_is_exact"] = (top1 == target_nsid)
                tpath, tunits = gallery.id_to_pose_mesh.get(top1, (None, False))
                if tpath and os.path.isfile(tpath):
                    if obj_id not in tgt_samples:
                        tgt_samples[obj_id] = m["pts"]
                    try:
                        Rt, tt, conf = estimate_pose(tpath, rgb_np, depth_m, mask,
                                                     K, mesh_units_m=tunits,
                                                     refine_iter=refine_iter)
                        if top1 not in prx_samples:
                            prx_samples[top1] = sample_surface_mm(tpath, units_m=tunits)
                        ds = d_sym(tgt_samples[obj_id], R_gt, t_gt,
                                   prx_samples[top1], Rt, tt, m["diameter"])
                        rec.update({"d_posed": round(ds["d_sym"], 3),
                                    "d_sym_norm": round(ds["d_sym_norm"], 4),
                                    "fscore": ds["fscore"],
                                    "top1_pose_conf": round(conf, 4),
                                    "diameter": m["diameter"],
                                    "top1_R": np.asarray(Rt).reshape(9).tolist(),
                                    "top1_t": np.asarray(tt).reshape(3).tolist()})
                        key = instance_key(rec)
                        drec = {**ds, "obj_id": obj_id, "_key": key}
                        if gt_by_key and key in gt_by_key:
                            rec["delta"] = round(ds["d_sym"] - gt_by_key[key], 3)
                        dsym_recs.append(drec)
                    except Exception as exc:
                        logger.warning("FP top-1 pose failed (%s): %s", top1, exc)
                else:
                    logger.warning("top-1 mesh missing for %s", top1)

            records.append(rec)

    summary = _summarize(dataset, mode, G, records, ranks, dsym_recs,
                         n_missing_rgb, include_target, do_pose,
                         gt_by_key=gt_by_key,
                         fused_ranks=fused_ranks if use_dgedi else None)
    return {"summary": summary, "records": records, "ranks": ranks,
            "fused_ranks": fused_ranks, "dsym_recs": dsym_recs}


def _summarize(dataset, mode, G, records, ranks, dsym_recs, n_missing_rgb,
               include_target, do_pose, gt_by_key=None, fused_ranks=None):
    summary = {"dataset": dataset, "mode": mode, "gallery_size": G,
               "target_in_gallery": include_target,
               "n_queries_evaluated": len(records),
               "n_missing_rgb": n_missing_rgb}
    if include_target:
        summary.update(summarize_retrieval(ranks))
        if fused_ranks:
            summary["pre_geometry"] = summarize_retrieval(fused_ranks)
    if any(r and "pc_query_fallback" in r for r in records):
        n_pc = sum(1 for r in records if r and "pc_query_fallback" in r)
        summary["pc_query_fallback"] = {
            "n_fell_back": sum(1 for r in records if r and r.get("pc_query_fallback")),
            "n_pc_query": n_pc}
    if any(r and "geo_applied" in r for r in records):
        oks = [r.get("dgedi_n_ok", 0) for r in records if r and "geo_applied" in r]
        summary["geometry_coverage"] = {
            "n_geo_applied": sum(1 for r in records if r and r.get("geo_applied")),
            "n_dgedi_query": len(oks),
            "mean_n_registered": float(np.mean(oks)) if oks else 0.0}
    if do_pose:
        n_att = sum(1 for r in records if r and "top1" in r)
        summary["dsym"] = summarize_dsym(dsym_recs, n_attempted=n_att)
        summary["per_object"] = _per_object(dsym_recs)
        if gt_by_key:
            summary["delta"] = summarize_delta(dsym_recs, gt_by_key)
    return summary


def _print_summary(tag, s):
    print(f"\n[stage3] {tag} — {s['n_queries_evaluated']} queries")
    if "recall@1" in s:
        print(f"  Recall@1={s['recall@1']:.3f}  Recall@5={s['recall@5']:.3f}  "
              f"Recall@10={s['recall@10']:.3f}  MRR={s['mrr']:.3f}  "
              f"(found {s['n_target_found']}/{s['n_queries_evaluated']})")
        if "pre_geometry" in s:
            p = s["pre_geometry"]
            print(f"  pre-geometry: Recall@1={p['recall@1']:.3f} "
                  f"Recall@5={p['recall@5']:.3f} MRR={p['mrr']:.3f}")
    if "geometry_coverage" in s:
        g = s["geometry_coverage"]
        print(f"  geometry: applied {g['n_geo_applied']}/{g['n_dgedi_query']}, "
              f"mean #registered={g['mean_n_registered']:.2f}")
    if "dsym" in s and s["dsym"].get("n_estimated"):
        d = s["dsym"]
        line = (f"  D_sym mean={d['d_sym_mean']:.2f}mm median={d['d_sym_median']:.2f} "
                f"/diam={d['d_sym_norm_mean']:.3f} (n={d['n_estimated']}, "
                f"cov={d.get('coverage', 1.0):.2f})")
        if "fscore" in d:
            fs = " ".join(f"F@{k}={v['f']:.3f}" for k, v in d["fscore"].items())
            line += f"  {fs}"
        print(line)
    if "delta" in s and s["delta"].get("n_paired"):
        dl = s["delta"]
        print(f"  Delta mean={dl['delta_mean']:.2f}mm median={dl['delta_median']:.2f} "
              f"(paired n={dl['n_paired']})")
    if "provenance" in s:
        pv = s["provenance"]
        c = pv.get("counts", {})
        print(f"  next-best provenance: target_cad={c.get('n_target_cad',0)} "
              f"proxy={c.get('n_proxy',0)} same-ds={c.get('n_same_dataset',0)} "
              f"(target-was-top1={c.get('n_target_was_top1',0)})")
        for prov in ("target_cad", "proxy"):
            b = pv.get(prov)
            if b and b.get("n"):
                print(f"    {prov:<10} n={b['n']:>5}  D_sym mean={b['d_sym_mean']:.2f}mm "
                      f"median={b['d_sym_median']:.2f}")


# ============================================================================
# Driver
# ============================================================================

def _load_gt_by_key(path):
    """Build instance_key -> D_posed_gt (mm) from a gt-benchmark records/combined
    file, so 3b can pair each proxy D_posed with its exact-CAD reference."""
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    recs = data.get("all_records") if isinstance(data, dict) else data
    if recs is None and isinstance(data, dict):
        recs = data.get("records", [])
    out = {}
    for r in (recs or []):
        if r.get("d_posed_gt") is None:
            continue
        k = (r.get("dataset"), r["scene_id"], r["im_id"], r["obj_id"], r["gt_idx"])
        out[k] = float(r["d_posed_gt"])
    return out


def run_stage3(datasets, mode="3a", max_targets=0,
               output_dir="results_bop_stage3", refine_iter=5,
               use_uni3d=False, use_dgedi=False, dgedi_top_k=10,
               use_pc_query=False, dgedi_repo=False, gt_records=None,
               from_3a=None, oscar_baseline=False, use_fullmesh=False):
    for d in datasets:
        if d not in DATASET_TEST:
            raise ValueError(f"Unknown dataset {d}; choose {list(DATASET_TEST)}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*64}\nStage-3 {mode} — queries {datasets}\n{'='*64}")

    # ---- mode gt: no gallery, no retrieval — FP with the GT CAD ----
    if mode == "gt":
        per_dataset, all_records, all_dsym = {}, [], []
        for dataset in datasets:
            res = _eval_gt_dataset(dataset, refine_iter, max_targets)
            per_dataset[dataset] = res["summary"]
            rdir = os.path.join(output_dir, f"{dataset}_gt")
            os.makedirs(rdir, exist_ok=True)
            with open(os.path.join(rdir, "records.json"), "w") as f:
                json.dump(res["records"], f, indent=2)
            with open(os.path.join(rdir, "summary.json"), "w") as f:
                json.dump(res["summary"], f, indent=2)
            _print_summary(f"{dataset} gt", res["summary"])
            all_records += res["records"]
            all_dsym += res["dsym_recs"]
        combined = {"mode": "gt", "datasets": datasets,
                    "n_queries_evaluated": len(all_records),
                    "dsym": summarize_dsym(all_dsym, n_attempted=len(all_records)),
                    "per_dataset": per_dataset, "all_records": all_records}
        with open(os.path.join(output_dir, "combined_gt.json"), "w") as f:
            json.dump(combined, f, indent=2)
        _print_summary("COMBINED gt", combined)
        print(f"\n[stage3] saved -> {output_dir}")
        return combined

    # ---- mode 3c: reuse the stored 3a ranking; pose the next-best-non-GT ----
    if mode == "3c":
        if not from_3a:
            raise ValueError("mode 3c/decompose requires --from-retrieval "
                             "<--mode retrieval output dir> (e.g. "
                             "runs/stage3_bop_retrieval_cross_partial)")
        gt_by_key = _load_gt_by_key(gt_records)
        if gt_records:
            print(f"[stage3] paired Delta against {len(gt_by_key)} gt records "
                  f"from {gt_records}")
        print(f"[stage3] 3c reuses 3a rankings from {from_3a}")
        per_dataset, all_records, all_dsym = {}, [], []
        nb_samples = {}
        for dataset in datasets:
            recs3a = _load_3a_records(from_3a, dataset)
            res = _eval_3c_dataset(dataset, recs3a, refine_iter, max_targets,
                                   gt_by_key, nb_samples)
            per_dataset[dataset] = res["summary"]
            rdir = os.path.join(output_dir, f"{dataset}_stage3c")
            os.makedirs(rdir, exist_ok=True)
            with open(os.path.join(rdir, "records.json"), "w") as f:
                json.dump(res["records"], f, indent=2)
            with open(os.path.join(rdir, "summary.json"), "w") as f:
                json.dump(res["summary"], f, indent=2)
            _print_summary(f"{dataset} 3c", res["summary"])
            all_records += res["records"]
            all_dsym += res["dsym_recs"]
        n_att = sum(1 for r in all_records if r and "nb_id" in r
                    and not r.get("failed"))
        combined = {"mode": "3c", "datasets": datasets, "from_3a": from_3a,
                    "n_queries_evaluated": len(all_records),
                    "dsym": summarize_dsym(all_dsym, n_attempted=n_att),
                    "provenance": _provenance_breakdown(all_records, all_dsym),
                    "per_dataset": per_dataset, "all_records": all_records}
        if gt_by_key:
            combined["delta"] = summarize_delta(all_dsym, gt_by_key)
        with open(os.path.join(output_dir, "combined_stage3c.json"), "w") as f:
            json.dump(combined, f, indent=2)
        _print_summary("COMBINED 3c", combined)
        print(f"\n[stage3] saved -> {output_dir}")
        return combined

    # ---- modes 3a / 3b: assemble the union gallery once ----
    gt_by_key = _load_gt_by_key(gt_records) if mode == "3b" else {}
    if mode == "3b" and gt_records:
        print(f"[stage3] paired Delta against {len(gt_by_key)} gt records "
              f"from {gt_records}")
    target_datasets = TARGET_DATASETS if mode == "3a" else ()
    print(f"[stage3] assembling gallery (targets in gallery: "
          f"{list(target_datasets) or 'none (proxy-only)'})"
          f"{'  [shape arm: Uni3D]' if use_uni3d else ''}...")
    gallery = assemble_gallery(target_datasets=target_datasets,
                               extra_overrides=(UNI3D_OVERRIDES if use_uni3d else None),
                               oscar_cascade=oscar_baseline,
                               use_partial=(not use_fullmesh))
    components = gallery.components()
    G = len(gallery.gallery_ids)
    print(f"[stage3] |gallery| = {G}  clip_rows = {len(components[1]._desc_labels)}")
    if use_dgedi:
        h = dgedi_health()
        print(f"[stage3] dGeDi geometry re-rank ON (top_k={dgedi_top_k}, "
              f"{'repo 6000kp/10k/+ICP' if dgedi_repo else 'fast 512kp/5k'}); "
              f"service: {h if h else 'UNREACHABLE — will degrade to fused'}")

    prx_samples = {}
    per_dataset = {}
    pooled = {"ranks": [], "dsym_recs": [], "all_records": []}
    for dataset in datasets:
        res = _eval_retrieval_dataset(
            dataset, gallery, components, mode, max_targets, refine_iter,
            prx_samples, gt_by_key, use_uni3d=use_uni3d, use_dgedi=use_dgedi,
            dgedi_top_k=dgedi_top_k, use_pc_query=use_pc_query,
            dgedi_repo=dgedi_repo, oscar_baseline=oscar_baseline)
        s = res["summary"]
        per_dataset[dataset] = s
        rdir = os.path.join(output_dir, f"{dataset}_stage{mode}")
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, "records.json"), "w") as f:
            json.dump(res["records"], f, indent=2)
        with open(os.path.join(rdir, "summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        _print_summary(f"{dataset} {mode}", s)
        pooled["all_records"] += res["records"]
        pooled["ranks"] += res["ranks"]
        pooled["dsym_recs"] += res["dsym_recs"]

    combined = _summarize("ALL", mode, G, pooled["all_records"], pooled["ranks"],
                          pooled["dsym_recs"], 0, mode == "3a", mode == "3b",
                          gt_by_key=gt_by_key,
                          fused_ranks=[r.get("fused_rank") for r in
                                       pooled["all_records"] if "fused_rank" in r]
                                      if use_dgedi else None)
    combined["datasets"] = datasets
    combined["per_dataset"] = per_dataset
    with open(os.path.join(output_dir, f"combined_stage{mode}.json"), "w") as f:
        json.dump(combined, f, indent=2)
    if len(datasets) > 1:
        _print_summary(f"COMBINED {mode}", combined)
    print(f"\n[stage3] saved -> {output_dir}")
    return combined


# ============================================================================
# Release naming: the archive-named copies of a finished run
# ============================================================================
# results/stage3_bop/ names each configuration by its RELEASE name and keeps two
# artefacts per configuration:
#
#   <release>.json              the combined summary of the run
#   per_query/<release>/<ds>.json   the per-dataset records
#
# A run reproduces both next to its own internal-named files, so the record and
# a re-run can be diffed directly instead of translated by hand.

def _link_or_copy(src: str, dst: str) -> str:
    """Hard-link ``src`` to ``dst``, copying only if that is impossible.

    The per-dataset records are 1.5-7 MB each, 13-25 MB per configuration; the
    archive-named artefact is by definition the SAME bytes as the file the run
    just wrote, so a hard link states that identity at zero disk cost.  A hard
    link (not a symlink) because it stays a plain regular file for every reader:
    nothing dangles if the source folder is later renamed or the records are
    moved.  Falls back to a copy across filesystems or on filesystems without
    link support.  Returns "link" or "copy" for the log.
    """
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        import shutil
        shutil.copyfile(src, dst)
        return "copy"


def export_release_artifacts(output_dir, mode, release, smoke=False):
    """Write ``<release>.json`` and ``per_query/<release>/<ds>.json``."""
    import shutil

    combined = os.path.join(
        output_dir, "combined_gt.json" if mode == "gt"
        else f"combined_stage{mode}.json")
    if not os.path.isfile(combined):
        print(f"[release] {combined} is missing — no archive copy.")
        return
    tag = "  [SMOKE]" if smoke else ""
    # A real copy, not a link: this is the file a reader diffs against the
    # record, and it must keep the numbers of THIS run even if the run folder is
    # reused for another configuration later (which truncates combined_*.json).
    shutil.copyfile(combined, os.path.join(output_dir, f"{release}.json"))
    print(f"[release] {release}.json <- "
          f"{os.path.basename(combined)}{tag}")

    with open(combined) as fh:
        datasets = json.load(fh).get("datasets") or []
    if not datasets:
        return
    pq_dir = os.path.join(output_dir, "per_query", release)
    os.makedirs(pq_dir, exist_ok=True)
    how = []
    for ds in datasets:
        sub = f"{ds}_gt" if mode == "gt" else f"{ds}_stage{mode}"
        src = os.path.join(output_dir, sub, "records.json")
        if not os.path.isfile(src):
            print(f"[release] WARNING: {src} is missing — "
                  f"per_query/{release}/{ds}.json not written.")
            continue
        how.append(_link_or_copy(src, os.path.join(pq_dir, f"{ds}.json")))
    if how:
        print(f"[release] per_query/{release}/: {len(how)} datasets "
              f"({'/'.join(sorted(set(how)))}){tag}")


# ============================================================================
# CLI (the argparse section lives in the prelude at the top of the file; the
# readable flags are translated onto the internal attributes there)
# ============================================================================

def main():
    args = _ARGS
    _seed_everything(args.seed)
    datasets = (list(TARGET_DATASETS) if args.datasets == "all"
                else [d.strip() for d in args.datasets.split(",") if d.strip()])

    logging.basicConfig(level=logging.WARNING)
    run_stage3(datasets, mode=args.mode, max_targets=args.max_targets,
               output_dir=args.output, refine_iter=args.refine_iter,
               use_uni3d=args.uni3d, use_dgedi=args.dgedi,
               dgedi_top_k=args.dgedi_top_k, use_pc_query=args.pc_query,
               dgedi_repo=args.dgedi_repo, gt_records=args.gt_records,
               from_3a=args.from_3a, oscar_baseline=args.oscar_baseline,
               use_fullmesh=args.fullmesh)


if __name__ == "__main__":
    import datetime
    import subprocess

    # Full run provenance next to the results (argv + derived internal values
    # incl. STAGE3_GEO_SIGNAL + git rev), written BEFORE the run starts.
    os.makedirs(_ARGS.output, exist_ok=True)
    try:
        _rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        _rev = ""
    _derived = {"mode": _ARGS.mode, "mode_word": _ARGS.mode_word,
                "query": _ARGS.query, "gallery": _ARGS.gallery,
                "pc_query": _ARGS.pc_query, "fullmesh": _ARGS.fullmesh,
                "geometry": _ARGS.geometry, "dgedi": _ARGS.dgedi,
                "dgedi_repo": _ARGS.dgedi_repo,
                "dgedi_top_k": _ARGS.dgedi_top_k,
                "oscar_baseline": _ARGS.oscar_baseline, "uni3d": _ARGS.uni3d,
                "datasets": _ARGS.datasets, "max_targets": _ARGS.max_targets,
                "refine_iter": _ARGS.refine_iter, "seed": _ARGS.seed,
                "gt_records": _ARGS.gt_records, "from_3a": _ARGS.from_3a,
                "output": _ARGS.output,
                "STAGE3_GEO_SIGNAL": os.environ.get("STAGE3_GEO_SIGNAL"),
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}
    json.dump({"argv": sys.argv[1:], "derived": _derived, "git": _rev,
               "time": datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(_ARGS.output, "run_config.json"), "w"),
              indent=1)

    main()

    # Immediate headline check against the frozen thesis record
    # (results/stage3_bop) — only comparable with --datasets all and without
    # --max-targets, and only for the documented reference configurations.
    _comb = os.path.join(_ARGS.output,
                         "combined_gt.json" if _ARGS.mode == "gt"
                         else f"combined_stage{_ARGS.mode}.json")
    if not os.path.isfile(_comb):
        sys.exit(f"[stage3] no {_comb} — incomplete run.")

    # Archive-named copies, so this run is shaped like results/stage3_bop.
    _RELEASE = release_configuration(_ARGS)
    if _RELEASE:
        export_release_artifacts(_ARGS.output, _ARGS.mode, _RELEASE,
                                 smoke=bool(_ARGS.max_targets))
    else:
        print(f"[release] --mode {_ARGS.mode_word} --query {_ARGS.query} "
              f"--gallery {_ARGS.gallery}"
              + (f" --geometry {_ARGS.geometry}" if _ARGS.geometry else "")
              + " is not a reported configuration — no archive copy "
                "(only the run's combined_*/records.json).")

    _d = json.load(open(_comb))
    _WHAT = (f"{_RELEASE} ({_ARGS.mode_word})" if _RELEASE
             else _ARGS.mode_word)
    if _ARGS.mode == "3a":
        _line = f"RESULT {_WHAT}: R@1 {_d['recall@1']:.4f}"
    else:
        _line = (f"RESULT {_WHAT}: D_sym median "
                 f"{_d['dsym']['d_sym_median']:.2f} mm")
    _ref = None
    if _ARGS.mode == "3a" and not _ARGS.geometry:
        if _ARGS.oscar_baseline:
            _ref = ("retrieval_oscar_baseline.json", "R@1 0.3198")
        else:
            _ref = {("cross", "partial"):
                        ("retrieval_cross_partial.json", "R@1 0.4818"),
                    ("cross", "fullmesh"):
                        ("retrieval_cross_fullmesh.json", "R@1 0.5151"),
                    ("pc", "partial"):
                        ("retrieval_pc_partial.json", "R@1 0.4636"),
                    ("pc", "fullmesh"):
                        ("retrieval_pc_fullmesh.json", "R@1 0.3878"),
                    }.get((_ARGS.query, _ARGS.gallery))
    elif _ARGS.mode == "gt":
        _ref = ("pose_gt.json", "D_sym median 1.72 mm")
    elif (_ARGS.mode == "3b" and (_ARGS.query, _ARGS.gallery) ==
            ("cross", "partial") and not _ARGS.geometry
            and not _ARGS.oscar_baseline):
        _ref = ("pose_proxy_cross_partial.json", "D_sym median 18.37 mm")
    _comparable = (_ARGS.datasets == "all" and not _ARGS.max_targets
                   and not _ARGS.uni3d)
    if _ref and _comparable:
        _line += f"   (reference results/stage3_bop/{_ref[0]}: {_ref[1]})"
    elif _ARGS.max_targets:
        _line += "   [SMOKE — not comparable]"
    print(_line, flush=True)
