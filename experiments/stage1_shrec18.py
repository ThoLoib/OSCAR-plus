#!/usr/bin/env python3
"""
stage1_shrec18.py
=================
Experiment 1 of the thesis — **Stage 1: RGB-D Retrieval Tuning** on
SHREC'18 ObjectNN+ [phamSHREC18RGBDObjecttoCAD2018].

One invocation = one configuration = one *arm* of the grid.  The readable
flags (``--appearance``, ``--shape``, ``--shape-mode``, ``--gallery``,
``--views``, ``--weights``, ``--fusion``, ``--scope``, ``--cascade``,
``--clip-select``, ``--geometry``) are translated onto the internal arm names
of the registry by :func:`derive_arm`; the flag -> arm table is printed at the
end of ``--help`` and again whenever a combination is not covered.  Only the
configurations REPORTED in the thesis are runnable — any other combination
exits with that table instead of quietly evaluating something unreported.

Paths come from ``config/paths.yaml`` (overridable per flag, see ``--help``).
Outside the container the script wraps itself into ``docker compose run``
automatically; geometry arms first check that the dgedi service is up with the
SHREC gallery loaded (``n_gallery`` = 3308).

Thesis references (chapters/evaluation.tex):
  * Stage 1 protocol .......... subsec:eval_stage1_retrieval
  * Ablation definitions ...... subsec:eval_baselines  (E1..E7, O1..O5)
  * Metrics ................... subsec:eval_retrieval_metrics and
                                background.tex subsec:retrieval_metrics
  * Result placeholders ....... tab:eval_stage1_ablation_grid,
                                fig:eval_stage1_ablation_bars

What this script does
---------------------
Runs the *retrieval-side* Stage-1 ablation grid of the OSCAR+ pipeline
[pulliOSCAROpenSetCAD2025] on SHREC'18 — one arm per invocation, see "How to
run" below — from which the best configuration is selected by **highest DCG,
tie-break mAP** (thesis subsec:eval_stage1_retrieval).  The selected
configuration is frozen and reused unchanged in Stages 2, 3a, 3b and 5.

The grid covered here (pose-side ablations E3/E5 are deferred to Stage 3a):

  E1   channel set (each channel alone / S_text+S_view / full fusion /
       CLIP-pruned / faithful OSCAR cascade)
  E2   local geometry re-ranking: none / RANSAC fitness / trimmed distance
       after RANSAC alignment / both by mean rank            [--geometry]
  E2b  shape reference: partial rendered views vs. full mesh
  E4   appearance encoder: DINOv2 vs. SigLIP  [zhaiSigmoidLossLanguage2023]
  E6   fusion: weighted sum vs. reciprocal rank fusion
       [cormackReciprocalRankFusion2009]
  E7   shape encoder: ULIP-2 vs. Uni3D
       [xueULIP2ScalableMultimodal2024, zhouUni3DExploringUnified2023]
  O1   S_shape redundancy vs. S_GeDi (4 configs)     [--geometry for c/e]
  O2   scope/ordering: full-database fusion vs. CLIP-pruned cascade
       (OSCAR) vs. visual-first (Stubborn Strawberries,
       [nguyenSHREC2025Retrieval2025])
  O4   number of reference views V ∈ {8, 16, 32, 42}
       [nguyenCNOSStrongBaseline2023]
  O5   query point cloud XYZ+RGB vs. XYZ-only

Execution model (cost control)
------------------------------
Two tiers:

  * Tier 1 — *channel-score passes*: the expensive encoder work.  Each
    pass computes, per query, a full score vector over the gallery for
    one (encoder x reference) combination and caches it under
    ``<results>/_cache/scores_<pass>.pt``.  The appearance channel stores
    one vector per view budget V, so ablation O4 needs **no** extra
    encoder work: all views are encoded once and re-aggregated over the
    FPS-ordered prefix (see pipeline/step4_dino_reranking.py).
  * Tier 2 — *derivations*: fusion weights/method, candidate scoping and
    geometry re-ranking are cheap post-processing of the cached vectors.
    Fusion semantics are NOT re-implemented — the cached vectors are
    wrapped into the pipeline's own result dataclasses and fused by
    ``pipeline.step6_fusion.ScoreFusion``, so every ablation exercises
    the exact production fusion code.

Dataset expectations (data is provided by the user, this script never
downloads anything; the roots below are the ``config/paths.yaml`` entries):

  <datasets_root>/shrec18/shrec18_full/  raw SHREC'18 distribution
      cad/*.obj            3,308 ShapeNetSem CAD models (gallery)
      rgbd/*.ply           2,101 query scans (colored triangle meshes)
      results/rgbd.*.txt   per-TRAINING-query relevance lists
      train.csv, test.csv  split files (category column is "unknown")
  <datasets_root>/shrec18/shrec18_official/
                           official evaluation kit (rgbd.csv, cad.csv,
                           metrics.py), see OFFICIAL_DIR below
  <gallery_root>/shrec18/<cad_id>/    rendered gallery views
      <cad_id>_<v>.png                   42 views (FPS-ordered icosphere,
                                         preprocessing/render_views.py)
      .ulip_partial_cache_*.pt           per-view partial-cloud embeddings
                                         (step5 finds them on its own)
  <cad_root>/shrec18/descriptions_attributes.json
                                         LLaVA view descriptions
                                         (preprocessing/generate_descriptions.py)

Ground truth
------------
The public SHREC'18 dump ships *without* category labels (train.csv /
test.csv say "unknown"; the official GT was released separately by the
organisers and is not part of this distribution).  The labels are
reconstructed here by union-find over the ``results/`` relevance lists:
per the dataset README every listed result "is from the correct
category", so queries and CADs co-occurring in a list share a category,
and the transitive closure over all 1,452 training lists yields exactly
the 20 track categories.  Only the *training* queries are recoverable
this way — Stage 1 therefore tunes on the 1,452 labeled training
queries; test queries have no local GT.  (State this in the thesis; see
"Open risks" in the accompanying plan/docs.)

How to run
----------
    python3 experiments/stage1_shrec18.py
        # BASE full fusion (E1c_full_fusion): nDCG 0.5868 / hit@1 0.341
    python3 experiments/stage1_shrec18.py --weights 0,0,1
        # shape channel ALONE, isolated (E1_shape_only)
    python3 experiments/stage1_shrec18.py --geometry trimmed-distance
        # dGeDi-RANSAC re-rank of the fused top-K (E2_chamfer_ransac):
        # nDCG 0.6405 / hit@1 0.472
    python3 experiments/stage1_shrec18.py --gallery fullmesh
        # full-mesh S_shape reference (E2b_fullmesh): nDCG 0.5935 / hit@1 0.360
    python3 experiments/stage1_shrec18.py --gallery fullmesh \\
            --geometry trimmed-distance
        # strongest Stage-1 arm (E2b_fullmesh_geo): nDCG 0.6417 / hit@1 0.481
    python3 experiments/stage1_shrec18.py --shape-mode cross
        # ULIP-2 image cross query instead of the query point cloud
    python3 experiments/stage1_shrec18.py --subset test-split
        # score only the 649 reconstructed SHREC'18 TEST queries
    python3 experiments/stage1_shrec18.py --limit 20 --out stage1_smoke
        # smoke run (NOT comparable to the reported numbers)

Outputs
-------
  <runs_root>/<--out>/            (default --out: stage1_shrec18)
      run_config.json                     argv, derived arm, env, git revision
      <config>/metrics_summary.json       all Stage-1 metrics + config
      <config>/results_per_query.json     per-query top-10 + ranks
      per_query/<config>.json             copy of the above, archive layout
      all_arms.csv                        archive row of this configuration
      stage1_summary.csv                  aggregate over the arms present
      stage1_summary.tex                  booktabs table for the thesis
      best_config.json                    frozen best configuration
      _cache/                             score vectors, geometry cache, ...

  ``<config>`` is the RELEASE name of the configuration (``fusion``,
  ``fusion_geo_trimmed_distance``, ...): the arm is computed under its internal
  ablation key and the folder is renamed once the run finishes, so the output is
  named and shaped exactly like ``results/stage1_shrec18/``.  The internal key
  stays visible in ``run_config.json`` and in the headline print.
  With ``--subset test-split`` the two CSV/JSON artefacts are the test-split
  counterparts (``official_track_testsplit.csv``, ``per_query/testsplit/``).

  The frozen thesis record of ALL arms stays in
  ``results/stage1_shrec18/all_arms.csv`` (test split:
  ``results/stage1_shrec18/official_track_testsplit.csv``); a fresh run is
  compared against it by the headline print at the end.

Bibliography keys used in comments (references.bib):
  phamSHREC18RGBDObjecttoCAD2018, pulliOSCAROpenSetCAD2025,
  nguyenCNOSStrongBaseline2023, zhaiSigmoidLossLanguage2023,
  voSAMURAIShapeAwareMultimodal2025, xueULIP2ScalableMultimodal2024,
  zhouUni3DExploringUnified2023, caraffaFreeZeTrainingfreeZeroshot2025,
  diUREDUnsupervised3D2023, nguyenSHREC2025Retrieval2025,
  shilanePrincetonShapeBenchmark2004, chuOPENOcclusionInvariantPerception2024a
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ===========================================================================
# 0. CLI prelude
# ===========================================================================
# Everything here runs BEFORE the heavy imports (numpy right below; torch,
# open3d, eval_common are deferred into the functions that need them), so
# ``--help``, the flag -> arm translation and the dgedi healthcheck work on the
# host without the container dependencies.  The readable CLI is translated into
# the driver's own argv (``--ablations <arm> --data-root ... --results-root``)
# and handed to :func:`main`; the internal argparse and the whole computation
# below are UNCHANGED.
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if os.path.join(_REPO, "evaluation") not in sys.path:
    sys.path.insert(1, os.path.join(_REPO, "evaluation"))

ROOT = _REPO           # repo root; used for the git revision and log strings

# Internal ablation key <-> release configuration name (pure-python table, no
# heavy imports).  results/ and every artefact this script writes are named by
# the RELEASE name; the internal key only survives in run_config.json and in the
# parenthesis of the headline print.
from configuration_names import to_release  # noqa: E402

_PATHS: Optional[Dict[str, str]] = None   # config/paths.yaml, see pipeline.paths
_ARGS = None           # parsed readable CLI (stays None on module import)
_ARM = ""              # internal ablation name derived from the readable flags
_ENV: Dict[str, str] = {}      # env this run sets (provenance)
_INTERNAL_ARGV: Optional[List[str]] = None   # translated argv for main()
_RESULTS_ROOT = ""     # <runs_root>/<--out>

# Arms whose ranking is produced by the dGeDi geometry re-rank: they need the
# geometry env and a running dgedi service holding the SHREC gallery.  Env set
# and healthcheck are taken verbatim from repro_experiment.py's Stage-1 branch.
_GEOMETRY_ARMS = {"E2_fitness", "E2_chamfer_ransac", "E2_both",
                  "O1c_gedi_post_fusion", "O1e_gedi_with_base",
                  "E2b_fullmesh_geo"}
_DGEDI_CACHE = "caches/dgedi/shrec"   # DGEDI_CACHE_DIR of the SHREC gallery
_DGEDI_N_GALLERY = 3308               # SHREC'18 gallery size (all 3,308 CADs)

# The frozen thesis record (for the pointer in messages below).
_RECORD_CSV = "results/stage1_shrec18/all_arms.csv"
_RECORD_CSV_TESTSPLIT = "results/stage1_shrec18/official_track_testsplit.csv"

_DEFAULT_WEIGHTS = (0.3, 0.4, 0.3)        # BASE fusion (thesis Table 4.1)
_TEXT_VIEW_WEIGHTS = (0.43, 0.57, 0.0)    # OSCAR's channel set, renormalised

# Readable flag combination -> internal arm.  This tuple IS the supported set:
# it is printed whenever a combination is not covered, it is the ``--help``
# epilog, and every name in it is cross-checked against the ablation registry
# at import time (see the guard right below ABLATIONS).
_SUPPORTED = (
    [("(defaults)", "E1c_full_fusion")]
    + [(f"--views {v}", f"O4_V{v}") for v in (8, 16, 32)]
    + [("--gallery fullmesh", "E2b_fullmesh"),
       ("--shape-mode cross", "E7_ulip2_cross"),
       ("--shape-mode cross --gallery fullmesh", "E7_ulip2_cross_fullmesh"),
       ("--shape uni3d", "E7_uni3d"),
       ("--shape ulip2-xyz", "O5_xyz_only"),
       ("--appearance siglip", "E4_siglip"),
       ("--weights 1,0,0", "E1a_text_only"),
       ("--weights 0.43,0.57,0", "E1b_text_view"),
       ("--weights 0,1,0", "E1_view_only")]
    + [(f"--weights 0,1,0 --views {v}", f"A2_view_only_V{v}")
       for v in (8, 16, 32)]
    + [("--weights 0,1,0 --appearance siglip", "E4_siglip_only"),
       ("--weights 0,0,1", "E1_shape_only")]
    + [(f"--weights 0,0,1 --views {v}", f"A7_shape_only_V{v}")
       for v in (8, 16, 32)]
    + [("--weights 0,0,1 --gallery fullmesh", "E2b_fullmesh_shape_only"),
       ("--weights 0,0,1 --shape-mode cross", "E7_ulip2_cross_shape_only"),
       ("--weights 0,0,1 --shape-mode cross --gallery fullmesh",
        "E7_ulip2_cross_fullmesh_shape_only"),
       ("--weights 0,0,1 --shape uni3d", "E7_uni3d_shape_only"),
       ("--weights 0,0,1 --shape ulip2-xyz", "O5_xyz_shape_only"),
       ("--fusion rrf", "E6_rrf"),
       ("--scope clip-top20", "E1d_clip_pruned"),
       ("--cascade", "E1_oscar_cascade"),
       ("--clip-select threshold", "O2_clip_threshold"),
       ("--clip-select threshold-calibrated", "O2_clip_threshold_cal"),
       ("--clip-select visual-first", "O2_visual_first"),
       ("--geometry fitness", "E2_fitness"),
       ("--geometry trimmed-distance", "E2_chamfer_ransac"),
       ("--geometry trimmed-distance --gallery fullmesh", "E2b_fullmesh_geo"),
       ("--geometry mean-rank", "E2_both"),
       ("--geometry mean-rank-with-base", "O1e_gedi_with_base"),
       ("--geometry post-fusion --weights 0.43,0.57,0",
        "O1c_gedi_post_fusion")]
)
_DERIVABLE_ARMS = tuple(dict.fromkeys(arm for _flags, arm in _SUPPORTED))

# Default of every knob derive_arm looks at.  A rule matches only if all the
# knobs it does NOT consume sit at their default (see _fixed).
_FLAG_DEFAULTS = {"appearance": "dinov2", "shape": "ulip2", "shape_mode": "pc",
                  "gallery": "partial", "views": 42, "fusion": "weighted",
                  "scope": "full", "geometry": None, "cascade": False,
                  "clip_select": None, "weights": _DEFAULT_WEIGHTS}


def _supported_table() -> str:
    width = max(len(f) for f, _ in _SUPPORTED)
    return "\n".join(f"  {f:<{width}}  ->  {arm}" for f, arm in _SUPPORTED)


def _unsupported(reason: str):
    """Exit with the reason AND the full list of runnable configurations."""
    sys.exit(f"[stage1] {reason}.\n"
             f"Runnable are EXACTLY the configurations reported in the "
             f"thesis (flags -> internal arm):\n"
             f"{_supported_table()}\n"
             f"Reference values of all arms: {_RECORD_CSV}")


def _parse_weights(text: str) -> Tuple[float, float, float]:
    """``"0.3,0.4,0.3"`` -> (0.3, 0.4, 0.3); fusion weights (text, view, shape)."""
    try:
        parts = [round(float(x), 4) for x in text.split(",")]
    except ValueError:
        parts = []
    if len(parts) != 3 or abs(sum(parts) - 1.0) > 1e-6:
        sys.exit(f"[stage1] --weights {text}: three numbers T,V,S summing to 1 "
                 f"(e.g. 0.3,0.4,0.3 or 0,0,1).")
    return (parts[0], parts[1], parts[2])


def _shown(args, name: str) -> str:
    value = args.weights_raw if name == "weights" else getattr(args, name)
    flag = "--" + name.replace("_", "-")
    return flag if value is True else f"{flag} {value}"


def _fixed(args, *names: str) -> None:
    """Require the listed knobs at their default for the matched rule."""
    off = [_shown(args, n) for n in names
           if getattr(args, n) != _FLAG_DEFAULTS[n]]
    if off:
        _unsupported("combination not covered — " + " ".join(off)
                     + " is not reported together with the remaining flags")


def derive_arm(args) -> str:
    """Translate the readable flags onto the internal ablation name.

    The registry holds exactly ONE arm per reported configuration, so each rule
    also requires the knobs it does not consume to sit at their default — a
    combination that was never reported must fail loudly rather than run a
    neighbouring arm under a misleading name.
    """
    w = args.weights
    mech = [n for n in ("cascade", "clip_select", "scope", "fusion", "geometry")
            if getattr(args, n) != _FLAG_DEFAULTS[n]]
    if len(mech) > 1:
        _unsupported("several mechanism flags at once ("
                     + ", ".join(_shown(args, n) for n in mech)
                     + ") — exactly one per run is reported")

    # ---- scope / ordering / fusion mechanism (BASE channels otherwise) ----
    _base_knobs = ("appearance", "shape", "shape_mode", "gallery", "views",
                   "weights")
    if args.cascade:
        _fixed(args, *_base_knobs)
        return "E1_oscar_cascade"
    if args.clip_select:
        _fixed(args, *_base_knobs)
        return {"threshold": "O2_clip_threshold",
                "threshold-calibrated": "O2_clip_threshold_cal",
                "visual-first": "O2_visual_first"}[args.clip_select]
    if args.scope == "clip-top20":
        _fixed(args, *_base_knobs)
        return "E1d_clip_pruned"
    if args.fusion == "rrf":
        _fixed(args, *_base_knobs)
        return "E6_rrf"

    # ---- geometry re-ranking of the fused shortlist (E2 / O1c / O1e) ------
    if args.geometry:
        _fixed(args, "appearance", "shape", "shape_mode", "views")
        if args.geometry == "post-fusion":
            if w != _TEXT_VIEW_WEIGHTS:
                _unsupported("--geometry post-fusion is only reported without "
                             "S_shape (--weights 0.43,0.57,0)")
            _fixed(args, "gallery")
            return "O1c_gedi_post_fusion"
        if w != _DEFAULT_WEIGHTS:
            _unsupported(f"--geometry {args.geometry} is only reported with the "
                         f"BASE weights 0.3,0.4,0.3")
        if args.geometry == "trimmed-distance":
            return ("E2b_fullmesh_geo" if args.gallery == "fullmesh"
                    else "E2_chamfer_ransac")
        _fixed(args, "gallery")
        return {"fitness": "E2_fitness",
                "mean-rank": "E2_both",
                "mean-rank-with-base": "O1e_gedi_with_base"}[args.geometry]

    # ---- channel sets and fusion weights ---------------------------------
    if w == (1.0, 0.0, 0.0):                      # S_text alone
        _fixed(args, "appearance", "shape", "shape_mode", "gallery", "views")
        return "E1a_text_only"
    if w == _TEXT_VIEW_WEIGHTS:                   # OSCAR's channel set
        _fixed(args, "appearance", "shape", "shape_mode", "gallery", "views")
        return "E1b_text_view"
    if w == (0.0, 1.0, 0.0):                     # appearance channel alone
        _fixed(args, "shape", "shape_mode", "gallery")
        if args.appearance == "siglip":
            _fixed(args, "views")
            return "E4_siglip_only"
        return ("E1_view_only" if args.views == 42
                else f"A2_view_only_V{args.views}")
    if w == (0.0, 0.0, 1.0):                     # shape channel alone
        _fixed(args, "appearance")
        if args.shape == "uni3d":
            _fixed(args, "shape_mode", "gallery", "views")
            return "E7_uni3d_shape_only"
        if args.shape == "ulip2-xyz":
            _fixed(args, "shape_mode", "gallery", "views")
            return "O5_xyz_shape_only"
        if args.shape_mode == "cross":
            _fixed(args, "views")
            return ("E7_ulip2_cross_fullmesh_shape_only"
                    if args.gallery == "fullmesh"
                    else "E7_ulip2_cross_shape_only")
        if args.gallery == "fullmesh":
            _fixed(args, "views")
            return "E2b_fullmesh_shape_only"
        return ("E1_shape_only" if args.views == 42
                else f"A7_shape_only_V{args.views}")
    if w == _DEFAULT_WEIGHTS:                    # three-channel fusion
        if args.appearance == "siglip":
            _fixed(args, "shape", "shape_mode", "gallery", "views")
            return "E4_siglip"
        if args.shape == "uni3d":
            _fixed(args, "shape_mode", "gallery", "views")
            return "E7_uni3d"
        if args.shape == "ulip2-xyz":
            _fixed(args, "shape_mode", "gallery", "views")
            return "O5_xyz_only"
        if args.shape_mode == "cross":
            _fixed(args, "views")
            return ("E7_ulip2_cross_fullmesh" if args.gallery == "fullmesh"
                    else "E7_ulip2_cross")
        if args.gallery == "fullmesh":
            _fixed(args, "views")
            return "E2b_fullmesh"
        return ("E1c_full_fusion" if args.views == 42
                else f"O4_V{args.views}")
    _unsupported(f"--weights {args.weights_raw}: reported are only 1,0,0 / "
                 f"0.43,0.57,0 / 0,1,0 / 0,0,1 / 0.3,0.4,0.3")


def _check_dgedi(expected_n: int, cache_hint: str) -> None:
    """The dGeDi service must be up AND hold the right gallery
    (behaviour modelled on repro_experiment.check_dgedi)."""
    import urllib.request
    start_cmd = (f"DGEDI_CACHE_DIR={cache_hint} docker compose up -d "
                 "--force-recreate dgedi")
    for url in ("http://localhost:5061/health", "http://dgedi:5061/health"):
        try:
            health = json.load(urllib.request.urlopen(url, timeout=5))
        except Exception:
            continue
        n = health.get("n_gallery", -1)
        if n == expected_n:
            print(f"[stage1] dGeDi ok: n_gallery={n}", flush=True)
            return
        sys.exit(f"[stage1] dGeDi is up but has n_gallery={n} instead of "
                 f"{expected_n}. Switch it over: {start_cmd}   — then start "
                 "again.")
    sys.exit(f"[stage1] dGeDi service unreachable. Start: {start_cmd}")


if __name__ == "__main__":
    from pipeline import paths as _paths_mod

    _parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="flags -> internal arm (nothing else is runnable):\n"
               + _supported_table(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _parser.add_argument("--appearance", choices=["dinov2", "siglip"],
                         default="dinov2",
                         help="appearance encoder of S_view (default: dinov2)")
    _parser.add_argument("--shape", choices=["ulip2", "ulip2-xyz", "uni3d"],
                         default="ulip2",
                         help="shape encoder of S_shape; ulip2-xyz = the "
                              "XYZ-only ULIP-2 checkpoint, i.e. the query "
                              "cloud without colours (default: ulip2)")
    _parser.add_argument("--shape-mode", choices=["pc", "cross"], default="pc",
                         help="shape query: pc = query point cloud, cross = "
                              "ULIP-2 image tower on the query crop "
                              "(default: pc)")
    _parser.add_argument("--gallery", choices=["partial", "fullmesh"],
                         default="partial",
                         help="shape gallery representation: rendered partial "
                              "views or the full CAD mesh (default: partial)")
    _parser.add_argument("--views", type=int, choices=[8, 16, 32, 42],
                         default=42,
                         help="number of FPS-ordered gallery views pooled by "
                              "the ablated channel (default: 42)")
    _parser.add_argument("--weights", default="0.3,0.4,0.3", metavar="T,V,S",
                         help="fusion weights text,view,shape; a single 1 "
                              "isolates that channel (default: 0.3,0.4,0.3)")
    _parser.add_argument("--fusion", choices=["weighted", "rrf"],
                         default="weighted",
                         help="score fusion: weighted sum or reciprocal rank "
                              "fusion (default: weighted)")
    _parser.add_argument("--scope", choices=["full", "clip-top20"],
                         default="full",
                         help="candidate scope: full database or the CLIP "
                              "top-20 shortlist (default: full)")
    _parser.add_argument("--cascade", action="store_true",
                         help="faithful OSCAR cascade: CLIP top-20 shortlist, "
                              "DINOv2 arg-max within it, no shape channel")
    _parser.add_argument("--clip-select", default=None,
                         choices=["threshold", "threshold-calibrated",
                                  "visual-first"],
                         help="how the shortlist is selected: OSCAR's text "
                              "threshold (tau=0.37), the same threshold "
                              "calibrated on SHREC'18, or visual-first "
                              "pruning by S_view")
    _parser.add_argument("--geometry", default=None,
                         choices=["fitness", "trimmed-distance", "mean-rank",
                                  "mean-rank-with-base", "post-fusion"],
                         help="dGeDi geometry re-rank of the fused top-K: "
                              "RANSAC fitness, trimmed surface distance after "
                              "alignment, both by mean rank, both plus the "
                              "base-fusion rank, or post-fusion re-rank of the "
                              "shape-less text+view arm (needs the dgedi "
                              "service; default: no geometry)")
    _parser.add_argument("--geom-k", type=int, choices=[5, 20, 50], default=50,
                         help="geometry re-ranking depth K (default: 50)")
    _parser.add_argument("--subset", choices=["test-split"], default=None,
                         help="restrict scoring to the reconstructed SHREC'18 "
                              "TEST split (649 queries); pure Tier-2 filter, "
                              "the score caches stay unchanged")
    _parser.add_argument("--limit", type=int, default=0, metavar="N",
                         help="smoke runs: score only the first N queries "
                              "(0 = all queries; default: 0)")
    _parser.add_argument("--out", default="",
                         help="result folder name below runs_root "
                              "(default: stage1_shrec18)")
    _parser.add_argument("--overwrite", action="store_true",
                         help="recompute even if results already exist")
    _parser.add_argument("--no-docker", action="store_true",
                         help="do not auto-wrap into the oscar-plus container")
    _paths_mod.add_path_args(_parser)
    _args = _parser.parse_args()

    _args.weights_raw = _args.weights
    _args.weights = _parse_weights(_args.weights)
    _ARM = derive_arm(_args)
    if _ARM not in _DERIVABLE_ARMS:      # derive_arm vs _SUPPORTED drift
        sys.exit(f"[stage1] internal error: derive_arm -> {_ARM} is missing "
                 f"from the table of supported combinations.")
    _PATHS = _paths_mod.from_args(_args)
    _RESULTS_ROOT = os.path.join(_PATHS["runs_root"],
                                 _args.out or "stage1_shrec18")

    # --- env, exactly as repro_experiment.py's Stage-1 branch set it -------
    # SHREC_DINO_POOLING=mean is the pooling ALL published Stage-1 numbers were
    # produced with (the module default is "mean" too, pinned here so the value
    # is recorded rather than inherited).  No SHREC_FORCE_PARTIAL_CACHE: step5
    # finds the partial-view cache by itself (_load_partial_cache_by_dim) and
    # raises instead of silently falling back to full-mesh.
    _ENV = {"PYTHONHASHSEED": "0", "SHREC_DINO_POOLING": "mean"}
    if _args.subset == "test-split":
        _subset_file = os.path.join(_REPO, "evaluation", "shrec18_splits",
                                    "test_split_ids.json")
        if not os.path.isfile(_subset_file):
            sys.exit(f"[stage1] {_subset_file} missing — repo incomplete?")
        _ENV["SHREC_QUERY_SUBSET"] = _subset_file
    _GEO = _ARM in _GEOMETRY_ARMS
    if _GEO:
        # GEOM_VOXEL mirrors the module constant (0.02); carried as env so the
        # value that produced the run is in run_config.json, like repro_experiment.
        _ENV.update({"STAGE1_GEOMETRY_BACKEND": "dgedi",
                     "DGEDI_CACHE_DIR": _DGEDI_CACHE,
                     "GEOM_VOXEL": "0.02"})
    os.environ.update(_ENV)
    if _GEO:      # fail fast on the host before wrapping, and again inside
        _check_dgedi(_DGEDI_N_GALLERY, _DGEDI_CACHE)

    print(f"[stage1] arm {_ARM}"
          + (f" (geometry K={_args.geom_k})" if _GEO else "")
          + (f" [SMOKE: {_args.limit} queries]" if _args.limit else ""),
          flush=True)

    if not os.path.exists("/.dockerenv") and not _args.no_docker:
        if not os.path.isfile(os.path.join(_REPO, "docker-compose.yml")):
            sys.exit(f"[stage1] {_REPO}/docker-compose.yml missing — "
                     "repo incomplete?")
        print("[stage1] runs in the oscar-plus container — wrapping automatically.",
              flush=True)
        os.chdir(_REPO)  # docker compose needs the repo as CWD
        os.execvp("docker", ["docker", "compose", "run", "--rm", "oscar-plus",
                             "python3", "/app/experiments/stage1_shrec18.py"]
                  + sys.argv[1:])

    # --- readable flags -> the driver's own argv (logic below unchanged) ---
    _INTERNAL_ARGV = [
        "--ablations", _ARM,
        "--data-root", os.path.join(_PATHS["datasets_root"], "shrec18",
                                    "shrec18_full"),
        "--images-dir", os.path.join(_PATHS["gallery_root"], "shrec18"),
        "--desc-file", os.path.join(_PATHS["cad_root"], "shrec18",
                                    "descriptions_attributes.json"),
        "--results-root", _RESULTS_ROOT,
    ]
    if _GEO:
        _INTERNAL_ARGV += ["--with-geometry", "--geom-k", str(_args.geom_k)]
    if _args.limit:
        _INTERNAL_ARGV += ["--limit-queries", str(_args.limit)]
    if _args.overwrite:
        _INTERNAL_ARGV += ["--overwrite"]
    _ARGS = _args

if _PATHS is None:   # imported as a module (no CLI): plain paths.yaml defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()

import numpy as np    # noqa: E402  (after the prelude: --help needs no numpy)

# ---------------------------------------------------------------------------
# Input and output locations.  Every entry comes from the central path registry
# (config/paths.yaml) so nothing is hard-coded to one machine; the internal
# flags below can still override each one (the readable CLI sets them from
# _PATHS, see the prelude).
# ---------------------------------------------------------------------------
DEFAULTS = {
    # raw SHREC'18 distribution (cad/, rgbd/, results/, train.csv, test.csv)
    "data_root":    os.path.join(_PATHS["datasets_root"], "shrec18",
                                 "shrec18_full"),
    # derived per-query artefacts: rendered crops, query clouds, GT cache
    "stage1_root":  os.path.join(_PATHS["datasets_root"], "shrec18", "stage1"),
    "images_dir":   os.path.join(_PATHS["gallery_root"], "shrec18"),
    "desc_file":    os.path.join(_PATHS["cad_root"], "shrec18",
                                 "descriptions_attributes.json"),
    "results_root": os.path.join(_PATHS["runs_root"], "stage1_shrec18"),
}

# SHREC'18 constants [phamSHREC18RGBDObjecttoCAD2018]
N_CATEGORIES = 20      # 20 manually grouped categories (dataset README)
N_CADS = 3308          # gallery size (ShapeNetSem indoor subset)
N_QUERIES_TOTAL = 2101 # all queries; only the ~70% train split has local GT
E_MEASURE_K = 32       # E-measure cut-off, Princeton Shape Benchmark
                       # convention [shilanePrincetonShapeBenchmark2004]

# OSCAR+ Stage-1 defaults (thesis Table 4.1 / methodology Step B1):
# full-database scoring, weighted-sum fusion with w = (0.3, 0.4, 0.3).
BASE_WEIGHTS = (0.3, 0.4, 0.3)
CLIP_PRUNE_K = 20      # OSCAR cascade shortlist size (O2 / E1d)
                       # [pulliOSCAROpenSetCAD2025]
# OSCAR's ACTUAL candidate filter (Pulli et al. 2025, Sec. 3.2): keep every
# object with sim_text(s_i) >= tau_text, i.e. a VARIABLE-sized candidate set;
# top-k is only the fallback when nothing clears the threshold.  Our
# step3_clip_retrieval.py stores raw cosine similarities (line 306 normalises
# both sides, line 345 stores sims[idx]), so 0.37 is directly applicable.
#
# Note the repo default is NOT this: pipeline/config.py:74 sets
# clip_threshold = 0.25 with a stale "noch nicht implementiert" comment, and
# retrieve()'s threshold defaults to None, so the shipped pipeline runs pure
# top-k.  The paper value is used here deliberately.
CLIP_TAU_TEXT = 0.37
# Measured on SHREC'18 (O2_clip_threshold, shortlist_stats.fallback_rate):
# tau_text = 0.37 admits NOTHING on 98.3% of queries, so the arm falls back to
# top-k and reproduces the cascade rather
# than exercising the threshold at all.  That is a real result about transfer
# — the constant was fitted to MI3DOR/YCB-V caption similarities — but it says
# nothing about whether threshold pruning *works*.  So a second arm calibrates
# tau to this dataset's own similarity distribution.
# (Bis 2026-09-19 stand hier 96.9% mit Datum 2026-07-31 — das ist der
# MI3DOR-Wert aus Stage 2, 10174/10500 = 96.9%, nicht der SHREC-Wert.
# SHREC liefert 0.98334 in JEDEM archivierten Lauf: k20/k50 wie 42v/k5.)
#
# Rule: tau = the percentile of the per-query MAXIMUM similarity that leaves
# at most this fraction of queries with an empty candidate set.  Calibrating
# on "the mechanism must actually run" rather than on a target set size keeps
# it non-circular — it is not tuned to match top-k, and it is not tuned on the
# retrieval metric it will later be judged by.
CLIP_TAU_FALLBACK_TARGET = 0.05
CLIP_TAU_CAL: Optional[float] = None   # set by calibrate_tau() at run time
VIEW_BUDGETS = (8, 16, 32, 42)  # O4; thesis lists {8,16,32}, 42 added as
                       # the CNOS/repo default (full icosphere)
                       # [nguyenCNOSStrongBaseline2023]
GEOM_SHORTLIST = 5     # legacy default (pipeline/config.py); kept only as the
                       # fallback for --geom-k.  The 2026-07-30 grid ran at 5,
                       # which reaches just 12.7% of the official DCG weight
                       # mass (25.9% at K=20) — see STAGE1_EVALUATION_DESIGN §1.
GEOM_K = GEOM_SHORTLIST  # effective depth; set from --geom-k in main()
# Depths at which the base-fusion hit-rate curve is reported.  That curve is
# the ceiling on what re-ranking at depth K can reach, and choose_geom_k()
# turns it into the K actually used.
HITRATE_KS = (1, 5, 10, 20, 50, 100)
HITRATE_TOL = 0.02     # "within 2 percentage points of the deepest K"
# Set from --no-icp.  ICP is ~1.1 s of the ~2.9 s per-pair budget and moved
# nDCG by 0.0001 at K=5, so the deep run may skip it and cite that shallow
# measurement rather than spend tens of hours re-confirming a null.
SKIP_ICP = False
# Full-database S_GeDi would need |queries| x |gallery| RANSAC fits.
GEDI_FULL_DB_PAIRS = N_QUERIES_TOTAL * N_CADS
# Number of (FPS-ordered) partial views the shape channel AGGREGATES over at
# retrieval time.  The reference cache still stores all encoded views (whole
# dataset); this only trims the top-k_v pooling input.  16 is sufficient for
# the first experiment and matches the DINO/CNOS view budget; set to None to
# aggregate over every cached view.
# Optionaler Query-Filter: JSON-Datei mit einer Liste von qids. Damit laesst
# sich die Auswertung auf eine Teilmenge einschraenken — z. B. den
# rekonstruierten SHREC'18-TEST-Split (649 Queries; die Trainings-Queries sind
# daran erkennbar, dass fuer sie eine Beispiel-Rangliste veroeffentlicht wurde).
# Reine Tier-2-Einschraenkung: die Score-Caches bleiben unveraendert.
SHREC_QUERY_SUBSET = os.environ.get("SHREC_QUERY_SUBSET", "").strip()

SHAPE_AGG_VIEWS = 42   # shape gallery views pooled = DINOv2's 42 (parity with the
                       # appearance channel and with Stage-2/MI3DOR, which pools all 42).
                       # Was 16 (a Stage-1-script-only cap); back-ported to 42 2026-08-26.

ULIP_CKPT_DEFAULT = _PATHS["checkpoint_ulip2"]
# O5 XYZ-only arm uses the released ULIP-2 *xyz* PointBERT (8,192 pts, no RGB,
# input_dim=3, SLIP ViT-B tower) — the colored 10k checkpoint has a 6-channel
# input conv and cannot encode xyz-only clouds. Its basename is part of the
# cache fingerprint (step5 _get_partial_cache_path), so the eval PC must stage
# the same file at the same path/name to reuse these caches.
ULIP_CKPT_XYZ = _PATHS["checkpoint_ulip2_xyz"]
# Uni3D (E7 shape-encoder swap); pipeline/config.py carries the same default,
# pinned here so BOTH shape encoders are named by the path registry.
UNI3D_CKPT = _PATHS["checkpoint_uni3d"]

# Official SHREC'18 evaluation kit (git clone of hkust-vgd/shrec18): the
# real category+subcategory GT (rgbd.csv/cad.csv, all 2,101 queries and
# 3,308 CADs) and the exact metric implementations (metrics.py) used to
# score the published track participants.  Using these makes every OSCAR+
# number leaderboard-comparable by construction, for tuning and final runs
# alike [phamSHREC18RGBDObjecttoCAD2018].
OFFICIAL_DIR = os.path.join(_PATHS["datasets_root"], "shrec18",
                            "shrec18_official")


# ===========================================================================
# 1. Ground-truth reconstruction (union-find over results/ lists)
# ===========================================================================

def build_gt(data_root: str, stage1_root: str, force: bool = False) -> dict:
    """Reconstruct category labels from the ``results/`` relevance lists.

    The SHREC'18 distribution withholds category labels, but ships, for
    every *training* query, a (non-exhaustive) retrieval list whose
    entries are guaranteed to be from the correct category (dataset
    README).  Treating queries and CADs as nodes and every
    (query, listed CAD) pair as an edge, the connected components of
    this co-occurrence graph are exactly the track categories: 20
    components emerge, covering all 1,452 training queries and 3,305 of
    the 3,308 CADs.  The 3 CADs never listed for any query stay
    unlabeled and act as gallery distractors.

    Category names are synthetic (``cat_00``.. by descending CAD count)
    since the official names are not in the dump.  Category-level
    metrics only need the partition, not the names.  If the official GT
    is ever obtained, drop a file with the same JSON schema at
    ``<stage1>/gt/category_labels.json`` and delete the reconstructed
    one — the rest of the script is agnostic to the label source.
    """
    gt_dir = os.path.join(stage1_root, "gt")
    out_path = os.path.join(gt_dir, "category_labels.json")
    if os.path.isfile(out_path) and not force:
        with open(out_path) as f:
            return json.load(f)

    results_dir = os.path.join(data_root, "results")
    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".txt"))
    if not files:
        raise SystemExit(f"[gt] no relevance lists in {results_dir}")

    # --- Union-find --------------------------------------------------------
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]     # path halving
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    query_ids: List[str] = []
    for fname in files:
        # file name:  rgbd.<hash>.txt ;  lines:  cad.<hash> <distance>
        qid = fname[:-4]                       # strip ".txt"
        query_ids.append(qid)
        with open(os.path.join(results_dir, fname)) as f:
            for line in f:
                tok = line.split()
                if tok:
                    union(qid, tok[0])

    # --- Components -> categories -----------------------------------------
    comps: Dict[str, dict] = defaultdict(lambda: {"cad": [], "rgbd": []})
    for node in parent:
        kind = "cad" if node.startswith("cad.") else "rgbd"
        comps[find(node)][kind].append(node.split(".", 1)[1])

    comps_list = sorted(comps.values(),
                        key=lambda c: (-len(c["cad"]), min(c["cad"] or [""])))
    if len(comps_list) != N_CATEGORIES:
        raise SystemExit(
            f"[gt] union-find produced {len(comps_list)} components, "
            f"expected {N_CATEGORIES}.  The results/ folder is incomplete "
            f"or corrupted — refusing to fabricate ground truth.")

    cad_labels: Dict[str, str] = {}
    query_labels: Dict[str, str] = {}
    sizes: Dict[str, dict] = {}
    for i, comp in enumerate(comps_list):
        cat = f"cat_{i:02d}"
        for h in comp["cad"]:
            cad_labels[h] = cat
        for h in comp["rgbd"]:
            query_labels[h] = cat
        sizes[cat] = {"cad": len(comp["cad"]), "queries": len(comp["rgbd"])}

    all_cads = {os.path.splitext(f)[0]
                for f in os.listdir(os.path.join(data_root, "cad"))
                if f.endswith(".obj")}
    unlabeled = sorted(all_cads - set(cad_labels))

    gt = {
        "provenance": (
            "Reconstructed via union-find over results/*.txt "
            "(SHREC'18 README: every listed result is from the correct "
            "category).  Covers the training split only; official "
            "category names are not available in this dump."),
        "cad": cad_labels,
        "queries": query_labels,
        "unlabeled_cads": unlabeled,
        "component_sizes": sizes,
    }
    os.makedirs(gt_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"[gt] {len(comps_list)} categories, {len(query_labels)} labeled "
          f"queries, {len(cad_labels)}/{len(all_cads)} labeled CADs "
          f"({len(unlabeled)} distractors) -> {out_path}")
    return gt


def _read_shrec_csv(path: str) -> Dict[str, tuple]:
    """Parse an official rgbd.csv / cad.csv: 'rgbd.<hash>,cat,subcat' rows.

    The leading 'rgbd.' / 'cad.' prefix is stripped so the keys match our
    query stems (ply basename) and gallery ids (obj basename).
    """
    out: Dict[str, tuple] = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            key = row[0].split(".", 1)[1] if "." in row[0] else row[0]
            out[key] = (row[1], row[2])
    return out


def load_official_gt(data_root: str, stage1_root: str) -> dict:
    """Load the official SHREC'18 category+subcategory ground truth.

    Uses the track's own rgbd.csv / cad.csv (see :data:`OFFICIAL_DIR`),
    which cover all 2,101 queries and 3,308 CADs with real category and
    subcategory labels — the same GT the published participants were
    scored against.  Falls back to the union-find reconstruction only if
    the official kit is absent.

    Returns::

        {"queries": {qid: (cat, subcat)},   # 2,101
         "cad":     {cadid: (cat, subcat)},  # 3,308
         "freqs":   {cat: n_cads_in_cat},    # official recall denominators
         "provenance": "official hkust-vgd/shrec18 rgbd.csv + cad.csv"}
    """
    rgbd_csv = os.path.join(OFFICIAL_DIR, "rgbd.csv")
    cad_csv = os.path.join(OFFICIAL_DIR, "cad.csv")
    if not (os.path.isfile(rgbd_csv) and os.path.isfile(cad_csv)):
        raise SystemExit(
            f"[gt] official GT not found at {OFFICIAL_DIR}. Clone it with:\n"
            f"  git clone https://github.com/hkust-vgd/shrec18 "
            f"{OFFICIAL_DIR}\n"
            f"(or fall back to the union-find reconstruction via build_gt).")

    queries = _read_shrec_csv(rgbd_csv)
    cad = _read_shrec_csv(cad_csv)
    freqs: Dict[str, int] = {}
    for cat, _sub in cad.values():
        freqs[cat] = freqs.get(cat, 0) + 1

    gt = {"queries": queries, "cad": cad, "freqs": freqs,
          "provenance": "official hkust-vgd/shrec18 rgbd.csv + cad.csv"}
    gt_dir = os.path.join(stage1_root, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    with open(os.path.join(gt_dir, "official_labels.json"), "w") as f:
        json.dump({"queries": {k: list(v) for k, v in queries.items()},
                   "cad": {k: list(v) for k, v in cad.items()},
                   "freqs": freqs, "provenance": gt["provenance"]}, f)
    print(f"[gt] official: {len(queries)} queries, {len(cad)} CADs, "
          f"{len(freqs)} categories "
          f"({sum(1 for v in cad.values() if v[1] and v[1] != v[0])} have "
          f"subcategories) -> {gt_dir}/official_labels.json")
    return gt


def score_official(ranking_cad_ids: Sequence[str], q_label: tuple,
                   cad_labels: Dict[str, tuple], freqs: Dict[str, int]
                   ) -> Optional[Dict[str, float]]:
    """Score one query's ranking with the OFFICIAL SHREC'18 metrics.

    Faithful replication of evaluate.py's per-query loop, reusing the
    unchanged official ``metrics.py`` so the numbers are bug-for-bug
    identical to the published leaderboard:
      * graded relevance — 2 if (category, subcategory) match, 1 if only
        category, else 0 (``categories_to_rel``);
      * every metric computed on the top-f results, f = category size in
        cad.csv (``freqs``);
      * precision/recall/F1/AP/NNT1/NNT2 count any category match
        (np.count_nonzero); nDCG uses the graded gains (subcategory=2).
    """
    if OFFICIAL_DIR not in sys.path:
        sys.path.insert(0, OFFICIAL_DIR)
    import metrics as MM  # official, unchanged

    qc, qs = q_label
    f = freqs.get(qc, 0)
    if f == 0:
        return None
    rel = []
    for cid in ranking_cad_ids:
        lab = cad_labels.get(cid)
        if lab is None:
            rel.append(0.0)
        elif lab[0] == qc and lab[1] == qs:
            rel.append(2.0)
        elif lab[0] == qc:
            rel.append(1.0)
        else:
            rel.append(0.0)
    x = rel[:f]
    if not x:
        return None
    return {
        "nDCG":      float(MM.ndcg(x)),
        "precision": float(MM.precision(x)),
        "recall":    float(MM.recall(x, f)),
        "F1":        float(MM.f1score(x, f)),
        "AP":        float(MM.average_precision(x, f)),
        "NNT1":      float(MM.nnt1(x, f)),
        "NNT2":      float(MM.nnt2(x, f)),
    }


def _graded_relevance(ranking_cad_ids: Sequence[str], q_label: tuple,
                      cad_labels: Dict[str, tuple]) -> np.ndarray:
    """SHREC'18 graded relevance over a ranking: 2 = subcategory, 1 = category.

    Same rule as ``score_official`` / the official ``categories_to_rel``,
    factored out so Table B can read the sub-category grade too.
    """
    qc, qs = q_label
    out = np.zeros(len(ranking_cad_ids), dtype=np.float64)
    for i, cid in enumerate(ranking_cad_ids):
        lab = cad_labels.get(cid)
        if lab is None or lab[0] != qc:
            continue
        out[i] = 2.0 if lab[1] == qs else 1.0
    return out


def score_depth_matched(ranking_cad_ids: Sequence[str], q_label: tuple,
                        cad_labels: Dict[str, tuple], k: int
                        ) -> Dict[str, float]:
    """TABLE B — the metrics a top-K re-ranking can actually move.

    The official scalar metrics (``score_official``) are cut at f = category
    size (~165) and, except for nDCG and AP, are ``np.count_nonzero`` over
    that prefix — so they are *algebraically* invariant to any permutation of
    the first K < f entries, i.e. blind to the whole B2 geometry stage.
    Everything
    here is therefore cut at the geometry depth K instead.

    Two deliberate differences from Table A:

    * ``NN_sub`` reads the **sub-category** grade.  Category assignment is
      largely solved by the language channel; whether geometry picks the right
      *variant* within the category is the open question, so NN_sub is the
      headline for the geometry arms.
    * ``nDCG@K`` uses a **corrected** DCG.  The official ``metrics.dcg`` has an
      off-by-one (``total = x[0]`` and then the loop adds ``x[i]`` again at
      ``i = 0``, double-counting rank 1 and dropping the last element).  It is
      left unpatched in Table A for leaderboard comparability; here it is
      correct.  The two nDCGs are different quantities and never share a
      column.

    ``hit_cat@N`` / ``hit_sub@N`` for N in :data:`HITRATE_KS` are emitted on
    every call so that the BASE arm's run yields the hit-rate curve that
    :func:`choose_geom_k` reads — geometry re-ranks but never inserts, so base
    hit-rate@K is a hard ceiling on every top-1 metric at depth K.
    """
    rel = _graded_relevance(ranking_cad_ids, q_label, cad_labels)
    cat = rel >= 1.0
    sub = rel >= 2.0
    n = len(rel)
    kk = min(k, n)
    if kk == 0:
        return {}

    def _mrr(mask: np.ndarray) -> float:
        hit = np.flatnonzero(mask[:kk])
        return 1.0 / (int(hit[0]) + 1) if hit.size else 0.0

    # AP@K over category relevance, normalised by the reachable number of
    # relevant items (min(K, R)) so a query with R < K is not capped below 1.
    hits = np.flatnonzero(cat[:kk])
    if hits.size:
        precs = (np.arange(hits.size) + 1) / (hits + 1)
        ap = float(precs.sum() / min(kk, int(cat.sum())))
    else:
        ap = 0.0

    # Corrected DCG with graded gains; ideal = the same gains sorted desc.
    disc = 1.0 / np.log2(np.arange(kk) + 2.0)
    dcg = float((rel[:kk] * disc).sum())
    idcg = float((np.sort(rel)[::-1][:kk] * disc).sum())

    out = {
        "NN_cat": float(cat[0]),
        "NN_sub": float(sub[0]),
        "MRR":     _mrr(cat),
        "MRR_sub": _mrr(sub),
        "mAP_K":  ap,
        "nDCG_K": (dcg / idcg) if idcg > 0 else 0.0,
    }
    for n_k in HITRATE_KS:
        m = min(n_k, n)
        out[f"hit_cat@{n_k}"] = float(cat[:m].any())
        out[f"hit_sub@{n_k}"] = float(sub[:m].any())
    return out


BASE_ABLATION = "E1c_full_fusion"   # the shortlist source for every geometry arm


def arm_dir(results_root: str, arm: str) -> str:
    """Folder of one arm inside a run/record root — either naming accepted.

    While an arm is being computed it lives under its INTERNAL key; once the run
    finishes the folder is renamed to the RELEASE name (that is what ``results/``
    holds).  Every in-driver reader goes through here so a finished arm stays
    findable, and so ``--results-root results/stage1_shrec18`` can be read back.
    Returns the internal-named folder when it exists, else the release-named one
    (which is also what a not-yet-existing arm is reported as).
    """
    internal = os.path.join(results_root, arm)
    if os.path.isdir(internal):
        return internal
    return os.path.join(results_root, to_release(arm, "stage1"))


def load_hitrate_curve(results_root: str, grade: str = "sub"
                       ) -> Dict[int, float]:
    """Read the base-fusion hit-rate curve out of the BASE arm's summary.

    No extra computation: :func:`score_depth_matched` emits ``hit_cat@N`` /
    ``hit_sub@N`` for every arm, so running BASE already produced the curve.
    """
    p = os.path.join(arm_dir(results_root, BASE_ABLATION),
                     "metrics_summary.json")
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        md = json.load(f).get("metrics_depth", {})
    return {k: md[f"hit_{grade}@{k}"]
            for k in HITRATE_KS if f"hit_{grade}@{k}" in md}


def _no_curve_reason(results_root: str) -> str:
    """Why the hit-rate curve is unavailable — the two cases differ.

    A summary that merely *predates* Table B looks identical to a missing one
    at the call site, but the fix is different (re-run vs run), so say which.
    """
    p = os.path.join(arm_dir(results_root, BASE_ABLATION),
                     "metrics_summary.json")
    if not os.path.isfile(p):
        return (f"no {BASE_ABLATION} summary yet — run the BASE ablation "
                f"first; the curve is a by-product of it")
    return (f"{BASE_ABLATION} predates the depth-matched metrics (no "
            f"'metrics_depth' key) — re-run it with --overwrite to emit the "
            f"curve; the expensive score passes are cached, so this is cheap")


def print_hitrate_curve(results_root: str) -> None:
    """Report the ceiling curve and the K it implies."""
    curves = {g: load_hitrate_curve(results_root, g) for g in ("cat", "sub")}
    if not curves["sub"]:
        print(f"[hit-rate] {_no_curve_reason(results_root)}.")
        return
    print(f"[hit-rate] base-fusion ceiling (from {BASE_ABLATION}):")
    print(f"{'K':>6}  {'hit_cat@K':>10}  {'hit_sub@K':>10}")
    for k in HITRATE_KS:
        c, s = curves["cat"].get(k), curves["sub"].get(k)
        print(f"{k:>6}  {c:>10.4f}  {s:>10.4f}"
              if c is not None and s is not None else f"{k:>6}  {'-':>10}")
    k = choose_geom_k(curves["sub"])
    print(f"[hit-rate] --geom-k {k}  (smallest K within "
          f"{HITRATE_TOL:.0%} of hit_sub@{max(curves['sub'])})")


def choose_geom_k(hit_curve: Dict[int, float],
                  tol: float = HITRATE_TOL) -> int:
    """Smallest K whose hit-rate is within ``tol`` of the deepest measured K.

    The geometry stage re-orders a shortlist; it can never pull a relevant CAD
    into it.  So base-fusion hit-rate@K bounds every top-1 metric the
    re-ranking can reach, and the point where that curve flattens is the
    cheapest depth that does not throw away reachable headroom.  Fixing K by
    this rule (rather than by taste) is what makes the constant defensible in
    the write-up.
    """
    if not hit_curve:
        return GEOM_SHORTLIST
    deepest = max(hit_curve)
    ceiling = hit_curve[deepest]
    for k in sorted(hit_curve):
        if hit_curve[k] >= ceiling - tol:
            return k
    return deepest


# ===========================================================================
# 2. Query preparation (PLY -> RGB crop + raw point cloud)
# ===========================================================================
# Each SHREC'18 query is a colored triangle mesh reconstructed from a
# SceneNN RGB-D scan.  The OSCAR+ image channels (S_text via CLIP, S_view
# via DINOv2/SigLIP, S_shape in cross-modal mode via the ULIP-2 image
# branch) expect an RGB object crop on neutral grey — the Step-A ROI
# convention (pipeline/step1_localization.py, thesis Step A4).  Since the
# original camera pose is not part of the dump, the crop is re-rendered
# from the mean-vertex-normal direction (scan meshes are one-sided, so
# this approximates the original viewpoint).  The point-cloud channels
# (S_shape pc-mode for O5/E7, S_GeDi/S_chamfer for E2/O1) use the raw
# vertices + colors stored as .npz; normalization to the encoder input
# convention happens inside the pipeline (step5 normalize_pointcloud).

_GREY = 205  # neutral background grey, matches eval_common.crop_with_mask


def _offscreen_available() -> bool:
    """Whether Open3D's GL OffscreenRenderer can be used safely.

    OffscreenRenderer needs EGL/GL; on a headless CPU container it does
    not merely raise — it *segfaults* the interpreter (uncatchable). It is
    therefore auto-enabled only when libEGL is confirmed loadable (true on
    the GPU compose run, which gives clean solid mesh crops), and skipped
    otherwise in favour of the CPU-only, headless-safe point-splat
    renderer. Set OSCAR_QUERY_OFFSCREEN=0 to force the splat even with a
    GPU present.
    """
    if os.environ.get("OSCAR_QUERY_OFFSCREEN") == "0":
        return False
    try:
        import ctypes
        ctypes.CDLL("libEGL.so.1")
        return True
    except OSError:
        return False


def _view_basis(points: np.ndarray, normals: Optional[np.ndarray]):
    """Camera basis from the mean vertex normal (fallback: PCA)."""
    n = None
    if normals is not None and len(normals):
        n = normals.mean(axis=0)
        if np.linalg.norm(n) < 1e-6:
            n = None
    if n is None:
        # Degenerate normals: view along the least-extent PCA axis.
        c = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 5000)], full_matrices=False)
        n = vt[-1]
    n = n / np.linalg.norm(n)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, n)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    u = np.cross(up, n); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v


def _render_query_offscreen(mesh, n, size: int):
    """Preferred path: Open3D OffscreenRenderer (needs EGL/GPU)."""
    import open3d as o3d
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    try:
        g = _GREY / 255.0
        r.scene.set_background([g, g, g, 1.0])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"           # show raw vertex colors
        r.scene.add_geometry("query", mesh, mat)
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        radius = np.linalg.norm(bbox.get_extent()) / 2.0
        pts = np.asarray(mesh.vertices)
        nrm = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
        n_dir, _, v = _view_basis(pts, nrm)
        eye = center + n_dir * (2.5 * max(radius, 1e-3))
        r.setup_camera(60.0, center.astype(np.float32),
                       eye.astype(np.float32), v.astype(np.float32))
        img = np.asarray(r.render_to_image())
        return img
    finally:
        del r


def _render_query_splat(points, colors, n, u, v, size: int):
    """Headless fallback: orthographic depth-ordered point splatting.

    Purely numpy — no GL required.  The splat radius adapts to point
    density (sparse scans get larger dots) so the object reads as a solid
    surface rather than a dotted texture; points are drawn far->near,
    nearer ones overwriting farther ones (painter's algorithm — NO depth buffer).  Coarser than the mesh render but
    adequate for category-level encoder crops; verify with --viz-check.
    """
    x = points @ u
    y = points @ v
    depth = points @ n                          # camera on +n side
    pad = size // 12
    draw = size - 2 * pad
    span = max(x.max() - x.min(), y.max() - y.min(), 1e-9)
    px = ((x - x.min()) / span * draw + pad).astype(np.int32)
    py = (size - 1 - ((y - y.min()) / span * draw + pad)).astype(np.int32)

    # Adaptive disk radius ≈ mean inter-point pixel spacing, so gaps close
    # for sparse clouds without over-blurring dense ones.
    n_pts = max(len(points), 1)
    spacing = draw / max(np.sqrt(n_pts), 1.0)
    radius = int(np.clip(round(0.7 * spacing), 1, 8))

    order = np.argsort(depth)                   # far first, near overwrites
    xs0, ys0, c8 = px[order], py[order], (np.clip(colors, 0, 1) * 255).astype(np.uint8)[order]
    img = np.full((size, size, 3), _GREY, dtype=np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:   # disk, not square
                continue
            xs = np.clip(xs0 + dx, 0, size - 1)
            ys = np.clip(ys0 + dy, 0, size - 1)
            img[ys, xs] = c8
    return img


def _render_query_surface(mesh, n, u, v, size: int,
                          n_samples: int = 200000, seed: int = 0):
    """Headless SOLID-surface render — pure numpy, no GL.

    The raw SHREC'18 query scans carry only ~10^3 vertices, so splatting the
    vertices alone leaves a dotted, holey cloud that does not read as a rigid
    segmented object (bad crop for the DINOv2/CLIP/SigLIP appearance
    channels).  Here we area-weight-sample points *on the triangle faces*
    (barycentric), so the whole surface is covered, then reuse the depth-ordered
    point splatter.  The result is a filled, shaded object crop.

    Deterministic (fixed ``seed``) so each query renders identically on every
    run/machine.  Returns ``None`` if the mesh has no usable faces (caller
    then falls back to vertex splatting).
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles)
    if len(faces) == 0 or len(verts) == 0:
        return None
    vcols = (np.asarray(mesh.vertex_colors, dtype=np.float32)
             if mesh.has_vertex_colors()
             else np.full((len(verts), 3), 0.5, dtype=np.float32))
    tri = verts[faces]                                   # (F, 3, 3)
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = float(areas.sum())
    if total <= 0:
        return None
    rng = np.random.RandomState(seed)
    fidx = rng.choice(len(faces), size=n_samples, p=areas / total)
    r1 = np.sqrt(rng.random_sample(n_samples).astype(np.float32))
    r2 = rng.random_sample(n_samples).astype(np.float32)
    bary = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1)  # (S, 3)
    fv = faces[fidx]                                     # (S, 3)
    s_pts = (bary[:, :, None] * verts[fv]).sum(axis=1)   # (S, 3)
    s_cols = (bary[:, :, None] * vcols[fv]).sum(axis=1)  # (S, 3)
    # Keep the raw vertices too so silhouette edges stay crisp.
    pts = np.concatenate([verts, s_pts], axis=0)
    cols = np.concatenate([vcols, s_cols], axis=0)
    return _render_query_splat(pts, cols, n, u, v, size)


def _crop_to_content(img: np.ndarray, margin: int = 8) -> np.ndarray:
    mask = np.any(np.abs(img.astype(np.int16) - _GREY) > 6, axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, img.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def prepare_queries(data_root: str, stage1_root: str, gt: dict,
                    size: int = 448, force: bool = False) -> List[dict]:
    """Produce <hash>.png + <hash>.npz for every labeled query (cached).

    Returns the query index [{id, category, png, npz}, ...] and writes it
    to <stage1>/gt/queries_index.json.
    """
    import open3d as o3d
    from PIL import Image

    qdir = os.path.join(stage1_root, "queries")
    os.makedirs(qdir, exist_ok=True)
    index_path = os.path.join(stage1_root, "gt", "queries_index.json")
    if os.path.isfile(index_path) and not force:
        with open(index_path) as f:
            index = json.load(f)
        missing = [q for q in index
                   if not (os.path.isfile(q["png"]) and os.path.isfile(q["npz"]))]
        # Only trust a cached index if it covers exactly the current GT
        # query set — a stale/partial index (e.g. from an earlier limited
        # test) must trigger a full regenerate, not silently shrink the run.
        covers = {q["id"] for q in index} == set(gt["queries"])
        if not missing and covers:
            return index
        print(f"[prepare] cached index covers {len(index)}/"
              f"{len(gt['queries'])} queries, {len(missing)} missing files "
              f"— regenerating.")

    offscreen_ok = _offscreen_available()  # default False -> numpy splat
    print(f"[prepare] query renderer: "
          f"{'Open3D OffscreenRenderer (GL)' if offscreen_ok else 'numpy point-splat (headless)'}")
    index = []
    qids = sorted(gt["queries"])
    print(f"[prepare] rendering {len(qids)} query crops -> {qdir}")
    for i, qid in enumerate(qids):
        png = os.path.join(qdir, f"{qid}.png")
        npz = os.path.join(qdir, f"{qid}.npz")
        entry = {"id": qid, "category": gt["queries"][qid],
                 "png": png, "npz": npz}
        index.append(entry)
        if os.path.isfile(png) and os.path.isfile(npz) and not force:
            continue

        ply = os.path.join(data_root, "rgbd", f"{qid}.ply")
        mesh = o3d.io.read_triangle_mesh(ply)
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        cols = (np.asarray(mesh.vertex_colors, dtype=np.float32)
                if mesh.has_vertex_colors() else
                np.full((len(pts), 3), 0.5, dtype=np.float32))
        if len(pts) == 0:
            print(f"[prepare] WARNING: empty mesh {ply}, skipped")
            index.pop()
            continue

        # Raw (unnormalized, metric) points + colors: the shape channel
        # normalizes internally (step5 normalize_pointcloud, thesis
        # Sec. 3.3); geometry re-ranking normalizes to the unit sphere in
        # this script; the scale gate needs the raw metric extents.
        np.savez_compressed(npz, points=pts, colors=cols)

        if not mesh.has_vertex_normals():
            try:
                mesh.compute_vertex_normals()
            except Exception:
                pass
        nrm = (np.asarray(mesh.vertex_normals, dtype=np.float32)
               if mesh.has_vertex_normals() else None)
        n, u, v = _view_basis(pts, nrm)

        img = None
        # Primary: pure-numpy solid-surface render from the mesh faces (no GL,
        # deterministic) — gives a filled rigid-object crop for the appearance
        # encoders instead of the sparse dotted vertex splat.
        if mesh.has_triangles():
            try:
                img = _render_query_surface(mesh, n, u, v, size)
            except Exception as exc:
                print(f"[prepare] surface render failed ({exc}); "
                      f"trying other renderers.")
        # Optional GL path (only if libEGL is present and forced on).
        if img is None and offscreen_ok and mesh.has_triangles():
            try:
                img = _render_query_offscreen(mesh, n, size)
            except Exception as exc:
                print(f"[prepare] OffscreenRenderer unavailable ({exc}); "
                      f"falling back to point splatting for all queries.")
                offscreen_ok = False
        # Fallback: vertex point-splat (face-less clouds).
        if img is None:
            img = _render_query_splat(pts, cols, n, u, v, size)
        Image.fromarray(_crop_to_content(img)).save(png)

        if (i + 1) % 100 == 0:
            print(f"[prepare]   {i + 1}/{len(qids)}")

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[prepare] done: {len(index)} queries indexed -> {index_path}")
    return index


def viz_check(index: List[dict], stage1_root: str, n: int) -> None:
    """Contact sheet of n random query crops for visual sanity."""
    from PIL import Image
    rng = np.random.RandomState(42)
    picks = [index[i] for i in rng.choice(len(index), min(n, len(index)),
                                          replace=False)]
    cell = 224
    cols = int(np.ceil(np.sqrt(len(picks))))
    rows = int(np.ceil(len(picks) / cols))
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i, q in enumerate(picks):
        img = Image.open(q["png"]).convert("RGB")
        img.thumbnail((cell, cell))
        sheet.paste(img, ((i % cols) * cell, (i // cols) * cell))
    out = os.path.join(stage1_root, "viz_check.png")
    sheet.save(out)
    print(f"[viz] contact sheet of {len(picks)} crops -> {out}")


# ===========================================================================
# 3. Retrieval metrics (SHREC'18 / Princeton Shape Benchmark conventions)
# ===========================================================================
# NN, FT, ST, E-measure and DCG follow the PSB definitions
# [shilanePrincetonShapeBenchmark2004] used by the SHREC'18 track
# [phamSHREC18RGBDObjecttoCAD2018]; Recall@k and mAP are added per thesis
# subsec:eval_retrieval_metrics.  All metrics are category-level: a
# retrieved CAD is relevant iff it belongs to the query's category.
# Implementation reuses evaluation/eval_common.py helpers where
# they exist (dcg_at_k, ideal_dcg_at_k, average_precision_from_binary).

def evaluate_ranking(rels: np.ndarray, num_rel: int) -> Dict[str, float]:
    """All Stage-1 metrics for one query.

    Args:
        rels:    binary relevance in rank order over the FULL gallery.
        num_rel: |C| — number of relevant models in the scored gallery.
    """
    from eval_common import (dcg_at_k, ideal_dcg_at_k,
                             average_precision_from_binary)
    n = len(rels)
    ft_k = num_rel
    st_k = min(2 * num_rel, n)
    r32 = int(rels[:E_MEASURE_K].sum())
    p = r32 / min(E_MEASURE_K, n)
    r = r32 / num_rel
    idcg = ideal_dcg_at_k(num_rel, n)
    return {
        "NN":  float(rels[0]),
        "FT":  float(rels[:ft_k].sum()) / num_rel,
        "ST":  float(rels[:st_k].sum()) / num_rel,
        # E-measure = F-score at the fixed cut-off K=32 (PSB convention).
        "E":   (2 * p * r / (p + r)) if (p + r) > 0 else 0.0,
        # DCG over the full ranked list, normalized by the ideal DCG —
        # the Stage-1 model-selection metric (subsec:eval_stage1_retrieval).
        "DCG": dcg_at_k(rels.tolist(), n) / idcg if idcg > 0 else 0.0,
        "R@1": float(rels[0]),
        "R@5": float(rels[:5].max()) if n else 0.0,
        "AP":  average_precision_from_binary(rels.tolist()),
    }


# ===========================================================================
# 4. Tier 1 — channel-score passes
# ===========================================================================
# Each pass builds the pipeline via eval_common.build_pipeline (which owns
# all on-disk embedding caches) and produces per-query score vectors over
# the canonical gallery list.  Pass definitions:
#
#   base          CLIP + DINOv2 (all views, per-V aggregation) — the
#                 appearance/semantic half of the OSCAR+ channel stack.
#                 S_shape is NOT part of this pass: Stage-1 scores the shape
#                 channel in pc-mode (see ulip_pc_rgb / _BASE_CH).
#   siglip        appearance channel re-scored with SigLIP (E4).
#   ulip_pc_fullmesh  S_shape with full-mesh reference embeddings (E2b),
#                 pc-mode query.
#   ulip_pc_rgb   S_shape in pc-mode, query = XYZ+RGB point cloud
#                 (E7 ULIP-2 arm / O5 RGB arm).  pc-mode is forced for
#                 these ablations so that the *point cloud* is what varies
#                 (the default cross-modal mode encodes the query image,
#                 which would make O5 a no-op).
#   ulip_pc_xyz   as above with colors disabled on both sides (O5).
#   uni3d         S_shape scored by Uni3D (pc-mode only; E7).

PASS_DEFS: "OrderedDict[str, dict]" = OrderedDict([
    ("base",          dict(channels=("clip", "dino"),
                           ulip2_mode="cross", partial=True, no_shape=True,
                           overrides={})),
    ("siglip",        dict(channels=("dino",), ulip2_mode="cross",
                           partial=True, no_shape=True,
                           overrides={"appearance_encoder": "siglip"})),
    # Full-mesh S_shape reference in pc-mode (E2b).  Mode only affects the
    # QUERY side, so this reuses both existing caches unchanged: the
    # full-mesh gallery cache (cad/.ulip_cache_*.pt) and the pc query cache
    # (keyed on encoder config, not on `partial` — see _pc_query_cache_path).
    ("ulip_pc_fullmesh", dict(channels=("shape",), ulip2_mode="pc",
                           partial=False, overrides={})),
    ("ulip_pc_rgb",   dict(channels=("shape",), ulip2_mode="pc",
                           partial=True, overrides={})),
    # Die vierte Zelle der Matrix Query-Modus x Gallery-Repraesentation:
    # cross-Query (ULIP-2-Bildturm) gegen die FULL-MESH-Gallery. Sie fehlte bis
    # 2026-09-03, obwohl genau diese Kombination auf BOP der beste Arm ist
    # (3a_cross_fullmesh_v2, R@1 0.5151). Beide Caches bestehen bereits: die
    # Full-Mesh-Gallery aus ulip_pc_fullmesh, die cross-Query aus
    # ulip_cross_rgb — der Modus betrifft nur die Query-Seite, `partial` nur
    # die Gallery-Seite. Es faellt also nur die Fusion an.
    ("ulip_cross_fullmesh", dict(channels=("shape",), ulip2_mode="cross",
                           partial=False, overrides={})),
    # ULIP-2 shape channel scored in CROSS mode: the query IMAGE is encoded by
    # ULIP-2's image tower against the SAME partial-view CAD gallery, instead
    # of the query point cloud.  Reuses the ulip_pc_rgb gallery cache unchanged
    # (mode only affects the query side), so it just encodes the query images.
    # Isolated arm E7_ulip2_cross_shape_only so pc-vs-cross is the only
    # variable — this is the mode Stage-2/MI3DOR must use (no depth), measured
    # here on SHREC for a direct comparison against ulip_pc_rgb.
    ("ulip_cross_rgb", dict(channels=("shape",), ulip2_mode="cross",
                           partial=True, overrides={})),
    # A7: ULIP-2 shape gallery-view aggregation sweep (pc-mode). Same encoder
    # and caches as ulip_pc_rgb; only the number of FPS-ordered gallery views
    # pooled differs (shape_agg_views). Gallery embeddings already hold all 42,
    # so these just re-pool cached sims — no re-encode. (v16 == ulip_pc_rgb.)
    ("ulip_pc_rgb_v8",  dict(channels=("shape",), ulip2_mode="pc", partial=True,
                           shape_agg_views=8,  overrides={})),
    ("ulip_pc_rgb_v16", dict(channels=("shape",), ulip2_mode="pc", partial=True,
                           shape_agg_views=16, overrides={})),
    ("ulip_pc_rgb_v32", dict(channels=("shape",), ulip2_mode="pc", partial=True,
                           shape_agg_views=32, overrides={})),
    ("ulip_pc_rgb_v42", dict(channels=("shape",), ulip2_mode="pc", partial=True,
                           shape_agg_views=42, overrides={})),
    ("ulip_pc_xyz",   dict(channels=("shape",), ulip2_mode="pc",
                           partial=True,
                           overrides={"ulip2_use_colors": False,
                                      "ulip2_backbone": "pointbert",
                                      "ulip2_checkpoint": ULIP_CKPT_XYZ,
                                      "ulip2_num_points": 8192,
                                      # SLIP ViT-B tower → 512-d space (the
                                      # colored 10k arm is ViT-g / 1280-d)
                                      "ulip2_embed_dim": 512})),
    ("uni3d",         dict(channels=("shape",), ulip2_mode="pc",
                           partial=True,
                           overrides={"shape_encoder": "uni3d",
                                      "uni3d_checkpoint": UNI3D_CKPT})),
])


def _stack_view_embeddings(emb_map: Dict[str, list], object_ids: List[str]):
    """Stack {obj: [(emb, path)...] | tensor} into one (M, D) matrix.

    Returns (big, ranges) with ranges[i] = (start, end) row span of
    object_ids[i].  Objects without embeddings get an empty span.
    """
    import torch
    chunks, ranges, pos = [], [], 0
    for oid in object_ids:
        v = emb_map.get(oid)
        if v is None:
            ranges.append((pos, pos))
            continue
        if isinstance(v, list):                      # DINO: [(emb, path)]
            t = torch.stack([e for e, _ in v], dim=0)
        else:                                        # ULIP: tensor
            t = v if v.dim() == 2 else v.unsqueeze(0)
        chunks.append(t.float().cpu())
        ranges.append((pos, pos + t.shape[0]))
        pos += t.shape[0]
    big = torch.cat(chunks, dim=0) if chunks else torch.zeros(0, 1)
    return big, ranges


def _aggregate_groups(sims, ranges, prefix: Optional[int],
                      method: str, top_k: int, tau: float) -> np.ndarray:
    """Per-object aggregation of per-view similarities.

    Vectorized re-implementation of step4/_aggregate_view_scores
    ("topk_softmax": softmax-weighted top-k_v pooling, the OPEN-style
    query-conditioned aggregation [chuOPENOcclusionInvariantPerception2024a]
    with k_v = 5, tau = 0.5 per CNOS [nguyenCNOSStrongBaseline2023]).
    Exact for uniform aggregation params; falls back to the pipeline's
    own scalar helper for non-default methods.

    ``prefix`` keeps only the first N views per object — valid because
    views are FPS-ordered (preprocessing/render_views.py; O4 relies on this and
    on the natural-sort fix in step4_dino_reranking).
    """
    import torch
    from pipeline.step4_dino_reranking import _aggregate_view_scores

    out = np.full(len(ranges), -np.inf, dtype=np.float32)
    # group object indices by effective view count for batched pooling
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, (s, e) in enumerate(ranges):
        cnt = e - s
        if prefix is not None:
            cnt = min(cnt, prefix)
        if cnt > 0:
            groups[cnt].append(i)

    for cnt, idxs in groups.items():
        rows = torch.stack([sims[ranges[i][0]:ranges[i][0] + cnt]
                            for i in idxs])                    # (G, cnt)
        if method == "topk_softmax":
            k = min(top_k, cnt)
            vals, _ = rows.topk(k, dim=1)
            w = torch.softmax(vals / tau, dim=1)
            agg = (w * vals).sum(dim=1)
        elif method == "mean":
            agg = rows.mean(dim=1)
        elif method == "max":
            agg = rows.max(dim=1).values
        else:                                   # exact but slow fallback
            agg = torch.tensor([_aggregate_view_scores(
                r_, method=method, top_k=top_k, temperature=tau)[0]
                for r_ in rows])
        out[np.asarray(idxs)] = agg.cpu().numpy()
    return out


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _pass_provenance(pass_key: str, pipe_cfg, need) -> dict:
    """Encoder identity of a precomputed pass, so the eval PC can confirm the
    shipped caches were built with a matching encoder (same weights + code)."""
    return {
        "pass": pass_key,
        "channels": sorted(need),
        "appearance_encoder": pipe_cfg.appearance_encoder,
        "shape_encoder": pipe_cfg.shape_encoder,
        "dino_model": pipe_cfg.dino_model_name,
        "siglip_model": pipe_cfg.siglip_model_name,
        "ulip2_checkpoint": os.path.basename(pipe_cfg.ulip2_checkpoint or ""),
        "ulip2_backbone": pipe_cfg.ulip2_backbone,
        "ulip2_use_colors": bool(pipe_cfg.ulip2_use_colors),
        "ulip2_mode": pipe_cfg.ulip2_mode,
        "ulip2_num_points": pipe_cfg.ulip2_num_points,
        "ulip2_embed_dim": pipe_cfg.ulip2_embed_dim,
        "code_commit": _git_commit(),
    }


def _pc_query_cache_path(paths: dict, pipe_cfg) -> str:
    """Cache path for pc-mode query point-cloud embeddings.

    Same fingerprint fields as step5_shape_matching's gallery partial-view
    cache (encoder identity + checkpoint + dims) — content-based, not
    path/mtime-based, so it stays valid across --results-root and
    --limit-queries changes and never collides between ulip_pc_rgb,
    ulip_pc_xyz, and uni3d (different checkpoints/dims each).
    """
    import hashlib
    encoder = getattr(pipe_cfg, "shape_encoder", "ulip2")
    if encoder == "uni3d":
        meta = (f"encoder=uni3d|model={pipe_cfg.uni3d_model_name}|"
               f"npts={pipe_cfg.uni3d_num_points}|"
               f"ckpt={os.path.basename(pipe_cfg.uni3d_checkpoint or '')}")
    else:
        meta = (f"encoder=ulip2|backbone={pipe_cfg.ulip2_backbone}|"
               f"npts={pipe_cfg.ulip2_num_points}|"
               f"colors={int(pipe_cfg.ulip2_use_colors)}|"
               f"edim={pipe_cfg.ulip2_embed_dim}|"
               f"ckpt={os.path.basename(pipe_cfg.ulip2_checkpoint or '')}")
    digest = hashlib.sha256(meta.encode("utf-8")).hexdigest()[:16]
    # Lives under stage1_root (dataset-scoped, same as queries_index.json),
    # not results_root, so it survives switching --results-root between runs.
    cache_dir = os.path.join(paths["stage1_root"], "query_pc_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"pc_query_cache_{digest}.pt")


def _load_or_build_pc_query_cache(paths: dict, pipe_cfg, index: List[dict],
                                  shape_m) -> Dict[str, "torch.Tensor"]:
    """Load (or build+save) query point-cloud embeddings for a pc-mode pass.

    Mirrors eval_common.pre_encode_ulip_queries' cache-then-encode pattern
    for the image branch, extended to the point-cloud branch, which
    previously had no cache at all — this is the expensive step (one
    point-cloud forward pass per query, not batched), so this is worth far
    more than the image-branch cache in wall-clock time saved on reruns.
    """
    import torch

    cache_path = _pc_query_cache_path(paths, pipe_cfg)
    if os.path.isfile(cache_path):
        print(f"[pc-query-cache] loading {cache_path}...")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        missing = [q for q in index if q["id"] not in cache]
        if not missing:
            print(f"[pc-query-cache] {len(cache)} embeddings loaded.")
            return cache
        print(f"[pc-query-cache] {len(missing)}/{len(index)} queries "
              f"missing from cache — encoding those.")
    else:
        cache = {}
        missing = index

    from tqdm import tqdm
    for q in tqdm(missing, desc="pc-query pre-enc", unit="query"):
        data = np.load(q["npz"])
        emb = shape_m.encode_pointcloud(data["points"], colors=data["colors"])
        cache[q["id"]] = emb.detach().cpu()

    torch.save(cache, cache_path)
    print(f"[pc-query-cache] saved {len(cache)} embeddings -> {cache_path}")
    return cache


def precompute_gallery(paths: dict, object_ids: List[str],
                       resume: bool) -> None:
    """Build every gallery *reference* cache with no query scoring.

    This is the precompute entry point meant to run on the
    gallery-generating PC (see --precompute).  build_pipeline writes the
    DINO/SigLIP/ULIP/Uni3D caches into the gallery dir (--images-dir) and the
    cad/ dir; here we just drive all pass configs and record a provenance
    manifest that the eval PC checks before trusting the shipped caches.
    """
    manifest = {"code_commit": _git_commit(),
                "gallery_size": len(object_ids), "passes": []}
    for pkey in PASS_DEFS:
        try:
            prov = run_pass(pkey, paths, index=None, object_ids=object_ids,
                            limit=None, resume=resume, build_only=True)
            manifest["passes"].append(prov)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[precompute:{pkey}] FAILED ({exc})")
            manifest["passes"].append({"pass": pkey, "error": str(exc)})
    out = os.path.join(paths["images_dir"], "precompute_manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    n_ok = len([p for p in manifest["passes"] if "error" not in p])
    print(f"\n[precompute] done — {n_ok}/{len(PASS_DEFS)} passes built.\n"
          f"[precompute] manifest -> {out}\n"
          f"[precompute] caches: {paths['images_dir']}/*cache*.pt + "
          f"precompute_manifest.json, and "
          f"{os.path.join(paths['data_root'], 'cad')}/.ulip_cache_*.pt")


def verify_precompute_provenance(paths: dict) -> None:
    """Warn (loudly, non-fatally) if shipped caches were built at a different
    code commit than this PC — the divergence guard for cross-machine caches."""
    man = os.path.join(paths["images_dir"], "precompute_manifest.json")
    if not os.path.isfile(man):
        return
    with open(man) as f:
        m = json.load(f)
    remote, local = m.get("code_commit", "unknown"), _git_commit()
    if remote != local and "unknown" not in (remote, local):
        print(f"[provenance] WARNING: gallery caches were precomputed at "
              f"commit {remote}, but this PC is at {local}. If the encoder "
              f"path (pipeline/step3-5, config, eval_common) differs between "
              f"those commits, gallery and query embeddings may be "
              f"inconsistent — verify those files match before trusting "
              f"results.")
    else:
        print(f"[provenance] precompute manifest present (commit {remote}) — OK.")


def run_pass(pass_key: str, paths: dict, index: List[dict],
             object_ids: List[str], limit: Optional[int],
             resume: bool, build_only: bool = False) -> dict:
    """Compute (or load) the score store for one channel pass.

    Store layout::

        {"pass": key, "object_ids": [...],
         "queries": {qid: {"clip": (N,), "dino": {V: (N,)}, "shape": (N,)}}}

    With ``build_only=True`` only the gallery *reference* caches are built
    (DINO/SigLIP/ULIP/Uni3D embeddings, written by build_pipeline) and a
    provenance record is returned — no queries are scored.  This is the
    precompute path meant to run on the gallery-generating PC.
    """
    import torch

    tag = f"_n{limit}" if limit else ""
    cache = os.path.join(paths["results_root"], "_cache",
                         f"scores_{pass_key}{tag}.pt")
    qids = ([q["id"] for q in (index or [])][:limit] if limit
            else [q["id"] for q in (index or [])])
    if resume and not build_only and os.path.isfile(cache):
        store = torch.load(cache, map_location="cpu", weights_only=False)
        if store.get("object_ids") == object_ids and \
                all(q in store["queries"] for q in qids):
            print(f"[pass:{pass_key}] loaded cache ({len(store['queries'])} "
                  f"queries) <- {cache}")
            return store

    import eval_common as ec
    from PIL import Image

    pdef = PASS_DEFS[pass_key]
    need = set(pdef["channels"])
    print(f"[pass:{pass_key}] computing channels {sorted(need)} "
          f"for {len(qids)} queries ...")

    cfg = ec.EvalConfig(
        ref_dir=paths["images_dir"],
        desc_file=paths["desc_file"],
        # shape passes need CAD meshes (id source / full-mesh encoding);
        # appearance-only passes skip ULIP entirely.
        cad_mesh_glob=("" if pdef.get("no_shape")
                       else os.path.join(paths["data_root"], "cad", "*.obj")),
        result_folder=os.path.join(paths["results_root"], "_cache"),
        clip_top_k=10 ** 6, dino_top_k=10 ** 6,
        ulip2_top_k=10 ** 6, fusion_top_k=10 ** 6,
        weight_clip=BASE_WEIGHTS[0], weight_dino=BASE_WEIGHTS[1],
        weight_ulip=BASE_WEIGHTS[2],
        ulip2_mode=pdef["ulip2_mode"],
        ulip2_use_partial_views=pdef["partial"],
        ulip2_checkpoint=ULIP_CKPT_DEFAULT,
        # Encode/keep ALL views; O4 trimming happens at derivation time.
        # SHREC_DINO_POOLING selects the DINO view-embedding pooling: default
        # "cls" reproduces the archived Stage-1 winner; "mean" (mean-over-tokens,
        # the pipeline's mean-patch pooling) matches Stage 2/3 for the cross-stage
        # comparable rerun. The DINO gallery cache is keyed by pooling
        # (step4._cache_path), so cls and mean never collide.
        pipeline_overrides={"num_views": None,
                            "dino_pooling": os.environ.get("SHREC_DINO_POOLING", "mean"),
                            **pdef["overrides"]},
    )
    if "shape" not in need:
        cfg.cad_mesh_glob = ""

    pipe_cfg, clip_retr, dino_rer, _fusion, shape_m = ec.build_pipeline(cfg)
    if "shape" in need and shape_m is None:
        raise SystemExit(f"[pass:{pass_key}] shape encoder failed to load "
                         f"(checkpoint/repo missing?) — cannot compute "
                         f"the S_shape channel.")

    if build_only:
        # Gallery reference caches are now written by build_pipeline; return
        # provenance and stop (no query scoring in the precompute path).
        prov = _pass_provenance(pass_key, pipe_cfg, need)
        del clip_retr, dino_rer, shape_m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[precompute:{pass_key}] reference caches built.")
        return prov

    # --- pre-stack reference embeddings into one matrix per channel -------
    dino_big = dino_ranges = None
    if "dino" in need:
        dino_big, dino_ranges = _stack_view_embeddings(
            dino_rer._ref_embeddings, object_ids)
        dino_big = dino_big.to(pipe_cfg.device)
    shape_big = shape_ranges = None
    if "shape" in need:
        shape_big, shape_ranges = _stack_view_embeddings(
            shape_m._cad_embeddings, object_ids)
        shape_big = shape_big.to(pipe_cfg.device)

    # --- ULIP image-branch query cache (cross-modal passes) ---------------
    ulip_img_cache = None
    if "shape" in need and pdef["ulip2_mode"] == "cross":
        cache_img = os.path.join(paths["results_root"], "_cache",
                                 "ulip_query_img_cache.pt")
        ulip_img_cache = ec.load_ulip_query_cache(cache_img)
        if ulip_img_cache is None:
            pngs = [q["png"] for q in index]
            ulip_img_cache = ec.pre_encode_ulip_queries(pngs, shape_m)
            os.makedirs(os.path.dirname(cache_img), exist_ok=True)
            torch.save(ulip_img_cache, cache_img)

    # --- point-cloud query cache (pc-mode passes) --------------------------
    # Unlike the cross-modal image branch above, this had NO cache at all:
    # every pc-mode pass (ulip_pc_rgb, ulip_pc_xyz, uni3d) re-encoded every
    # query's point cloud from scratch, every single run — the single most
    # expensive step in the whole ablation grid (~1-2s/query, one at a time,
    # no batching). Keyed by encoder config (not path/mtime), same fields as
    # the gallery partial-view cache, so ulip_pc_rgb/xyz/uni3d never collide
    # and stay valid across --results-root / --limit-queries changes.
    pc_query_cache = None
    if "shape" in need and pdef["ulip2_mode"] != "cross":
        pc_query_cache = _load_or_build_pc_query_cache(paths, pipe_cfg, index, shape_m)

    by_id = {q["id"]: q for q in index}
    store = {"pass": pass_key, "object_ids": object_ids, "queries": {}}
    for qi, qid in enumerate(qids):
        q = by_id[qid]
        rec: dict = {}
        roi = None
        if "clip" in need or "dino" in need:
            roi = Image.open(q["png"]).convert("RGB")

        if "clip" in need:
            res = clip_retr.retrieve(roi, top_k=10 ** 6)
            smap = {c.object_id: c.score for c in res.candidates}
            rec["clip"] = np.array([smap.get(o, -np.inf) for o in object_ids],
                                   dtype=np.float32)

        if "dino" in need:
            with torch.no_grad():
                qe = dino_rer.encode_image(roi)                  # (1, D)
                sims = (qe @ dino_big.T).squeeze(0).float().cpu()
            rec["dino"] = {
                V: _aggregate_groups(sims, dino_ranges, V,
                                     pipe_cfg.dino_view_aggregation,
                                     pipe_cfg.dino_view_topk,
                                     pipe_cfg.dino_view_temperature)
                for V in VIEW_BUDGETS}

        if "shape" in need:
            if pdef["ulip2_mode"] == "cross":
                qe = ec.lookup_ulip_query_emb(ulip_img_cache, q["png"])
                if qe is None:                       # encoded on the fly
                    qe = shape_m.encode_image(roi or Image.open(q["png"]))
                qe = qe.to(pipe_cfg.device)
            else:
                qe = pc_query_cache[qid].to(pipe_cfg.device)
            with torch.no_grad():
                sims = (qe.float() @ shape_big.T).squeeze(0).float().cpu()
            # partial refs: multi-view pooling over the first SHAPE_AGG_VIEWS
            # (FPS-ordered) of the cached views — the cache still holds all 42
            # encoded; full-mesh refs have a single embedding per object.
            rec["shape"] = _aggregate_groups(
                sims, shape_ranges,
                pdef.get("shape_agg_views", SHAPE_AGG_VIEWS) if pdef["partial"] else None,
                pipe_cfg.ulip_view_aggregation if pdef["partial"] else "max",
                pipe_cfg.ulip_view_topk, pipe_cfg.ulip_view_temperature)

        store["queries"][qid] = rec
        if (qi + 1) % 50 == 0 or qi + 1 == len(qids):
            print(f"[pass:{pass_key}]   {qi + 1}/{len(qids)}")

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(store, cache)
    print(f"[pass:{pass_key}] cached -> {cache}")

    # Free GPU memory before the next pass (encoders + big matrices).
    del clip_retr, dino_rer, shape_m, dino_big, shape_big
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return store


# ===========================================================================
# 5. Ablation registry
# ===========================================================================

@dataclass(frozen=True)
class AblationSpec:
    """One cell of the Stage-1 grid (thesis tab:eval_stage1_ablation_grid).

    ``channels`` maps channel name -> (pass_key, view_budget|None).  A
    channel that is absent is excluded from fusion.  ``alias_of`` marks
    grid cells that are numerically identical to another cell (e.g. the
    "default" arm of most ablations is the BASE config E1c) — they are
    not recomputed, only cross-referenced in the summary.
    """
    name: str
    group: str
    question: str
    thesis_ref: str = "subsec:eval_baselines"
    bib: str = ""
    channels: Dict[str, tuple] = field(default_factory=dict)
    weights: Tuple[float, float, float] = BASE_WEIGHTS
    fusion_method: str = "weighted_sum"
    scope: str = "full"          # "full" | "clip_topk" | "dino_topk"
    geometry: Optional[str] = None  # None | fitness | chamfer_unaligned |
                                    # chamfer_ransac | chamfer_icp | scale_gate
    alias_of: Optional[str] = None
    notes: str = ""

    @property
    def passes(self) -> Tuple[str, ...]:
        return tuple(sorted({p for p, _ in self.channels.values()}))

    @property
    def needs_gedi(self) -> bool:
        # Every signal except the unaligned diagnostic needs GeDi, since
        # alignment is what GeDi correspondences are for.
        return self.geometry in ("fitness", "chamfer_ransac", "chamfer_icp",
                                 "both_borda", "both_borda_base")


# Stage-1 BASE.  S_shape is scored in **pc-mode**: SHREC'18 queries are real
# RGB-D scans, so the shape channel encodes the query *point cloud* rather
# than the query image.  (Stage 2 / MI3DOR keeps cross-mode because it has no
# depth — thesis subsec:eval_datasets_mi3dor_restricted_role.)
_BASE_CH = {"clip": ("base", None), "dino": ("base", 42),
            "shape": ("ulip_pc_rgb", None)}
_TV_CH = {"clip": ("base", None), "dino": ("base", 42)}   # text+view only
# A7f: BASE fusion but the S_shape channel pools all 42 partial views instead
# of the frozen SHAPE_AGG_VIEWS=16.  Reuses the same 42-view gallery cache
# (only the pooled count differs), so it re-pools cached sims — no re-encode.
# Sensitivity arm for "does feeding the shape channel 42 views move the frozen
# BASE?"  Isolated A7 gave shape-only +0.013 (16->42); this measures fusion.
_BASE_CH_V42 = {"clip": ("base", None), "dino": ("base", 42),
                "shape": ("ulip_pc_rgb_v42", None)}


def _spec(name, group, question, **kw) -> Tuple[str, AblationSpec]:
    return name, AblationSpec(name=name, group=group, question=question, **kw)


ABLATIONS: "OrderedDict[str, AblationSpec]" = OrderedDict([
    # --- E1: does the global shape channel help? (thesis E1) -------------
    # "S_text alone; (S_text,S_view) OSCAR-equivalent; full fusion; plus a
    #  CLIP-pruned variant — all variants produced in a single pass."
    _spec("E1a_text_only", "E1", "S_text alone",
          bib="pulliOSCAROpenSetCAD2025",
          channels={"clip": ("base", None)}, weights=(1.0, 0.0, 0.0)),
    _spec("E1_view_only", "E1", "S_view alone (appearance channel)",
          bib="nguyenCNOSStrongBaseline2023",
          channels={"dino": ("base", 42)}, weights=(0.0, 1.0, 0.0)),
    _spec("E1_shape_only", "E1", "S_shape alone (pc-mode)",
          bib="xueULIP2ScalableMultimodal2024",
          channels={"shape": ("ulip_pc_rgb", None)}, weights=(0.0, 0.0, 1.0)),
    _spec("E1b_text_view", "E1",
          "(S_text, S_view) — OSCAR's channel set, full-database fusion",
          bib="pulliOSCAROpenSetCAD2025",
          channels=dict(_TV_CH), weights=(0.43, 0.57, 0.0)),
    # Faithful OSCAR reproduction: CLIP selects the top-k shortlist, DINOv2
    # arg-maxes *within* it, and there is no shape channel at all
    # (thesis background.tex, "OSCAR" paragraph).  Weight (0,1,0) + scope
    # clip_topk makes derive_ranking take the single-channel shortcut: the
    # pool is ordered by S_view alone, the tail by S_text.  This differs from
    # E1d/O2_clip_cascade, which score S_view AND S_shape on the shortlist.
    _spec("E1_oscar_cascade", "E1",
          "OSCAR cascade: CLIP top-20 shortlist, DINOv2 arg-max within it",
          bib="pulliOSCAROpenSetCAD2025",
          channels=dict(_TV_CH), weights=(0.0, 1.0, 0.0), scope="clip_topk"),
    _spec("E1c_full_fusion", "E1",
          "full fusion (S_text, S_view, S_shape) — OSCAR+ BASE config",
          bib="pulliOSCAROpenSetCAD2025, zhouCrossModal3DRepresentation",
          channels=dict(_BASE_CH)),
    _spec("A7f_full_fusion_shape_V42", "E1",
          "BASE fusion, S_shape pools 42 partial views (vs frozen 16) — "
          "fusion sensitivity of the shape view budget",
          channels=dict(_BASE_CH_V42)),
    _spec("E1d_clip_pruned", "E1",
          "S_view/S_shape scored only on the CLIP top-20 shortlist",
          bib="pulliOSCAROpenSetCAD2025",
          channels=dict(_BASE_CH), scope="clip_topk"),
    # --- E2: local geometry re-ranking (thesis E2) -----------------------
    # The five configurations of subsec:eval_baselines.  Alignment requires
    # correspondences, so every *aligned* variant necessarily runs GeDi ->
    # RANSAC first; a distance without GeDi is by construction unaligned and
    # serves only as the diagnostic control.
    _spec("E2_none", "E2", "no geometry re-ranking (= BASE)",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    _spec("E2_fitness", "E2", "RANSAC fitness only (no surface distance)",
          bib="caraffaFreeZeTrainingfreeZeroshot2025",
          channels=dict(_BASE_CH), geometry="fitness"),
    _spec("E2_chamfer_unaligned", "E2",
          "unaligned trimmed distance — diagnostic control, not a method",
          bib="diUREDUnsupervised3D2023",
          channels=dict(_BASE_CH), geometry="chamfer_unaligned",
          notes=("Expected to underperform E2_chamfer_ransac; it exists to "
                 "verify that the gain comes from evaluating the distance "
                 "AFTER alignment (thesis subsec:eval_baselines).")),
    _spec("E2_chamfer_ransac", "E2",
          "GeDi-RANSAC alignment, then trimmed surface distance",
          bib="caraffaFreeZeTrainingfreeZeroshot2025, diUREDUnsupervised3D2023",
          channels=dict(_BASE_CH), geometry="chamfer_ransac"),
    _spec("E2_chamfer_icp", "E2",
          "GeDi-RANSAC + ICP refinement, then trimmed surface distance",
          bib="caraffaFreeZeTrainingfreeZeroshot2025",
          channels=dict(_BASE_CH), geometry="chamfer_icp",
          notes=("ICP is OFF in the frozen config: it adds a refinement DOF "
                 "that can partly launder a wrong retrieval into a plausible "
                 "fit, which is the confusion to avoid in a *retrieval* "
                 "evaluation. This arm is retained as the evidence for that "
                 "default (measured 2026-07-30: +0.0001 nDCG for ~5.4 s per "
                 "query), not as a competitor to it.")),
    # Both geometry signals combined by mean rank (Borda, ties averaged)
    # [Aslam & Montague, SIGIR 2001].  Reads the same cached registration as
    # the three aligned arms above, so it costs no extra RANSAC.  See
    # apply_geometry() for why a raw sum of fitness and D_trim is not a fusion.
    _spec("E2_both", "E2",
          "fitness + D_trim combined by mean rank (Borda)",
          bib="aslamModelsMetasearch2001, "
              "caraffaFreeZeTrainingfreeZeroshot2025, "
              "diUREDUnsupervised3D2023",
          channels=dict(_BASE_CH), geometry="both_borda"),
    # --- E2b: partial-view vs full-mesh shape reference ------------------
    _spec("E2b_partial", "E2b", "partial rendered views as S_shape "
          "reference (= BASE)", bib="linSAM6DSegmentAnything2024",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    _spec("E2b_fullmesh", "E2b", "full-mesh S_shape reference",
          bib="diUREDUnsupervised3D2023",
          channels={**_TV_CH, "shape": ("ulip_pc_fullmesh", None)}),
    # Geometrie auf dem jeweils staerksten Arm OHNE Geometrie. Die E2-Arme
    # sitzen alle auf dem BASE-Ranking (0.5868); seit dem Farb-Fix ist aber
    # E2b_fullmesh mit 0.5935 vorn. Ein besseres Ausgangsranking hebt die Latte,
    # die das Re-Ranking ueberspringen muss — Stage 3 zeigt denselben
    # Mechanismus. Beide Kandidaten sind vordefiniert, gefahren wird per Gate
    # nur der, der sich am Ende als staerkster erweist.
    _spec("E2b_fullmesh_geo", "E2b",
          "GeDi-RANSAC re-rank ON TOP of the full-mesh fusion",
          bib="caraffaFreeZeTrainingfreeZeroshot2025",
          channels={**_TV_CH, "shape": ("ulip_pc_fullmesh", None)},
          geometry="chamfer_ransac"),
    _spec("E7_ulip2_cross_fullmesh_geo", "E7",
          "GeDi-RANSAC re-rank ON TOP of the cross x full-mesh fusion",
          bib="caraffaFreeZeTrainingfreeZeroshot2025",
          channels={**_TV_CH, "shape": ("ulip_cross_fullmesh", None)},
          geometry="chamfer_ransac"),
    # Isolated (shape-only) counterparts: the partial-vs-fullmesh effect read
    # on the shape channel ALONE, not masked by the shared text+view fusion.
    # E2b_partial_shape_only == E1_shape_only (pc-mode partial views).
    _spec("E2b_partial_shape_only", "E2b",
          "partial-view S_shape ALONE (isolated) = E1_shape_only",
          channels={"shape": ("ulip_pc_rgb", None)}, weights=(0.0, 0.0, 1.0),
          alias_of="E1_shape_only"),
    _spec("E2b_fullmesh_shape_only", "E2b",
          "full-mesh S_shape ALONE (isolated)",
          bib="diUREDUnsupervised3D2023",
          channels={"shape": ("ulip_pc_fullmesh", None)},
          weights=(0.0, 0.0, 1.0)),
    # --- E4: appearance encoder ------------------------------------------
    _spec("E4_dinov2", "E4", "DINOv2 appearance channel (= BASE)",
          bib="nguyenCNOSStrongBaseline2023",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    _spec("E4_siglip", "E4", "SigLIP appearance channel",
          bib="zhaiSigmoidLossLanguage2023, nguyenSHREC2025Retrieval2025",
          channels={**_BASE_CH, "dino": ("siglip", 42)}),
    # Isolated appearance arms: DINOv2 ALONE vs SigLIP ALONE (MAP-pooled, the
    # pooler_output fix f50a844f), so the encoder swap is compared with no
    # text/shape fusion masking it.  E4_dino_only == E1_view_only.
    _spec("E4_dino_only", "E4",
          "DINOv2 appearance ALONE (isolated) = E1_view_only",
          channels={"dino": ("base", 42)}, weights=(0.0, 1.0, 0.0),
          alias_of="E1_view_only"),
    _spec("E4_siglip_only", "E4",
          "SigLIP appearance ALONE (isolated, MAP-pooled)",
          bib="zhaiSigmoidLossLanguage2023",
          channels={"dino": ("siglip", 42)}, weights=(0.0, 1.0, 0.0)),
    # --- E6: fusion strategy ---------------------------------------------
    _spec("E6_weighted", "E6", "weighted-sum fusion (= BASE)",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    # Rank-based fusion = Reciprocal Rank Fusion (k=60), the standard
    # scale-free rank combiner.  SAMURAI/COMPASS motivate *why* rank-based
    # fusion is worth testing, but SAMURAI's own vote runs over text/
    # silhouette retrieval strategies, not over our three channels — so RRF
    # is cited on its own terms rather than as a SAMURAI reproduction.
    _spec("E6_rrf", "E6", "reciprocal rank fusion (RRF, k=60)",
          bib="cormackReciprocalRankFusion2009",
          channels=dict(_BASE_CH), fusion_method="rank_fusion"),
    # --- E7: 3D foundation model in S_shape ------------------------------
    # Both arms run in pc-mode so encoder choice is the only difference
    # (Uni3D has no ULIP-style image branch; cross-modal would confound).
    # Since Stage-1 BASE is itself pc-mode, the ULIP-2 arm *is* the BASE
    # config — cross-referenced rather than recomputed.
    _spec("E7_ulip2_pc", "E7", "ULIP-2 shape encoder (pc-mode) (= BASE)",
          bib="xueULIP2ScalableMultimodal2024",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    _spec("E7_uni3d", "E7", "Uni3D shape encoder (pc-mode)",
          bib="zhouUni3DExploringUnified2023, "
              "vandenherrewegenFinetuning3DFoundation2024",
          channels={**_TV_CH, "shape": ("uni3d", None)}),
    # Isolated shape-encoder arms: ULIP-2(pc) vs Uni3D(pc) vs ULIP-2(cross),
    # each scored ALONE so the encoder / query-mode is the only variable and
    # the number is directly comparable to the fused E7 arms above.  The fused
    # E7_uni3d confounds encoder-swap with the text+view floor; these isolate
    # it (resolves the solo-uni3d-vs-ulip2 question).  E7_ulip2_pc_shape_only
    # == E1_shape_only.
    _spec("E7_ulip2_pc_shape_only", "E7",
          "ULIP-2 shape ALONE, pc-mode (isolated) = E1_shape_only",
          bib="xueULIP2ScalableMultimodal2024",
          channels={"shape": ("ulip_pc_rgb", None)}, weights=(0.0, 0.0, 1.0),
          alias_of="E1_shape_only"),
    _spec("E7_uni3d_shape_only", "E7",
          "Uni3D shape ALONE, pc-mode (isolated)",
          bib="zhouUni3DExploringUnified2023",
          channels={"shape": ("uni3d", None)}, weights=(0.0, 0.0, 1.0)),
    _spec("E7_ulip2_cross_shape_only", "E7",
          "ULIP-2 shape ALONE, cross-mode query image (isolated)",
          bib="xueULIP2ScalableMultimodal2024",
          channels={"shape": ("ulip_cross_rgb", None)},
          weights=(0.0, 0.0, 1.0)),
    # Vierte Zelle der FUSIONIERTEN Matrix. Isoliert waren alle vier Zellen seit
    # 2026-09-04 da, fusioniert fehlte cross x partial — dadurch liess sich
    # E7_ulip2_cross_fullmesh nur gegen die beiden pc-Arme stellen, nicht gegen
    # sein eigenes Gegenstueck. Reine Tier-2-Ableitung: der Pass ulip_cross_rgb
    # ist bereits gecacht, es faellt nur die Fusion an.
    _spec("E7_ulip2_cross", "E7",
          "full fusion with cross-mode shape vs PARTIAL-view gallery",
          bib="xueULIP2ScalableMultimodal2024",
          channels={**_TV_CH, "shape": ("ulip_cross_rgb", None)}),
    # Vierte Zelle der 2x2-Matrix, siehe Pass ulip_cross_fullmesh.
    _spec("E7_ulip2_cross_fullmesh_shape_only", "E7",
          "ULIP-2 shape ALONE, cross-mode query vs FULL-MESH gallery (isolated)",
          bib="xueULIP2ScalableMultimodal2024",
          channels={"shape": ("ulip_cross_fullmesh", None)},
          weights=(0.0, 0.0, 1.0)),
    _spec("E7_ulip2_cross_fullmesh", "E7",
          "full fusion with cross-mode shape vs FULL-MESH gallery",
          bib="xueULIP2ScalableMultimodal2024",
          channels={**_TV_CH, "shape": ("ulip_cross_fullmesh", None)}),
    # --- O1: is S_shape redundant once S_GeDi exists? --------------------
    _spec("O1a_no_geometry", "O1", "neither S_shape nor S_GeDi",
          channels=dict(_TV_CH), weights=(0.43, 0.57, 0.0),
          alias_of="E1b_text_view"),
    _spec("O1b_shape_in_fusion", "O1", "S_shape in fusion (= BASE)",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    _spec("O1c_gedi_post_fusion", "O1",
          "S_GeDi as post-fusion re-ranker (no S_shape)",
          channels=dict(_TV_CH), weights=(0.43, 0.57, 0.0),
          geometry="fitness"),
    _spec("O1d_shape_plus_gedi", "O1", "S_shape in fusion + S_GeDi rerank",
          channels=dict(_BASE_CH), geometry="fitness",
          alias_of="E2_fitness"),
    # "S_GeDi replaces S_shape inside the fusion score" is NOT implementable
    # as originally specified: it needs a GeDi score for EVERY gallery entry,
    # i.e. 2,101 x 3,308 ~ 6.95M RANSAC fits ~ 830 h per cell at the measured
    # 0.430 s/fit (--bench-gedi).  The asymmetry is structural — CLIP/DINOv2/
    # ULIP-2 score the full gallery with a matrix multiply, geometry cannot —
    # and the thesis states it rather than working around it.
    #
    # What IS answerable at shortlist level, and is what O1e now means: once
    # geometry is available, does the base fusion score still carry
    # information?  E2_both discards it inside the shortlist and lets geometry
    # rank alone; O1e keeps it as a third Borda voter.  Note this inverts the
    # usual framing — E2_both is the *aggressive* arm (a CAD ranked first by
    # all three channels gets no credit if RANSAC disagrees), O1e is the
    # conventional cascade that retains the earlier-stage score.  Same cached
    # registrations, so O1e is free.
    _spec("O1e_gedi_with_base", "O1",
          "geometry + base fusion rank, combined by mean rank (Borda)",
          bib="aslamModelsMetasearch2001, pulliOSCAROpenSetCAD2025",
          channels=dict(_BASE_CH), geometry="both_borda_base",
          notes=("Renamed from the original 'S_GeDi inside the fusion score': "
                 "full-database S_GeDi is infeasible (see comment above), so "
                 "this evaluates the same question restricted to the top-K "
                 "shortlist. Documented as an approximation.")),
    # --- O2: scope and ordering ------------------------------------------
    _spec("O2_full_database", "O2", "simultaneous full-database fusion "
          "(= BASE)", channels=dict(_BASE_CH),
          alias_of="E1c_full_fusion"),
    _spec("O2_clip_cascade", "O2", "OSCAR cascade: CLIP top-20 shortlist",
          bib="pulliOSCAROpenSetCAD2025",
          channels=dict(_BASE_CH), scope="clip_topk",
          alias_of="E1d_clip_pruned"),
    # The paper's ACTUAL filter.  O2_clip_cascade above reproduces OSCAR's
    # *fallback* (top-k); this reproduces its mechanism (threshold), so the
    # pair together is what the write-up compares.  Reported two ways: this
    # padded row in Table A, plus a short faithful paragraph built from the
    # shortlist_size / fallback_rate fields recorded per query.
    _spec("O2_clip_threshold", "O2",
          f"OSCAR threshold pruning: sim_text >= {CLIP_TAU_TEXT} "
          f"(top-{CLIP_PRUNE_K} fallback)",
          bib="pulliOSCAROpenSetCAD2025",
          channels=dict(_BASE_CH), scope="clip_threshold",
          notes=("tau_text = 0.37 is Pulli et al.'s value, calibrated on "
                 "MI3DOR/YCB-V caption-similarity distributions. Whether it "
                 "transfers to SHREC'18 is exactly what the |S'| statistics "
                 "test: if it admits ~3 or ~3,000 CADs, this arm measures a "
                 "mistuned constant rather than the method, and the write-up "
                 "must say so.")),
    # Same mechanism, tau fitted to THIS dataset's similarity distribution.
    # Without it the grid can only report that OSCAR's constant fails to
    # transfer; with it, the question "does threshold pruning work at all on
    # SHREC'18?" is separable from "does 0.37 transfer?".
    _spec("O2_clip_threshold_cal", "O2",
          "threshold pruning with tau calibrated to SHREC'18",
          bib="pulliOSCAROpenSetCAD2025",
          channels=dict(_BASE_CH), scope="clip_threshold_cal",
          notes=("tau is the 5th percentile of the per-query max CLIP "
                 "similarity, i.e. the highest threshold that still leaves "
                 "95% of queries with a non-empty candidate set. Calibrated "
                 "on coverage rather than on set size or on a retrieval "
                 "metric, so it is neither a restatement of top-k nor tuned "
                 "on the evaluation.")),
    _spec("O2_visual_first", "O2", "visual-first: S_view prunes to top-20",
          bib="nguyenSHREC2025Retrieval2025",
          channels=dict(_BASE_CH), scope="dino_topk"),
    # --- O4: number of reference views (appearance channel) --------------
    *[_spec(f"O4_V{v}", "O4", f"V = {v} FPS-ordered reference views",
            bib="nguyenCNOSStrongBaseline2023",
            channels={**_BASE_CH, "dino": ("base", v)},
            **({"alias_of": "E1c_full_fusion"} if v == 42 else {}))
      for v in VIEW_BUDGETS],
    # --- O5: query point cloud colors (pc-mode, see E7 note) -------------
    _spec("O5_xyzrgb", "O5", "XYZ+RGB query point cloud (pc-mode)",
          bib="xueULIP2ScalableMultimodal2024",
          channels={**_TV_CH, "shape": ("ulip_pc_rgb", None)},
          alias_of="E7_ulip2_pc"),
    _spec("O5_xyz_only", "O5", "XYZ-only query point cloud (colors zeroed)",
          channels={**_TV_CH, "shape": ("ulip_pc_xyz", None)}),
    # Isolated colour ablation: XYZ+RGB vs XYZ-only query PC, shape channel
    # ALONE, so the colour effect is not diluted by the text+view fusion.
    # O5_xyzrgb_shape_only == E1_shape_only.
    _spec("O5_xyzrgb_shape_only", "O5",
          "XYZ+RGB query PC, S_shape ALONE (isolated) = E1_shape_only",
          channels={"shape": ("ulip_pc_rgb", None)}, weights=(0.0, 0.0, 1.0),
          alias_of="E1_shape_only"),
    _spec("O5_xyz_shape_only", "O5",
          "XYZ-only query PC, S_shape ALONE (isolated)",
          channels={"shape": ("ulip_pc_xyz", None)}, weights=(0.0, 0.0, 1.0)),
    # === A2: appearance view count, ISOLATED (view-only, no fusion) ========
    # DINOv2 alone at each render budget, so the view-count effect is read on
    # the appearance channel itself (the O4_V* arms measure it inside fusion).
    # V=42 == E1_view_only. Pure Tier-2 derivation from scores_base.pt.
    *[_spec(f"A2_view_only_V{v}", "A2",
            f"DINOv2 appearance ALONE, V={v} FPS-ordered views (isolated)",
            channels={"dino": ("base", v)}, weights=(0.0, 1.0, 0.0),
            **({"alias_of": "E1_view_only"} if v == 42 else {}))
      for v in VIEW_BUDGETS],
    # === A7: ULIP-2 shape gallery-view count, ISOLATED (shape-only) ========
    # ULIP-2 shape alone, sweeping how many FPS-ordered gallery partial-views
    # are pooled. V=42 == E1_shape_only (BASE SHAPE_AGG_VIEWS=42).
    *[_spec(f"A7_shape_only_V{v}", "A7",
            f"ULIP-2 shape ALONE, {v}-view gallery aggregation (isolated, pc)",
            channels={"shape": (f"ulip_pc_rgb_v{v}", None)},
            weights=(0.0, 0.0, 1.0),
            **({"alias_of": "E1_shape_only"} if v == 42 else {}))
      for v in (8, 16, 32, 42)],
])

# --- guard: the readable CLI may only name arms that really exist -----------
# Every arm derive_arm can emit (i.e. every entry of _SUPPORTED) is checked
# against the registry above at import time, so a rename here can never leave
# the translation table pointing at a nonexistent ablation.
_unregistered = [a for a in _DERIVABLE_ARMS if a not in ABLATIONS]
if _unregistered:
    raise SystemExit("[stage1] CLI table and ablation registry have drifted "
                     "apart — these arms do not exist (any more): "
                     + ", ".join(_unregistered))
# Second half of the same guard: an arm that re-ranks geometrically MUST be in
# _GEOMETRY_ARMS, otherwise the prelude would skip the geometry env, the dgedi
# healthcheck and --geom-k — the run would still produce numbers, just at a
# different depth than the reported ones (silent misconfiguration).
_ungated = [a for a in _DERIVABLE_ARMS
            if ABLATIONS[a].geometry and a not in _GEOMETRY_ARMS]
if _ungated:
    raise SystemExit("[stage1] geometry arms without geometry env in the CLI: "
                     + ", ".join(_ungated))


def select_ablations(arg: Optional[str], run_all: bool,
                     with_geometry: bool) -> List[AblationSpec]:
    """Resolve --ablations/--all into an ordered spec list."""
    if run_all:
        picked = list(ABLATIONS.values())
    else:
        picked, wanted = [], [t.strip() for t in (arg or "").split(",") if t.strip()]
        if not wanted:
            raise SystemExit("Nothing selected — use --all, --ablations or --list.")
        for tok in wanted:
            hits = [s for s in ABLATIONS.values()
                    if s.name == tok or s.group == tok]
            if not hits:
                raise SystemExit(f"Unknown ablation '{tok}' (see --list).")
            picked.extend(h for h in hits if h not in picked)
    if SKIP_ICP:
        # E2_chamfer_icp IS the ICP measurement, so running it under --no-icp
        # would either fabricate its result (d_icp = d_ransac) or quietly
        # re-enable the very cost the flag exists to avoid.  Drop it and say
        # so, rather than emitting a row that looks like an ICP result.
        icp_cells = [s.name for s in picked if s.geometry == "chamfer_icp"]
        picked = [s for s in picked if s.geometry != "chamfer_icp"]
        if icp_cells:
            print(f"[select] --no-icp: dropped {', '.join(icp_cells)} — that "
                  f"arm measures ICP, so it cannot run with ICP disabled. "
                  f"Cite the K=5 measurement (Delta 0.0001 nDCG) instead.")
    if not with_geometry:
        dropped = [s.name for s in picked if s.geometry]
        picked = [s for s in picked if not s.geometry]
        if dropped:
            print(f"[select] geometry ablations skipped (add "
                  f"--with-geometry): {', '.join(dropped)}")
    return picked


# ===========================================================================
# 6. Tier 2 — derivations (fusion / scoping / geometry on cached vectors)
# ===========================================================================

def _synthetic_results(spec: AblationSpec, vecs: Dict[str, np.ndarray],
                       pool: np.ndarray, object_ids: List[str],
                       cad_dir: str):
    """Wrap cached score vectors into the pipeline's result dataclasses.

    This is what makes tier 2 faithful: the production ScoreFusion
    consumes CLIPRetrievalResult / DINOReRankingResult /
    ShapeMatchingResult exactly as in the live pipeline (step6_fusion).
    Candidates are sorted per channel because majority voting (E6)
    consumes rank order.
    """
    from pipeline.step3_clip_retrieval import (CLIPCandidate,
                                               CLIPRetrievalResult)
    from pipeline.step4_dino_reranking import (DINOCandidate,
                                               DINOReRankingResult)
    from pipeline.step5_shape_matching import (ShapeCandidate,
                                               ShapeMatchingResult)

    dummy = np.zeros(1, dtype=np.float32)

    def ordered(ch):
        v = vecs[ch]
        return pool[np.argsort(-v[pool], kind="stable")]

    clip_r = dino_r = shape_r = None
    if "clip" in vecs:
        clip_r = CLIPRetrievalResult(
            candidates=[CLIPCandidate(object_id=object_ids[i],
                                      score=float(vecs["clip"][i]))
                        for i in ordered("clip")],
            query_embedding=dummy)
    if "dino" in vecs:
        dino_r = DINOReRankingResult(
            candidates=[DINOCandidate(object_id=object_ids[i],
                                      dino_score=float(vecs["dino"][i]),
                                      clip_score=0.0)
                        for i in ordered("dino")],
            query_embedding=dummy)
    if "shape" in vecs:
        shape_r = ShapeMatchingResult(
            candidates=[ShapeCandidate(
                object_id=object_ids[i],
                shape_score=float(vecs["shape"][i]),
                cad_model_path=os.path.join(cad_dir, object_ids[i] + ".obj"))
                for i in ordered("shape")],
            query_embedding=dummy)
    return clip_r, dino_r, shape_r


def derive_ranking(spec: AblationSpec, qid: str, stores: Dict[str, dict],
                   object_ids: List[str], fusion_mod, cad_dir: str,
                   diag: Optional[dict] = None) -> List[int]:
    """Rank the gallery for one query under one ablation config.

    Returns gallery indices in rank order (full list — the tail beyond a
    pruned shortlist is ordered by the pruning channel, mirroring how a
    cascade would leave non-shortlisted items ranked by its first stage).

    Always returning the FULL list is what makes the threshold scope
    reportable: OSCAR's candidate set is variable-sized, and the official
    ``precision(x)`` divides by the *submitted* list length, so submitting a
    short list would inflate precision while recall collapsed.  The padded
    form goes in Table A; ``diag`` carries the numbers needed to characterise
    the faithful variable-length form alongside it.

    ``diag``, when given, receives ``shortlist_size`` and ``fallback`` for the
    threshold scope.
    """
    n = len(object_ids)
    vecs: Dict[str, np.ndarray] = {}
    for ch, (pkey, budget) in spec.channels.items():
        rec = stores[pkey]["queries"][qid][ch]
        v = rec[budget] if isinstance(rec, dict) else rec
        # -inf marks gallery objects a channel could not score (missing
        # views/embeddings).  Clamp to the channel minimum so min-max
        # normalization in ScoreFusion stays well-defined; such objects
        # simply rank last in that channel.
        if not np.isfinite(v).all():
            finite = v[np.isfinite(v)]
            v = np.nan_to_num(v, nan=0.0, posinf=0.0,
                              neginf=float(finite.min()) if finite.size else 0.0)
        vecs[ch] = v

    # --- scope: candidate pool + tail ordering (thesis O2) ---------------
    if spec.scope in ("clip_topk", "clip_threshold", "clip_threshold_cal"):
        prune = vecs["clip"]
    elif spec.scope == "dino_topk":
        prune = vecs["dino"]
    else:
        prune = None
    if prune is not None:
        order = np.argsort(-prune, kind="stable")
        tau = spec_tau(spec)
        if tau is not None:
            # OSCAR Sec. 3.2: keep everything above tau; top-k only when
            # the set comes back empty.  |S'| varies per query by design.
            n_keep = int((prune >= tau).sum())
            fell_back = n_keep == 0
            if fell_back:
                n_keep = min(CLIP_PRUNE_K, n)
            if diag is not None:
                diag["shortlist_size"] = n_keep
                diag["fallback"] = fell_back
        else:
            n_keep = CLIP_PRUNE_K
        pool, tail = order[:n_keep], order[n_keep:]
    else:
        pool, tail = np.arange(n), np.array([], dtype=int)

    # --- single-channel shortcut (E1a) ------------------------------------
    active = {ch for ch, w in zip(("clip", "dino", "shape"), spec.weights)
              if ch in vecs and w > 0}
    if spec.fusion_method == "weighted_sum" and len(active) == 1:
        ch = active.pop()
        ranked = pool[np.argsort(-vecs[ch][pool], kind="stable")]
        return list(ranked) + list(tail)

    # --- production fusion (step6_fusion.ScoreFusion) ---------------------
    sub = {ch: v for ch, v in vecs.items() if ch in active}
    clip_r, dino_r, shape_r = _synthetic_results(
        spec, sub, pool, object_ids, cad_dir)
    fused = fusion_mod.fuse(clip_r, dino_r, shape_r,
                            method=spec.fusion_method, top_k=n)
    idx = {oid: i for i, oid in enumerate(object_ids)}
    ranked = [idx[c.object_id] for c in fused.candidates]
    seen = set(ranked)
    ranked += [i for i in pool if i not in seen]     # fusion dropped none, but be safe
    return ranked + list(tail)


def spec_tau(spec: AblationSpec) -> Optional[float]:
    """The similarity threshold this spec prunes with, if any."""
    if spec.scope == "clip_threshold":
        return CLIP_TAU_TEXT
    if spec.scope == "clip_threshold_cal":
        return CLIP_TAU_CAL
    return None


def calibrate_tau(stores: Dict[str, dict], qids: Sequence[str]) -> float:
    """Pick tau from this dataset's own CLIP similarity distribution.

    Returns the ``CLIP_TAU_FALLBACK_TARGET`` percentile of the per-query
    maximum similarity, i.e. the highest threshold at which all but that
    fraction of queries still get a non-empty candidate set.

    Deliberately calibrated on *coverage*, not on set size or on any
    retrieval metric: matching a target |S'| would make the arm a restatement
    of top-k, and tuning on nDCG/NN would be tuning on the evaluation.
    """
    sims = []
    for qid in qids:
        v = stores["base"]["queries"][qid]["clip"]
        v = v[np.isfinite(v)]
        if v.size:
            sims.append(float(v.max()))
    if not sims:
        return CLIP_TAU_TEXT
    tau = float(np.percentile(sims, 100.0 * CLIP_TAU_FALLBACK_TARGET))
    arr = np.array(sims)
    print(f"[calibrate] per-query max CLIP similarity: "
          f"min={arr.min():.4f} p05={np.percentile(arr, 5):.4f} "
          f"median={np.median(arr):.4f} max={arr.max():.4f}")
    print(f"[calibrate] tau_cal = {tau:.4f} (target fallback "
          f"{100 * CLIP_TAU_FALLBACK_TARGET:.0f}%); paper tau = "
          f"{CLIP_TAU_TEXT} clears {100 * (arr >= CLIP_TAU_TEXT).mean():.1f}% "
          f"of queries")
    return tau


def make_fusion_module(spec: AblationSpec):
    """ScoreFusion configured with this spec's weights/method.

    Reuses pipeline/step6_fusion.py — weights follow thesis Step B1
    (w_text, w_view, w_shape) = (0.3, 0.4, 0.3) unless the ablation
    renormalizes after dropping a channel (E1b/O1a: (0.43, 0.57, 0)).
    """
    from pipeline.config import PipelineConfig
    from pipeline.step6_fusion import ScoreFusion
    cfg = PipelineConfig(
        weight_clip=spec.weights[0], weight_dino=spec.weights[1],
        weight_ulip=spec.weights[2], fusion_method=spec.fusion_method)
    return ScoreFusion(cfg)


# ===========================================================================
# 7. Geometry re-ranking (E2, O1c-e) — Sub-step B2 on the SHREC'18 pairs
# ===========================================================================
# SHREC'18 queries are metric SceneNN crops while ShapeNetSem CADs use
# arbitrary model units, so both clouds are normalized to the unit sphere
# before GeDi/Chamfer — scale is not a retrieval cue here (it is ablated
# separately by the scale gate).  voxel/correspondence radii are set for
# unit-sphere scale.  GeDi = RANSAC inlier count [FreeZe:
# caraffaFreeZeTrainingfreeZeroshot2025]; Chamfer = trimmed one-sided
# distance (trim 10%, U-RED-style [diUREDUnsupervised3D2023]).

def _cad_sample_seed(object_id: str) -> int:
    """Stable per-CAD seed for Open3D's global sampling RNG.

    Derived from the object id (not a counter) so the sampled cloud is a pure
    function of which CAD it is — independent of run order, shortlist depth,
    or how many other CADs were processed first.

    Masked into the non-negative int32 range: Open3D's seed() is bound to a
    32-bit signed int, and a full 8 hex digits overflows it for roughly half
    of all ids (observed: 2 of 4 CADs rejected with "incompatible function
    arguments" before the mask).
    """
    return int(hashlib.sha1(object_id.encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


GEOM_VOXEL = 0.02   # unit-sphere-scale voxel size for B2 (repo default
                    # 0.002 assumes metric tabletop scenes)
GEDI_RETRIES = 4    # retries when the GeDi service drops out mid-run
GEDI_WAIT_S = 300   # how long to wait for it to come back before aborting
QUERY_MAX_PTS = 500_000   # cap on query cloud size fed to GeDi/RANSAC
                          # (larger clouds crash the service; see _query_cloud)

# --- dGeDi geometry backend (cross-stage: the SAME service Stage 3 uses) -------
# STAGE1_GEOMETRY_BACKEND: "dgedi" (default — der dGeDi-HTTP-Dienst,
# evaluation/dgedi_bridge; ALLE publizierten Geometrie-Zahlen) oder
# "gedi" (das archivierte In-Process-UnitSphere-Legacy, nur historisch).
#
# SCALE: SHREC'18 queries are SceneNN-metric crops, CADs are ShapeNet arbitrary
# units — they share NO physical scale, so dGeDi's native per-candidate diameter
# co-scaling (built for metric datasets like BOP) is inapplicable. Instead we do
# exactly what the GeDi UnitSphereReRanker does: make the match SCALE-INVARIANT
# by normalizing BOTH sides to unit diameter — the gallery descriptors are
# self-normalized by prep_cloud, and here the query is divided by its own
# diameter. The SHREC gallery's diameters.json MUST be 1.0 so the server's
# per-candidate co-scale is a no-op (built that way in run_reruns.sh).
STAGE1_GEOMETRY_BACKEND = os.environ.get("STAGE1_GEOMETRY_BACKEND", "dgedi")
DGEDI_KP = int(os.environ.get("DGEDI_KP", "6000"))        # repo config
DGEDI_MAXIT = int(os.environ.get("DGEDI_MAXIT", "10000"))
DGEDI_USE_ICP = os.environ.get("DGEDI_USE_ICP", "1") != "0"


class _GeometryEngine:
    """Lazy wrapper around pipeline.step7_geometry_reranking (vormals step_b2) with
    per-(query, cad) score caching (append-only jsonl, resumable)."""

    def __init__(self, paths: dict):
        self.paths = paths
        # Per-cloud GeDi descriptors.  Separate from the per-pair score cache
        # below because the two have different cardinalities: 5,409 clouds vs
        # |queries| x K pairs.
        self.descriptor_cache_dir = os.path.join(
            paths["results_root"], "_cache", "gedi_descriptors")
        self.cache_path = os.path.join(paths["results_root"], "_cache",
                                       "geometry_scores.jsonl")
        self.cache: Dict[Tuple[str, str], dict] = {}
        if os.path.isfile(self.cache_path):
            with open(self.cache_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.cache[(r["qid"], r["cad"])] = r
                    except (json.JSONDecodeError, KeyError):
                        continue
        self._reranker = None
        self._gedi_available = None
        self._cad_clouds: "OrderedDict[str, object]" = OrderedDict()
        self._extents_path = os.path.join(paths["results_root"], "_cache",
                                          "cad_extents.json")
        self._extents: Dict[str, list] = {}
        if os.path.isfile(self._extents_path):
            with open(self._extents_path) as f:
                self._extents = json.load(f)

    # -- infrastructure ----------------------------------------------------
    def _get_reranker(self):
        if self._reranker is None:
            from pipeline.config import PipelineConfig
            from pipeline.step7_geometry_reranking import GeometryReRanker

            cad_dir = os.path.join(self.paths["data_root"], "cad")
            cloud_cache = self._cad_clouds

            class UnitSphereReRanker(GeometryReRanker):
                """B2 re-ranker with unit-sphere CAD normalization.

                Overrides the parent's metric-scale CAD loader: SHREC'18
                CADs (ShapeNetSem) have arbitrary units, so every cloud
                is centered and scaled to the unit sphere (same frame as
                the query cloud built in _query_cloud)."""

                def _load_cad_pointcloud(self, cad_path, n_points=10000):
                    import open3d as o3d
                    oid = os.path.splitext(os.path.basename(cad_path))[0]
                    if oid in cloud_cache:
                        cloud_cache.move_to_end(oid)
                        return cloud_cache[oid]
                    mesh = o3d.io.read_triangle_mesh(
                        os.path.join(cad_dir, oid + ".obj"))
                    if mesh.is_empty():
                        return None
                    # sample_points_uniformly() draws from Open3D's GLOBAL RNG
                    # and (as of 0.19) takes no seed argument, so two runs
                    # sample different points from the same mesh.  That made
                    # the geometry results irreproducible run-to-run and, once
                    # descriptor caching was added, made every CAD entry miss:
                    # a re-sampled cloud has a different fingerprint by
                    # construction.  Seed per object id rather than once
                    # globally, so a cloud is a pure function of its id and
                    # does not depend on how many CADs were sampled before it.
                    o3d.utility.random.seed(_cad_sample_seed(oid))
                    pcd = mesh.sample_points_uniformly(n_points)
                    pts = np.asarray(pcd.points)
                    pts -= pts.mean(axis=0)
                    r = np.linalg.norm(pts, axis=1).max()
                    if r > 0:
                        pts /= r
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(
                            radius=GEOM_VOXEL * 4, max_nn=30))
                    cloud_cache[oid] = pcd
                    while len(cloud_cache) > 400:     # ~50 MB ceiling
                        cloud_cache.popitem(last=False)
                    return pcd

            cfg = PipelineConfig(voxel_size=GEOM_VOXEL,
                                 geometry_reranking_top_k=GEOM_K,
                                 gedi_cache_dir=self.descriptor_cache_dir,
                                 geometry_skip_icp=SKIP_ICP)
            self._reranker = UnitSphereReRanker(cfg)
        return self._reranker

    def cad_cloud(self, cad_id: str):
        """The unit-sphere CAD cloud exactly as the re-ranker builds it.

        Shared with --precompute-gedi so the precomputed descriptors are
        fingerprint-identical to what the evaluation later asks for; building
        the cloud a second, subtly different way would make every entry miss.
        """
        return self._get_reranker()._load_cad_pointcloud(cad_id + ".obj")

    def gedi_available(self) -> bool:
        if self._gedi_available is None:
            if STAGE1_GEOMETRY_BACKEND == "dgedi":
                # dGeDi backend: the geometry ablations register against the
                # dGeDi service (port 5061), NOT the legacy GeDi service (5060).
                # Gating them on GeDi health skipped E2_both whenever only dGeDi
                # was up — the exact Stage-1 dGeDi rerun config — silently
                # falling back to mean-only (2026-08-25 gate FAIL). Probe dGeDi.
                self._gedi_available = self._dgedi_healthy()
                if not self._gedi_available:
                    print("[geometry] dGeDi service unreachable "
                          "(docker compose up -d dgedi) — geometry "
                          "ablations will be skipped.")
            else:
                # A SINGLE 5s probe is not enough to decide whether to skip four
                # ablations: on 2026-07-28 it timed out against a service that was
                # healthy throughout (the GeDi Flask server is single-threaded, so
                # /health blocks behind any in-flight descriptor call, and a
                # cold container's first DNS lookup adds to that).  The cost of a
                # false negative here is silently dropping E2_fitness /
                # chamfer_ransac / chamfer_icp / O1c from the grid, so retry.
                self._gedi_available = self._wait_for_gedi(GEDI_WAIT_S)
                if not self._gedi_available:
                    print("[geometry] GeDi service unreachable "
                          "(docker compose up -d gedi) — GeDi-signal "
                          "ablations will be skipped.")
        return self._gedi_available

    def _dgedi_healthy(self) -> bool:
        """LIVE probe of the dGeDi service (port 5061), for the dgedi backend —
        the analogue of _gedi_healthy for the GeDi service."""
        try:
            from dgedi_bridge import dgedi_health
            return bool(dgedi_health())
        except Exception:
            return False

    def _gedi_healthy(self) -> bool:
        """LIVE probe of the GeDi service (unlike the cached gedi_available).

        Needed to tell a genuine registration failure apart from the service
        having died mid-run.  The two are indistinguishable at the call site
        — both surface as ``registration_failed`` — but only the first may be
        written to the resumable cache.
        """
        try:
            import requests
            from pipeline.config import PipelineConfig
            url = getattr(PipelineConfig(), "gedi_url", "http://gedi:5060")
            # Generous timeout: the service is a single-threaded Flask dev
            # server, so /health queues behind any in-flight descriptor
            # computation (~5s each) rather than answering immediately.
            return requests.get(f"{url}/health", timeout=30).status_code == 200
        except Exception:
            return False

    def _wait_for_gedi(self, timeout_s: int) -> bool:
        """Block until the GeDi service answers again, or give up.

        Covers the compose auto-restart window (model reload ~60-90s).
        """
        import time
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._gedi_healthy():
                print("[geometry] GeDi is back — resuming.", flush=True)
                time.sleep(5)          # let the model finish warming up
                return True
            time.sleep(10)
        return False

    def _query_cloud(self, npz_path: str):
        import open3d as o3d
        data = np.load(npz_path)
        pts = data["points"].astype(np.float64)
        pts -= pts.mean(axis=0)
        r = np.linalg.norm(pts, axis=1).max()
        if r > 0:
            pts /= r
        # Cap only the pathological clouds.  SHREC'18 scans run from 540 to
        # 5.7M points (median 27.5k); 145 of the 2,101 exceed 1M, and handing
        # one of those to GeDi kills the service outright (silent `exit=0`,
        # observed 5x on query 0d2ff0fffe... / 4.45M points).
        #
        # Deliberately NOT voxel_down_sample(GEOM_VOXEL): that was tried on
        # 2026-07-28 and, while it did stop the crashes, it decimated every
        # query to 0.5-14k points and RANSAC then ran to its iteration limit
        # on the sparse correspondences — 40 min/query vs 26 s (~58 days for
        # the grid).  Dense queries are what makes registration converge fast.
        # 999k points is the largest cloud observed to work; 500k keeps a 2x
        # margin under that and 9x under the smallest crash.
        if len(pts) > QUERY_MAX_PTS:
            sel = np.random.default_rng(0).choice(      # deterministic
                len(pts), QUERY_MAX_PTS, replace=False)
            pts = pts[np.sort(sel)]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=GEOM_VOXEL * 4, max_nn=30))
        return pcd

    def _append_cache(self, rec: dict) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # -- scoring -----------------------------------------------------------
    def pair_scores(self, qid: str, npz_path: str, cad_ids: Sequence[str],
                    signal: str) -> Dict[str, dict]:
        """Geometry scores for (qid x cad_ids); cached pairs are free.

        Cache records are keyed per SIGNAL, because D_trim under different
        alignments is a different quantity: the unaligned diagnostic, the
        RANSAC-aligned distance and the ICP-refined distance must never be
        read from one another's entries.
        """
        if STAGE1_GEOMETRY_BACKEND == "dgedi":
            return self._pair_scores_dgedi(qid, npz_path, cad_ids)
        # Field names are signal-scoped; "fitness" is shared by every signal
        # that performs a registration (it is the same RANSAC either way).
        #
        # The three ALIGNED signals all require the same GeDi + RANSAC step,
        # which at ~35 s/query dominates everything else (the trimmed distance
        # on its own — the chamfer_unaligned ablation — is ~0.5 s/query, and
        # ICP is local refinement).  So they are computed together in one pass
        # (rerank(all_aligned=True)) and cached together: whichever aligned
        # ablation runs first pays the registration, the other two are free
        # cache hits.  Scoring them separately cost ~60 h instead of ~26 h
        # over the 2,101 official queries.
        _ALIGNED_FIELDS = ("fitness", "d_ransac", "d_icp")
        need_fields = {
            "fitness": _ALIGNED_FIELDS,
            "chamfer_unaligned": ("d_unaligned",),
            "chamfer_ransac": _ALIGNED_FIELDS,
            "chamfer_icp": _ALIGNED_FIELDS,
        }[signal]
        all_aligned = need_fields is _ALIGNED_FIELDS
        missing = [c for c in cad_ids
                   if any(self.cache.get((qid, c), {}).get(f) is None
                          for f in need_fields)]
        if missing:
            from pipeline.step6_fusion import FusedCandidate
            rr = self._get_reranker()
            obs = self._query_cloud(npz_path)
            cands = [FusedCandidate(object_id=c, fused_score=0.0,
                                    cad_model_path=c) for c in missing]
            rr.config.geometry_reranking_top_k = len(cands)
            # A dead GeDi service fails every fit, and those failures must NOT
            # be cached as if they were real (the B2 policy then ranks the
            # candidate last, permanently).  On 2026-07-27 this poisoned 2,845
            # pairs across 10h before anyone noticed.
            #
            # The service also restarts on its own (compose `restart:
            # unless-stopped`) and needs ~60-90s to reload the model, so a
            # blanket abort would end a 15h run over a 90s blip.  Wait for it
            # to come back and retry; abort only if it stays down.
            for attempt in range(1, GEDI_RETRIES + 1):
                res = rr.rerank(cands, obs, signal=signal,
                                all_aligned=all_aligned, query_id=qid)
                n_bad = sum(1 for gc in res.candidates
                            if gc.registration_failed)
                if (signal == "chamfer_unaligned" or not n_bad
                        or self._gedi_healthy()):
                    break                      # genuine result — cache it
                print(f"[geometry] GeDi unreachable ({n_bad}/"
                      f"{len(res.candidates)} fits failed for query {qid}) — "
                      f"waiting for it to come back "
                      f"(attempt {attempt}/{GEDI_RETRIES})", flush=True)
                if not self._wait_for_gedi(GEDI_WAIT_S):
                    raise SystemExit(
                        f"[geometry] GeDi service still down after "
                        f"{GEDI_WAIT_S}s — aborting rather than caching bogus "
                        f"failures.\n"
                        f"  Restart it:  docker compose up -d gedi\n"
                        f"  Then re-run; completed pairs are already cached.")
            else:
                raise SystemExit(
                    f"[geometry] GeDi kept failing across {GEDI_RETRIES} "
                    f"attempts on query {qid} — aborting rather than caching "
                    f"bogus failures.")
            dist_field = {"chamfer_unaligned": "d_unaligned",
                          "chamfer_ransac": "d_ransac",
                          "chamfer_icp": "d_icp"}.get(signal)

            def _num(x):
                return None if x is None or np.isinf(x) else float(x)

            for gc in res.candidates:
                rec = dict(self.cache.get((qid, gc.object_id),
                                          {"qid": qid, "cad": gc.object_id}))
                rec["failed"] = bool(gc.registration_failed)
                if "fitness" in need_fields:
                    rec["fitness"] = float(gc.ransac_fitness)
                if all_aligned:
                    # One registration, all three aligned readouts.
                    rec["d_ransac"] = _num(gc.d_ransac)
                    rec["d_icp"] = _num(gc.d_icp)
                    rec["icp_fitness"] = float(gc.icp_fitness)
                    rec["icp_rmse"] = float(gc.icp_inlier_rmse)
                elif dist_field:
                    rec[dist_field] = _num(gc.chamfer_score)
                self.cache[(qid, gc.object_id)] = rec
                self._append_cache(rec)
        return {c: self.cache.get((qid, c), {}) for c in cad_ids}

    # -- dGeDi backend (cross-stage geometry arm) --------------------------
    @staticmethod
    def _cloud_diameter(pts: np.ndarray) -> float:
        """Max pairwise distance = unit-diameter normalizer, matching the
        gallery's server._diameter. Computed over the convex hull so it stays
        O(H^2) rather than O(N^2) on large query clouds."""
        from scipy.spatial.distance import pdist
        h = pts
        if len(pts) > 64:
            try:
                from scipy.spatial import ConvexHull
                h = pts[ConvexHull(pts).vertices]
            except Exception:
                h = pts
        if len(h) > 4000:      # bound pdist memory if the hull fallback is large
            h = h[np.random.RandomState(0).choice(len(h), 4000, replace=False)]
        return float(pdist(h).max()) if len(h) > 1 else 1.0

    def _pair_scores_dgedi(self, qid: str, npz_path: str,
                           cad_ids: Sequence[str]) -> Dict[str, dict]:
        """Per-(query, cad) geometry signals via the dGeDi HTTP service — the
        exact backend Stage 3 uses (evaluation/dgedi_bridge). Produces the
        SAME cache fields (``fitness`` = RANSAC inlier fraction, ``d_ransac`` =
        trimmed one-sided Chamfer after RANSAC[->ICP]) that ``geometry_score``
        and the both_borda arm already consume, so no downstream change.

        Scale-invariant: the query is normalized to unit diameter to match the
        self-normalized gallery descriptors (server co-scale is a no-op because
        the SHREC diameters.json is 1.0). See STAGE1_GEOMETRY_BACKEND note."""
        need = ("fitness", "d_ransac")
        missing = [c for c in cad_ids
                   if any(self.cache.get((qid, c), {}).get(f) is None
                          for f in need)]
        if missing:
            eval_dir = os.path.join(_REPO, "evaluation")
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
            from dgedi_bridge import dgedi_rerank

            pts = np.load(npz_path)["points"].astype(np.float64)
            if len(pts) > QUERY_MAX_PTS:
                idx = np.random.RandomState(0).choice(
                    len(pts), QUERY_MAX_PTS, replace=False)
                pts = pts[idx]
            pts = pts - pts.mean(axis=0)
            diam = self._cloud_diameter(pts)
            q_unit = (pts / diam if diam > 0 else pts).astype(np.float32)

            geo = dgedi_rerank(q_unit, list(missing),
                               ransac_keypoints=DGEDI_KP,
                               ransac_max_iter=DGEDI_MAXIT,
                               use_icp=DGEDI_USE_ICP)
            if geo is None:
                # Service unreachable — a dead dGeDi fails EVERY candidate, and
                # caching those as real failures would poison the pair cache
                # permanently (mirrors the GeDi path's guard). Abort instead.
                raise SystemExit(
                    "[geometry] dGeDi service unreachable — aborting rather "
                    "than caching bogus failures.\n"
                    "  Bring it up on the SHREC gallery:  docker compose up -d dgedi\n"
                    "  Completed pairs are already cached; re-run to resume.")
            for cid in missing:
                g = geo.get(cid, {})
                ok = bool(g.get("ok"))
                rec = dict(self.cache.get((qid, cid),
                                          {"qid": qid, "cad": cid}))
                rec["failed"] = not ok
                rec["fitness"] = float(g.get("ransac_fitness", 0.0)) if ok else 0.0
                dr = g.get("d_ransac")
                dv = float(dr) if (ok and dr is not None) else None
                rec["d_ransac"] = dv
                rec["d_icp"] = dv          # dGeDi repo config already runs RANSAC->ICP
                self.cache[(qid, cid)] = rec
                self._append_cache(rec)
        return {c: self.cache.get((qid, c), {}) for c in cad_ids}

    def geometry_score(self, rec: dict, signal: str) -> Tuple[float, float]:
        """Sort key mirroring GeometryReRanker's ranking rule.

        Returns (primary, tiebreak) with higher = better: failed
        registrations sink to -inf, distance signals rank by -D_trim, and
        RANSAC fitness breaks ties (thesis Sec. 3.3 — no arbitrary scale
        factor combining an inlier statistic with a distance).
        """
        fitness = rec.get("fitness") or 0.0
        if rec.get("failed"):
            return (float("-inf"), fitness)
        if signal == "fitness":
            return (fitness, fitness)
        field = {"chamfer_unaligned": "d_unaligned",
                 "chamfer_ransac": "d_ransac",
                 "chamfer_icp": "d_icp"}[signal]
        d = rec.get(field)
        if d is None:
            return (float("-inf"), fitness)
        return (-float(d), fitness)

    # -- scale gate (legacy E2 variant, thesis "Scale gate (legacy)") ------
    def cad_extent(self, cad_id: str) -> Optional[np.ndarray]:
        if cad_id not in self._extents:
            try:
                import open3d as o3d
                mesh = o3d.io.read_triangle_mesh(os.path.join(
                    self.paths["data_root"], "cad", cad_id + ".obj"))
                ext = mesh.get_axis_aligned_bounding_box().get_extent()
                self._extents[cad_id] = [float(x) for x in ext]
            except Exception:
                self._extents[cad_id] = None
            with open(self._extents_path, "w") as f:
                json.dump(self._extents, f)
        e = self._extents.get(cad_id)
        return None if e is None else np.array(e)

    def scale_gate(self, npz_path: str, cad_ids: List[str]) -> List[str]:
        """Promote the first candidate whose sorted-bbox scale ratio lies
        in [scale_gate_min, scale_gate_max] (rotation-invariant estimate,
        cf. pipeline/config.py scale_gate_*).  reject_policy =
        fallback_best: order unchanged when nothing passes.  ShapeNetSem
        units are heterogeneous, so frequent rejection is an *expected*
        outcome of this ablation, not a failure."""
        q_ext = np.sort(np.ptp(np.load(npz_path)["points"], axis=0))[::-1]
        for i, cid in enumerate(cad_ids):
            c_ext = self.cad_extent(cid)
            if c_ext is None or (c_ext <= 0).any():
                continue
            scale = float(np.median(q_ext / np.sort(c_ext)[::-1]))
            if 0.8 <= scale <= 1.2:
                return [cid] + [c for j, c in enumerate(cad_ids) if j != i]
        return cad_ids


def precompute_gedi(paths: dict, index: List[dict], object_ids: List[str],
                    limit: Optional[int] = None) -> None:
    """Extract and cache GeDi descriptors for every gallery CAD and query.

    This is the change that makes the re-ranking depth a nearly free knob.
    Descriptors are a **per-cloud** cost (3.446 s measured) while the RANSAC
    fit that consumes them is **0.430 s per pair**, so without a cache the
    descriptor bill scales with |queries| x K and dominates everything:
    K = 20 over 2,101 queries is ~82 h uncached versus ~5 h on top of a
    ~5.2 h one-time pass here.

    Resumable and idempotent: entries validate against a fingerprint of the
    input cloud, so an interrupted run is simply restarted, and a
    preprocessing change invalidates the affected entries instead of silently
    serving descriptors computed for a different cloud.
    """
    import time

    geom = _GeometryEngine(paths)
    if not geom.gedi_available():
        raise SystemExit("[precompute-gedi] GeDi service unavailable — start "
                         "it with `docker compose up -d gedi` and retry.")
    gedi_mod = geom._get_reranker()._get_gedi()
    cache_dir = geom.descriptor_cache_dir

    cads = object_ids[:limit] if limit else object_ids
    queries = index[:limit] if limit else index
    total = len(cads) + len(queries)
    print(f"[precompute-gedi] {len(cads)} CADs + {len(queries)} queries "
          f"= {total} clouds -> {cache_dir}")

    done = failed = 0
    t0 = time.perf_counter()
    for kind, items in (("cad", cads), ("query", queries)):
        for n, item in enumerate(items, 1):
            key = item if kind == "cad" else item["id"]
            path = os.path.join(cache_dir, kind, f"{key}.npz")
            try:
                pcd = (geom.cad_cloud(key) if kind == "cad"
                       else geom._query_cloud(item["npz"]))
                if pcd is None:
                    failed += 1
                    continue
                res = gedi_mod.compute_and_cache(
                    pcd, path, provenance={"kind": kind, "key": key,
                                           "voxel_size": GEOM_VOXEL,
                                           "normalization": "unit_sphere"})
                if res.descriptors_np.size == 0:
                    failed += 1
                else:
                    done += 1
            except Exception as exc:      # noqa: BLE001 — report and continue
                print(f"[precompute-gedi] {kind}/{key}: {exc}")
                failed += 1
            if n % 100 == 0:
                el = time.perf_counter() - t0
                print(f"[precompute-gedi] {kind} {n}/{len(items)}  "
                      f"({el / 60:.1f} min elapsed, "
                      f"{el / max(done, 1):.2f} s/cloud)", flush=True)

    el = time.perf_counter() - t0
    print(f"[precompute-gedi] {done} cached, {failed} failed, "
          f"{el / 3600:.2f} h total ({el / max(done, 1):.2f} s/cloud).")
    if failed:
        print(f"[precompute-gedi] {failed} clouds have no descriptors; the "
              f"evaluation will fall back to computing those on the fly. "
              f"Re-run to retry them — cached clouds are skipped.")


def bench_rerank(paths: dict, index: List[dict], object_ids: List[str],
                 n_pairs: int) -> None:
    """Measure real re-ranking throughput and turn K into a budget decision.

    ``--bench-gedi`` answers a different question (is full-database S_GeDi
    feasible? no).  This one answers the question that actually has to be
    decided: at the production settings, on THIS machine, how long does the
    grid take at each shortlist depth?

    Modelling this from component timings has been wrong twice — the RANSAC
    fit is only ~0.43 s but the surrounding work (descriptor cache load,
    voxel downsampling, two trimmed-distance evaluations, optional ICP)
    dominates it.  So measure the whole per-pair path exactly as the grid
    runs it, with the descriptor cache warm, and extrapolate from that.
    """
    import time

    geom = _GeometryEngine(paths)
    if not geom.gedi_available():
        raise SystemExit("[bench] GeDi service unavailable — start it with "
                         "`docker compose up -d gedi` and retry.")

    # Spread the sample over SEVERAL queries.  Per-pair cost is driven by the
    # query cloud (the trimmed distance and RANSAC both scale with it), and
    # SHREC'18 scans run from 540 to 5.7M points — timing a single query would
    # extrapolate that one scan's size to the whole grid.
    rng = np.random.RandomState(42)
    n_q = int(min(4, max(1, n_pairs // 3), len(index)))
    per_q = max(1, n_pairs // n_q)
    queries = [index[i] for i in rng.choice(len(index), n_q, replace=False)]
    cads = [object_ids[i] for i in rng.choice(
        len(object_ids), min(n_q * per_q, len(object_ids)), replace=False)]

    # Warm the descriptor cache first so the measurement reflects steady
    # state, not the one-time extraction that --precompute-gedi covers.
    print(f"[bench] warming descriptor cache: {len(cads)} CADs + {n_q} "
          f"queries (this is the --precompute-gedi cost, not per-pair cost)")
    gedi_mod = geom._get_reranker()._get_gedi()
    for cad in cads:
        pcd = geom.cad_cloud(cad)
        if pcd is not None:
            gedi_mod.compute_and_cache(
                pcd, os.path.join(geom.descriptor_cache_dir, "cad",
                                  f"{cad}.npz"))
    for q in queries:
        gedi_mod.compute_and_cache(
            geom._query_cloud(q["npz"]),
            os.path.join(geom.descriptor_cache_dir, "query", f"{q['id']}.npz"))

    print(f"[bench] timing {n_q} x {per_q} registrations "
          f"(ICP {'OFF' if SKIP_ICP else 'ON'}) ...")
    per_query_rates, n_done = [], 0
    t0 = time.perf_counter()
    for qi, q in enumerate(queries):
        chunk = cads[qi * per_q:(qi + 1) * per_q]
        if not chunk:
            continue
        tq = time.perf_counter()
        for cad in chunk:
            geom.cache.pop((q["id"], cad), None)   # bypass pair-score cache
            geom.pair_scores(q["id"], q["npz"], [cad], "chamfer_ransac")
        dt = (time.perf_counter() - tq) / len(chunk)
        n_pts = len(np.load(q["npz"])["points"])
        per_query_rates.append((q["id"], n_pts, dt))
        n_done += len(chunk)
    elapsed = time.perf_counter() - t0

    per_pair = elapsed / max(n_done, 1)
    print(f"\n[bench] per-query breakdown (cost tracks cloud size):")
    print(f"{'query':<14} {'points':>9}  {'s/pair':>7}")
    for qid, n_pts, dt in sorted(per_query_rates, key=lambda r: r[1]):
        print(f"{qid[:12]:<14} {n_pts:>9,}  {dt:>7.2f}")
    if len(per_query_rates) > 1:
        rates = [r[2] for r in per_query_rates]
        print(f"[bench] spread across queries: {min(rates):.2f}-"
              f"{max(rates):.2f} s/pair — the projection below uses the mean, "
              f"so treat it as +/-{100 * (max(rates) - min(rates)) / (2 * per_pair):.0f}%.")
    cads = cads[:n_done]
    nq = len(index)
    print(f"\n[bench] {n_done} pairs in {elapsed:.1f}s -> "
          f"{per_pair:.2f} s/pair  (ICP {'OFF' if SKIP_ICP else 'ON'})")
    print(f"[bench] projected wall-clock over {nq:,} queries:\n")
    print(f"{'K':>6}  {'registrations':>14}  {'hours':>8}  {'days':>6}")
    for k in HITRATE_KS:
        pairs = nq * k
        h = pairs * per_pair / 3600.0
        print(f"{k:>6}  {pairs:>14,}  {h:>8.1f}  {h / 24:>6.1f}")
    curve = load_hitrate_curve(paths["results_root"], "sub")
    if curve:
        print(f"\n{'K':>6}  {'hit_sub@K':>10}  {'hours':>8}  "
              f"{'pts/hour':>9}   (marginal return)")
        prev_k = prev_v = prev_h = None
        for k in sorted(curve):
            h = nq * k * per_pair / 3600.0
            marg = ("" if prev_k is None or h <= prev_h
                    else f"{100 * (curve[k] - prev_v) / (h - prev_h):9.2f}")
            print(f"{k:>6}  {curve[k]:>10.4f}  {h:>8.1f}  {marg:>9}")
            prev_k, prev_v, prev_h = k, curve[k], h
        print("\n[bench] pick K where pts/hour stops justifying the wall "
              "clock; the ceiling column is what re-ranking can reach at all.")
    else:
        print(f"\n[bench] run {BASE_ABLATION} first to get the hit_sub@K "
              f"ceiling alongside these costs.")
    if not SKIP_ICP:
        print("[bench] re-run with --no-icp to see the cost without ICP "
              "(~38% cheaper at K=5; ICP moved nDCG by 0.0001 there).")


def bench_gedi(paths: dict, index: List[dict], object_ids: List[str],
               n_pairs: int) -> None:
    """Measure the per-pair GeDi+RANSAC cost and extrapolate to full-database.

    Backs the thesis claim that using S_GeDi *inside* the fusion score (rather
    than as a post-fusion re-ranker) is computationally infeasible: it would
    need one registration per (query, gallery entry) pair.  Descriptor
    extraction on the gallery side is amortisable; the RANSAC fit is not, so
    that is what is timed here.
    """
    import time

    geom = _GeometryEngine(paths)
    if not geom.gedi_available():
        raise SystemExit("[bench] GeDi service unavailable — start it with "
                         "`docker compose up -d gedi` and retry.")

    rng = np.random.RandomState(42)
    q = index[int(rng.randint(len(index)))]
    cads = [object_ids[i] for i in rng.choice(len(object_ids),
                                              min(n_pairs, len(object_ids)),
                                              replace=False)]
    print(f"[bench] timing {len(cads)} GeDi+RANSAC fits "
          f"(query {q['id']}) ...")

    t0 = time.perf_counter()
    for cad in cads:
        # One pair at a time, bypassing the cache, so the timing reflects a
        # single registration rather than a batched shortlist.
        geom.cache.pop((q["id"], cad), None)
        geom.pair_scores(q["id"], q["npz"], [cad], "chamfer_ransac")
    elapsed = time.perf_counter() - t0

    per_fit = elapsed / max(len(cads), 1)
    total_s = per_fit * GEDI_FULL_DB_PAIRS
    print(f"\n[bench] {len(cads)} fits in {elapsed:.1f}s "
          f"-> {per_fit * 1000:.0f} ms per fit")
    print(f"[bench] full-database S_GeDi = {N_QUERIES_TOTAL:,} queries x "
          f"{N_CADS} CADs = {GEDI_FULL_DB_PAIRS:,} fits")
    print(f"[bench] extrapolated: {total_s / 3600:.0f} h "
          f"({total_s / 86400:.1f} days) of RANSAC alone, per ablation cell.")
    print("[bench] -> S_GeDi is reported as a post-fusion re-ranker only "
          "(thesis O1c/O1d).")


def _average_ranks(scores: Sequence[float]) -> np.ndarray:
    """Ranks with 1 = best; tied scores share their mean rank.

    Averaging ties matters here because ``-inf`` is the failure sentinel: a
    query whose whole shortlist failed to register would otherwise get an
    arbitrary stable-sort order presented as a ranking.
    """
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


# Combined geometry signals -> the registration signal they read from.  Both
# use the aligned pass, so they are pure cache hits once any of E2_fitness /
# E2_chamfer_ransac / E2_chamfer_icp has run on the same shortlist.
COMBINED_SIGNALS = {"both_borda": "chamfer_ransac",
                    "both_borda_base": "chamfer_ransac"}


def apply_geometry(spec: AblationSpec, qid: str, npz_path: str,
                   ranking: List[int], object_ids: List[str],
                   geom: _GeometryEngine) -> List[int]:
    """Apply the spec's geometry stage to a derived base ranking.

    Only the first :data:`GEOM_K` entries are touched; the tail keeps its
    base-fusion order and the full ranking is always returned.  Truncating to
    K here would make the official ``precision(x)`` divide by K and report a
    meaningless inflated number (STAGE1_EVALUATION_DESIGN §3).
    """
    idx = {oid: i for i, oid in enumerate(object_ids)}

    if spec.geometry == "scale_gate":
        top = [object_ids[i] for i in ranking[:GEOM_K]]
        new = geom.scale_gate(npz_path, top)
        return [idx[o] for o in new] + ranking[GEOM_K:]

    # Sub-step B2: re-order the fusion top-k shortlist.
    top = [object_ids[i] for i in ranking[:GEOM_K]]
    signal = COMBINED_SIGNALS.get(spec.geometry, spec.geometry)
    recs = geom.pair_scores(qid, npz_path, top, signal)

    if spec.geometry in COMBINED_SIGNALS:
        # Mean rank (Borda count) over the two geometry signals — and, for
        # O1e, over the base fusion rank as a third voter.
        #
        # A raw sum is NOT an option: RANSAC fitness spans ~0.23 while D_trim
        # spans ~0.09, so fitness + (-D_trim) reproduces the fitness-only
        # ranking exactly and the "fusion" fuses nothing.  Borda is scale-free
        # and parameter-free [Aslam & Montague, SIGIR 2001].  RRF was rejected
        # for this: its k=60 was calibrated on TREC lists of thousands, and at
        # K=20 its weights span only 1/61..1/80 (1.3x), so it degenerates into
        # exactly this mean rank while implying a tuned constant.
        fit = [geom.geometry_score(recs[o], "fitness")[0] for o in top]
        dist = [geom.geometry_score(recs[o], "chamfer_ransac")[0] for o in top]
        votes = [_average_ranks(fit), _average_ranks(dist)]
        if spec.geometry == "both_borda_base":
            # Candidates arrive in base-fusion order, so position IS the base
            # rank; negate so that higher = better like the other two.
            votes.append(_average_ranks([-i for i in range(len(top))]))
        mean_rank = np.mean(votes, axis=0)
        scored = [top[i] for i in np.argsort(mean_rank, kind="stable")]
    else:
        scored = sorted(top, key=lambda o: geom.geometry_score(
            recs[o], spec.geometry), reverse=True)
    return [idx[o] for o in scored] + ranking[GEOM_K:]


# ===========================================================================
# 8. Per-ablation evaluation loop
# ===========================================================================

def _simplex_grid(step: float) -> List[Tuple[float, float, float]]:
    """All (w_text, w_view, w_shape) on the probability simplex at ``step``
    (weights sum to 1). Includes the corners (single-channel) and edges
    (two-channel), so the sweep also re-derives the E1 baselines."""
    n = int(round(1.0 / step))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((round(i * step, 4), round(j * step, 4),
                        round(k * step, 4)))
    return pts


def run_rrf_c_sweep(paths: dict, index: List[dict], object_ids: List[str],
                    cad_labels: Dict[str, tuple], freqs: Dict[str, int],
                    stores: Dict[str, dict], cs: List[int],
                    limit: Optional[int]) -> List[dict]:
    """RRF SENSITIVITY over the Cormack factor c (E6 robustness check).

    Recomputes the E6_rrf arm on the cached BASE score vectors for several
    c — byte-identical pipeline (derive_ranking + score_official), only
    k_param of the reciprocal rank fusion varies — and compares each c
    per-query paired against the weighted sum (E1c_full_fusion from the
    results folder). Reported as a sensitivity, NOT a selection procedure:
    c=60 [cormackReciprocalRankFusion2009] stays the frozen reported
    value; the sweep only shows whether "weighted sum > RRF" depends on
    c. Writes rrf_c_sweep.csv into the results folder."""
    import pipeline.step6_fusion as _s6
    cad_dir = os.path.join(paths["data_root"], "cad")
    qlist = index[:limit] if limit else index
    ref_p = os.path.join(arm_dir(paths["results_root"], "E1c_full_fusion"),
                         "results_per_query.json")
    with open(ref_p) as fh:
        ref = {r["id"]: r["nDCG"] for r in json.load(fh)}
    print(f"[rrf-sweep] weighted-sum reference: {len(ref)} queries from "
          f"{ref_p}")
    orig = _s6.ScoreFusion._reciprocal_rank_fusion
    rows = []
    try:
        for c in cs:
            def _rrf(self, clip_result, dino_result, shape_result,
                     top_k, _c=c):
                return orig(self, clip_result, dino_result, shape_result,
                            top_k, k_param=_c)
            _s6.ScoreFusion._reciprocal_rank_fusion = _rrf
            spec = AblationSpec(name=f"E6_rrf_c{c}", group="E6SWEEP",
                                question="RRF c sensitivity",
                                channels=dict(_BASE_CH),
                                fusion_method="rank_fusion")
            fusion_mod = make_fusion_module(spec)
            sums = defaultdict(float)
            nq = wins = losses = ties = 0
            dsum = 0.0
            for q in qlist:
                q_label = tuple(q["category"])
                if freqs.get(q_label[0], 0) == 0:
                    continue
                ranking = derive_ranking(spec, q["id"], stores, object_ids,
                                         fusion_mod, cad_dir, None)
                m = score_official([object_ids[i] for i in ranking], q_label,
                                   cad_labels, freqs)
                if m is None:
                    continue
                for k, v in m.items():
                    sums[k] += v
                nq += 1
                r = ref.get(q["id"])
                if r is not None:
                    d = r - m["nDCG"]        # > 0: gewichtete Summe vorn
                    dsum += d
                    if d > 1e-9:
                        wins += 1
                    elif d < -1e-9:
                        losses += 1
                    else:
                        ties += 1
            row = {"c": c, "n": nq,
                   "nDCG_rrf": round(sums["nDCG"] / nq, 4) if nq else None,
                   "mAP_rrf": round(sums["AP"] / nq, 4) if nq else None,
                   "delta_weighted_minus_rrf": round(dsum / nq, 4) if nq else None,
                   "wins_weighted": wins, "wins_rrf": losses, "ties": ties}
            rows.append(row)
            print(f"[rrf-sweep] c={c:>4}  nDCG={row['nDCG_rrf']:.4f}  "
                  f"Δ(weighted−rrf)={row['delta_weighted_minus_rrf']:+.4f}  "
                  f"record {wins}:{losses} ({ties} ties)")
    finally:
        _s6.ScoreFusion._reciprocal_rank_fusion = orig
    import csv as _csv
    out = os.path.join(paths["results_root"], "rrf_c_sweep.csv")
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[rrf-sweep] wrote -> {out}")
    return rows


def run_weight_sweep(paths: dict, index: List[dict], object_ids: List[str],
                     cad_labels: Dict[str, tuple], freqs: Dict[str, int],
                     stores: Dict[str, dict], step: float,
                     limit: Optional[int],
                     shape_pass: str = "ulip_pc_rgb") -> List[dict]:
    """Fusion-weight SENSITIVITY sweep on the cached BASE score vectors.

    Reuses derive_ranking + score_official, so the fusion and official scoring
    are byte-identical to the grid — the only thing that varies is (w_text,
    w_view, w_shape). Each point drops its zero-weight channels from
    ``channels`` so a corner maps exactly onto E1a/E1_view_only/E1_shape_only
    and an edge onto E1b, i.e. the sweep reproduces those arms at its extremes.

    Reported as a sensitivity surface (span of nDCG across the plausible region
    around BASE). NOT a selection procedure: choosing the argmax here and then
    reporting SHREC with it would be tuning on the SHREC test set. BASE
    (0.3/0.4/0.3) stays the frozen choice; the sweep only shows how flat/peaked
    the surface around it is."""
    cad_dir = os.path.join(paths["data_root"], "cad")
    qlist = index[:limit] if limit else index
    full = {"clip": ("base", None), "dino": ("base", 42),
            "shape": (shape_pass, None)}
    mode = "cross-mode (query image)" if shape_pass == "ulip_cross_rgb" else "pc-mode (query point cloud)"
    grid = _simplex_grid(step)
    print(f"[wsweep] S_shape pass = {shape_pass} — {mode}")
    print(f"[wsweep] {len(grid)} weight points (step={step}) x {len(qlist)} queries")
    rows = []
    for (wt, wv, ws) in grid:
        chan = {ch: full[ch] for ch, w in
                zip(("clip", "dino", "shape"), (wt, wv, ws)) if w > 0}
        spec = AblationSpec(name=f"W_{wt:.2f}_{wv:.2f}_{ws:.2f}", group="WSWEEP",
                            question="fusion-weight sensitivity",
                            channels=chan, weights=(wt, wv, ws))
        fusion_mod = make_fusion_module(spec)
        sums = defaultdict(float)
        nq = 0
        for q in qlist:
            q_label = tuple(q["category"])
            if freqs.get(q_label[0], 0) == 0:
                continue
            ranking = derive_ranking(spec, q["id"], stores, object_ids,
                                     fusion_mod, cad_dir, None)
            m = score_official([object_ids[i] for i in ranking], q_label,
                               cad_labels, freqs)
            if m is None:
                continue
            for k, v in m.items():
                sums[k] += v
            nq += 1
        row = {"w_text": wt, "w_view": wv, "w_shape": ws, "n": nq,
               "nDCG": round(sums["nDCG"] / nq, 4) if nq else float("nan"),
               "mAP": round(sums["AP"] / nq, 4) if nq else float("nan")}
        rows.append(row)
        print(f"[wsweep] w=({wt:.2f},{wv:.2f},{ws:.2f})  "
              f"nDCG={row['nDCG']:.4f}  mAP={row['mAP']:.4f}")

    import csv
    os.makedirs(paths["results_root"], exist_ok=True)
    out = os.path.join(paths["results_root"], "weight_sweep.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["w_text", "w_view", "w_shape",
                                          "n", "nDCG", "mAP"])
        w.writeheader()
        w.writerows(rows)

    nd = [r["nDCG"] for r in rows if r["nDCG"] == r["nDCG"]]  # drop nan
    best = max(rows, key=lambda r: (r["nDCG"] if r["nDCG"] == r["nDCG"] else -1))
    base = next((r for r in rows if (r["w_text"], r["w_view"], r["w_shape"])
                 == (0.3, 0.4, 0.3)), None)
    print(f"\n[wsweep] nDCG over {len(nd)} points: "
          f"[{min(nd):.4f}, {max(nd):.4f}]  span={max(nd) - min(nd):.4f}")
    if base:
        gap = best["nDCG"] - base["nDCG"]
        print(f"[wsweep] BASE (0.3,0.4,0.3): nDCG={base['nDCG']:.4f}")
        print(f"[wsweep] argmax: w=({best['w_text']},{best['w_view']},"
              f"{best['w_shape']}) nDCG={best['nDCG']:.4f}  "
              f"(argmax-BASE = {gap:+.4f} — SENSITIVITY only, not a selection)")
    print(f"[wsweep] -> {out}")
    return rows


def run_ablation(spec: AblationSpec, paths: dict, index: List[dict],
                 stores: Dict[str, dict], object_ids: List[str],
                 cad_labels: Dict[str, tuple], freqs: Dict[str, int],
                 limit: Optional[int],
                 geom: Optional[_GeometryEngine]) -> dict:
    out_dir = os.path.join(paths["results_root"], spec.name)
    os.makedirs(out_dir, exist_ok=True)

    fusion_mod = make_fusion_module(spec)
    cad_dir = os.path.join(paths["data_root"], "cad")
    # Category of every scored gallery model (for the per-query record /
    # first-relevant rank); official scoring itself uses cad_labels+freqs.
    obj_cats = np.array([cad_labels.get(o, (None, None))[0]
                         for o in object_ids], dtype=object)

    qlist = index[:limit] if limit else index
    sums = defaultdict(float)          # Table A — official scorer
    sums_b = defaultdict(float)        # Table B — depth-matched
    shortlist_sizes: List[int] = []    # |S'| for the threshold scope
    n_fallback = 0
    per_query = []
    if SHREC_QUERY_SUBSET:
        _keep = set(json.load(open(SHREC_QUERY_SUBSET)))
        qlist = [q for q in qlist if q["id"] in _keep]
        print(f"[subset] restricted to {len(qlist)} queries "
              f"<- {os.path.basename(SHREC_QUERY_SUBSET)}")
    for q in qlist:
        qid = q["id"]
        q_label = tuple(q["category"])          # (category, subcategory)
        qc = q_label[0]
        if freqs.get(qc, 0) == 0:
            continue        # query category unknown to the official GT
        diag: dict = {}
        ranking = derive_ranking(spec, qid, stores, object_ids,
                                 fusion_mod, cad_dir, diag)
        if spec.geometry:
            ranking = apply_geometry(spec, qid, q["npz"], ranking,
                                     object_ids, geom)
        ranked_ids = [object_ids[i] for i in ranking]
        m = score_official(ranked_ids, q_label, cad_labels, freqs)
        if m is None:
            continue
        for k, v in m.items():
            sums[k] += v
        # Table B: cut at the geometry depth, where re-ranking is visible.
        mb = score_depth_matched(ranked_ids, q_label, cad_labels, GEOM_K)
        for k, v in mb.items():
            sums_b[k] += v
        if "shortlist_size" in diag:
            shortlist_sizes.append(diag["shortlist_size"])
            n_fallback += int(diag["fallback"])
        rels_cat = obj_cats[np.asarray(ranking)] == qc
        first_rel = int(np.argmax(rels_cat)) + 1 if rels_cat.any() else -1
        rec = {
            "id": qid, "category": list(q_label),
            "top10": [[object_ids[i],
                       list(cad_labels.get(object_ids[i], (None, None)))]
                      for i in ranking[:10]],
            "first_relevant_rank": first_rel,
            "AP": round(m["AP"], 4), "nDCG": round(m["nDCG"], 4),
            # Kept per query so the paired statistics (deltas vs BASE,
            # N_changed, bootstrap CI) can be computed post hoc without
            # re-running anything — geometry arms differ from BASE only by a
            # K-element permutation, so unpaired means would be swamped by the
            # unchanged majority.
            "NN_cat": mb.get("NN_cat"), "NN_sub": mb.get("NN_sub"),
            "MRR": round(mb.get("MRR", 0.0), 4),
            "nDCG_K": round(mb.get("nDCG_K", 0.0), 4),
        }
        if diag:
            rec["shortlist_size"] = diag.get("shortlist_size")
            rec["fallback"] = diag.get("fallback")
        per_query.append(rec)

    nq = len(per_query)
    metrics = {k: (sums[k] / nq if nq else float("nan"))
               for k in ("nDCG", "precision", "recall", "F1", "AP",
                         "NNT1", "NNT2")}
    # Table B. Reported for EVERY arm, not just the geometry ones: it is
    # post-hoc arithmetic over the same ranking, so it costs nothing, and it
    # lets a geometry arm be compared against any baseline on equal footing.
    metrics_depth = {k: (sums_b[k] / nq if nq else float("nan"))
                     for k in sorted(sums_b)}
    summary = {
        "ablation": spec.name, "group": spec.group,
        "question": spec.question, "thesis_ref": spec.thesis_ref,
        "bib": spec.bib, "notes": spec.notes,
        "num_queries": nq, "gallery_size": len(object_ids),
        "metrics": metrics,
        "metrics_depth": metrics_depth,
        "config": {
            "channels": {c: list(v) for c, v in spec.channels.items()},
            "weights": list(spec.weights),
            "fusion_method": spec.fusion_method,
            "scope": spec.scope, "geometry": spec.geometry,
            "clip_prune_k": CLIP_PRUNE_K,
            "clip_tau": spec_tau(spec),
            "geometry_k": GEOM_K,
            "skip_icp": SKIP_ICP,
            "geom_voxel": GEOM_VOXEL,
            # Derived, not chosen: step_b2 uses 1.5 x voxel_size (Open3D
            # convention) and the clouds are unit-sphere, so diameter = 2.
            "inlier_threshold_pct_diameter": 100.0 * GEOM_VOXEL * 1.5 / 2.0,
        },
        "limit_queries": limit,
    }
    if shortlist_sizes:
        # The faithful (variable-length) reading of the OSCAR threshold arm.
        # These are NOT a table row — precision() divides by the submitted
        # list length, so a short list is not the same quantity as the padded
        # rows. They are reported as a characterising paragraph instead.
        s = np.array(shortlist_sizes, dtype=float)
        summary["shortlist_stats"] = {
            "tau_text": spec_tau(spec),
            "median": float(np.median(s)),
            "q1": float(np.percentile(s, 25)),
            "q3": float(np.percentile(s, 75)),
            "min": int(s.min()), "max": int(s.max()),
            "fallback_rate": n_fallback / len(s),
        }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "results_per_query.json"), "w") as f:
        json.dump(per_query, f)
    print(f"[run:{spec.name}] n={nq}  nDCG={metrics['nDCG']:.4f}  "
          f"P={metrics['precision']:.4f}  mAP={metrics['AP']:.4f}")
    print(f"[run:{spec.name}] K={GEOM_K}  "
          f"NN_sub={metrics_depth.get('NN_sub', float('nan')):.4f}  "
          f"NN_cat={metrics_depth.get('NN_cat', float('nan')):.4f}  "
          f"MRR={metrics_depth.get('MRR', float('nan')):.4f}  "
          f"nDCG@K={metrics_depth.get('nDCG_K', float('nan')):.4f}")
    if "shortlist_stats" in summary:
        st = summary["shortlist_stats"]
        print(f"[run:{spec.name}] |S'| median={st['median']:.0f} "
              f"IQR=[{st['q1']:.0f},{st['q3']:.0f}] "
              f"range=[{st['min']},{st['max']}] "
              f"fallback={100 * st['fallback_rate']:.1f}%")
    return summary


def _resolve_alias_root(name: str) -> str:
    """Follow an alias chain to the cell that is actually computed.

    Aliases may chain (O5_xyzrgb -> E7_ulip2_pc -> E1c_full_fusion once BASE
    is pc-mode).  Resolving to the root means a cell can be materialized even
    when the intermediate alias is not part of the selected run.
    """
    seen = set()
    while name in ABLATIONS and ABLATIONS[name].alias_of:
        if name in seen:            # defensive: never loop on a cyclic chain
            break
        seen.add(name)
        name = ABLATIONS[name].alias_of
    return name


def write_alias(spec: AblationSpec, paths: dict) -> Optional[dict]:
    """Materialize an alias cell by copying its canonical metrics."""
    root = _resolve_alias_root(spec.alias_of)
    src = os.path.join(paths["results_root"], root,
                       "metrics_summary.json")
    if not os.path.isfile(src):
        return None
    with open(src) as f:
        summary = json.load(f)
    summary.update({"ablation": spec.name, "group": spec.group,
                    "question": spec.question, "alias_of": spec.alias_of})
    out_dir = os.path.join(paths["results_root"], spec.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ===========================================================================
# 9. Aggregation: CSV + LaTeX + best-config selection
# ===========================================================================

# Official SHREC'18 metric set (metrics.py, all at top-f).  nDCG is the
# graded model-selection metric; AP averaged over queries is the mAP.
# TABLE A — official SHREC'18 scorer, cut at f = category size.
METRIC_COLS = ("nDCG", "precision", "recall", "F1", "AP", "NNT1", "NNT2")
# TABLE B — depth-matched, cut at GEOM_K.  Reported separately and never in
# the same table as METRIC_COLS: precision/recall/F1/NNT1/NNT2 above are
# invariant to any re-ranking at K < f, and the two nDCGs are different
# quantities (Table A's is the official off-by-one one over the f-prefix,
# Table B's is corrected and cut at K).  See STAGE1_EVALUATION_DESIGN §2.
DEPTH_COLS = ("NN_cat", "NN_sub", "MRR", "MRR_sub", "mAP_K", "nDCG_K")


def aggregate(paths: dict) -> None:
    """Collect every metrics_summary.json into the Stage-1 deliverables:

      * stage1_summary.csv — machine-readable grid
      * stage1_summary.tex — booktabs rows for tab:eval_stage1_ablation_grid
                             / data for fig:eval_stage1_ablation_bars
      * best_config.json   — argmax DCG (tie-break mAP,
                             subsec:eval_stage1_retrieval), frozen for
                             Stages 2/3a/3b/5.
    """
    rows = []
    for name, spec in ABLATIONS.items():
        p = os.path.join(arm_dir(paths["results_root"], name),
                         "metrics_summary.json")
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            s = json.load(f)
        rows.append((spec, s))
    if not rows:
        print("[aggregate] no results yet.")
        return

    csv_path = os.path.join(paths["results_root"], "stage1_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ablation", "group", "alias_of", "num_queries",
                    *METRIC_COLS])
        for spec, s in rows:
            w.writerow([spec.name, spec.group, s.get("alias_of", ""),
                        s["num_queries"],
                        *[f"{s['metrics'][m]:.4f}" for m in METRIC_COLS]])
    print(f"[aggregate] {len(rows)} configs -> {csv_path}")

    # --- best config: computed (non-alias) cells only ---------------------
    real = [(spec, s) for spec, s in rows if not s.get("alias_of")]
    if not real:
        print("[aggregate] only alias entries present — no best config.")
        return
    best_spec, best_sum = max(
        real, key=lambda t: (t[1]["metrics"]["nDCG"], t[1]["metrics"]["AP"]))
    best = {
        "selection_rule": ("highest (graded) nDCG on SHREC'18 ObjectNN+ "
                           "under the official evaluation, mAP as "
                           "tie-breaker (thesis subsec:eval_stage1_retrieval)"),
        "name": best_spec.name,
        "metrics": best_sum["metrics"],
        "config": best_sum["config"],
        "frozen_for": ["Stage 2 (MI3DOR)", "Stage 3a/3b (BOP pose)",
                       "Stage 5 (efficiency)"],
    }
    best_path = os.path.join(paths["results_root"], "best_config.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"[aggregate] best: {best_spec.name} "
          f"(nDCG={best_sum['metrics']['nDCG']:.4f}) -> {best_path}")

    # --- LaTeX (booktabs, grouped; \_ escaping for names) -----------------
    def tex(s):
        return str(s).replace("_", r"\_")

    lines = [
        "% Auto-generated by experiments/stage1_shrec18.py",
        "% Stage 1 results (thesis tab:eval_stage1_ablation_grid /",
        "% fig:eval_stage1_ablation_bars).  Official SHREC'18 metrics",
        "% (hkust-vgd/shrec18 metrics.py); nDCG is the selection metric.",
        r"\begin{tabular}{ll" + "r" * len(METRIC_COLS) + "}",
        r"\toprule",
        "Config & Ablation & " + " & ".join(
            c.replace("_", r"\_") for c in METRIC_COLS) + r" \\",
        r"\midrule",
    ]
    last_group = None
    for spec, s in rows:
        if last_group is not None and spec.group != last_group:
            lines.append(r"\addlinespace")
        last_group = spec.group
        name = tex(spec.name)
        if spec.name == best_spec.name:
            name = r"\textbf{" + name + "}"
        vals = " & ".join(f"{s['metrics'][m]:.3f}" for m in METRIC_COLS)
        lines.append(f"{name} & {tex(spec.group)} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    tex_path = os.path.join(paths["results_root"], "stage1_summary.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[aggregate] LaTeX table -> {tex_path}")

    # --- TABLE B: depth-matched metrics -----------------------------------
    # A separate file on purpose.  Merging the two would put the official
    # (buggy, f-prefix) nDCG next to the corrected nDCG@K and invite the
    # reader to compare them, and would imply the invariant Table A columns
    # discriminate between geometry arms when they provably cannot.
    depth_rows = [(spec, s) for spec, s in rows if s.get("metrics_depth")]
    if depth_rows:
        k_used = {s["config"].get("geometry_k") for _, s in depth_rows}
        csv_b = os.path.join(paths["results_root"],
                             "stage1_summary_depth.csv")
        with open(csv_b, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ablation", "group", "alias_of", "num_queries", "K",
                        *DEPTH_COLS,
                        *[f"hit_sub@{k}" for k in HITRATE_KS]])
            for spec, s in depth_rows:
                md = s["metrics_depth"]
                w.writerow([
                    spec.name, spec.group, s.get("alias_of", ""),
                    s["num_queries"], s["config"].get("geometry_k", ""),
                    *[f"{md.get(m, float('nan')):.4f}" for m in DEPTH_COLS],
                    *[f"{md.get(f'hit_sub@{k}', float('nan')):.4f}"
                      for k in HITRATE_KS]])
        print(f"[aggregate] {len(depth_rows)} configs (Table B) -> {csv_b}")
        if len(k_used) > 1:
            print(f"[aggregate] WARNING: Table B mixes geometry depths "
                  f"{sorted(x for x in k_used if x is not None)} — the "
                  f"@K columns are not comparable across those rows. "
                  f"Re-run with a single --geom-k.")

        blines = [
            "% Auto-generated by experiments/stage1_shrec18.py",
            "% Stage 1, TABLE B: depth-matched metrics at the geometry depth K.",
            "% The official scalar metrics (stage1_summary.tex) are cut at",
            "% f = category size and are order-insensitive within that prefix,",
            "% so they cannot resolve a top-K re-ranking; these can.",
            "% NN_sub (sub-category grade) is the headline for the geometry arms.",
            r"\begin{tabular}{ll" + "r" * len(DEPTH_COLS) + "}",
            r"\toprule",
            "Config & Ablation & " + " & ".join(
                c.replace("_", r"\_") for c in DEPTH_COLS) + r" \\",
            r"\midrule",
        ]
        last_group = None
        for spec, s in depth_rows:
            if last_group is not None and spec.group != last_group:
                blines.append(r"\addlinespace")
            last_group = spec.group
            md = s["metrics_depth"]
            vals = " & ".join(f"{md.get(m, float('nan')):.3f}"
                              for m in DEPTH_COLS)
            blines.append(f"{tex(spec.name)} & {tex(spec.group)} & {vals} \\\\")
        blines += [r"\bottomrule", r"\end{tabular}", ""]
        tex_b = os.path.join(paths["results_root"],
                             "stage1_summary_depth.tex")
        with open(tex_b, "w") as f:
            f.write("\n".join(blines))
        print(f"[aggregate] LaTeX Table B -> {tex_b}")

    # --- OSCAR threshold arm: the faithful (variable-length) reading -------
    # Deliberately NOT a table row: precision() divides by the submitted list
    # length, so a short list is not the same quantity as a padded one.
    for spec, s in rows:
        st = s.get("shortlist_stats")
        if not st:
            continue
        print(f"[aggregate] {spec.name}: tau_text={st['tau_text']}  "
              f"|S'| median={st['median']:.0f} "
              f"IQR=[{st['q1']:.0f},{st['q3']:.0f}] "
              f"range=[{st['min']},{st['max']}]  "
              f"fallback={100 * st['fallback_rate']:.1f}%")
        # The fallback rate, NOT the median, is the degeneracy test.  When the
        # threshold admits nothing the arm falls back to top-k, so |S'| reads
        # back as exactly k and the median looks perfectly healthy while the
        # threshold mechanism is in fact never running.  Measured on SHREC'18:
        # tau=0.37 falls back on 98.3% of queries, and the arm then reproduces
        # the top-k cascade to within 0.0015 nDCG.  (MI3DOR/Stage 2 sits at
        # 96.9% — verwechselte Zuordnung bis 2026-09-19.)
        if st["fallback_rate"] > 0.25:
            print(f"[aggregate] WARNING: tau_text={st['tau_text']} admits "
                  f"nothing on {100 * st['fallback_rate']:.1f}% of queries — "
                  f"those fall back to top-{CLIP_PRUNE_K}, so this arm is "
                  f"mostly reproducing the top-k cascade, not the threshold "
                  f"mechanism. Report it as a *negative transfer* result "
                  f"(tau was calibrated on MI3DOR/YCB-V) rather than as "
                  f"OSCAR's pruning.")
        elif st["median"] < 5 or st["median"] > 0.5 * N_CADS:
            print(f"[aggregate] WARNING: tau_text={st['tau_text']} is "
                  f"degenerate on this dataset (median |S'| = "
                  f"{st['median']:.0f} of {N_CADS}). The arm measures a "
                  f"mistuned constant, not OSCAR's method — say so in the "
                  f"write-up or report a recalibrated tau alongside it.")


# ===========================================================================
# 9b. Release naming: rename the arm folder and emit the archive-form files
# ===========================================================================
# ``results/stage1_shrec18/`` names every configuration by its RELEASE name and
# keeps exactly three shapes of artefact.  A fresh run reproduces all three, so
# the record and a re-run can be diffed column by column instead of translated
# by hand.  The two headers below were read off the archive files (do NOT
# "improve" them — they define the column order of the record):
#
#   per_query/<release>.json   copy of <release>/results_per_query.json
#   all_arms.csv               one row per configuration, full-query runs
#   official_track_testsplit.csv + per_query/testsplit/<release>.json
#                              the --subset test-split counterpart
_ALL_ARMS_HEADER = ("configuration", "num_queries", "nDCG", "AP", "precision",
                    "NN_sub_hit1", "NN_cat", "MRR", "MRR_sub",
                    "hit_sub@5", "hit_sub@50")
_TESTSPLIT_HEADER = ("configuration", "num_queries", "nDCG", "precision",
                     "recall", "F1", "AP", "NNT1", "NNT2")
# Column -> (block of metrics_summary.json, key inside it).  Determined by
# comparing each archive row against that arm's summary: everything in
# all_arms.csv except NN_sub_hit1 carries its own name, and NN_sub_hit1 is the
# depth-matched NN_sub (sub-category hit@1).  ``configuration`` is the release
# name and ``num_queries`` sits at the top level of the summary.
_ALL_ARMS_SOURCE = {
    "nDCG": ("metrics", "nDCG"),
    "AP": ("metrics", "AP"),
    "precision": ("metrics", "precision"),
    "NN_sub_hit1": ("metrics_depth", "NN_sub"),
    "NN_cat": ("metrics_depth", "NN_cat"),
    "MRR": ("metrics_depth", "MRR"),
    "MRR_sub": ("metrics_depth", "MRR_sub"),
    "hit_sub@5": ("metrics_depth", "hit_sub@5"),
    "hit_sub@50": ("metrics_depth", "hit_sub@50"),
}
_TESTSPLIT_SOURCE = {m: ("metrics", m) for m in
                     ("nDCG", "precision", "recall", "F1", "AP",
                      "NNT1", "NNT2")}


def _release_row(release: str, summary: dict, header: Sequence[str],
                 source: Dict[str, Tuple[str, str]]) -> List[object]:
    """One archive row; values are round(x, 4) exactly as in the record."""
    row: List[object] = []
    missing = []
    for col in header:
        if col == "configuration":
            row.append(release)
        elif col == "num_queries":
            row.append(summary.get("num_queries", ""))
        elif col in source:
            block, key = source[col]
            val = summary.get(block, {}).get(key)
            row.append("" if val is None else round(float(val), 4))
            if val is None:
                missing.append(f"{col} ({block}.{key})")
        else:
            row.append("")
            missing.append(f"{col} (not derivable from metrics_summary.json)")
    if missing:
        print(f"[release] WARNING: left empty because not derivable from "
              f"metrics_summary.json: {', '.join(missing)}")
    return row


def _upsert_csv(path: str, header: Sequence[str], row: Sequence[object],
                sort_by: Optional[str] = None,
                lineterminator: str = "\r\n") -> None:
    """Rewrite ``path`` with ``row`` replacing/appending its configuration.

    One invocation of this script produces one arm, but several invocations may
    share an ``--out`` folder; the record holds all arms in one file, so the rows
    accumulate here too.  ``sort_by`` (a column name) sorts descending with a
    stable sort, which is the ordering of ``all_arms.csv``.
    """
    rows: List[List[str]] = []
    if os.path.isfile(path):
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            old_head = next(rd, None)
            if old_head == list(header):
                rows = [r for r in rd if r and r[0] != row[0]]
    rows.append([("" if v is None else str(v)) for v in row])
    if sort_by and sort_by in header:
        i = list(header).index(sort_by)

        def _key(r):
            try:
                return -float(r[i])
            except (TypeError, ValueError, IndexError):
                return 0.0
        rows.sort(key=_key)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator=lineterminator)
        w.writerow(list(header))
        w.writerows(rows)


def export_release_artifacts(results_root: str, arm: str, testsplit: bool,
                             overwrite: bool) -> dict:
    """Rename ``<results_root>/<arm>/`` to the release name and emit the
    archive-form artefacts next to it.  Returns the arm's metrics summary."""
    release = to_release(arm, "stage1")
    src, dst = os.path.join(results_root, arm), os.path.join(results_root,
                                                             release)
    if src != dst and os.path.isdir(src):
        if os.path.exists(dst):
            if not overwrite:
                sys.exit(f"[release] {dst} already exists — the run sits "
                         f"next to it as {src} and was NOT renamed. "
                         f"Either clear the old result away or repeat the run "
                         f"with --overwrite.")
            shutil.rmtree(dst)
            print(f"[release] {release}/ replaced (--overwrite).")
        os.rename(src, dst)
        print(f"[release] {arm}/ -> {release}/ (release name as in results/)")
    elif not os.path.isdir(dst):
        sys.exit(f"[release] no results folder {src} — run incomplete.")

    ms = os.path.join(dst, "metrics_summary.json")
    if not os.path.isfile(ms):
        sys.exit(f"[release] no {ms} — run incomplete.")
    with open(ms) as fh:
        summary = json.load(fh)

    # per_query/<release>.json — byte copy of the run's own per-query file.
    pq_dir = os.path.join(results_root, "per_query",
                          *(("testsplit",) if testsplit else ()))
    pq_src = os.path.join(dst, "results_per_query.json")
    if os.path.isfile(pq_src):
        os.makedirs(pq_dir, exist_ok=True)
        shutil.copyfile(pq_src, os.path.join(pq_dir, f"{release}.json"))
        print(f"[release] per_query/"
              + ("testsplit/" if testsplit else "")
              + f"{release}.json")
    else:
        print(f"[release] WARNING: {pq_src} missing — per_query/{release}.json "
              f"not written.")

    # the record CSV of this run type
    if testsplit:
        header = tuple(_TESTSPLIT_HEADER)
        row = _release_row(release, summary, header, _TESTSPLIT_SOURCE)
        out = os.path.join(results_root, "official_track_testsplit.csv")
        # LF, wie results/stage1_shrec18/official_track_testsplit.csv
        # (all_arms.csv dagegen ist CRLF — jede Datei wie ihr Archiv).
        _upsert_csv(out, header, row, lineterminator="\n")  # record order
    else:
        header = tuple(_ALL_ARMS_HEADER)
        row = _release_row(release, summary, header, _ALL_ARMS_SOURCE)
        out = os.path.join(results_root, "all_arms.csv")
        _upsert_csv(out, header, row, sort_by="nDCG")
    print(f"[release] {os.path.basename(out)} <- {release}")
    return summary


# ===========================================================================
# 10. Input validation (data is provided by the user — never downloaded)
# ===========================================================================

def validate_inputs(paths: dict, allow_partial: bool) -> List[str]:
    """Check that the user-provided data is in place; return the canonical
    gallery id list (intersection of CADs, rendered views, descriptions)."""
    problems = []
    data_root = paths["data_root"]
    for sub in ("cad", "rgbd", "results"):
        if not os.path.isdir(os.path.join(data_root, sub)):
            problems.append(f"missing {os.path.join(data_root, sub)} "
                            f"(raw SHREC'18 distribution)")
    if problems:
        raise SystemExit("[validate] " + "; ".join(problems))

    cads = {os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(data_root, "cad"))
            if f.endswith(".obj")}

    imgs_dir = paths["images_dir"]
    rendered = set()
    if os.path.isdir(imgs_dir):
        for d in os.listdir(imgs_dir):
            full = os.path.join(imgs_dir, d)
            if os.path.isdir(full) and any(
                    f.endswith(".png") and not f.endswith("_bg.png")
                    for f in os.listdir(full)):
                rendered.add(d)

    desc_ids = set()
    if os.path.isfile(paths["desc_file"]):
        with open(paths["desc_file"]) as f:
            desc_ids = set(json.load(f).keys())

    object_ids = sorted(cads & rendered & desc_ids)
    print(f"[validate] CADs: {len(cads)}  rendered: {len(rendered)}  "
          f"described: {len(desc_ids)}  -> scored gallery: "
          f"{len(object_ids)}")

    # Partial-view point clouds: without them the S_shape channel falls
    # back to full-mesh encoding (eval_common.build_pipeline), which
    # would silently collapse ablation E2b (partial vs. full mesh).
    with_partial = sum(
        1 for d in object_ids
        if any(f.endswith("_partial.npz")
               for f in os.listdir(os.path.join(imgs_dir, d))))
    if with_partial < len(object_ids):
        print(f"[validate] WARNING: only {with_partial}/{len(object_ids)} "
              f"rendered models have *_partial.npz partial point clouds "
              f"(preprocess_gallery.py --step partial).  E2b's "
              f"partial-view arm needs them; without them the base pass "
              f"degrades to full-mesh references.")

    if len(object_ids) < len(cads):
        msg = (f"[validate] gallery incomplete: {len(object_ids)}/{len(cads)} "
               f"models have renders+descriptions.\n"
               f"  renders:      python3 preprocess_gallery.py "
               f"--dataset shrec18 --step render\n"
               f"  descriptions: python3 preprocess_gallery.py "
               f"--dataset shrec18 --step describe")
        if allow_partial and object_ids:
            print(msg + "\n[validate] continuing (--allow-partial-gallery); "
                  "metrics are NOT final Stage-1 numbers.")
        else:
            raise SystemExit(msg + "\nProvide the missing data or pass "
                             "--allow-partial-gallery for a smoke test.")
    return object_ids


# ===========================================================================
# 11. CLI
# ===========================================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Thesis Experiment 1 — Stage 1 RGB-D retrieval tuning "
                    "on SHREC'18 ObjectNN+ (ablation grid; see module "
                    "docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ablations", help="comma list of groups/cells, e.g. "
                                        "'E1,E4,O4_V8'")
    ap.add_argument("--all", action="store_true",
                    help="run the whole registered grid")
    ap.add_argument("--list", action="store_true",
                    help="print the ablation registry and exit")
    ap.add_argument("--resume", action="store_true",
                    help="skip ablations/passes whose outputs already exist")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if results exist")
    ap.add_argument("--with-geometry", action="store_true",
                    help="include E2/O1c-e (GeDi service: docker compose "
                         "up -d gedi; chamfer/scale-gate run without it)")
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="smoke test on the first N queries")
    ap.add_argument("--allow-partial-gallery", action="store_true",
                    help="run even if renders/descriptions are incomplete")
    ap.add_argument("--precompute", action="store_true",
                    help="gallery PC: build all reference embedding caches "
                         "(no queries/ablations) + a provenance manifest to "
                         "ship to the eval PC")
    ap.add_argument("--geom-k", type=int, default=None, metavar="K",
                    help="geometry re-ranking depth. Default: derived from the "
                         "base-fusion hit-rate curve (smallest K within 2 pts "
                         "of the deepest measured K); falls back to "
                         f"{GEOM_SHORTLIST} if BASE has not been run yet")
    ap.add_argument("--hit-rate-curve", action="store_true",
                    help="print the base-fusion hit-rate@K ceiling and the K "
                         "it implies, then exit")
    ap.add_argument("--no-icp", action="store_true",
                    help="skip ICP refinement in B2 (~38%% of the per-pair "
                         "cost for a measured 0.0001 nDCG effect at K=5); "
                         "E2_chamfer_icp is dropped from the selection since "
                         "that arm IS the ICP measurement")
    ap.add_argument("--bench-rerank", type=int, default=0, metavar="N",
                    help="time N real registrations at the production "
                         "settings (warm descriptor cache) and print the "
                         "K -> wall-clock table, then exit")
    ap.add_argument("--precompute-gedi", action="store_true",
                    help="extract and cache GeDi descriptors for all gallery "
                         "CADs and queries, then exit (~5.2 h, ~4.1 GB; makes "
                         "the geometry depth K nearly free afterwards)")
    ap.add_argument("--bench-gedi", type=int, default=0, metavar="N",
                    help="time N GeDi+RANSAC fits and extrapolate the cost of "
                         "full-database S_GeDi, then exit (see thesis O1)")
    ap.add_argument("--viz-check", type=int, default=0, metavar="N",
                    help="save a contact sheet of N query crops")
    ap.add_argument("--rrf-c-sweep", nargs="?", const="1,10,30,60,100,300",
                    default="", metavar="C_LIST",
                    help="E6 robustness check: recompute RRF for several "
                         "Cormack c on the cached scores and pair each c "
                         "against the weighted sum (default list "
                         "1,10,30,60,100,300). Writes rrf_c_sweep.csv.")
    ap.add_argument("--weight-sweep", action="store_true",
                    help="fusion-weight SENSITIVITY sweep over the simplex on "
                         "the cached BASE score vectors (Tier-2, no GPU). "
                         "Writes weight_sweep.csv. Report as sensitivity, do "
                         "NOT select weights on it (would be tuning on test).")
    ap.add_argument("--weight-step", type=float, default=0.1, metavar="S",
                    help="grid step for --weight-sweep (default 0.1)")
    ap.add_argument("--sweep-shape-pass", default="ulip_pc_rgb",
                    choices=["ulip_pc_rgb", "ulip_cross_rgb", "ulip_pc_xyz", "uni3d"],
                    help="which S_shape pass the weight sweep fuses (default "
                         "ulip_pc_rgb = pc-mode BASE; ulip_cross_rgb = cross-mode "
                         "query image, the Stage-2/MI3DOR depth-free setting)")
    ap.add_argument("--data-root", default=DEFAULTS["data_root"])
    ap.add_argument("--images-dir", default=DEFAULTS["images_dir"])
    ap.add_argument("--desc-file", default=DEFAULTS["desc_file"])
    ap.add_argument("--results-root", default=DEFAULTS["results_root"])
    args = ap.parse_args(argv)

    if args.list:
        print(f"{'name':<24} {'group':<4} {'passes':<28} geometry  question")
        for s in ABLATIONS.values():
            extra = f"alias of {s.alias_of}" if s.alias_of else \
                    (s.geometry or "")
            print(f"{s.name:<24} {s.group:<4} {','.join(s.passes):<28} "
                  f"{extra:<18} {s.question}")
        return

    paths = {"data_root": args.data_root, "images_dir": args.images_dir,
             "desc_file": args.desc_file, "results_root": args.results_root,
             "stage1_root": DEFAULTS["stage1_root"]}
    os.makedirs(os.path.join(paths["results_root"], "_cache"), exist_ok=True)

    if args.hit_rate_curve:
        print_hitrate_curve(paths["results_root"])
        return

    global SKIP_ICP
    SKIP_ICP = args.no_icp

    # ---- geometry depth K ------------------------------------------------
    # Geometry re-ranks a shortlist but can never pull a relevant CAD into it,
    # so base hit-rate@K bounds every top-1 metric the stage can reach.  K is
    # therefore read off that curve rather than chosen, which is what makes the
    # constant defensible in the write-up (STAGE1_EVALUATION_DESIGN §3).
    global GEOM_K
    if args.geom_k is not None:
        GEOM_K = args.geom_k
        print(f"[config] geometry depth K = {GEOM_K} (explicit --geom-k)")
    else:
        curve = load_hitrate_curve(paths["results_root"], "sub")
        if curve:
            GEOM_K = choose_geom_k(curve)
            print(f"[config] geometry depth K = {GEOM_K} (from the "
                  f"{BASE_ABLATION} hit_sub@K curve, tol {HITRATE_TOL:.0%})")
        else:
            GEOM_K = GEOM_SHORTLIST
            print(f"[config] geometry depth K = {GEOM_K} (fallback: "
                  f"{_no_curve_reason(paths['results_root'])})")

    # Make repo modules importable regardless of CWD (the prelude already does
    # this at import time; kept for direct main() calls from other scripts).
    for p in (_REPO, os.path.join(_REPO, "evaluation")):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Make repo modules importable regardless of CWD (before the selection,
    # which is skipped in --precompute mode).
    # torch bundles libgomp.so.1 that open3d also needs — import it first.
    import torch  # noqa: F401

    # ---- gallery PC: precompute all reference caches and exit ------------
    if args.precompute:
        object_ids = validate_inputs(paths, args.allow_partial_gallery)
        precompute_gallery(paths, object_ids,
                           resume=args.resume and not args.overwrite)
        return

    # ---- one-time per-cloud descriptor extraction and exit ---------------
    if args.precompute_gedi:
        object_ids = validate_inputs(paths, args.allow_partial_gallery)
        gt = load_official_gt(paths["data_root"], paths["stage1_root"])
        index = prepare_queries(paths["data_root"], paths["stage1_root"], gt)
        precompute_gedi(paths, index, object_ids, args.limit_queries)
        return

    # ---- re-ranking throughput: turns K into a measured budget decision ---
    if args.bench_rerank:
        object_ids = validate_inputs(paths, args.allow_partial_gallery)
        gt = load_official_gt(paths["data_root"], paths["stage1_root"])
        index = prepare_queries(paths["data_root"], paths["stage1_root"], gt)
        bench_rerank(paths, index, object_ids, args.bench_rerank)
        return

    # ---- GeDi cost benchmark (thesis O1 feasibility claim) and exit ------
    if args.bench_gedi:
        object_ids = validate_inputs(paths, args.allow_partial_gallery)
        gt = load_official_gt(paths["data_root"], paths["stage1_root"])
        index = prepare_queries(paths["data_root"], paths["stage1_root"], gt)
        bench_gedi(paths, index, object_ids, args.bench_gedi)
        return

    # ---- fusion-weight sensitivity sweep (Tier-2 only) and exit ----------
    if args.weight_sweep:
        object_ids = validate_inputs(paths, args.allow_partial_gallery)
        gt = load_official_gt(paths["data_root"], paths["stage1_root"])
        cad_labels, freqs = gt["cad"], gt["freqs"]
        index = prepare_queries(paths["data_root"], paths["stage1_root"], gt)
        stores = {}
        for pkey in ("base", args.sweep_shape_pass):   # BASE config's passes
            stores[pkey] = run_pass(pkey, paths, index, object_ids,
                                    args.limit_queries, resume=True)
        run_weight_sweep(paths, index, object_ids, cad_labels, freqs, stores,
                         args.weight_step, args.limit_queries,
                         shape_pass=args.sweep_shape_pass)
        return

    # ---- RRF-c-Sensitivitaet (E6-Robustheitscheck) und exit --------------
    if args.rrf_c_sweep:
        object_ids = validate_inputs(paths, args.allow_partial_gallery)
        gt = load_official_gt(paths["data_root"], paths["stage1_root"])
        cad_labels, freqs = gt["cad"], gt["freqs"]
        index = prepare_queries(paths["data_root"], paths["stage1_root"], gt)
        stores = {}
        for pkey in ("base", "ulip_pc_rgb"):
            stores[pkey] = run_pass(pkey, paths, index, object_ids,
                                    args.limit_queries, resume=True)
        run_rrf_c_sweep(paths, index, object_ids, cad_labels, freqs, stores,
                        [int(x) for x in args.rrf_c_sweep.split(",")],
                        args.limit_queries)
        return

    specs = select_ablations(args.ablations, args.all, args.with_geometry)
    if not specs:
        raise SystemExit("Nothing to run after selection.")

    # ---- inputs, GT, queries (lazy + cached) -----------------------------
    # Official SHREC'18 GT (rgbd.csv/cad.csv): real category+subcategory for
    # all 2,101 queries and 3,308 CADs, scored with the track's own
    # metrics.py so results are leaderboard-comparable (tuning + final).
    object_ids = validate_inputs(paths, args.allow_partial_gallery)
    verify_precompute_provenance(paths)
    gt = load_official_gt(paths["data_root"], paths["stage1_root"])
    cad_labels = gt["cad"]          # {cadid: (category, subcategory)}
    freqs = gt["freqs"]             # {category: n_cads}  (official denominators)
    index = prepare_queries(paths["data_root"], paths["stage1_root"], gt)
    if args.viz_check:
        viz_check(index, paths["stage1_root"], args.viz_check)

    # ---- tier 1: passes --------------------------------------------------
    resume = args.resume and not args.overwrite
    todo = [s for s in specs if not s.alias_of]
    if resume:
        todo = [s for s in todo if not os.path.isfile(os.path.join(
            arm_dir(paths["results_root"], s.name), "metrics_summary.json"))]
        done = len([s for s in specs if not s.alias_of]) - len(todo)
        if done:
            print(f"[resume] {done} ablations already computed.")

    needed_passes: List[str] = [
        p for p in PASS_DEFS
        if any(p in s.passes for s in todo)]
    stores: Dict[str, dict] = {}
    for pkey in needed_passes:      # PASS_DEFS order: 'base' always first
        try:
            stores[pkey] = run_pass(pkey, paths, index, object_ids,
                                    args.limit_queries, resume=True)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[pass:{pkey}] FAILED ({exc}) — dependent ablations "
                  f"will be skipped.")

    # ---- calibrate tau, if any selected arm needs it ----------------------
    if any(s.scope == "clip_threshold_cal" for s in todo) and "base" in stores:
        global CLIP_TAU_CAL
        CLIP_TAU_CAL = calibrate_tau(
            stores, [q["id"] for q in (index[:args.limit_queries]
                                       if args.limit_queries else index)])

    # ---- tier 2: ablations ----------------------------------------------
    geom = _GeometryEngine(paths) if any(s.geometry for s in todo) else None
    if geom and any(s.needs_gedi for s in todo) and not geom.gedi_available():
        skipped = [s.name for s in todo if s.needs_gedi]
        print(f"[run] skipping GeDi-signal ablations: {', '.join(skipped)}")
        todo = [s for s in todo if not s.needs_gedi]

    for spec in todo:
        if any(p not in stores for p in spec.passes):
            print(f"[run:{spec.name}] skipped (missing pass "
                  f"{[p for p in spec.passes if p not in stores]}).")
            continue
        run_ablation(spec, paths, index, stores, object_ids, cad_labels,
                     freqs, args.limit_queries, geom)

    # ---- aliases + aggregate (always refreshed) --------------------------
    for spec in specs:
        if spec.alias_of:
            write_alias(spec, paths)
    aggregate(paths)


if __name__ == "__main__":
    import datetime
    import subprocess

    # The arm was already validated against _SUPPORTED in the prelude and
    # _SUPPORTED against the registry at import time; this is the last check
    # against a stale translation reaching select_ablations.
    if _ARM not in ABLATIONS:
        sys.exit(f"[stage1] arm {_ARM} is missing from the ablation registry.")

    # Full run provenance next to the results (readable argv, derived arm, the
    # argv actually handed to the driver, effective env, git revision).
    try:
        _rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        _rev = ""
    os.makedirs(_RESULTS_ROOT, exist_ok=True)
    with open(os.path.join(_RESULTS_ROOT, "run_config.json"), "w") as _f:
        json.dump({"argv": sys.argv[1:],
                   "arm": _ARM,
                   "internal_argv": _INTERNAL_ARGV,
                   "env": _ENV,
                   "git": _rev,
                   "time": datetime.datetime.now().isoformat(timespec="seconds")},
                  _f, indent=1)

    main(_INTERNAL_ARGV)

    # Release naming: the arm folder becomes <release>/ and the archive-form
    # artefacts (per_query/<release>.json, all_arms.csv) land next to it, so this
    # run is shaped exactly like results/stage1_shrec18.
    _RELEASE = to_release(_ARM, "stage1")
    _summary = export_release_artifacts(
        _RESULTS_ROOT, _ARM, testsplit=bool(_ARGS.subset),
        overwrite=_ARGS.overwrite)

    _line = (f"RESULT {_RELEASE} ({_ARM}): "
             f"nDCG {_summary['metrics']['nDCG']:.4f} / "
             f"hit@1 {_summary['metrics_depth']['NN_sub']:.4f}")
    if _ARGS.limit:
        _line += f"   [SMOKE ({_ARGS.limit} queries) — not comparable]"
    else:
        _line += ("   (reported values: "
                  + (_RECORD_CSV_TESTSPLIT if _ARGS.subset else _RECORD_CSV) + ")")
    print(_line, flush=True)
