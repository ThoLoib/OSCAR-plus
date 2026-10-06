"""
stage2_mi3dor.py
================
OSCAR+ category-level retrieval evaluation on MI3DOR (Stage 2).

Drives CLIP + DINOv2 + ULIP-2 + Score Fusion. All shared logic lives in
``evaluation/eval_common.py``; this file only contains MI3DOR-specific
iteration, label mapping, and configuration. Paths come from
``config/paths.yaml`` (overridable per flag, see ``--help``). Outside the
container the script wraps itself into ``docker compose run`` automatically.

How to run
----------
    python3 experiments/stage2_mi3dor.py --gallery partial
        # BASE result: NN 88.44 / FT 0.6918
    python3 experiments/stage2_mi3dor.py --gallery fullmesh
        # NN 86.57 / FT 0.6818
    python3 experiments/stage2_mi3dor.py --gallery fullmesh --views 8
        # OSCAR legacy baseline (8 rendered views)

Outputs land in ``<runs_root>/<--out>/<mode>/``; a ``run_config.json`` with
argv, env, and git revision is written next to them. Per-query JSON records
match the baseline format (category, filename, gt, pred, clip_candidates,
matched_files) with ULIP and fusion fields added. Per reported configuration the
metrics summary is additionally copied to ``<runs_root>/<--out>/<config>.json``
under the RELEASE name used by ``results/stage2_mi3dor/`` — ``fused_partial``,
``fused_fullmesh`` and (with ``--views 8``) ``legacy_oscar_views8``; content
unchanged. See ``evaluation/configuration_names.py``.
"""

import json
import os
import glob

# ---------------------------------------------------------------------------
# CLI prelude — before the heavy imports (PIL/tqdm/eval_common) so that
# `--help` works on the host without the container dependencies.
# ---------------------------------------------------------------------------
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(1, os.path.join(_REPO, "evaluation"))

from configuration_names import to_release  # noqa: E402

_OUT_NAME = "stage2_mi3dor"
_PATHS = None

# (shape mode, gallery views) -> internal run key of the reported configuration.
# The internal keys are the run names the Stage-2 results were produced under;
# to_release() turns them into the names results/stage2_mi3dor/ uses
# (fused_partial / fused_fullmesh / legacy_oscar_views8).  Combinations outside
# this table were never reported and get no archive-named copy — see
# evaluation/configuration_names.py.
_MODE_ARM = {
    ("partial", 42): "oscarplus_v2_tau037_dinomean_partialforce",
    ("fullmesh", 42): "oscarplus_v2_tau037_dinomean_ulipfix",
    ("fullmesh", 8): "oscar_legacy_v8",
}

if __name__ == "__main__":
    import argparse

    from pipeline import paths as _paths_mod

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gallery", choices=["partial", "fullmesh", "both"],
                        default="partial",
                        help="shape gallery representation for the ULIP "
                             "channel (default: partial)")
    parser.add_argument("--views", type=int, choices=[8, 42], default=42,
                        help="number of gallery views; 8 = OSCAR legacy "
                             "baseline (default: 42)")
    parser.add_argument("--dino-pooling", choices=["mean", "cls"],
                        default="mean",
                        help="DINO patch pooling; cls reproduces the old CLS "
                             "numbers (default: mean)")
    parser.add_argument("--max-queries-per-category", type=int, default=0,
                        help="smoke runs: cap queries per category "
                             "(0 = all queries; default: 0)")
    parser.add_argument("--out", default="",
                        help="result folder name below runs_root (default: "
                             "stage2_mi3dor, or stage2_mi3dor_views<N> for "
                             "non-default view counts)")
    parser.add_argument("--no-docker", action="store_true",
                        help="do not auto-wrap into the oscar-plus container")
    _paths_mod.add_path_args(parser)
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["MI3DOR_MODES"] = ("fullmesh,partial" if args.gallery == "both"
                                  else args.gallery)
    os.environ["MI3DOR_NUM_VIEWS"] = str(args.views)
    os.environ["MI3DOR_DINO_POOLING"] = args.dino_pooling
    if args.max_queries_per_category > 0:
        os.environ["MI3DOR_MAX_QUERIES_PER_CAT"] = str(args.max_queries_per_category)
    _OUT_NAME = args.out or ("stage2_mi3dor" if args.views == 42
                             else f"stage2_mi3dor_views{args.views}")
    _PATHS = _paths_mod.from_args(args)

    if not os.path.exists("/.dockerenv") and not args.no_docker:
        if not os.path.isfile(os.path.join(_REPO, "docker-compose.yml")):
            sys.exit(f"[stage2] {_REPO}/docker-compose.yml is missing — "
                     "incomplete repo?")
        print("[stage2] runs in the oscar-plus container — wrapping automatically.",
              flush=True)
        os.chdir(_REPO)  # docker compose needs the repo as CWD
        os.execvp("docker", ["docker", "compose", "run", "--rm", "oscar-plus",
                             "python3", "/app/experiments/stage2_mi3dor.py"]
                  + sys.argv[1:])

if _PATHS is None:  # imported as a module (no CLI): plain paths.yaml defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()

from PIL import Image
from tqdm import tqdm

from eval_common import (
    EvalConfig,
    build_pipeline,
    load_ulip_query_cache,
    pre_encode_ulip_queries,
    run_evaluation,
)

# ============================================================================
# CONFIG — all locations come from the central path registry (_PATHS)
# ============================================================================
ref_dir               = os.path.join(_PATHS["gallery_root"], "MI3DOR")
bop_root              = os.path.join(_PATHS["datasets_root"], "mi3dor/image/test")
desc_file             = os.path.join(_PATHS["cad_root"], "MI3DOR/descriptions_attributes.json")
cad_mesh_glob         = os.path.join(_PATHS["cad_root"], "MI3DOR/model/test/*/*.obj")
result_folder         = os.path.join(_PATHS["runs_root"], _OUT_NAME)
ulip_query_cache_path = os.path.join(_PATHS["caches_root"], "ulip_query_cache_mi3dor.pt")
os.makedirs(_PATHS["caches_root"], exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

# DINO pooling for MI3DOR. Default is now "mean" (Pulli's mean-patch pooling):
# the 2026-08-07 full-set ablation showed mean beats CLS on MI3DOR
# (dino_only_full FT 0.587->0.629 / NN 78.0->83.0; 3-way fusion FT 0.620->0.648
# partial). This default is MI3DOR-SCOPED — the global PipelineConfig.dino_pooling
# stays "cls" so SHREC/other benchmarks are unaffected until separately tested.
# Set MI3DOR_DINO_POOLING=cls to reproduce the old CLS numbers. The gallery DINO
# cache is keyed by pooling (step4._cache_path), so the two never collide.
dino_pooling = os.environ.get("MI3DOR_DINO_POOLING", "mean")

# Number of gallery views. Default 42 (our config). MI3DOR_NUM_VIEWS=8 gives the
# **OSCAR legacy baseline**: the published cascade is described with 8 rendered
# views, so the clip-pruned/oscar_maxview arms are re-run at V=8 for a like-for-
# like mechanism comparison. The DINO gallery cache holds all 42 views and
# step4._apply_view_limit() trims to the first N, so this needs NO re-encoding.
num_views = int(os.environ.get("MI3DOR_NUM_VIEWS", "42"))


cfg = EvalConfig(
    ref_dir=ref_dir,
    desc_file=desc_file,
    cad_mesh_glob=cad_mesh_glob,
    result_folder=result_folder,
    topk=[15],
    # CLIP shortlist S' for the pruned / OSCAR arms: threshold pruning (Pulli
    # et al. arXiv:2601.07333), tau_text=0.37 on the image<->text cosine, with
    # a top-20 fallback when no candidate clears tau. The full-DB arms
    # (clip_only / dino_only_full / ulip_only_full / clip_dino_ulip_full) rank
    # the whole gallery regardless (CLIP retrieval auto-expanded to |gallery|).
    clip_top_k=20,               # = fallback depth; full CLIP ranking auto-expands
    clip_prune_mode="threshold",
    clip_tau=0.37,
    clip_fallback_k=20,
    dino_top_k=9999,
    ulip2_top_k=9999,
    fusion_top_k=9999,
    TOP_F=20,
    # Frozen Stage-1 E2 full-fusion weights. MUST be set explicitly: EvalConfig
    # defaults to 0/0.5/0.5 (CLIP OFF), which silently made clip_dino_ulip_full a
    # DINO+ULIP run and the pruned arm DINO:ULIP 1:1 instead of 4:3 (cross-stage
    # comparability audit 2026-08-13 P0.1 — same bug that hit Stage 3).
    weight_clip=0.3, weight_dino=0.4, weight_ulip=0.3,
    ulip_query_cache_path=ulip_query_cache_path,
    # Best settings from the shrec18 experiment: the O4 view-count sweep peaked
    # at V=42 (nDCG 0.597; V8 0.580 < V16 0.593 < V42 0.597), with topk_softmax
    # over the top-5 views (the pipeline default used there). Pinned explicitly.
    pipeline_overrides={
        "num_views": num_views,
        "dino_view_aggregation": "topk_softmax",
        "dino_view_topk": 5,
        "dino_view_temperature": 0.5,
        # Partial-view shape mode: the per-view ULIP clouds are soft-k-max
        # combined (topk_softmax, tau=0.5). top-k set to 5 to MATCH the DINO
        # view aggregation (user request 2026-08-04) — was top-8 in shrec18's
        # partial shape channel (experiment1:1386); equalised here so both
        # channels pool the same way. Full-mesh mode has one embedding, so
        # aggregation is moot there.
        "ulip_view_aggregation": "topk_softmax",
        "ulip_view_topk": 5,
        "ulip_view_temperature": 0.5,
        # CLS (frozen default) vs mean (Pulli) DINO pooling — ablation knob.
        "dino_pooling": dino_pooling,
    },
)


# ============================================================================
# MI3DOR helpers
# ============================================================================

def to_category_label(object_id: str) -> str:
    """``airplane_test_0001`` -> ``airplane``."""
    if "_test" in object_id:
        return object_id.split("_test")[0]
    return object_id


def _get_categories():
    """Return categories that have CLIP descriptions."""
    all_cats = sorted(
        c for c in os.listdir(bop_root)
        if os.path.isdir(os.path.join(bop_root, c))
    )
    desc_cats = set()
    if desc_file and os.path.isfile(desc_file):
        with open(desc_file) as f:
            for k in json.load(f).keys():
                desc_cats.add(k.split("_test")[0] if "_test" in k else k)
    if desc_cats:
        cats = [c for c in all_cats if c in desc_cats]
        skipped = set(all_cats) - set(cats)
        if skipped:
            print(f"[mi3dor] {len(all_cats)} categories on disk, "
                  f"{len(cats)} have CLIP descriptions. "
                  f"Skipping: {sorted(skipped)}")
        return cats
    return all_cats


def _collect_filtered_cad_mesh_items(allowed_categories):
    """Collect CAD meshes only for categories that have descriptions."""
    allowed = set(allowed_categories)
    mesh_items = []
    for mesh_path in sorted(glob.glob(cad_mesh_glob)):
        obj_id = os.path.splitext(os.path.basename(mesh_path))[0]
        category = obj_id.split("_test")[0] if "_test" in obj_id else obj_id
        if category in allowed:
            mesh_items.append((obj_id, mesh_path))
    print(f"[mi3dor] Using {len(mesh_items)} CAD meshes after category filter")
    return mesh_items


def _collect_query_paths(categories):
    # Respect the same MI3DOR_MAX_QUERIES_PER_CAT cap as the query factory so
    # the ULIP query pre-encoding only touches the subset actually evaluated
    # (identical first-N-sorted-per-category selection).
    _cap = int(os.environ.get("MI3DOR_MAX_QUERIES_PER_CAT", "0") or "0")
    paths = []
    for cat in categories:
        cat_dir = os.path.join(bop_root, cat)
        cat_files = sorted(
            f for f in os.listdir(cat_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if _cap > 0:
            cat_files = cat_files[:_cap]
        for fname in cat_files:
            paths.append(os.path.join(cat_dir, fname))
    return paths


def _make_query_factory(categories):
    # Small-run knob: cap queries PER CATEGORY for a fast directional read
    # (e.g. CLS-vs-mean pooling). |C| is the gallery class size, unaffected by
    # how many queries we sample, so a capped run yields the same metric
    # definitions on a smaller (noisier) query sample. 0/unset = all queries.
    _cap = int(os.environ.get("MI3DOR_MAX_QUERIES_PER_CAT", "0") or "0")

    def factory(k):
        for category in tqdm(categories,
                             desc=f"Top-k={k} Categories", unit="cat"):
            cat_dir = os.path.join(bop_root, category)
            query_files = sorted(
                f for f in os.listdir(cat_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            if _cap > 0:
                query_files = query_files[:_cap]
            for fname in tqdm(query_files, desc=category,
                              unit="img", leave=False):
                img_path = os.path.join(cat_dir, fname)
                roi = Image.open(img_path).convert("RGB")
                yield roi, category, img_path, category, fname
    return factory


# ============================================================================
# Main
# ============================================================================

def _quarantine_foreign_views(ref_root: str) -> int:
    """Some gallery instances contain 2 FOREIGN reference images that are NOT
    our produced renders (3-digit zero-padded names like ``_002.png``). Our 42
    real views are ``_0.png``..``_41.png`` (1-2 digits, no leading zero). The
    DINO view-index parser reads ``_002.png`` as index 2, colliding with
    ``_2.png`` and displacing ``_41.png`` from the top-42 — so a foreign image
    would silently replace a real view. Rename any 3-digit-named png to
    ``.foreign`` so the loader (which matches ``*.png``) skips it. Idempotent;
    non-destructive (rename, not delete). Returns how many were quarantined."""
    import re
    pat = re.compile(r"_[0-9]{3,}\.png$")
    n = 0
    if not os.path.isdir(ref_root):
        return 0
    for inst in os.listdir(ref_root):
        d = os.path.join(ref_root, inst)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".png") and not fn.endswith("_bg.png") and pat.search(fn):
                os.rename(os.path.join(d, fn), os.path.join(d, fn + ".foreign"))
                n += 1
    if n:
        print(f"[mi3dor] quarantined {n} FOREIGN reference views (3-digit names) "
              f"-> *.foreign; keeping only the 42 produced views per instance.")
    return n


def main():
    _quarantine_foreign_views(ref_dir)
    categories = _get_categories()
    if not categories:
        raise RuntimeError(f"No categories found under {bop_root}")
    print(f"[mi3dor] Evaluating {len(categories)} categories")
    print(
        "[mi3dor] Both full-set and CLIP-pruned DINO/ULIP variants are "
        "computed in one run (six variants reported)."
    )

    cad_mesh_items = _collect_filtered_cad_mesh_items(categories)

    # Shape-source ablation: full CAD mesh vs partial rendered views (mirrors
    # shrec18 E2b_fullmesh vs E2b_partial; there full-mesh 0.5985 > partial
    # 0.5970). Only the ULIP shape channel differs — the query-IMAGE ULIP
    # embeddings are identical across modes, so encode them once and reuse.
    # Results land in <result_folder>/{fullmesh,partial}/ so one sync of the
    # parent captures both. The `partial` mode needs the per-view *_partial.npz
    # in ref_dir; if they are absent build_pipeline warns and falls back to
    # full-mesh (which would make the two tables identical — check the log).
    base_result_folder = cfg.result_folder
    # MI3DOR_MODES lets the DINO-pooling ablation run fullmesh-only: the DINO
    # arms are shape-mode-independent, so fullmesh alone isolates the pooling
    # effect at half the runtime. Default runs both.
    _modes_env = os.environ.get("MI3DOR_MODES", "fullmesh,partial")
    _wanted = {m.strip() for m in _modes_env.split(",") if m.strip()}
    ulip_cache = None
    for mode_name, use_partial in (("fullmesh", False), ("partial", True)):
        if mode_name not in _wanted:
            print(f"[mi3dor] skipping mode {mode_name} (MI3DOR_MODES={_modes_env})")
            continue
        cfg.ulip2_use_partial_views = use_partial
        cfg.result_folder = os.path.join(base_result_folder, mode_name)
        print(f"\n===== MI3DOR shape-source ablation: {mode_name} "
              f"(ulip2_use_partial_views={use_partial}) =====")
        components = build_pipeline(cfg, cad_mesh_items=cad_mesh_items)
        _, _, _, _, shape_m = components
        if shape_m is not None and ulip_cache is None:
            ulip_cache = load_ulip_query_cache(cfg.ulip_query_cache_path)
            if ulip_cache is None:
                print("[mi3dor] No pre-computed ULIP cache — encoding query "
                      "images on the fly (shared across both shape modes). "
                      "The reported numbers were produced with the cache from "
                      "evaluation/precompute_ulip_query_embeddings.py; the "
                      "on-the-fly encoding can differ in the fifth decimal.")
                ulip_cache = pre_encode_ulip_queries(
                    _collect_query_paths(categories), shape_m)
        run_evaluation(cfg, to_category_label,
                       _make_query_factory(categories),
                       components, ulip_cache)


if __name__ == "__main__":
    import datetime
    import subprocess

    # Full run provenance next to the results (argv + effective env + git rev).
    try:
        _rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        _rev = ""
    _run_env = {k: os.environ[k] for k in (
        "PYTHONHASHSEED", "MI3DOR_MODES", "MI3DOR_NUM_VIEWS",
        "MI3DOR_DINO_POOLING", "MI3DOR_MAX_QUERIES_PER_CAT")
        if k in os.environ}
    json.dump({"argv": sys.argv[1:], "env": _run_env, "git": _rev,
               "time": datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(result_folder, "run_config.json"), "w"),
              indent=1)

    main()

    # Immediate check against the frozen thesis record (results/stage2_mi3dor)
    # — only comparable without a query cap and at the default 42 views.
    _capped = int(os.environ.get("MI3DOR_MAX_QUERIES_PER_CAT", "0") or "0") > 0
    _refs = {
        "partial": "(reference results/stage2_mi3dor/fused_partial.json: "
                   "NN 88.44 / FT 0.6918)",
        "fullmesh": "(reference results/stage2_mi3dor/fused_fullmesh.json: "
                    "NN 86.57 / FT 0.6818)",
    }
    _ran = {m.strip() for m in
            os.environ.get("MI3DOR_MODES", "fullmesh,partial").split(",")
            if m.strip()}
    import shutil

    for _mode in ("fullmesh", "partial"):
        if _mode not in _ran:
            continue
        _ms = os.path.join(result_folder, _mode, "metrics_summary_topk_15.json")
        if not os.path.isfile(_ms):
            sys.exit(f"[stage2] no {_ms} — incomplete run.")

        # Archive-named copy of the summary: results/stage2_mi3dor/ keeps this
        # exact file per configuration, under the RELEASE name and with
        # unchanged content, so the record and a re-run can be diffed directly.
        _arm = _MODE_ARM.get((_mode, num_views))
        if _arm:
            _release = to_release(_arm, "stage2")
            shutil.copyfile(_ms, os.path.join(result_folder,
                                              f"{_release}.json"))
            print(f"[stage2] {_release}.json ({_mode}, {num_views} views) "
                  f"<- {os.path.relpath(_ms, result_folder)}", flush=True)
        else:
            print(f"[stage2] {_mode} @ {num_views} views is not a reported "
                  f"configuration — no archive copy (only "
                  f"{_mode}/metrics_summary_topk_15.json).", flush=True)

        _v = json.load(open(_ms))["variants"]["clip_dino_ulip_full"]
        _line = (f"RESULT full fusion ({_mode}"
                 + (f" -> {to_release(_arm, 'stage2')}" if _arm else "")
                 + f"): NN {_v['NN_accuracy']:.2f} / FT {_v['FT_mean']:.4f}")
        if not _capped and num_views == 42:
            _line += "   " + _refs[_mode]
        print(_line, flush=True)
