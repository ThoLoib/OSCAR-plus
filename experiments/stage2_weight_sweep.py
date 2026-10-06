#!/usr/bin/env python3
"""
stage2_weight_sweep.py
======================
MI3DOR weight sweep (Stage 2) over the three fusion weights.

Each query is scored once; the per-channel scores are then fused for all 66
simplex points (step 0.1, as in the Stage-1 sweep). The score assembly
replicates pipeline/step6_fusion._weighted_sum — insertion order (CLIP, then
DINO, then shape candidates), clip = max(clip_res.score, dino.clip_score),
ulip NaN stays 0, min-max over the assembled values (zeros included), stable
sort. weight_sweep_manifest.json records the commit and the configuration.

How to run
----------
    python3 experiments/stage2_weight_sweep.py
    # Smoke: MI3DOR_MAX_QUERIES_PER_CAT=15 python3 experiments/stage2_weight_sweep.py

Results land in ``<runs_root>/<--out>/`` (default: stage2_weight_sweep).
Outside the container the script wraps itself into ``docker compose run``
automatically.
"""
import csv
import datetime as _dt
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# CLI prelude — before the heavy imports (numpy/stage2_mi3dor/eval_common) so
# that `--help` works on the host without the container dependencies.
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(1, os.path.join(_REPO, "evaluation"))

_OUT_NAME = "stage2_weight_sweep"
_PATHS = None

if __name__ == "__main__":
    import argparse

    from pipeline import paths as _paths_mod

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="stage2_weight_sweep",
                        help="result folder name below runs_root "
                             "(default: stage2_weight_sweep)")
    parser.add_argument("--no-docker", action="store_true",
                        help="do not auto-wrap into the oscar-plus container")
    _paths_mod.add_path_args(parser)
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = "0"
    _OUT_NAME = args.out or "stage2_weight_sweep"
    _PATHS = _paths_mod.from_args(args)

    if not os.path.exists("/.dockerenv") and not args.no_docker:
        if not os.path.isfile(os.path.join(_REPO, "docker-compose.yml")):
            sys.exit(f"[sweep2] {_REPO}/docker-compose.yml is missing — "
                     "incomplete repo?")
        print("[sweep2] runs in the oscar-plus container — wrapping automatically.",
              flush=True)
        os.chdir(_REPO)  # docker compose needs the repo as CWD
        _cmd = ["docker", "compose", "run", "--rm"]
        # Smoke-Cap in den Container durchreichen (env wird sonst nicht vererbt).
        _cap = os.environ.get("MI3DOR_MAX_QUERIES_PER_CAT")
        if _cap:
            _cmd += ["-e", f"MI3DOR_MAX_QUERIES_PER_CAT={_cap}"]
        _cmd += ["oscar-plus", "python3", "/app/experiments/stage2_weight_sweep.py"]
        os.execvp("docker", _cmd + sys.argv[1:])

if _PATHS is None:  # imported as a module (no CLI): plain paths.yaml defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()

import numpy as np

from stage2_mi3dor import (  # noqa: E402
    cfg, to_category_label, _get_categories,
    _collect_filtered_cad_mesh_items, _make_query_factory, _collect_query_paths,
)
from eval_common import (  # noqa: E402
    build_pipeline, run_query, make_accum, update_accum, finalize_accum,
    load_ulip_query_cache, lookup_ulip_query_emb, pre_encode_ulip_queries,
)

BASE_W = (0.3, 0.4, 0.3)                # Produktionsgewichte (Berichtspunkt)
STEP = 0.1                              # 66 Punkte — wie der Stage-1-Sweep
OUT_DIR = os.path.join(_PATHS["runs_root"], _OUT_NAME)


def _simplex(step):
    n = int(round(1.0 / step))
    return [(round(i * step, 4), round(j * step, 4), round((n - i - j) * step, 4))
            for i in range(n + 1) for j in range(n + 1 - i)]


def assemble(out):
    """Score assembly EXACTLY as ScoreFusion._weighted_sum (step6:172-197):
    insertion order clip -> dino -> shape; clip=max(., dino.clip_score);
    ulip NaN is skipped (stays 0). Returns: (ids, clip, dino, ulip)."""
    order, idx = [], {}

    def ent(oid):
        if oid not in idx:
            idx[oid] = len(order)
            order.append(oid)
            for a in (c_v, d_v, u_v):
                a.append(0.0)
        return idx[oid]

    c_v, d_v, u_v = [], [], []
    if out["clip_res"]:
        for c in out["clip_res"].candidates:
            i = ent(c.object_id)
            c_v[i] = max(c_v[i], float(c.score))
    if out["dino_res_full"]:
        for c in out["dino_res_full"].candidates:
            i = ent(c.object_id)
            d_v[i] = max(d_v[i], float(c.dino_score))
            c_v[i] = max(c_v[i], float(getattr(c, "clip_score", 0.0) or 0.0))
    if out["shape_res_full"]:
        for c in out["shape_res_full"].candidates:
            i = ent(c.object_id)
            s = c.shape_score
            if not (isinstance(s, float) and np.isnan(s)):
                u_v[i] = max(u_v[i], float(s))
    # float64 wie die Produktions-Python-Floats — float32 kippte bei
    # Beinahe-Gleichstaenden die Reihenfolge (2/300 Proben am 24.09.)
    return (order, np.asarray(c_v, np.float64), np.asarray(d_v, np.float64),
            np.asarray(u_v, np.float64))


def _norm(v):
    """Min-max as in step6._minmax on NaN-free vectors (zeros included)."""
    lo, hi = float(v.min()), float(v.max())
    rng = hi - lo
    if rng <= 0:
        return np.zeros_like(v)
    return (v - lo) / rng


def rank_ids(ids, c, d, u, w):
    fused = w[0] * _norm(c) + w[1] * _norm(d) + w[2] * _norm(u)
    order = np.argsort(-fused, kind="stable")     # stabil == Python-sort
    return [(ids[i], float(fused[i])) for i in order]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    categories = _get_categories()
    cad_mesh_items = _collect_filtered_cad_mesh_items(categories)
    cfg.ulip2_use_partial_views = True          # Partial-Galerie (berichtete Config)
    cfg.result_folder = OUT_DIR
    print(f"[sweep2] pipeline (partial) over {len(categories)} categories ...")
    components = build_pipeline(cfg, cad_mesh_items=cad_mesh_items)
    pipeline_cfg, clip_retr, dino_rer, fusion_mod, shape_m = components

    ulip_cache = load_ulip_query_cache(cfg.ulip_query_cache_path)
    if ulip_cache is None:
        # Deliberately NOT saved to cfg.ulip_query_cache_path: the on-the-fly
        # encoding batches differently (32, native dtype) than
        # evaluation/precompute_ulip_query_embeddings.py (8, float16), which
        # shifts the embeddings in the fifth decimal. Persisting it under the
        # canonical name would silently feed later runs the wrong variant.
        print("[sweep2] ULIP query cache missing — encoding on the fly. For "
              "the reported numbers run "
              "evaluation/precompute_ulip_query_embeddings.py first.")
        ulip_cache = pre_encode_ulip_queries(_collect_query_paths(categories), shape_m)

    gallery_ids = set(getattr(dino_rer, "_ref_embeddings", {}) or {})
    if shape_m is not None and getattr(shape_m, "_cad_embeddings", None):
        gallery_ids |= set(shape_m._cad_embeddings)
    glc = {}
    for oid in gallery_ids:
        glc[to_category_label(oid)] = glc.get(to_category_label(oid), 0) + 1
    ref_objects = len(getattr(dino_rer, "_ref_embeddings", {}) or {})
    cad_objects = len(shape_m._cad_embeddings) if (shape_m and shape_m._cad_embeddings) else 0
    dino_k = max(cfg.dino_top_k, ref_objects) if ref_objects else cfg.dino_top_k
    ulip_k = max(cfg.ulip2_top_k, cad_objects) if cad_objects else cfg.ulip2_top_k
    clip_rows = len(getattr(clip_retr, "_desc_labels", []) or [])
    clip_k = max(cfg.clip_top_k, clip_rows, ref_objects, 1_000_000 if clip_rows == 0 else 0)

    # ---- Scoring: je Query EINMAL run_query; Produktions-Assemblierung ----
    print("[sweep2] scoring (run_query) + assembly as in _weighted_sum ...")
    cache = []                   # (ids, c, d, u, gt_label, |C|)
    n = 0
    for roi, gt_label, img_path, category, fname in _make_query_factory(categories)(cfg.topk[0]):
        try:
            emb = lookup_ulip_query_emb(ulip_cache, img_path)
            out = run_query(pipeline_cfg, clip_retr, dino_rer, fusion_mod, shape_m,
                            roi, cfg, ulip_query_emb=emb,
                            dino_full_top_k=dino_k, ulip_full_top_k=ulip_k,
                            clip_full_top_k=clip_k)
        except Exception as exc:                       # noqa: BLE001
            print(f"[sweep2][warn] query failed ({img_path}): {exc}")
            continue
        ids, c, d, u = assemble(out)
        cache.append((ids, c, d, u, gt_label, glc.get(gt_label, 0)))
        n += 1
        if n % 1000 == 0:
            print(f"[sweep2]   {n} queries scored", flush=True)
    print(f"[sweep2] {len(cache)} queries scored")

    # ---- Sweep (66 Punkte) ------------------------------------------------
    def eval_point(w):
        acc = make_accum()
        for ids, c, d, u, gt, nr in cache:
            update_accum(acc, rank_ids(ids, c, d, u, w), gt,
                         to_category_label, cfg.TOP_F, nr)
        m = finalize_accum(acc)
        return float(m["FT_mean"]), float(m["NN_accuracy"])

    bft, bnn = eval_point(BASE_W)

    rows, best = [], (-1.0, None)
    for w in _simplex(STEP):
        ft, nn = eval_point(w)
        rows.append((*w, round(ft, 4), round(nn, 4)))
        if ft > best[0]:
            best = (ft, w)
    out_csv = os.path.join(OUT_DIR, "weight_sweep.csv")
    with open(out_csv, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["w_clip", "w_dino", "w_ulip", "FT", "NN"])
        wtr.writerows(rows)

    manifest = dict(
        ts=_dt.datetime.now().isoformat(timespec="seconds"),
        git=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, cwd=_REPO).stdout.strip(),
        grid=f"{len(rows)} points, step {STEP} (as in the Stage-1 sweep)",
        config=dict(partial=True, dino_pooling=os.environ.get("MI3DOR_DINO_POOLING", "mean"),
                    num_views=int(os.environ.get("MI3DOR_NUM_VIEWS", "42")),
                    weights_prod=BASE_W, ulip_ckpt=cfg.ulip2_checkpoint
                    if hasattr(cfg, "ulip2_checkpoint") else ""),
        queries=len(cache),
        fusion="pipeline/step6_fusion._weighted_sum (assembly replicated)")
    json.dump(manifest, open(os.path.join(OUT_DIR, "weight_sweep_manifest.json"),
                             "w"), indent=1)
    print(f"[sweep2] {len(rows)} points -> {out_csv}")
    print(f"[sweep2] BASE FT={bft:.4f} | optimum FT={best[0]:.4f} at w={best[1]} "
          f"(Δ={best[0]-bft:+.4f})")


if __name__ == "__main__":
    main()
