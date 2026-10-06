#!/usr/bin/env python3
"""
stage1_weight_sweep.py
======================
Stage-1 weight sweep (SHREC'18) — sensitivity of the fusion weights on the
PRODUCTION FUSION, not on a re-implementation.

66 simplex points (step 0.1) on the final 42v/k5 configuration; each point via
``AblationSpec`` + ``make_fusion_module`` + ``derive_ranking`` +
``score_official``/``score_depth_matched`` of the Stage-1 driver
(``experiments/stage1_shrec18.py``) — hence byte-identical to the regular
arms. Channels with weight 0 are dropped from the spec exactly as in its
``run_weight_sweep``, so the corners of the simplex are the isolated arms.

One invocation = one shape pass (``--track``):

  ``--track pc``     S_shape from the query point cloud (pass ``ulip_pc_rgb``,
                     production configuration ``fusion`` = E1c_full_fusion)
  ``--track cross``  S_shape from the ULIP-2 image tower (pass ``ulip_cross_rgb``,
                     production configuration ``fusion_cross`` = E7_ulip2_cross)

Configuration names: internally the arms are named like the driver's ablation
keys (``E1c_full_fusion``); reported and named in ``results/`` they carry their
release name (``fusion``) — see
``evaluation/configuration_names.py``.

Tier 2, no GPU run: the computation runs exclusively on the persisted score
stores (``<source>/_cache/scores_*.pt``) of a Stage-1 run. If they are missing
the script aborts with the generating command.

How to run
----------
    python3 experiments/stage1_weight_sweep.py --track pc
    python3 experiments/stage1_weight_sweep.py --track cross
    # Smoke (result NOT comparable):
    #   S1SWEEP_LIMIT=25 S1SWEEP_POINTS=3 \
    #       python3 experiments/stage1_weight_sweep.py --track pc

Input (``--source``, default ``<runs_root>/stage1_shrec18``): the folder of a
Stage-1 run with ``_cache/scores_base.pt`` and the score store of the chosen
shape pass. Produce it with:

    python3 experiments/stage1_shrec18.py                   # pc
    python3 experiments/stage1_shrec18.py --shape-mode cross # cross

Output (``--out``, default ``<runs_root>/stage1_weight_sweep``):
``weight_sweep_<track>.csv`` and ``weight_sweep_<track>_manifest.json``
(commit, configuration, optima). The frozen record of the thesis
lives in ``results/stage1_shrec18/weight_sweep_{pc,cross}.csv``.

Outside the container the script wraps itself automatically into
``docker compose run`` (the score stores are read with torch).
"""
import csv
import datetime as _dt
import json
import os
import subprocess
import sys

# ===========================================================================
# 0. CLI prelude
# ===========================================================================
# Laeuft VOR den schweren Imports (stage1_shrec18 zieht numpy, run_pass torch),
# damit ``--help`` und die Eingabepruefung auf dem Host ohne
# Container-Abhaengigkeiten funktionieren.
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if os.path.join(_REPO, "evaluation") not in sys.path:
    sys.path.insert(1, os.path.join(_REPO, "evaluation"))

STEP = 0.1                       # Simplex-Schrittweite -> 66 Punkte
GEOM_K = 5                       # Tiefe der Tabelle-B-Metriken (42v/k5-Config)
# Pro Track: der Score-Store des Shape-Passes und die Flags, mit denen der
# Stage-1-Lauf ihn erzeugt (fuer die Fehlermeldung in _require_stores).
TRACKS = {"pc":    {"pass": "ulip_pc_rgb",   "flags": ""},
          "cross": {"pass": "ulip_cross_rgb", "flags": " --shape-mode cross"}}
# Smoke-Regler (Env, damit sie in den Container durchgereicht werden koennen).
LIMIT = int(os.environ.get("S1SWEEP_LIMIT", "0") or "0") or None
MAX_PTS = int(os.environ.get("S1SWEEP_POINTS", "0") or "0") or None

_PATHS = None
_TRACK = ""
_SOURCE = ""
_OUT_DIR = ""


def _score_store(source: str, pass_key: str) -> str:
    """Path of the score store as ``stage1_shrec18.run_pass`` writes it."""
    tag = f"_n{LIMIT}" if LIMIT else ""
    return os.path.join(source, "_cache", f"scores_{pass_key}{tag}.pt")


def _require_stores(source: str, track: str) -> None:
    """Abort with the generating command if a score store is missing (no GPU run)."""
    g = TRACKS[track]
    missing = [p for p in (_score_store(source, "base"),
                           _score_store(source, g["pass"]))
               if not os.path.isfile(p)]
    if not missing:
        return
    smoke = f" --limit {LIMIT}" if LIMIT else ""
    sys.exit(f"[s1sweep] score store missing:\n  "
             + "\n  ".join(missing)
             + f"\nThe sweep computes exclusively on persisted scores. "
               f"Run the matching Stage-1 run first:\n"
               f"  python3 experiments/stage1_shrec18.py{g['flags']}{smoke}\n"
               f"(or point --source at a run that holds "
               f"_cache/scores_base.pt and _cache/scores_{g['pass']}.pt).")


if __name__ == "__main__":
    import argparse

    from pipeline import paths as _paths_mod

    _parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _parser.add_argument("--track", required=True, choices=sorted(TRACKS),
                         help="shape pass of the sweep: pc = query point cloud "
                              "(configuration fusion / E1c_full_fusion), cross = "
                              "ULIP-2 image tower on the query crop "
                              "(fusion_cross / E7_ulip2_cross)")
    _parser.add_argument("--source", default=None, metavar="RUN_DIR",
                         help="Stage-1 run with the score stores in _cache/ "
                              "(default: <runs_root>/stage1_shrec18)")
    _parser.add_argument("--out", default=None,
                         help="target folder for CSV + manifest "
                              "(default: <runs_root>/stage1_weight_sweep)")
    _parser.add_argument("--no-docker", action="store_true",
                         help="do not auto-wrap into the oscar-plus container")
    _paths_mod.add_path_args(_parser)
    _args = _parser.parse_args()

    # Gleiche Env wie der Stage-1-Treiber: die Score-Stores sind mit
    # SHREC_DINO_POOLING=mean entstanden, also kann ein (durch den
    # Eingabecheck ohnehin verhinderter) Neu-Lauf nicht abweichen.
    os.environ.update({"PYTHONHASHSEED": "0", "SHREC_DINO_POOLING": "mean"})
    _PATHS = _paths_mod.from_args(_args)
    _TRACK = _args.track
    _SOURCE = _args.source or os.path.join(_PATHS["runs_root"],
                                           "stage1_shrec18")
    _OUT_DIR = _args.out or os.path.join(_PATHS["runs_root"],
                                         "stage1_weight_sweep")
    _require_stores(_SOURCE, _TRACK)     # schon auf dem Host, vor dem Wrappen

    if not os.path.exists("/.dockerenv") and not _args.no_docker:
        if not os.path.isfile(os.path.join(_REPO, "docker-compose.yml")):
            sys.exit(f"[s1sweep] {_REPO}/docker-compose.yml missing — "
                     "repo incomplete?")
        print("[s1sweep] runs in the oscar-plus container — wrapping automatically.",
              flush=True)
        os.chdir(_REPO)  # docker compose needs the repo as CWD
        _cmd = ["docker", "compose", "run", "--rm"]
        for _var in ("S1SWEEP_LIMIT", "S1SWEEP_POINTS"):   # Smoke-Regler
            if os.environ.get(_var):
                _cmd += ["-e", f"{_var}={os.environ[_var]}"]
        _cmd += ["oscar-plus", "python3", "/app/experiments/stage1_weight_sweep.py"]
        os.execvp("docker", _cmd + sys.argv[1:])

if _PATHS is None:   # als Modul importiert: reine paths.yaml-Defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()
    _SOURCE = os.path.join(_PATHS["runs_root"], "stage1_shrec18")
    _OUT_DIR = os.path.join(_PATHS["runs_root"], "stage1_weight_sweep")

import stage1_shrec18 as E                                          # noqa: E402
from stage1_shrec18 import (                                        # noqa: E402
    AblationSpec, make_fusion_module, derive_ranking,
    run_pass, validate_inputs, load_official_gt, prepare_queries,
    score_official, score_depth_matched)

E.GEOM_K = GEOM_K
# Der Treiber loest seine Modulkonstanten beim Import aus paths.yaml auf (die
# CLI-Overrides dieses Skripts sieht er nicht) — die beiden von uns benutzten
# Eingaben deshalb explizit nachziehen.
E.OFFICIAL_DIR = os.path.join(_PATHS["datasets_root"], "shrec18",
                              "shrec18_official")

PATHS = {"data_root":   os.path.join(_PATHS["datasets_root"], "shrec18",
                                     "shrec18_full"),
         "images_dir":  os.path.join(_PATHS["gallery_root"], "shrec18"),
         "desc_file":   os.path.join(_PATHS["cad_root"], "shrec18",
                                     "descriptions_attributes.json"),
         "results_root": _SOURCE,
         "stage1_root": os.path.join(_PATHS["datasets_root"], "shrec18",
                                     "stage1")}


def simplex(step):
    n = int(round(1.0 / step))
    return [(round(i * step, 4), round(j * step, 4), round((n - i - j) * step, 4))
            for i in range(n + 1) for j in range(n + 1 - i)]


def main():
    mode, g = _TRACK, TRACKS[_TRACK]
    os.makedirs(_OUT_DIR, exist_ok=True)
    object_ids = validate_inputs(PATHS, True)
    gt = load_official_gt(PATHS["data_root"], PATHS["stage1_root"])
    cad_labels, freqs = gt["cad"], gt["freqs"]
    index = prepare_queries(PATHS["data_root"], PATHS["stage1_root"], gt)
    if LIMIT:
        index = index[:LIMIT]
    cad_dir = os.path.join(PATHS["data_root"], "cad")
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=_REPO).stdout.strip()
    except Exception:                                      # noqa: BLE001
        git = ""

    stores = {pk: run_pass(pk, PATHS, index, object_ids, LIMIT, resume=True)
              for pk in ("base", g["pass"])}
    full = {"clip": ("base", None), "dino": ("base", 42),
            "shape": (g["pass"], None)}

    def eval_point(wt, wv, ws):
        chan = {ch: full[ch] for ch, w in
                zip(("clip", "dino", "shape"), (wt, wv, ws)) if w > 0}
        spec = AblationSpec(name=f"W_{wt}_{wv}_{ws}", group="WSWEEP2",
                            question="weight sensitivity v2",
                            channels=chan, weights=(wt, wv, ws))
        fm = make_fusion_module(spec)
        s = dict(nDCG=0.0, hit1=0.0, NN_cat=0.0, MRR=0.0)
        nq = 0
        for q in index:
            ql = tuple(q["category"])
            if freqs.get(ql[0], 0) == 0:
                continue
            ranking = derive_ranking(spec, q["id"], stores, object_ids,
                                     fm, cad_dir, None)
            ranked = [object_ids[i] for i in ranking]
            mo = score_official(ranked, ql, cad_labels, freqs)
            if mo is None:
                continue
            mb = score_depth_matched(ranked, ql, cad_labels, E.GEOM_K)
            s["nDCG"] += mo["nDCG"]
            s["hit1"] += mb["NN_sub"]
            s["NN_cat"] += mb["NN_cat"]
            s["MRR"] += mb["MRR"]
            nq += 1
        return {k: v / nq for k, v in s.items()}, nq

    base, nq = eval_point(0.3, 0.4, 0.3)
    print(f"[s1sweep:{mode}] BASE (0.3, 0.4, 0.3): nDCG={base['nDCG']:.4f} "
          f"hit@1={base['hit1']:.4f} (n={nq})", flush=True)

    rows = []
    pts = simplex(STEP)[:MAX_PTS] if MAX_PTS else simplex(STEP)
    for (wt, wv, ws) in pts:
        m, _ = eval_point(wt, wv, ws)
        rows.append({"w_text": wt, "w_view": wv, "w_shape": ws,
                     **{k: round(v, 4) for k, v in m.items()}})
        print(f"[s1sweep:{mode}] w=({wt},{wv},{ws}) nDCG={m['nDCG']:.4f} "
              f"hit@1={m['hit1']:.4f}", flush=True)
    out_csv = os.path.join(_OUT_DIR, f"weight_sweep_{mode}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["w_text", "w_view", "w_shape",
                                          "nDCG", "hit1", "NN_cat", "MRR"])
        w.writeheader()
        w.writerows(rows)
    best_h = max(rows, key=lambda r: r["hit1"])
    best_n = max(rows, key=lambda r: r["nDCG"])
    noshape = [r for r in rows if r["w_shape"] == 0]
    manifest = dict(ts=_dt.datetime.now().isoformat(timespec="seconds"),
                    git=git, mode=mode, source=_SOURCE,
                    grid=f"{len(rows)} points, step {STEP}",
                    config="42v/k5 (SHAPE_AGG_VIEWS=42, ulip_view_topk=5, "
                           f"DINO 42v, partial references), GEOM_K={GEOM_K}",
                    base=base,
                    best_hit1=best_h, best_nDCG=best_n,
                    best_without_shape_hit1=max(noshape, key=lambda r: r["hit1"]) if noshape else None,
                    best_without_shape_nDCG=max(noshape, key=lambda r: r["nDCG"]) if noshape else None)
    json.dump(manifest, open(os.path.join(
        _OUT_DIR, f"weight_sweep_{mode}_manifest.json"), "w"), indent=1)
    print(f"[s1sweep:{mode}] {len(rows)} points -> {out_csv}", flush=True)
    print(f"[s1sweep:{mode}] reference of the thesis: "
          f"results/stage1_shrec18/weight_sweep_{mode}.csv", flush=True)


if __name__ == "__main__":
    main()
