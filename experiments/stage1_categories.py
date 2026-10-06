#!/usr/bin/env python3
"""
stage1_categories.py — which Stage-1 arm wins in WHICH category?

    # the 2x2 matrix query mode x gallery representation (isolated shape channel)
    python3 experiments/stage1_categories.py --preset shape-matrix

    # any two configurations, sorted by largest gap
    # (internal ablation key OR release name, both are recognised)
    python3 experiments/stage1_categories.py \\
        E1_shape_only E2b_fullmesh_shape_only --metric nDCG
    python3 experiments/stage1_categories.py \\
        shape_only shape_only_fullmesh --metric nDCG

    # sorted by category instead of by gap, CSV somewhere else
    python3 experiments/stage1_categories.py --preset shape-matrix \\
        --sort category --csv runs/shape_matrix_by_category.csv

Reads the per-query results of each arm — next to the metrics they carry the GT
category of every query, so the script needs no external GT file.
An aggregate only says WHICH arm is better; only the breakdown says WHERE and
hence why.

Paired by query id: both arms see the same 2101 queries, so the mean difference
per category is formed over the same queries, not as the difference of two
independent means.

Input (``--results-root``, default ``<runs_root>/stage1_shrec18``): a Stage-1
run or the frozen record itself (``--results-root
results/stage1_shrec18``). Both layouts AND both naming systems are read
— ``<config>/results_per_query.json`` as ``experiments/stage1_shrec18.py``
writes it and the flat ``per_query/<config>.json`` of the record, each under
the internal ablation key or the release name (see
``evaluation/configuration_names.py``). Pure JSON reading, runs without a
container directly on the host.

Output: the table on stdout and always a CSV — without ``--csv`` to
``<runs_root>/stage1_categories/categories_<preset or A_vs_B>.csv``, so that
every run leaves an artefact behind. In the CSV the compared configurations
appear under their RELEASE name (columns ``configuration_a`` /
``configuration_b``), as in ``results/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from typing import Dict, List, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evaluation.configuration_names import to_internal, to_release  # noqa: E402

PRESETS = {
    # Die vier Zellen der Matrix, isolierter Shape-Kanal.
    "shape-matrix": [
        ("E1_shape_only", "pc x partial"),
        ("E2b_fullmesh_shape_only", "pc x full-mesh"),
        ("E7_ulip2_cross_shape_only", "cross x partial"),
        ("E7_ulip2_cross_fullmesh_shape_only", "cross x full-mesh"),
    ],
    "fusion": [
        ("E1c_full_fusion", "BASE (pc, partial)"),
        ("E2b_fullmesh", "pc x full-mesh"),
        ("E7_ulip2_cross_fullmesh", "cross x full-mesh"),
    ],
}

_ROOT_DIR = ""        # --results-root, von main() gesetzt


def per_query_path(arm: str) -> Optional[str]:
    """Per-query JSON of one configuration, whatever it is called on disk.

    Four candidates: run layout (``<name>/results_per_query.json``) before
    archive layout (``per_query/<name>.json``), internal ablation key before
    release name. ``arm`` may be given in either naming system.
    """
    internal = to_internal(arm, "stage1")
    for name in dict.fromkeys((internal, to_release(internal, "stage1"))):
        for cand in (os.path.join(_ROOT_DIR, name, "results_per_query.json"),
                     os.path.join(_ROOT_DIR, "per_query", f"{name}.json")):
            if os.path.isfile(cand):
                return cand
    return None


def configuration_name(arm: str) -> str:
    """Release name this configuration is reported under."""
    return to_release(to_internal(arm, "stage1"), "stage1")


def load(arm: str) -> Dict[str, dict]:
    f = per_query_path(arm)
    if f is None:
        return {}
    d = json.load(open(f))
    rows = d if isinstance(d, list) else list(d.values())
    return {r["id"]: r for r in rows if "id" in r}


def category_of(rec) -> str:
    """The category is stored as the string of a list ("['keyboard', ...]").

    First entry = category, second = sub-category. We group on the category; a
    literal_eval would be fragile, so the string is cleaned up instead.
    """
    c = rec.get("category")
    if isinstance(c, (list, tuple)):
        return str(c[0])
    s = str(c or "?").strip("[]")
    return s.split(",")[0].strip().strip("'\"") or "?"


def compare(a: str, b: str, metric: str):
    ra, rb = load(a), load(b)
    common = sorted(set(ra) & set(rb))
    if not common:
        return None, []
    per: Dict[str, List[float]] = {}
    for q in common:
        va, vb = ra[q].get(metric), rb[q].get(metric)
        if va is None or vb is None:
            continue
        per.setdefault(category_of(ra[q]), []).append(float(va) - float(vb))
    rows = [(c, len(v), statistics.fmean(v),
             statistics.fmean([1.0 if d > 0 else 0.0 for d in v]))
            for c, v in per.items()]
    overall = statistics.fmean([d for v in per.values() for d in v])
    return overall, sorted(rows, key=lambda r: -abs(r[2]))


def main():
    global _ROOT_DIR
    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*",
                    help="exactly two configurations (internal ablation key "
                         "or release name), or --preset")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--results-root", default=None,
                    help="Stage-1 run or frozen record holding the "
                         "configurations (default: <runs_root>/stage1_shrec18)")
    ap.add_argument("--metric", default="nDCG",
                    help="nDCG | NN_sub | NN_cat | MRR | AP | nDCG_K")
    ap.add_argument("--sort", choices=["delta", "category"], default="delta")
    ap.add_argument("--top", type=int, default=0, help="only the N largest gaps")
    ap.add_argument("--csv", help="CSV target (default: <runs_root>/"
                                  "stage1_categories/categories_<preset>.csv)")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()
    _paths = _paths_mod.from_args(args)

    _ROOT_DIR = args.results_root or os.path.join(_paths["runs_root"],
                                                  "stage1_shrec18")
    if not os.path.isabs(_ROOT_DIR):
        _ROOT_DIR = os.path.join(_REPO, _ROOT_DIR)

    if args.preset:
        arms = PRESETS[args.preset]
        missing = [a for a, _ in arms if per_query_path(a) is None]
        if missing:
            print(f"[warn] missing configurations (not run yet?): "
                  f"{[configuration_name(a) for a in missing]}\n")
        arms = [x for x in arms if x[0] not in missing]
        if len(arms) < 2:
            sys.exit(f"[cat] fewer than two configurations of preset "
                     f"'{args.preset}' are present in {_ROOT_DIR} (expected "
                     f"<config>/results_per_query.json or "
                     f"per_query/<config>.json, internal or release name)."
                     f"\nProduce the arms — one per invocation, flag->arm table "
                     f"in the driver's --help:\n"
                     f"  python3 experiments/stage1_shrec18.py --weights 0,0,1\n"
                     f"  python3 experiments/stage1_shrec18.py --weights 0,0,1 "
                     f"--gallery fullmesh\n"
                     f"  ...\nThe frozen run of the thesis lives in "
                     f"results/stage1_shrec18 "
                     f"(--results-root results/stage1_shrec18).")
        pairs = [(arms[0], b) for b in arms[1:]]
        stem = args.preset.replace("-", "_")
    elif len(args.arms) == 2:
        a, b = args.arms
        pairs = [((a, configuration_name(a)), (b, configuration_name(b)))]
        stem = f"{configuration_name(a)}_vs_{configuration_name(b)}"
    else:
        ap.error("give either --preset or exactly two configurations")

    out_rows = []
    for (a, la), (b, lb) in pairs:
        overall, rows = compare(a, b, args.metric)
        ca, cb = configuration_name(a), configuration_name(b)
        if overall is None:
            print(f"!! no common queries: {ca} vs {cb}")
            continue
        if args.sort == "category":
            rows.sort(key=lambda r: r[0])
        if args.top:
            rows = rows[:args.top]
        _ta = la if la == ca else f"{la} [{ca}]"
        _tb = lb if lb == cb else f"{lb} [{cb}]"
        print(f"\n=== {_ta}  minus  {_tb}   ({args.metric}) ===")
        print(f"  Overall: {overall:+.4f}\n")
        print(f"  {'category':<22}{'n':>6}{'Δ ' + args.metric:>12}{'share won':>18}")
        for c, n, d, w in rows:
            bar = "+" * min(int(abs(d) * 40), 18)
            print(f"  {c:<22}{n:>6}{d:>+12.4f}{w:>17.0%}  {bar if d > 0 else ''}")
            out_rows.append((ca, cb, args.metric, c, n, round(d, 5), round(w, 4)))
        print(f"\n  Positive = '{_ta}' is better there.")

    if not out_rows:
        sys.exit(f"[cat] no common queries — nothing to write "
                 f"(arms from {_ROOT_DIR} incomplete?).")

    p = args.csv or os.path.join(_paths["runs_root"], "stage1_categories",
                                 f"categories_{stem}.csv")
    if not os.path.isabs(p):
        p = os.path.join(_REPO, p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["configuration_a", "configuration_b", "metric", "category",
                    "n", "delta_mean", "share_a_wins"])
        w.writerows(out_rows)
    print(f"\n  CSV: {p}")


if __name__ == "__main__":
    main()
