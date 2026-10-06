#!/usr/bin/env python3
"""
stage2_categories.py — Stage-2 category table (Fig. 6.3).

Per-category NN of the three isolated channels plus fusion, and the count
"fusion worse than the best single channel" (expected: 9 of 21, vase −0.132).
The fusion column is ``clip_dino_ulip_full`` of the given run — for the
published table the full-mesh run. Plain JSON reading, runs directly on the
host without a container.

    python3 experiments/stage2_categories.py
    python3 experiments/stage2_categories.py --results runs/stage2_mi3dor/partial --csv table.csv

Default input: ``<runs_root>/stage2_mi3dor/fullmesh`` (produce with:
``python3 experiments/stage2_mi3dor.py --gallery fullmesh``). Without ``--csv``
the table is written to ``<runs_root>/stage2_categories/categories.csv`` so
that every run leaves an artefact behind.
"""
import argparse
import collections
import csv
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

ARMS = ["clip_only", "dino_only_full", "ulip_only_full", "clip_dino_ulip_full"]
LBL = dict(zip(ARMS, ["text", "view", "shape", "fusion"]))


def main():
    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None,
                    help="run folder with results_topk_15.json (default: "
                         "<runs_root>/stage2_mi3dor/fullmesh)")
    ap.add_argument("--csv", help="write the table as CSV there (default: "
                                  "<runs_root>/stage2_categories/categories.csv)")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()
    _paths = _paths_mod.from_args(args)

    results_dir = args.results or os.path.join(_paths["runs_root"],
                                               "stage2_mi3dor", "fullmesh")
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(_ROOT, results_dir)
    p = os.path.join(results_dir, "results_topk_15.json")
    if not os.path.isfile(p):
        sys.exit(f"missing: {p}\nExpected a Stage-2 run folder with "
                 "results_topk_15.json — produce with:\n"
                 "  python3 experiments/stage2_mi3dor.py --gallery fullmesh")
    data = json.load(open(p))

    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in data:
        for a in ARMS:
            rp = r["eval_trace"]["arms"][a]["rel_positions"]
            acc[r["gt"]][a].append(1.0 if (rp and rp[0] == 1) else 0.0)

    rows = []
    for cat, per in acc.items():
        vals = {a: sum(per[a]) / len(per[a]) for a in ARMS}
        # Gleichstand: spaeterer Kanal (text<view<shape) gewinnt — wie die
        # eingefrorene Tabelle (car: view==shape==1.0 -> shape).
        best_v, _, best_n = max((vals[a], i, LBL[a])
                                for i, a in enumerate(ARMS[:3]))
        rows.append((cat, len(per[ARMS[0]]), vals["clip_only"],
                     vals["dino_only_full"], vals["ulip_only_full"],
                     best_n, best_v, vals["clip_dino_ulip_full"],
                     round(vals["clip_dino_ulip_full"] - best_v, 3)))
    # Reihenfolge der eingefrorenen Tabelle: NN_fusion absteigend,
    # Kategorie als Zweitschluessel (car vor guitar bei 1.0/1.0).
    rows.sort(key=lambda r: (-r[7], r[0]))

    print(f"{'category':<12}{'n':>5}{'text':>8}{'view':>8}{'shape':>8}"
          f"   {'best':<8}{'fusion':>8}{'delta':>9}")
    for r in rows:
        print(f"{r[0]:<12}{r[1]:>5}{r[2]:>8.3f}{r[3]:>8.3f}{r[4]:>8.3f}"
              f"   {r[5]:<2} {r[6]:<5.3f}{r[7]:>8.3f}{r[8]:>+9.3f}")
    neg = [r for r in rows if r[8] < -1e-9]
    print(f"\nFusion worse than the best single channel: {len(neg)} of {len(rows)}"
          f" (largest loss {min(r[8] for r in rows):+.3f})")
    win = collections.Counter(r[5] for r in rows)
    print("Best single channel per category:", dict(win))

    csv_path = args.csv or os.path.join(_paths["runs_root"],
                                        "stage2_categories", "categories.csv")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        # CRLF (csv default), wie results/stage2_mi3dor/categories.csv.
        w = csv.writer(fh)
        w.writerow(["category", "NN_text", "NN_view", "NN_shape",
                    "NN_fusion_fullmesh", "best_single", "best_single_NN",
                    "delta_fusion_minus_best"])
        w.writerows([[r[0], r[2], r[3], r[4], r[7], r[5], r[6], r[8]]
                     for r in rows])
    print(f"written: {csv_path}")


if __name__ == "__main__":
    main()
