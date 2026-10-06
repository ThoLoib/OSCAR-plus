#!/usr/bin/env python3
"""
stage1_shortlist_analysis.py
============================
**Conditional top-1 accuracy inside the geometry shortlist** — SHREC'18 and BOP.

The question
------------
Geometric re-ranking helps on SHREC'18 (+0.13 hit@1) and hurts on BOP 3a
(−0.06 R@1), although the two aggregate retrieval numbers are close (hit@1 0.341
vs R@1 0.482).  The aggregate therefore cannot be what decides it.  Re-ranking
only ever permutes the **shortlist**: it cannot pull in a target that is not in
the top K, and it cannot lose one that is below it.  So the quantity that decides
is the accuracy *conditioned on the target being reachable*:

    conditional top-1  =  Top-1 / Recall@K

measured for the **incumbent** (the fused score that produced the shortlist) and
for the **challenger** (the geometry signal that re-orders it).  Geometric
re-ranking is worth it exactly when the challenger's conditional top-1 beats the
incumbent's.  The **headroom** ``Recall@K − Top-1`` is the upper bound on what
any re-ranking inside that shortlist can win, and it is knowable **before** a
single registration is computed — both numbers already sit in the retrieval run.

Both rows of a benchmark share one denominator, the **incumbent's** Recall@K:
re-ranking inside the shortlist cannot change whether the target is in it, so the
challenger's Recall@K is the same number (the script verifies this and says so if
it is not).

Not comparable between the two benchmarks is the absolute level — SHREC counts
sub-category hits, BOP the exact target instance.  Only the difference **within**
one benchmark carries meaning.

Provenance of the definition
----------------------------
"Weak ranking" is defined operationally: conditional top-1 inside the
shortlist = Top-1 / Recall@K, checkable in advance via the headroom
Recall@K − Top-1. The reference values are
SHREC K=50 incumbent 0.396 / challenger 0.548 / headroom 0.520 and
BOP 3a cross K=5 incumbent 0.657 / challenger 0.577 / headroom 0.251.
There is **no archived CSV** for this table — it was derived by hand — so unlike
the other reading scripts in this folder there is nothing to diff against
byte-for-byte; the documented values above are the only reference.

Where the four numbers come from
--------------------------------
=====================  ===================================================
SHREC'18, Top-1        mean ``NN_sub`` over the per-query records of the arm
SHREC'18, Recall@K     ``hit_sub@K`` from the run's arm overview
                       (``all_arms.csv``).  **Not** derivable from the
                       per-query records: those store only the top-10, and
                       K = 50 lies below that.
BOP, Top-1/Recall@K    share of instances with ``target_rank <= 1`` / ``<= K``
                       over the per-instance records, cross-checked against
                       the run summary's published ``recall@1``.
=====================  ===================================================

Pure reading: no GPU, no container — only JSON/CSV of existing Stage-1 and
Stage-3 runs (host Python, no third-party imports).

How to run
----------
    python3 experiments/stage1_shortlist_analysis.py
    python3 experiments/stage1_shortlist_analysis.py \\
        --shrec-root results/stage1_shrec18 --bop-root results/stage3_bop
    # other pair, other depth (K must exist as hit_sub@K in the arm overview)
    python3 experiments/stage1_shortlist_analysis.py \\
        --shrec-arms fusion_fullmesh,fusion_fullmesh_geo_trimmed_distance \\
        --shrec-k 50

Input (``--shrec-root``, default ``<runs_root>/stage1_shrec18``;
``--bop-root``, default ``<runs_root>/stage3_bop``): runs or the frozen records
themselves (``results/stage1_shrec18`` / ``results/stage3_bop``).  Both layouts
AND both naming systems are accepted per stage — see
``evaluation/configuration_names.py``; a benchmark whose input is missing is
skipped with the command that produces it.

Output (``--out``, default ``<runs_root>/stage1_shortlist_analysis/
conditional_top1.csv``): one row per (benchmark, role).  ``headroom`` is a
property of the shortlist and is filled on the incumbent row only; ``delta_pp``
is the challenger's advantage in percentage points and is filled on the
challenger row only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evaluation.configuration_names import to_internal, to_release  # noqa: E402

HEADER = ("benchmark", "shortlist_k", "role", "configuration", "n", "top1",
          "recall_at_k", "conditional_top1", "headroom", "delta_pp")
INCUMBENT, CHALLENGER = "incumbent", "challenger"

# --- SHREC'18 (Stage 1) -----------------------------------------------------
SHREC_LABEL = "SHREC'18"
SHREC_ARMS = ("fusion", "fusion_geo_trimmed_distance")   # incumbent, challenger
SHREC_K = 50                    # BASE geometry shortlist depth (--geom-k 50)
SHREC_TOP1 = "NN_sub"           # the Stage-1 hit@1 (sub-category grade)
OVERVIEW_CSV = ("all_arms.csv", "stage1_summary_arms.csv")
OVERVIEW_TOP1 = ("NN_sub_hit1", "NN_sub")
_SHREC_HOWTO = ("  python3 experiments/stage1_shrec18.py\n"
                "  python3 experiments/stage1_shrec18.py --geometry "
                "trimmed-distance --geom-k 50\n"
                "The frozen run of the thesis lives in "
                "results/stage1_shrec18 (--shrec-root results/stage1_shrec18).")

# --- BOP 3a (Stage 3) -------------------------------------------------------
BOP_LABEL = "BOP 3a cross"
BOP_ARMS = ("retrieval_cross_partial", "retrieval_cross_geo_trimmed_distance")
BOP_K = 5                       # Stage-3 geometry shortlist depth (--dgedi-top-k 5)
BOP_DATASETS = ("ycbv", "tless", "lmo")
_BOP_HOWTO = ("  python3 experiments/stage3_bop.py --mode retrieval --query "
              "cross --gallery partial\n"
              "  python3 experiments/stage3_bop.py --mode retrieval --query "
              "cross --gallery partial --geometry trimmed-distance\n"
              "The frozen run of the thesis lives in results/stage3_bop "
              "(--bop-root results/stage3_bop).")


# ---------------------------------------------------------------------------
# Stage 1 — SHREC'18
# ---------------------------------------------------------------------------

def stage1_per_query(root: str, arm: str) -> Optional[List[dict]]:
    """Per-query records of one Stage-1 configuration, however they are named.

    Four candidates, because both the layout and the naming may differ: run
    layout (``<name>/results_per_query.json``) before archive layout
    (``per_query/<name>.json``), and the internal ablation key before the release
    configuration name.
    """
    internal = to_internal(arm, "stage1")
    for name in dict.fromkeys((internal, to_release(internal, "stage1"))):
        for cand in (os.path.join(root, name, "results_per_query.json"),
                     os.path.join(root, "per_query", f"{name}.json")):
            if os.path.isfile(cand):
                with open(cand) as fh:
                    d = json.load(fh)
                return d if isinstance(d, list) else list(d.values())
    return None


def stage1_overview(root: str) -> Tuple[Dict[str, dict], str]:
    """(configuration -> overview row, path) of the run's arm overview."""
    for name in OVERVIEW_CSV:
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        with open(p, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            key = list(rows[0])[0]
            return {r[key]: r for r in rows if r.get(key)}, p
    return {}, ""


def stage1_side(root: str, arm: str, k: int,
                overview: Dict[str, dict]) -> Tuple[str, int, float]:
    """(release name, n, Top-1) of one Stage-1 arm, self-checked."""
    recs = stage1_per_query(root, arm)
    release = to_release(to_internal(arm, "stage1"), "stage1")
    if not recs:
        sys.exit(f"[shortlist] {release}: no per-query file under {root} "
                 f"(expected <config>/results_per_query.json or "
                 f"per_query/<config>.json).\nProduce the arm:\n" + _SHREC_HOWTO)
    top1 = sum(float(e[SHREC_TOP1]) for e in recs) / len(recs)
    row = overview.get(release) or overview.get(to_internal(arm, "stage1")) or {}
    col = next((c for c in OVERVIEW_TOP1 if c in row), None)
    if col:
        published = float(row[col])
        if abs(top1 - published) > 1e-4:
            sys.exit(f"[shortlist] ABORT: {release}: own Top-1 "
                     f"{top1:.4f} differs from published {published} — the "
                     f"per-query file does not match the arm overview.")
    return release, len(recs), top1


def stage1_recall_at_k(root: str, arm: str, k: int, overview: Dict[str, dict],
                       overview_path: str) -> float:
    """Recall@K of one Stage-1 arm = ``hit_sub@K`` of the arm overview.

    Deliberately NOT derived from the per-query records: those hold the top-10
    only, so K = 50 is out of reach there. The overview value is the run's own
    ``metrics_depth`` number, rounded to 4 decimals in the record.
    """
    release = to_release(to_internal(arm, "stage1"), "stage1")
    row = overview.get(release) or overview.get(to_internal(arm, "stage1"))
    if not row:
        sys.exit(f"[shortlist] {release}: no arm overview "
                 f"({' / '.join(OVERVIEW_CSV)}) under {root}. Recall@{k} is "
                 f"only there — the per-query records reach rank 10 only.\n"
                 f"Produce the arm:\n" + _SHREC_HOWTO)
    col = f"hit_sub@{k}"
    if col not in row:
        have = sorted(c for c in row if c.startswith("hit_sub@"))
        sys.exit(f"[shortlist] {release}: {overview_path} has no column "
                 f"{col}; present: {have}. Set --shrec-k accordingly.")
    return float(row[col])


# ---------------------------------------------------------------------------
# Stage 3 — BOP
# ---------------------------------------------------------------------------

def stage3_records(root: str, configuration: str, ds: str) -> List[dict]:
    """Per-instance records of one Stage-3a configuration, however named.

    Archive layout (``per_query/<name>/<ds>.json``) and run layout
    (``<ds>_stage3a/records.json``, with or without a per-configuration folder),
    each under the internal ablation key or the release name.
    """
    internal = to_internal(configuration, "stage3")
    for name in dict.fromkeys((internal, to_release(internal, "stage3"))):
        for cand in (os.path.join(root, "per_query", name, f"{ds}.json"),
                     os.path.join(root, name, "per_query", name, f"{ds}.json"),
                     os.path.join(root, name, f"{ds}_stage3a", "records.json"),
                     os.path.join(root, f"{ds}_stage3a", "records.json")):
            if os.path.isfile(cand):
                with open(cand) as fh:
                    d = json.load(fh)
                return d["records"] if isinstance(d, dict) else d
    return []


def stage3_summary(root: str, configuration: str) -> dict:
    """Run summary of one Stage-3a configuration (for the self-check)."""
    internal = to_internal(configuration, "stage3")
    for name in dict.fromkeys((internal, to_release(internal, "stage3"))):
        for cand in (os.path.join(root, f"{name}.json"),
                     os.path.join(root, name, "combined_stage3a.json"),
                     os.path.join(root, "combined_stage3a.json")):
            if os.path.isfile(cand):
                with open(cand) as fh:
                    return json.load(fh)
    return {}


def stage3_side(root: str, configuration: str, k: int,
                datasets) -> Tuple[str, int, float, float]:
    """(release name, n, Top-1, Recall@K) of one Stage-3a run, self-checked."""
    release = to_release(to_internal(configuration, "stage3"), "stage3")
    n = h1 = hk = 0
    for ds in datasets:
        for r in stage3_records(root, configuration, ds):
            n += 1
            rank = r.get("target_rank")
            if rank is None:
                continue                       # Ziel gar nicht im Ranking
            rank = int(rank)
            if rank <= 1:
                h1 += 1
            if rank <= k:
                hk += 1
    if not n:
        sys.exit(f"[shortlist] {release}: no records under {root} (expected "
                 f"per_query/{release}/<ds>.json or <ds>_stage3a/"
                 f"records.json).\nProduce the run:\n" + _BOP_HOWTO)
    top1, recall = h1 / n, hk / n
    published = stage3_summary(root, configuration).get("recall@1")
    if published is not None:
        print(f"    self-check R@1 {release}: own computation {top1:.6f} "
              f"against published {float(published):.6f}")
        if abs(top1 - float(published)) > 1e-4:
            sys.exit(f"    ABORT: {release}: R@1 from target_rank does not "
                     f"match the published recall@1 — wrong records.")
    return release, n, top1, recall


# ---------------------------------------------------------------------------

def rows_for(label: str, k: int, inc, cha) -> List[List[object]]:
    """Two CSV rows for one benchmark; ``inc``/``cha`` are (name, n, top1, recall).

    The denominator is the INCUMBENT's Recall@K for both roles — re-ranking
    permutes the shortlist and cannot change whether the target is inside it.
    """
    (i_name, i_n, i_top1, i_rec) = inc
    (c_name, c_n, c_top1, c_rec) = cha
    if abs(i_rec - c_rec) > 1e-4:
        print(f"    WARNING: Recall@{k} differs between "
              f"{i_name} ({i_rec:.4f}) and {c_name} ({c_rec:.4f}) — the "
              f"challenger is then not reranking the same shortlist. "
              f"The incumbent's denominator is used.")
    i_cond, c_cond = i_top1 / i_rec, c_top1 / i_rec
    return [
        [label, k, INCUMBENT, i_name, i_n, round(i_top1, 4), round(i_rec, 4),
         round(i_cond, 4), round(i_rec - i_top1, 4), ""],
        [label, k, CHALLENGER, c_name, c_n, round(c_top1, 4), round(i_rec, 4),
         round(c_cond, 4), "", round(100.0 * (c_cond - i_cond), 2)],
    ]


def main():
    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shrec-root", default=None,
                    help="Stage-1 run (default: <runs_root>/stage1_shrec18; "
                         "the frozen record: results/stage1_shrec18)")
    ap.add_argument("--bop-root", default=None,
                    help="Stage-3 run (default: <runs_root>/stage3_bop; the "
                         "frozen record: results/stage3_bop)")
    ap.add_argument("--shrec-arms", default=",".join(SHREC_ARMS),
                    help=f"incumbent,challenger on SHREC'18 (default "
                         f"{','.join(SHREC_ARMS)})")
    ap.add_argument("--bop-arms", default=",".join(BOP_ARMS),
                    help=f"incumbent,challenger on BOP (default "
                         f"{','.join(BOP_ARMS)})")
    ap.add_argument("--shrec-k", type=int, default=SHREC_K,
                    help=f"shortlist depth on SHREC'18 (default {SHREC_K}; "
                         f"needs the column hit_sub@K in the arm overview)")
    ap.add_argument("--bop-k", type=int, default=BOP_K,
                    help=f"shortlist depth on BOP (default {BOP_K})")
    ap.add_argument("--datasets", default=",".join(BOP_DATASETS),
                    help=f"BOP datasets, comma-separated (default "
                         f"{','.join(BOP_DATASETS)})")
    ap.add_argument("--out", default=None,
                    help="result CSV (default: <runs_root>/"
                         "stage1_shortlist_analysis/conditional_top1.csv)")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()
    _paths = _paths_mod.from_args(args)

    def _abs(p, default):
        p = p or os.path.join(_paths["runs_root"], default)
        return p if os.path.isabs(p) else os.path.join(_REPO, p)

    shrec_root = _abs(args.shrec_root, "stage1_shrec18")
    bop_root = _abs(args.bop_root, "stage3_bop")
    out_csv = _abs(args.out, os.path.join("stage1_shortlist_analysis",
                                          "conditional_top1.csv"))
    shrec_arms = [a.strip() for a in args.shrec_arms.split(",") if a.strip()]
    bop_arms = [a.strip() for a in args.bop_arms.split(",") if a.strip()]
    for name, arms in (("--shrec-arms", shrec_arms), ("--bop-arms", bop_arms)):
        if len(arms) != 2:
            sys.exit(f"[shortlist] {name} needs exactly two names "
                     f"(incumbent,challenger), got {arms}.")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    rows: List[List[object]] = []

    print(f"\n=== {SHREC_LABEL}  K={args.shrec_k}   root={shrec_root}")
    overview, overview_path = stage1_overview(shrec_root)
    if overview_path:
        print(f"    Recall@{args.shrec_k} and control values from {overview_path}")
    sides = []
    for arm in shrec_arms:
        release, n, top1 = stage1_side(shrec_root, arm, args.shrec_k, overview)
        recall = stage1_recall_at_k(shrec_root, arm, args.shrec_k, overview,
                                    overview_path)
        sides.append((release, n, top1, recall))
    rows += rows_for(SHREC_LABEL, args.shrec_k, sides[0], sides[1])

    print(f"\n=== {BOP_LABEL}  K={args.bop_k}   root={bop_root}")
    sides = [stage3_side(bop_root, a, args.bop_k, datasets) for a in bop_arms]
    rows += rows_for(BOP_LABEL, args.bop_k, sides[0], sides[1])

    # --- the table of the thesis, on stdout ------------------------------
    print(f"\n  {'benchmark':<18}{'role':<15}{'configuration':<40}"
          f"{'Top-1':>8}{'Rec@K':>8}{'cond.':>9}{'Δ pp.':>8}")
    for r in rows:
        role = "incumbent" if r[2] == INCUMBENT else "challenger"
        d = f"{r[9]:+.2f}" if r[9] != "" else ""
        print(f"  {r[0] + ' K=' + str(r[1]):<18}{role:<15}{r[3]:<40}"
              f"{r[5]:>8.4f}{r[6]:>8.4f}{r[7]:>9.4f}{d:>8}")
        if r[8] != "":
            print(f"  {'':<33}headroom Recall@K − Top-1 = {r[8]:.4f} "
                  f"(upper bound of the re-ranking)")
    print("\n  The thesis rounds this table to three decimals and forms the "
          "difference\n  AFTERWARDS (0.548 − 0.396 = +15.2 pp. on SHREC'18); "
          "delta_pp here is computed\n  from the unrounded values and may "
          "differ from it in the last digit.")
    print("  The absolute level is NOT comparable between the benchmarks "
          "(SHREC counts\n  sub-category hits, BOP the exact target object) — "
          "only the difference per row pair.")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(HEADER))
        w.writerows(rows)
    print(f"\nwritten: {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
