#!/usr/bin/env python3
"""
stage3_significance.py — paired significance for the BOP retrieval arms.

Counterpart to ``experiments/stage1_significance.py`` (SHREC'18). Pairing is by
instance key (dataset, scene, image, object, gt_idx) — both configurations see
the same 12,284 BOP instances, so the difference is taken over THE SAME
instances and not between two independent means.

Metrics from ``target_rank``:
  hit1  1 if the target rank is 1, else 0    -> equals Recall@1
  mrr   1/rank                               -> equals MRR

**Both** tests are reported, because they measure different things:
  * the 95% bootstrap CI tests the MEAN (sensitive to outliers),
  * Wilcoxon tests the CONSISTENCY of the sign.
In close cases the win count decides — that is exactly what exposed the
apparent Uni3D lead in Stage 1 as noise (p=0.54 at 1009:1027), while a
similarly small margin elsewhere was real.

Pure reading: no container, no GPU (numpy, scipy optional).

Input (``--results-root``, default ``<runs_root>/stage3_bop``): both layouts AND
both naming systems are read — ``per_query/<configuration>/<ds>.json`` as in the
frozen record ``results/stage3_bop/`` and
``<configuration>/<ds>_stage3a/records.json`` as written by
``experiments/stage3_bop.py``, each under the internal ablation key
(``3a_cross_v2``) or the release name (``retrieval_cross_partial``); see
``evaluation/configuration_names.py``.

Output (``--csv``, default
``<runs_root>/stage3_significance/significance.csv``): columns as in the frozen
record ``results/stage3_bop/significance.csv`` — the compared configurations
always appear under their RELEASE name (``configuration_a`` /
``configuration_b``), the pooled group is called ``ALL``.

How to run
----------
    python3 experiments/stage3_significance.py                  # default pairs
    python3 experiments/stage3_significance.py --results-root results/stage3_bop
    python3 experiments/stage3_significance.py retrieval_cross_partial \\
        retrieval_cross_fullmesh --metric mrr
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evaluation.configuration_names import to_internal, to_release  # noqa: E402

DATASETS = ("ycbv", "tless", "lmo")

# The comparisons of the frozen record, labelled as in
# results/stage3_bop/significance.csv.
DEFAULT_PAIRS = [
    ("Gallery: partial vs full-mesh (cross)",
     "retrieval_cross_partial", "retrieval_cross_fullmesh"),
    ("Gallery: partial vs full-mesh (pc)",
     "retrieval_pc_partial", "retrieval_pc_fullmesh"),
    ("Query: cross vs pc (partial)",
     "retrieval_cross_partial", "retrieval_pc_partial"),
    ("Shape channel: OSCAR+ vs OSCAR cascade",
     "retrieval_cross_partial", "retrieval_oscar_baseline"),
    ("Geometry: without vs distance (cross)",
     "retrieval_cross_partial", "retrieval_cross_geo_trimmed_distance"),
    ("Geometry: without vs fitness (cross)",
     "retrieval_cross_partial", "retrieval_cross_geo_fitness"),
]


def _names(configuration: str):
    """Both spellings of a configuration, internal first, without duplicates."""
    internal = to_internal(configuration, "stage3")
    return list(dict.fromkeys((internal, to_release(internal, "stage3"))))


def records_path(root: str, configuration: str, ds: str):
    """Per-dataset 3a records of one configuration, whatever they are called.

    Four candidates per spelling, because both the layout and the naming may
    differ: archive layout (``per_query/<name>/<ds>.json``, as in
    ``results/stage3_bop/``), the same export inside a per-configuration run
    folder, and the run layout (``<ds>_stage3a/records.json``) either under a
    per-configuration folder or directly in ``root`` when ``root`` IS the run
    folder of ``experiments/stage3_bop.py``.
    """
    for name in _names(configuration):
        for cand in (os.path.join(root, "per_query", name, f"{ds}.json"),
                     os.path.join(root, name, "per_query", name, f"{ds}.json"),
                     os.path.join(root, name, f"{ds}_stage3a", "records.json"),
                     os.path.join(root, f"{ds}_stage3a", "records.json")):
            if os.path.isfile(cand):
                return cand
    return None


def load(root: str, configuration: str) -> dict:
    """{instance key: target_rank} of one configuration."""
    out = {}
    for ds in DATASETS:
        p = records_path(root, configuration, ds)
        if p is None:
            continue
        d = json.load(open(p))
        for r in (d["records"] if isinstance(d, dict) else d):
            rk = r.get("target_rank")
            if rk is None:
                continue
            key = (r.get("dataset") or ds, r.get("scene_id"), r.get("im_id"),
                   r.get("obj_id"), r.get("gt_idx"))
            out[key] = rk
    return out


def score(rank, metric):
    return (1.0 if rank == 1 else 0.0) if metric == "hit1" else 1.0 / rank


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(lo, hi, p, wa, wb):
    if lo > 0 or hi < 0:
        return "REAL"
    if p is not None and p < 0.05:
        return "CONSISTENT (small, systematic; mean noisy)"
    if abs(wa - wb) <= 0.02 * max(wa + wb, 1):
        return "tie"
    return "not supported"


def compare(ra: dict, rb: dict, metric: str, per_dataset: bool = True):
    common = sorted(set(ra) & set(rb))
    if not common:
        return None
    groups = {"ALL": common}
    if per_dataset:
        for k in common:
            groups.setdefault(k[0], []).append(k)
    out = []
    for g, keys in groups.items():
        d = np.array([score(ra[k], metric) - score(rb[k], metric) for k in keys])
        lo, hi = boot_ci(d)
        try:
            from scipy.stats import wilcoxon
            nz = d[d != 0]
            p = float(wilcoxon(nz).pvalue) if len(nz) else None
        except Exception:                                      # noqa: BLE001
            p = None
        wa, wb = int((d > 0).sum()), int((d < 0).sum())
        out.append((g, len(keys), float(d.mean()), float(statistics.median(d)),
                    lo, hi, p, wa, wb, verdict(lo, hi, p, wa, wb)))
    return out


def main():
    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configurations", nargs="*", metavar="CONFIGURATION",
                    help="exactly two configurations (release or internal "
                         "name), else the default pairs of the record")
    ap.add_argument("--results-root", default=None,
                    help="root holding the Stage-3 records (default: "
                         "<runs_root>/stage3_bop; the frozen record of the "
                         "thesis: results/stage3_bop)")
    ap.add_argument("--metric", default="hit1", choices=["hit1", "mrr"])
    ap.add_argument("--per-dataset", dest="per_dataset", action="store_true",
                    default=True,
                    help="additionally break down per dataset (default on, "
                         "as in the frozen record)")
    ap.add_argument("--pooled-only", dest="per_dataset", action="store_false",
                    help="only the pooled group ALL")
    ap.add_argument("--csv", default=None,
                    help="result CSV (default: <runs_root>/"
                         "stage3_significance/significance.csv)")
    _paths_mod.add_path_args(ap)
    a = ap.parse_args()
    _paths = _paths_mod.from_args(a)

    root = a.results_root or os.path.join(_paths["runs_root"], "stage3_bop")
    if not os.path.isabs(root):
        root = os.path.join(_REPO, root)
    out_csv = a.csv or os.path.join(_paths["runs_root"], "stage3_significance",
                                    "significance.csv")
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(_REPO, out_csv)

    if len(a.configurations) == 1:
        sys.exit("[sig3] give either TWO configurations or none "
                 "(then the default pairs of the record run).")
    pairs = ([("custom", a.configurations[0], a.configurations[1])]
             if len(a.configurations) == 2 else DEFAULT_PAIRS)

    print(f"[sig3] results-root={root}")
    print(f"[sig3] metric={a.metric} · CI = mean test · "
          f"Wilcoxon = consistency test · positive = first configuration better\n")
    cache: dict = {}
    rows = []
    for label, x, y in pairs:
        for c in (x, y):
            if c not in cache:
                cache[c] = load(root, c)
        if not cache[x] or not cache[y]:
            miss = [c for c in (x, y) if not cache[c]]
            print(f"[skip] {label}: no records for {miss}")
            continue
        res = compare(cache[x], cache[y], a.metric, a.per_dataset)
        if res is None:
            print(f"[skip] {label}: no common instances ({x} / {y})")
            continue
        ca, cb = (to_release(to_internal(c, "stage3"), "stage3") for c in (x, y))
        print(f"=== {label}\n    {ca}  minus  {cb}")
        for g, n, mean, med, lo, hi, p, wa, wb, v in res:
            ps = f"p={p:.4g}" if p is not None else "p=—"
            print(f"    {g:<8} n={n:<6} Δ={mean:+.4f} med={med:+.4f} "
                  f"CI[{lo:+.4f},{hi:+.4f}] {ps:<12} {wa}:{wb} -> {v}")
            rows.append((label, ca, cb, a.metric, g, n, round(mean, 5),
                         round(lo, 5), round(hi, 5), p, wa, wb, v))
        print()
    if not rows:
        sys.exit(f"[sig3] no comparable configurations below {root} — "
                 f"no CSV written. Either put Stage-3 runs there "
                 f"(experiments/stage3_bop.py --mode retrieval ...) or read the "
                 f"frozen record: --results-root results/stage3_bop")
    import csv as _csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        # LF, like results/stage3_bop/significance.csv — csv's default dialect
        # would write CRLF and the re-run would not diff byte-identically.
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["comparison", "configuration_a", "configuration_b",
                    "metric", "group", "n", "delta", "ci_lo", "ci_hi",
                    "wilcoxon_p", "wins_a", "wins_b", "verdict"])
        w.writerows(rows)
    print(f"CSV: {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
