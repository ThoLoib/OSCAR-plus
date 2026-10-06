#!/usr/bin/env python3
"""
stage1_significance.py
======================
Paired significance test for the OSCAR+ **Stage-1** arm deltas (SHREC'18).

For each (arm_A, arm_B) pair the per-query results of both arms are matched by
query id and the paired mean delta is reported with a 95% bootstrap CI and a
Wilcoxon signed-rank p on one metric.  A delta is "real" iff the 95% CI excludes
0 AND the sign split is consistent (see :func:`verdict` — for the near-ties this
grid is full of, sign consistency is the meaningful verdict).  This closes the
"is this small delta noise?" question the 42v/k5 rerun opened up (Uni3D ≈ ULIP,
XYZ ≈ XYZ+RGB, config change 0.5889 -> 0.5868, ...).

Pure reading: no GPU, no container — the script only parses the per-query JSONs
of an existing Stage-1 run (numpy + scipy on the host are enough).

How to run
----------
    python3 experiments/stage1_significance.py
        # both reported metrics of the frozen record:
        #   nDCG   -> significance_ndcg.csv
        #   NN_sub -> significance_hit1.csv   (hit@1 on the sub-category grade)
    python3 experiments/stage1_significance.py --metric nDCG --out /tmp/sig
    python3 experiments/stage1_significance.py \\
        --results-root runs/stage1_shrec18 \\
        --legacy-root runs/stage1_shrec18_16v_k8

Input (``--results-root``, default ``<runs_root>/stage1_shrec18``): a Stage-1 run
folder or the frozen record itself (``--results-root results/stage1_shrec18``).
Both layouts AND both naming systems are accepted — ``<config>/
results_per_query.json`` as written by ``experiments/stage1_shrec18.py`` and the
flat ``per_query/<config>.json`` of the record, each under the internal ablation
key (``E1c_full_fusion``) or the release configuration name (``fusion``); see
``evaluation/configuration_names.py``.  Arms are produced one per invocation of
the driver; the flag -> arm table is the epilog of ``python3
experiments/stage1_shrec18.py --help``.

Output (``--out``, default ``<runs_root>/stage1_significance``): one CSV per
metric, named and column-shaped as in the frozen record
``results/stage1_shrec18/`` — the compared configurations are always reported
under their RELEASE names (``configuration_a`` / ``configuration_b``).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, Optional

import numpy as np

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:                                          # noqa: BLE001
    HAVE_SCIPY = False

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evaluation.configuration_names import to_internal, to_release  # noqa: E402

N_BOOT = 10000
LEGACY = "legacy"          # root key of the pre-fix (16v/k8) baseline run
# The pre-fix 16v/k8 BASE is a different CONFIGURATION, not a different arm: its
# internal key is E1c_full_fusion too and only the run it comes from differs.
# results/stage1_shrec18/significance_*.csv names it in the configuration_a
# column as follows; reproduce that rather than print "fusion" on both sides.
LEGACY_CONFIGURATION = "fusion_views16_shape_topk8"
# Metric -> CSV name of the frozen record (results/stage1_shrec18/).
CSV_NAME = {"nDCG": "significance_ndcg.csv", "NN_sub": "significance_hit1.csv"}

# (label, (root|None, armA), (root|None, armB)); None -> --results-root
PAIRS = [
    ("shape enc: ULIP-2 vs Uni3D (isolated)", (None, "E1_shape_only"),   (None, "E7_uni3d_shape_only")),
    ("shape enc: ULIP-2 vs Uni3D (fused)",    (None, "E1c_full_fusion"), (None, "E7_uni3d")),
    ("colour: XYZ+RGB vs XYZ (isolated)",     (None, "E1_shape_only"),   (None, "O5_xyz_shape_only")),
    ("colour: XYZ+RGB vs XYZ (fused)",        (None, "E1c_full_fusion"), (None, "O5_xyz_only")),
    ("ref: partial vs full-mesh (isolated)",  (None, "E1_shape_only"),   (None, "E2b_fullmesh_shape_only")),
    # Seit dem Farb-Fix (2026-09-01) ist E2b_fullmesh der staerkste Arm ohne
    # Geometrie — 0.5935 gegen 0.5868 fuer BASE. Der Abstand liegt in der
    # Groessenordnung, in der sich der Uni3D-"Sieg" als Rauschen erwies,
    # gehoert also geprueft und nicht behauptet.
    ("ref: partial vs full-mesh (fused)",     (None, "E1c_full_fusion"), (None, "E2b_fullmesh")),
    ("appearance: DINOv2 vs SigLIP (isolated)", (None, "E1_view_only"),  (None, "E4_siglip_only")),
    ("combiner: weighted vs RRF (fused)",     (None, "E1c_full_fusion"), (None, "E6_rrf")),
    ("shape views: V32 vs V42 (isolated)",    (None, "A7_shape_only_V32"), (None, "A7_shape_only_V42")),
    ("geometry: none vs GeDi+RANSAC (fused)", (None, "E1c_full_fusion"), (None, "E2_chamfer_ransac")),
    ("config: BASE 16v/k8 -> 42v/k5 (fused)", (LEGACY, "E1c_full_fusion"),  (None, "E1c_full_fusion")),
]


def per_query_path(root: str, arm: str) -> Optional[str]:
    """Per-query JSON of one configuration, whatever it is called on disk.

    Four candidates, because both the layout and the naming may differ: run
    layout (``<name>/results_per_query.json``) before archive layout
    (``per_query/<name>.json``), and the internal ablation key before the release
    configuration name.  ``arm`` itself may be given in either naming.
    """
    internal = to_internal(arm, "stage1")
    for name in dict.fromkeys((internal, to_release(internal, "stage1"))):
        for cand in (os.path.join(root, name, "results_per_query.json"),
                     os.path.join(root, "per_query", f"{name}.json")):
            if os.path.isfile(cand):
                return cand
    return None


def configuration_name(root_key: Optional[str], arm: str) -> str:
    """Release name to report this side of a comparison under."""
    if root_key == LEGACY:
        return LEGACY_CONFIGURATION
    return to_release(to_internal(arm, "stage1"), "stage1")


def load(root: str, arm: str, metric: str) -> Optional[Dict[str, float]]:
    p = per_query_path(root, arm)
    if p is None:
        return None
    with open(p) as fh:
        d = json.load(fh)
    rows = d if isinstance(d, list) else list(d.values())
    return {e["id"]: float(e[metric]) for e in rows if metric in e}


def boot_ci(da, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(da), size=(n, len(da)))
    means = da[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(ci_sig: bool, w_sig: bool, wins_a: int, wins_b: int) -> str:
    """Combine the two tests honestly.

    The bootstrap CI tests the **mean** difference; Wilcoxon tests whether one arm
    wins *consistently* (signed ranks). They disagree when the per-query deltas are
    heavy-tailed: a handful of huge swings can move the mean while the win/loss
    split is ~50/50. For the near-ties this grid is full of, the **sign-consistency
    (Wilcoxon + the win counts) is the meaningful verdict** — a "win" carried by a
    few outliers is not a design argument.
    """
    if ci_sig and w_sig:
        return "REAL"                    # both agree
    if not ci_sig and not w_sig:
        return "tie"                     # both agree
    if w_sig and not ci_sig:
        return "CONSISTENT (small, systematic; mean noisy)"
    # ci_sig and not w_sig
    split = f"{wins_a}:{wins_b}"
    return f"outlier-driven ({split}) -> treat as tie"


def run_metric(metric: str, roots: Dict[Optional[str], str], out_dir: str) -> int:
    """One metric = one CSV; returns the number of comparisons written."""
    print(f"\n[sig] metric={metric} boot={N_BOOT} "
          f"scipy={'yes' if HAVE_SCIPY else 'no (bootstrap only)'}")
    print("[sig] CI = mean test · Wilcoxon = sign-consistency test; for near-ties "
          "the sign split is authoritative.")
    rows = []
    for label, (ra, aa), (rb, ab) in PAIRS:
        if (ra is not None and roots.get(ra) is None) or \
                (rb is not None and roots.get(rb) is None):
            print(f"[skip] {label}: no --legacy-root given (pre-fix 16v/k8 run)")
            continue
        A = load(roots[ra], aa, metric)
        B = load(roots[rb], ab, metric)
        if A is None or B is None:
            miss = [n for n, v in ((aa, A), (ab, B)) if v is None]
            print(f"[skip] {label}: missing arm(s) {miss}")
            continue
        ids = sorted(set(A) & set(B))
        if len(ids) < 30:
            print(f"[skip] {label}: only {len(ids)} paired queries")
            continue
        da = np.array([A[i] - B[i] for i in ids])
        mean, med = float(da.mean()), float(np.median(da))
        lo, hi = boot_ci(da)
        ci_sig = (lo > 0 or hi < 0)
        p = (float(wilcoxon(da).pvalue) if (HAVE_SCIPY and np.any(da)) else float("nan"))
        w_sig = (p == p and p < 0.05)
        wins_a, wins_b, tied = int((da > 0).sum()), int((da < 0).sum()), int((da == 0).sum())
        v = verdict(ci_sig, w_sig, wins_a, wins_b)
        ca, cb = configuration_name(ra, aa), configuration_name(rb, ab)
        rows.append((label, ca, cb, round(mean, 4), round(med, 4), round(lo, 4), round(hi, 4),
                     (round(p, 4) if p == p else ""), wins_a, wins_b, tied,
                     "YES" if ci_sig else "no", "YES" if w_sig else "no", v, len(ids)))
        pstr = f"{p:.4f}" if p == p else "  n/a "
        flag = "REAL" if v == "REAL" else ("~~~ " if v.startswith("outlier") else
                                           ("CONS" if v.startswith("CONSISTENT") else "    "))
        print(f"{flag} {label:42s} Δ={mean:+.4f} med={med:+.4f} "
              f"CI[{lo:+.4f},{hi:+.4f}] p={pstr} wins {wins_a}:{wins_b} -> {v}")
    if not rows:
        return 0
    out = os.path.join(out_dir, CSV_NAME.get(metric,
                                             f"significance_{metric.lower()}.csv"))
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["comparison", "configuration_a", "configuration_b",
                    f"mean_d_{metric}", f"median_d_{metric}",
                    "ci_lo", "ci_hi", "wilcoxon_p", "n_A_better", "n_B_better", "n_tied",
                    "ci_significant", "wilcoxon_significant", "verdict", "n_paired"])
        w.writerows(rows)
    print(f"[sig] wrote {len(rows)} comparisons -> {out}")
    return len(rows)


def main():
    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=None,
                    help="Stage-1 run folder holding the arms (default: "
                         "<runs_root>/stage1_shrec18)")
    ap.add_argument("--legacy-root", default=None,
                    help="run folder of the pre-fix 16v/k8 BASE used by the "
                         "'config' comparison (default: none -> that one pair "
                         "is skipped)")
    ap.add_argument("--metric", nargs="+", default=["nDCG", "NN_sub"],
                    metavar="M",
                    help="per-query metric(s), one CSV each: nDCG | NN_sub | "
                         "NN_cat | MRR | AP | nDCG_K (default: nDCG NN_sub)")
    ap.add_argument("--out", default=None,
                    help="output folder for the CSVs (default: "
                         "<runs_root>/stage1_significance)")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()
    _paths = _paths_mod.from_args(args)

    root = args.results_root or os.path.join(_paths["runs_root"],
                                             "stage1_shrec18")
    if not os.path.isabs(root):
        root = os.path.join(_REPO, root)
    legacy = args.legacy_root
    if legacy and not os.path.isabs(legacy):
        legacy = os.path.join(_REPO, legacy)
    out_dir = args.out or os.path.join(_paths["runs_root"],
                                       "stage1_significance")

    arms = {arm for _l, (_ra, arm_a), (_rb, arm_b) in PAIRS
            for arm in (arm_a, arm_b)}
    if not any(per_query_path(root, a) for a in arms):
        sys.exit(f"[sig] no results found in {root} (expected "
                 f"<config>/results_per_query.json or per_query/<config>.json,"
                 f" internal or release name).\n"
                 f"Produce the arms — one per invocation, flag->arm table in "
                 f"the driver's --help:\n"
                 f"  python3 experiments/stage1_shrec18.py\n"
                 f"  python3 experiments/stage1_shrec18.py --weights 0,0,1\n"
                 f"  python3 experiments/stage1_shrec18.py --gallery fullmesh\n"
                 f"  ...\n"
                 f"The frozen run of the thesis lives in "
                 f"results/stage1_shrec18 (--results-root results/stage1_shrec18).")

    roots = {None: root, LEGACY: legacy}
    print(f"[sig] results-root={root}"
          + (f" legacy-root={legacy}" if legacy else ""))
    written = sum(run_metric(m, roots, out_dir) for m in args.metric)
    if not written:
        sys.exit(f"[sig] no comparable arm pairs — no CSV written "
                 f"(are the counter-arms missing in {root}?).")


if __name__ == "__main__":
    main()
