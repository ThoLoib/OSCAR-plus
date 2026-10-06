#!/usr/bin/env python3
"""
stage1_subcategory_split.py
===========================
Stage-1 **hit@1 split by whether the query has a real sub-category** (SHREC'18).

Why this table exists
---------------------
The Stage-1 headline top-1 metric is ``NN_sub``: "is the rank-1 CAD in the
query's category **and** sub-category?".  But 426 of the 2,101 official queries
carry **no** sub-category — and the track's GT does not encode that as an empty
field, it **repeats the category** (``rgbd.<hash>,keyboard,keyboard``).  The
official ``evaluate.py`` grades those rows raw, so for such a query "grade 2"
degenerates to "same category and likewise sub-category-less" (~77 such gallery
CADs per query, median).  Our implementation inherits that behaviour verbatim —
which is correct for leaderboard fidelity, but it means the published ``NN_sub``
mixes two different questions.

This script separates them: per configuration it reports hit@1 over all 2,101
queries, over the 1,675 queries **with** a real sub-category, and over the 426
**without** one.  The conclusion the thesis draws from it is that the signs and
the ranking of the arms survive on the clean subset, while the geometry gain is
disproportionately large on the sub-category-less queries (``E2_chamfer_ransac``
+0.2324 there vs +0.1045 with a sub-category).

Pure reading: no GPU, no container — only the per-query JSONs of an existing
Stage-1 run plus that run's arm overview (host Python, no third-party imports).

How to run
----------
    python3 experiments/stage1_subcategory_split.py
    python3 experiments/stage1_subcategory_split.py \\
        --results-root results/stage1_shrec18 --out /tmp/subcategory_split.csv

Input (``--results-root``, default ``<runs_root>/stage1_shrec18``): a Stage-1 run
folder or the frozen record itself (``--results-root results/stage1_shrec18``).
Both layouts AND both naming systems are accepted — ``<config>/
results_per_query.json`` as written by ``experiments/stage1_shrec18.py`` and the
flat ``per_query/<config>.json`` of the record, each under the internal ablation
key (``E1c_full_fusion``) or the release configuration name (``fusion``); see
``evaluation/configuration_names.py``.

Two further inputs, in this order of preference:

* **which configurations, in which order** — the arm overview of the same run
  (``all_arms.csv``, written by ``experiments/stage1_shrec18.py``; the record's
  copy is ``results/stage1_shrec18/all_arms.csv``).  That file is sorted by nDCG
  descending, and this table follows its row order 1:1.  Without it the script
  discovers the per-query files itself and sorts by the mean per-query nDCG,
  which reproduces the same ordering **except** for the exact position of exact
  nDCG ties — it says so when it has to do that.
* **the sub-category criterion** — the track's own ``rgbd.csv``
  (``<datasets_root>/shrec18/shrec18_official/rgbd.csv``, see ``OFFICIAL_DIR`` in
  ``experiments/stage1_shrec18.py``; override with ``--rgbd-csv``).  That file is
  downloaded input and is absent from a fresh checkout, so the fallback is the
  per-query records themselves: every record carries its query's GT label as
  ``category = [category, sub-category]``, i.e. exactly the two columns of
  ``rgbd.csv``.  When both are available the two are cross-checked query by
  query and a disagreement aborts.

Output (``--out``, default ``<runs_root>/stage1_subcategory_split/
subcategory_split.csv``): columns as in the frozen record
``results/stage1_shrec18/subcategory_split.csv`` — configurations under their
RELEASE name in the column ``configuration``, values ``round(x, 4)``.

Self-check
----------
Per configuration the hit@1 over all queries is compared against that arm's
``NN_sub_hit1`` in the arm overview; a mismatch aborts, because a silently
mis-joined per-query file would otherwise look like a finding.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evaluation.configuration_names import to_internal, to_release  # noqa: E402

HEADER = ("configuration", "n_all", "hit1_all", "n_sub", "hit1_sub",
          "n_nosub", "hit1_nosub")
METRIC = "NN_sub"                     # the Stage-1 hit@1 (sub-category grade)
# Arm overview of a Stage-1 run / of the record, in that order.  First column
# names the configuration, the row order is nDCG-descending and IS the row order
# of this table.
OVERVIEW_CSV = ("all_arms.csv", "stage1_summary_arms.csv")
OVERVIEW_HIT1 = ("NN_sub_hit1", "NN_sub")     # hit@1 column, for the self-check
# Generating command printed when there is nothing to read.
_HOWTO = ("Produce the arms — one per invocation, flag->arm table in the "
          "driver's --help:\n"
          "  python3 experiments/stage1_shrec18.py\n"
          "  python3 experiments/stage1_shrec18.py --weights 0,0,1\n"
          "  python3 experiments/stage1_shrec18.py --gallery fullmesh\n"
          "  ...\n"
          "The frozen run of the thesis lives in results/stage1_shrec18 "
          "(--results-root results/stage1_shrec18).")


# ---------------------------------------------------------------------------
# input resolution — both layouts, both naming systems
# ---------------------------------------------------------------------------

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


def load_records(root: str, arm: str) -> Optional[List[dict]]:
    p = per_query_path(root, arm)
    if p is None:
        return None
    with open(p) as fh:
        d = json.load(fh)
    return d if isinstance(d, list) else list(d.values())


def overview(root: str) -> Tuple[List[str], Dict[str, float]]:
    """(configurations in record order, configuration -> published hit@1).

    Reads the arm overview of the run; returns ``([], {})`` when there is none.
    """
    for name in OVERVIEW_CSV:
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        with open(p, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        key = list(rows[0])[0]                      # configuration / ablation
        col = next((c for c in OVERVIEW_HIT1 if c in rows[0]), None)
        order, published = [], {}
        for r in rows:
            arm = r[key]
            if not arm:
                continue
            order.append(arm)
            if col:
                try:
                    published[arm] = float(r[col])
                except (TypeError, ValueError):
                    pass
        print(f"[subcat] arm order and control values from {p}")
        return order, published
    return [], {}


def discover(root: str) -> List[str]:
    """Configurations found on disk, ordered by mean per-query nDCG descending.

    Fallback for a run folder without an arm overview.  That ordering IS the
    rule ``all_arms.csv`` follows, but exact nDCG ties there sit in the order the
    arms happened to be produced in, which cannot be reconstructed from the
    per-query files — so the caller is warned.
    """
    names: List[str] = []
    for p in sorted(glob.glob(os.path.join(root, "per_query", "*.json"))):
        names.append(os.path.splitext(os.path.basename(p))[0])
    for p in sorted(glob.glob(os.path.join(root, "*", "results_per_query.json"))):
        n = os.path.basename(os.path.dirname(p))
        if to_release(to_internal(n, "stage1"), "stage1") not in \
                [to_release(to_internal(x, "stage1"), "stage1") for x in names]:
            names.append(n)
    scored = []
    for n in names:
        recs = load_records(root, n)
        if not recs:
            continue
        nd = [float(e["nDCG"]) for e in recs if "nDCG" in e]
        scored.append((-(sum(nd) / len(nd)) if nd else 0.0, n))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [n for _s, n in scored]


# ---------------------------------------------------------------------------
# the sub-category criterion
# ---------------------------------------------------------------------------

def read_rgbd_csv(path: str) -> Dict[str, Tuple[str, str]]:
    """Official ``rgbd.csv``: ``rgbd.<hash>,category,subcategory`` -> labels.

    The ``rgbd.`` prefix is stripped: the per-query records key on the bare hash.
    """
    out: Dict[str, Tuple[str, str]] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 3 or not row[0].strip():
                continue
            qid = row[0].strip()
            if qid.startswith("rgbd."):
                qid = qid[len("rgbd."):]
            out[qid] = (row[1].strip(), row[2].strip())
    return out


def labels_from_records(recs: List[dict]) -> Dict[str, Tuple[str, str]]:
    """``id -> (category, sub-category)`` from the per-query records.

    Each record carries its query's GT label as ``category``, a two-element list
    holding exactly the two label columns of ``rgbd.csv``.
    """
    out: Dict[str, Tuple[str, str]] = {}
    for e in recs:
        lab = e.get("category")
        if not isinstance(lab, (list, tuple)) or len(lab) != 2:
            sys.exit(f"[subcat] record {e.get('id')!r} carries no "
                     f"[category, sub-category] label ('category'); without "
                     f"rgbd.csv the criterion cannot be determined.")
        out[str(e["id"])] = (str(lab[0]), str(lab[1]))
    return out


def has_subcategory(label: Tuple[str, str]) -> bool:
    """A query has a real sub-category iff the GT does not repeat the category.

    The official GT never leaves the sub-category empty; "none" is encoded by
    writing the category twice (audit 2026-09-21: 426/2101 queries, 625/3308
    gallery CADs).
    """
    return label[0] != label[1]


# ---------------------------------------------------------------------------

def hit1(recs: List[dict]) -> float:
    return sum(float(e[METRIC]) for e in recs) / len(recs)


def split_row(configuration: str, recs: List[dict],
              labels: Dict[str, Tuple[str, str]]) -> List[object]:
    """One archive row; values are ``round(x, 4)`` exactly as in the record."""
    missing = [str(e["id"]) for e in recs if str(e["id"]) not in labels]
    if missing:
        sys.exit(f"[subcat] {configuration}: {len(missing)} queries without a "
                 f"GT label (e.g. {missing[:3]}) — the criterion source does "
                 f"not match the records.")
    sub = [e for e in recs if has_subcategory(labels[str(e["id"])])]
    nosub = [e for e in recs if not has_subcategory(labels[str(e["id"])])]
    row: List[object] = [configuration, len(recs), round(hit1(recs), 4)]
    for part in (sub, nosub):
        row += [len(part), (round(hit1(part), 4) if part else "")]
    return row


def main():
    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=None,
                    help="Stage-1 run folder holding the arms (default: "
                         "<runs_root>/stage1_shrec18; the frozen record of "
                         "the thesis: results/stage1_shrec18)")
    ap.add_argument("--rgbd-csv", default=None,
                    help="official rgbd.csv with category+sub-category per "
                         "query (default: <datasets_root>/shrec18/"
                         "shrec18_official/rgbd.csv; if it is missing the "
                         "criterion is taken from the records' 'category' "
                         "field)")
    ap.add_argument("--out", default=None,
                    help="result CSV (default: <runs_root>/"
                         "stage1_subcategory_split/subcategory_split.csv)")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()
    _paths = _paths_mod.from_args(args)

    root = args.results_root or os.path.join(_paths["runs_root"],
                                             "stage1_shrec18")
    if not os.path.isabs(root):
        root = os.path.join(_REPO, root)
    out_csv = args.out or os.path.join(_paths["runs_root"],
                                       "stage1_subcategory_split",
                                       "subcategory_split.csv")
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(_REPO, out_csv)
    print(f"[subcat] results-root={root}")

    order, published = overview(root)
    if not order:
        order = discover(root)
        if order:
            print("[subcat] WARNING: no arm overview "
                  f"({' / '.join(OVERVIEW_CSV)}) under {root} — order formed "
                  f"here by mean nDCG. On exact nDCG ties it may differ from "
                  f"the record's order (which is the order the arms were "
                  f"produced in).")
    if not order:
        sys.exit(f"[subcat] no results found in {root} (expected "
                 f"<config>/results_per_query.json or "
                 f"per_query/<config>.json, internal or release name).\n"
                 + _HOWTO)

    # --- the sub-category criterion -------------------------------------
    rgbd_csv = args.rgbd_csv or os.path.join(_paths["datasets_root"], "shrec18",
                                             "shrec18_official", "rgbd.csv")
    official = read_rgbd_csv(rgbd_csv) if os.path.isfile(rgbd_csv) else None
    if official is None:
        print(f"[subcat] rgbd.csv not at {rgbd_csv} — criterion from the "
              f"'category' field of the per-query records (the same two "
              f"label columns).")
    else:
        print(f"[subcat] criterion from {rgbd_csv} ({len(official)} queries)")

    rows: List[List[object]] = []
    n_sub = n_nosub = None
    for arm in order:
        recs = load_records(root, arm)
        if not recs:
            print(f"[skip] {arm}: no per-query file")
            continue
        labels = labels_from_records(recs)
        if official is not None:
            bad = [q for q, lab in labels.items()
                   if q in official and official[q] != lab]
            if bad:
                sys.exit(f"[subcat] ABORT: {arm}: labels of the records and "
                         f"{rgbd_csv} contradict each other for {len(bad)} "
                         f"queries (e.g. {bad[0]}: {labels[bad[0]]} vs "
                         f"{official[bad[0]]}).")
            labels = {q: official.get(q, lab) for q, lab in labels.items()}
        configuration = to_release(to_internal(arm, "stage1"), "stage1")
        row = split_row(configuration, recs, labels)
        # Selbstpruefung: hit@1 ueber alle Queries gegen den publizierten Wert.
        pub = published.get(arm, published.get(configuration))
        if pub is not None and abs(float(row[2]) - pub) > 1e-4:
            sys.exit(f"[subcat] ABORT: {configuration}: own hit@1 "
                     f"{row[2]} differs from published {pub} — the "
                     f"per-query file does not match the arm overview.")
        if n_sub is None:
            n_sub, n_nosub = row[3], row[5]
        elif (row[3], row[5]) != (n_sub, n_nosub):
            sys.exit(f"[subcat] ABORT: {configuration} splits {row[3]}/"
                     f"{row[5]} queries, other arms {n_sub}/{n_nosub} — the "
                     f"arms do not see the same query set.")
        rows.append(row)
        print(f"  {configuration:42s} all {row[2]:<7} with sub {row[4]:<7} "
              f"without sub {row[6]}")

    if not rows:
        sys.exit(f"[subcat] no per-query files under {root} — no CSV "
                 f"written.\n" + _HOWTO)
    print(f"[subcat] split: {n_sub} queries with a real sub-category, "
          f"{n_nosub} without (category repeated)")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        # LF, like results/stage1_shrec18/subcategory_split.csv — csv's default
        # dialect would write CRLF and the re-run would not diff byte-identically.
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(HEADER))
        w.writerows(rows)
    print(f"written: {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
