#!/usr/bin/env python3
"""
stage3_channel_recall.py
========================
Stage-3 **per-channel Recall@1** on BOP — what does each single channel find?

Why this table exists
---------------------
The Stage-3 retrieval runs report the Recall of the **fused** ranking.  That
hides which channel carries it, and it hides the effect the thesis needs most on
BOP: the isolated shape channel is the one that reacts to the gallery
representation (partial scan vs full mesh), and the fusion absorbs most of that
difference.  Running isolated channels as their own configurations would cost one
full pass each — so ``experiments/stage3_bop.py`` instead writes, per instance,
the rank of the target under **every** channel combination it computes anyway
(``_arm_rankings``) into the field ``arm_ranks``.  This script turns those ranks
into Recall@1 per channel, per dataset.

The seven variants are the ones ``_arm_rankings`` produces, in its order:

===========================  ==================================================
``clip_only``                semantic (CLIP text) alone, full gallery
``dino_only_full``           appearance (DINOv2) alone, full gallery
``ulip_only_full``           shape (ULIP-2 / Uni3D) alone, full gallery
``clip_dino_ulip_full``      the configuration's own 3-way fusion (= the run)
``oscar_maxview``            OSCAR's mechanism: CLIP-τ shortlist, appearance max
``oscar_softmax``            same shortlist, this thesis' top-k softmax
``clip_pruned_dino_ulip``    fusion of appearance+shape **inside** that shortlist
===========================  ==================================================

In the three shortlist arms the semantic channel only *selects*; below the
shortlist the ranking keeps the semantic order.  ``clip_only``,
``dino_only_full``, ``oscar_maxview`` and ``oscar_softmax`` do not read the
gallery's shape representation at all, which is why they come out identical
across the four configurations — that identity is a useful sanity check, not a
copy-paste error.

**The cascade row of the thesis table is NOT in this CSV.** The OSCAR baseline of
Chapter 6 is its own run (``retrieval_oscar_baseline`` / internal ``3a_oscar``,
R@1 0.3198 pooled), whose *fused* ranking IS ``oscar_maxview`` but which was
scored over its own gallery pass.  The ``oscar_maxview`` variant here is the
same mechanism derived inside an OSCAR+ run and reaches 0.3675 pooled.  They are
different numbers and this file contains only the latter; do not read a cascade
row out of it.

Pure reading: no GPU, no container — only the per-instance JSONs of existing
Stage-3 retrieval runs (host Python, no third-party imports).

How to run
----------
    python3 experiments/stage3_channel_recall.py
    python3 experiments/stage3_channel_recall.py --results-root results/stage3_bop
    python3 experiments/stage3_channel_recall.py \\
        --configurations retrieval_cross_partial --datasets ycbv

Input (``--results-root``, default ``<runs_root>/stage3_bop``): the Stage-3 runs
or the frozen record itself (``--results-root results/stage3_bop``).  Both
layouts AND both naming systems are accepted — archive layout
``per_query/<configuration>/<ds>.json`` and run layout
``<configuration>/<ds>_stage3a/records.json`` as written by
``experiments/stage3_bop.py``, each under the internal ablation key
(``3a_cross_v2``) or the release configuration name
(``retrieval_cross_partial``); see ``evaluation/configuration_names.py``.

Output (``--out``, default ``<runs_root>/stage3_channel_recall/
channel_recall1.csv``): columns as in the frozen record
``results/stage3_bop/channel_recall1.csv`` — configurations under their RELEASE
name in the column ``configuration``, the pooled dataset row is called ``all``.

Self-check
----------
``clip_dino_ulip_full`` IS the configuration's own fused ranking, so its
Recall@1 must equal the run's published ``recall@1``.  That is checked against
the run summary whenever one is reachable, and a mismatch aborts.

Input requirement
-----------------
Every record must carry the ``arm_ranks`` field; all four archived retrieval
configurations do, so ``--results-root results/stage3_bop`` reproduces the full
112 rows.  A run folder written before that field existed yields fewer rows, and
the script names the configuration it had to drop.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evaluation.configuration_names import to_internal, to_release  # noqa: E402

HEADER = ("configuration", "variant", "dataset", "n", "recall@1")
# The four retrieval configurations of the record, in its row order (the 2x2 of
# query mode x gallery representation, cross first).
CONFIGURATIONS = ("retrieval_cross_fullmesh", "retrieval_cross_partial",
                  "retrieval_pc_partial", "retrieval_pc_fullmesh")
DATASETS = ("lmo", "tless", "ycbv")
POOLED = "all"
# ``_arm_rankings`` writes arm_ranks as an ordered dict and the record follows
# that order; it is read off the records so a changed arm set still lines up.
# This is only the fallback for records that carry no arm_ranks at all.
VARIANTS_FALLBACK = ("clip_only", "dino_only_full", "ulip_only_full",
                     "clip_dino_ulip_full", "oscar_maxview", "oscar_softmax",
                     "clip_pruned_dino_ulip")
# The variant that IS the run's own fused ranking -> self-check against recall@1.
FUSED_VARIANT = "clip_dino_ulip_full"
_HOWTO = ("Produce the runs (one per configuration, --mode retrieval = 3a):\n"
          "  python3 experiments/stage3_bop.py --mode retrieval --query "
          "cross --gallery fullmesh\n"
          "  python3 experiments/stage3_bop.py --mode retrieval --query "
          "cross --gallery partial\n"
          "  python3 experiments/stage3_bop.py --mode retrieval --query "
          "pc --gallery partial\n"
          "  python3 experiments/stage3_bop.py --mode retrieval --query "
          "pc --gallery fullmesh\n"
          "The frozen run of the thesis lives in results/stage3_bop "
          "(--results-root results/stage3_bop).")


def _names(configuration: str) -> List[str]:
    """Both spellings of a configuration, internal first, without duplicates."""
    internal = to_internal(configuration, "stage3")
    return list(dict.fromkeys((internal, to_release(internal, "stage3"))))


def records_path(root: str, configuration: str, ds: str) -> Optional[str]:
    """Per-dataset records of one configuration, whatever they are called on disk.

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


def combined_path(root: str, configuration: str) -> Optional[str]:
    """Summary of the run, for the self-check (archive: ``<name>.json``)."""
    for name in _names(configuration):
        for cand in (os.path.join(root, f"{name}.json"),
                     os.path.join(root, name, "combined_stage3a.json"),
                     os.path.join(root, "combined_stage3a.json")):
            if os.path.isfile(cand):
                return cand
    return None


def load_records(root: str, configuration: str, ds: str) -> List[dict]:
    p = records_path(root, configuration, ds)
    if p is None:
        return []
    with open(p) as fh:
        d = json.load(fh)
    return d["records"] if isinstance(d, dict) else d


def variants_of(records: List[dict]) -> List[str]:
    """Variant names in the order ``_arm_rankings`` produced them."""
    for r in records:
        ranks = r.get("arm_ranks")
        if isinstance(ranks, dict) and ranks:
            return list(ranks)
    return []


def hits_at_1(records: List[dict], variant: str) -> int:
    """Instances whose target is rank 1 under ``variant``.

    A missing rank (``None``: target not in the ranking at all, or an
    ``arm_ranks_error`` on that instance) counts as a miss, exactly like
    ``experiments/stage3_bop.py`` counts an unfound target.
    """
    n = 0
    for r in records:
        rank = (r.get("arm_ranks") or {}).get(variant)
        if rank is not None and int(rank) == 1:
            n += 1
    return n


def selfcheck(root: str, configuration: str, release: str,
              pooled_recall: Dict[str, float]) -> None:
    """``clip_dino_ulip_full`` IS the run's fused ranking. Abort on mismatch."""
    mine = pooled_recall.get(FUSED_VARIANT)
    if mine is None:
        return
    p = combined_path(root, configuration)
    if p is None:
        print(f"    (no run summary found — "
              f"self-check skipped)")
        return
    with open(p) as fh:
        published = json.load(fh).get("recall@1")
    if published is None:
        print(f"    (no recall@1 in {p} — self-check skipped)")
        return
    print(f"    self-check R@1 ({FUSED_VARIANT}): own computation "
          f"{mine:.6f} against published {float(published):.6f}")
    if abs(mine - float(published)) > 1e-4:
        sys.exit(f"    ABORT: {release}: {FUSED_VARIANT} IS the run's fused "
                 f"ranking and should match its recall@1 — "
                 f"the arm_ranks do not belong to this run.")
    print("    -> matches.")


def main():
    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=None,
                    help="root holding the Stage-3 records (default: "
                         "<runs_root>/stage3_bop; the frozen record of the "
                         "thesis: results/stage3_bop)")
    ap.add_argument("--configurations", default=",".join(CONFIGURATIONS),
                    help="configurations, release or internal names, "
                         "comma-separated (default: the four of the record, in "
                         "its row order)")
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help=f"BOP datasets, comma-separated (default "
                         f"{','.join(DATASETS)}); the pooled row '{POOLED}' is "
                         f"always written in addition")
    ap.add_argument("--out", default=None,
                    help="result CSV (default: <runs_root>/"
                         "stage3_channel_recall/channel_recall1.csv)")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()
    _paths = _paths_mod.from_args(args)

    root = args.results_root or os.path.join(_paths["runs_root"], "stage3_bop")
    if not os.path.isabs(root):
        root = os.path.join(_REPO, root)
    out_csv = args.out or os.path.join(_paths["runs_root"],
                                       "stage3_channel_recall",
                                       "channel_recall1.csv")
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(_REPO, out_csv)
    configurations = [c.strip() for c in args.configurations.split(",") if c.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"[chan] results-root={root}")

    rows: List[List[object]] = []
    no_records, no_arm_ranks = [], []
    for configuration in configurations:
        release = to_release(to_internal(configuration, "stage3"), "stage3")
        per_ds = {ds: load_records(root, configuration, ds) for ds in datasets}
        if not any(per_ds.values()):
            no_records.append(release)
            print(f"\n=== {release}: no records found — skipped "
                  f"(expected per_query/{release}/<ds>.json or "
                  f"<ds>_stage3a/records.json below {root}).")
            continue
        variants = next((v for ds in datasets
                         for v in [variants_of(per_ds[ds])] if v), [])
        if not variants:
            no_arm_ranks.append(release)
            print(f"\n=== {release}: records without the field 'arm_ranks' — "
                  f"skipped. The single channels are only in runs that also "
                  f"wrote that field "
                  f"(experiments/stage3_bop.py, include_target).")
            continue
        print(f"\n=== {release}  ({len(variants)} channel variants)")
        pooled_recall: Dict[str, float] = {}
        for v in variants:
            hits = total = 0
            for ds in datasets:
                recs = per_ds[ds]
                if not recs:
                    continue
                h = hits_at_1(recs, v)
                rows.append([release, v, ds, len(recs), round(h / len(recs), 4)])
                hits += h
                total += len(recs)
            if not total:
                continue
            rows.append([release, v, POOLED, total, round(hits / total, 4)])
            pooled_recall[v] = hits / total
            print(f"    {v:24s} R@1 {hits / total:.4f}  (n={total})")
        selfcheck(root, configuration, release, pooled_recall)

    if not rows:
        sys.exit(f"[chan] no usable records below {root} — no CSV "
                 f"written.\n" + _HOWTO)
    if no_records or no_arm_ranks:
        print("\n[chan] WARNING: incomplete against "
              "results/stage3_bop/channel_recall1.csv (112 rows):")
        for r in no_records:
            print(f"  {r}: no records below {root}")
        for r in no_arm_ranks:
            print(f"  {r}: records without 'arm_ranks' (in the frozen record "
                  f"the field was dropped for retrieval_cross_partial during "
                  f"the export)")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        # LF, like results/stage3_bop/channel_recall1.csv — csv's default dialect
        # would write CRLF and the re-run would not diff byte-identically.
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(HEADER))
        w.writerows(rows)
    print(f"\nwritten: {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
