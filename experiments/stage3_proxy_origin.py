#!/usr/bin/env python3
"""
stage3_proxy_origin.py
======================
Stage-3c: **where the posed model came from, and what that costs** (BOP).

Why this table exists
---------------------
Stage 3c decomposes the pose error of the full pipeline: for each BOP instance the
retrieval picks a model, FoundationPose poses it, and ``D_sym`` measures the
distance to the GT-posed target.  Two very different cases hide in one median —
either retrieval returned the instance's **own** BOP CAD (``target_cad``) or it
returned a **substitute** from the foreign proxy gallery (GSO / HouseCat6D /
ITODD, ``proxy``).  Splitting the error by that provenance is what turns "the
pipeline is off by 2 cm" into "it is off by 1 cm when it finds the right model and
by 2 cm when it has to substitute one", per dataset.

Both an absolute and a size-normalised median are reported, because the three
datasets differ by an order of magnitude in object size: ``d_sym_median_mm`` is
the median ``D_sym`` in millimetres, ``d_sym_norm_median`` the median of the same
error divided by the object diameter.  They can disagree and do — T-LESS proxies
are the worst case normalised (0.2755) and not in millimetres.

Note on terminology: ``target_cad`` is the record's own field value and means
"the substitute came from the instance's own BOP dataset"; it does not claim the
proxies are less real CAD (docs agreement of 2026-09-06).

Pure reading: no GPU, no container — only the per-instance JSONs of existing
Stage-3c runs (host Python, no third-party imports).

How to run
----------
    python3 experiments/stage3_proxy_origin.py
    python3 experiments/stage3_proxy_origin.py --results-root results/stage3_bop
    python3 experiments/stage3_proxy_origin.py \\
        --configurations pose_decomposition_cross --datasets ycbv

Input (``--results-root``, default ``<runs_root>/stage3_bop``): the Stage-3c runs
or the frozen record itself (``--results-root results/stage3_bop``).  Both
layouts AND both naming systems are accepted — archive layout
``per_query/<configuration>/<ds>.json`` and run layout
``<configuration>/<ds>_stage3c/records.json`` as written by
``experiments/stage3_bop.py``, each under the internal ablation key
(``3c_cross``) or the release configuration name
(``pose_decomposition_cross``); see ``evaluation/configuration_names.py``.

Output (``--out``, default ``<runs_root>/stage3_proxy_origin/
proxy_origin_by_dataset.csv``): columns as in the frozen record
``results/stage3_bop/proxy_origin_by_dataset.csv`` — configurations under their
RELEASE name in the column ``configuration``, the pooled group is called ``ALL``.
``d_sym_median_mm`` is rounded to 3 decimals (µm resolution is noise),
``d_sym_norm_median`` to 4, exactly as in the record.

Self-check
----------
The provenance groups must partition the instances: their counts have to add up
to the number of records of that dataset, else a provenance value was missed and
the medians would silently be computed over a subset.
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

HEADER = ("configuration", "dataset", "provenance", "n", "d_sym_median_mm",
          "d_sym_norm_median")
# The two 3c configurations of the record, in its row order.
CONFIGURATIONS = ("pose_decomposition_cross", "pose_decomposition_cross_fullmesh")
DATASETS = ("ycbv", "tless", "lmo")
POOLED = "ALL"
# Provenance of the posed model, in the record's row order. The values are what
# experiments/stage3_bop.py writes into `nb_provenance`.
PROVENANCES = ("target_cad", "proxy")
PROVENANCE_FIELD = "nb_provenance"
VALUE_MM = "d_posed"            # D_sym of the posed neighbour, millimetres
VALUE_NORM = "d_sym_norm"       # the same error divided by the object diameter
_HOWTO = ("Produce the runs (one per configuration, --mode decompose = 3c; "
          "the diagnostic reads a finished retrieval run):\n"
          "  python3 experiments/stage3_bop.py --mode decompose --query "
          "cross --gallery partial \\\n"
          "      --from-retrieval runs/stage3_bop_retrieval_cross_partial\n"
          "  python3 experiments/stage3_bop.py --mode decompose --query "
          "cross --gallery fullmesh \\\n"
          "      --from-retrieval runs/stage3_bop_retrieval_cross_fullmesh\n"
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
    folder, and the run layout (``<ds>_stage3c/records.json``) either under a
    per-configuration folder or directly in ``root`` when ``root`` IS the run
    folder of ``experiments/stage3_bop.py``.
    """
    for name in _names(configuration):
        for cand in (os.path.join(root, "per_query", name, f"{ds}.json"),
                     os.path.join(root, name, "per_query", name, f"{ds}.json"),
                     os.path.join(root, name, f"{ds}_stage3c", "records.json"),
                     os.path.join(root, f"{ds}_stage3c", "records.json")):
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


def by_provenance(records: List[dict], provenance: str) -> List[dict]:
    return [r for r in records if r.get(PROVENANCE_FIELD) == provenance]


def medians(records: List[dict]):
    """(median D_sym in mm, median D_sym / diameter) over ``records``."""
    mm = statistics.median([float(r[VALUE_MM]) for r in records])
    norm = statistics.median([float(r[VALUE_NORM]) for r in records])
    return mm, norm


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
                         "comma-separated (default: the two 3c runs of the "
                         "record, in its row order)")
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help=f"BOP datasets, comma-separated (default "
                         f"{','.join(DATASETS)}); the pooled group '{POOLED}' "
                         f"is always written in addition")
    ap.add_argument("--out", default=None,
                    help="result CSV (default: <runs_root>/"
                         "stage3_proxy_origin/proxy_origin_by_dataset.csv)")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()
    _paths = _paths_mod.from_args(args)

    root = args.results_root or os.path.join(_paths["runs_root"], "stage3_bop")
    if not os.path.isabs(root):
        root = os.path.join(_REPO, root)
    out_csv = args.out or os.path.join(_paths["runs_root"],
                                       "stage3_proxy_origin",
                                       "proxy_origin_by_dataset.csv")
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(_REPO, out_csv)
    configurations = [c.strip() for c in args.configurations.split(",") if c.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"[origin] results-root={root}")

    rows: List[List[object]] = []
    for configuration in configurations:
        release = to_release(to_internal(configuration, "stage3"), "stage3")
        per_ds: Dict[str, List[dict]] = {}
        for ds in datasets:
            recs = load_records(root, configuration, ds)
            if recs:
                per_ds[ds] = recs
        if not per_ds:
            print(f"\n=== {release}: no records found — skipped "
                  f"(expected per_query/{release}/<ds>.json or "
                  f"<ds>_stage3c/records.json below {root}).")
            continue
        print(f"\n=== {release}")
        print(f"    {'dataset':<10}{'provenance':<12}{'n':>7}"
              f"{'D_sym mm':>11}{'D_sym/d':>10}")
        # per dataset first, then the pooled group — the record's row order
        groups = [(ds, per_ds[ds]) for ds in datasets if ds in per_ds]
        groups.append((POOLED, [r for ds in datasets if ds in per_ds
                                for r in per_ds[ds]]))
        for label, recs in groups:
            counted = 0
            for prov in PROVENANCES:
                sel = by_provenance(recs, prov)
                counted += len(sel)
                if not sel:
                    print(f"    {label:<10}{prov:<12}{0:>7}{'—':>11}{'—':>10}")
                    continue
                mm, norm = medians(sel)
                rows.append([release, label, prov, len(sel), round(mm, 3),
                             round(norm, 4)])
                print(f"    {label:<10}{prov:<12}{len(sel):>7}"
                      f"{mm:>11.3f}{norm:>10.4f}")
            # Selbstpruefung: die Herkunftsgruppen muessen die Instanzen
            # vollstaendig aufteilen, sonst rechnen die Mediane auf einer
            # unbemerkten Teilmenge.
            if counted != len(recs):
                other = sorted({str(r.get(PROVENANCE_FIELD)) for r in recs}
                               - set(PROVENANCES))
                sys.exit(f"    ABORT: {release}/{label}: {counted} of "
                         f"{len(recs)} instances carry one of the provenances "
                         f"{list(PROVENANCES)}; further values in "
                         f"'{PROVENANCE_FIELD}': {other}. The medians would be "
                         f"computed over a subset.")

    if not rows:
        sys.exit(f"[origin] no 3c records below {root} — no CSV "
                 f"written.\n" + _HOWTO)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        # LF, like results/stage3_bop/proxy_origin_by_dataset.csv — csv's default
        # dialect would write CRLF and the re-run would not diff byte-identically.
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(HEADER))
        w.writerows(rows)
    print(f"\nwritten: {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
