#!/usr/bin/env python3
"""
stage3_occlusion.py — what does occlusion do to retrieval and pose?

Question
--------
Stages 1-3 compare design decisions. This evaluation asks something else: how
much does the result depend on how much of the object is visible at all? BOP
ships the answer with it — no new run is needed.

Data source
-----------
* ``<datasets_root>/<ds>/test*/<scene>/scene_gt_info.json`` — BOP's own
  annotation. ``visib_fract = px_count_visib / px_count_valid``, i.e. the share
  of the projected object area that is visible in the image.
* the per-instance records of a Stage-3 run (``--results-root``, default
  ``<runs_root>/stage3_bop``). Both layouts AND both naming systems are read —
  ``per_query/<configuration>/<ds>.json`` as in the frozen record
  ``results/stage3_bop/`` and ``<configuration>/<ds>_stage3{a,b}/records.json``
  as written by ``experiments/stage3_bop.py``, each under the internal ablation
  key (``3a_cross_v2``) or the release name
  (``retrieval_cross_partial``); see ``evaluation/configuration_names.py``.

Joining is by ``(scene_id, im_id, gt_idx)``. CAUTION: the fields are strings in
3a and integers in 3b — both sides are cast with ``int()``.

Self-check
----------
R@1 is computed from ``target_rank`` HERE and checked against the published
value in the run's summary. On a mismatch the script aborts: a wrong join would
otherwise go unnoticed.

Examples
--------
    # default evaluation (3a retrieval + 3b pose, pooled and per dataset)
    python3 experiments/stage3_occlusion.py

    # against the frozen record of the thesis
    python3 experiments/stage3_occlusion.py --results-root results/stage3_bop

    # retrieval only, different configuration
    python3 experiments/stage3_occlusion.py --mode 3a \\
        --configuration retrieval_cross_fullmesh

    # own bin edges, own CSVs
    python3 experiments/stage3_occlusion.py --bins 0.25,0.5,0.75 \\
        --csv /tmp/o.csv --normalized-out /tmp/o_norm.csv

Output — TWO CSVs, columns as in the two frozen records:

* ``--csv`` (default ``<runs_root>/stage3_occlusion/
  occlusion_by_visibility.csv``), like
  ``results/stage3_bop/occlusion_by_visibility.csv``: one number per bin,
  ``value`` — in 3a the hit rate R@1, in 3b the median of ``D_sym`` in mm.
* ``--normalized-out`` (default: the same file with the suffix
  ``_normalized``, i.e. next to the first CSV), like
  ``results/stage3_bop/occlusion_by_visibility_normalized.csv``: the same
  median ONCE MORE (``median_mm``) and next to it the same error divided by the
  object diameter (``median_norm``, field ``d_sym_norm`` of the records).
  That is needed because the three datasets differ by an order of magnitude in
  object size: T-LESS has the SMALLEST error in mm and the LARGEST relative to
  object size. This CSV concerns 3b only — a retrieval rank has no length, so
  in 3a it stays empty and is not written.

In both the configuration appears under its RELEASE name in the column
``configuration``, the pooled group is called ``ALL``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evaluation.configuration_names import to_internal, to_release  # noqa: E402

# BOP test splits under <datasets_root>.
TEST_ROOTS = {
    "ycbv":  ["ycbv/test"],
    "tless": ["tless/test_primesense"],
    "lmo":   ["lmo/test"],
}
# Default configuration per mode = the one the frozen record reports.
DEFAULT_CONFIGURATION = {"3a": "retrieval_cross_partial",
                         "3b": "pose_proxy_cross_partial"}

_PATHS: dict = {}


def test_root(ds: str) -> str:
    """First test split of `ds` that exists and is non-empty (else the first)."""
    cands = [os.path.join(_PATHS["datasets_root"], c) for c in TEST_ROOTS[ds]]
    for c in cands:
        if os.path.isdir(c) and os.listdir(c):
            return c
    return cands[0]


def visibility_map(ds: str) -> dict:
    """(scene, im, gt_idx) -> visib_fract, over all test scenes of a dataset."""
    out = {}
    pattern = os.path.join(test_root(ds), "*", "scene_gt_info.json")
    for f in sorted(glob.glob(pattern)):
        scene = int(os.path.basename(os.path.dirname(f)))
        for im, entries in json.load(open(f)).items():
            for gt_idx, e in enumerate(entries):
                if "visib_fract" in e:
                    out[(scene, int(im), gt_idx)] = e["visib_fract"]
    if not out:
        sys.exit(f"no scene_gt_info.json below {pattern} — switch the root "
                 f"with --datasets-root.")
    return out


def _names(configuration: str):
    """Both spellings of a configuration, internal first, without duplicates."""
    internal = to_internal(configuration, "stage3")
    return list(dict.fromkeys((internal, to_release(internal, "stage3"))))


def records_path(root: str, configuration: str, ds: str, mode: str):
    """Per-dataset records of one configuration, whatever they are called on disk.

    Four candidates per spelling, because both the layout and the naming may
    differ: archive layout (``per_query/<name>/<ds>.json``, as in
    ``results/stage3_bop/``), the same export inside a per-configuration run
    folder, and the run layout (``<ds>_stage3{a,b}/records.json``) either under a
    per-configuration folder or directly in ``root`` when ``root`` IS the run
    folder of ``experiments/stage3_bop.py``.
    """
    m = mode[-1]
    for name in _names(configuration):
        for cand in (os.path.join(root, "per_query", name, f"{ds}.json"),
                     os.path.join(root, name, "per_query", name, f"{ds}.json"),
                     os.path.join(root, name, f"{ds}_stage3{m}", "records.json"),
                     os.path.join(root, f"{ds}_stage3{m}", "records.json")):
            if os.path.isfile(cand):
                return cand
    return None


def combined_path(root: str, configuration: str, mode: str):
    """Summary of the run, for the self-check (archive: ``<name>.json``)."""
    m = mode[-1]
    for name in _names(configuration):
        for cand in (os.path.join(root, f"{name}.json"),
                     os.path.join(root, name, f"combined_stage3{m}.json"),
                     os.path.join(root, f"combined_stage3{m}.json")):
            if os.path.isfile(cand):
                return cand
    return None


def load_records(root: str, configuration: str, ds: str, mode: str) -> list:
    p = records_path(root, configuration, ds, mode)
    if p is None:
        return []
    d = json.load(open(p))
    return d["records"] if isinstance(d, dict) else d


def value_of(rec: dict, mode: str):
    """The quantity to evaluate, per instance."""
    if mode == "3a":
        tr = rec.get("target_rank")
        if str(tr) in ("None", ""):
            return 0.0                      # Ziel gar nicht im Ranking
        return 1.0 if int(tr) == 1 else 0.0
    return float(rec["d_posed"])            # 3b: D_sym in mm


def collect(root: str, configuration: str, mode: str, datasets) -> dict:
    """dataset -> [(visib_fract, value, d_sym_norm|None), ...]"""
    out = {}
    for ds in datasets:
        recs = load_records(root, configuration, ds, mode)
        if not recs:
            continue
        vm = visibility_map(ds)
        rows, missed = [], 0
        for r in recs:
            key = (int(r["scene_id"]), int(r["im_id"]), int(r["gt_idx"]))
            v = vm.get(key)
            if v is None:
                missed += 1
                continue
            norm = float(r["d_sym_norm"]) if "d_sym_norm" in r else None
            rows.append((v, value_of(r, mode), norm))
        if missed:
            print(f"  WARNING {ds}: {missed} of {len(recs)} instances without "
                  f"a visibility annotation")
        out[ds] = rows
    return out


def selfcheck(root: str, configuration: str, data: dict, mode: str) -> None:
    """R@1 computed here against the published number. Aborts on a mismatch."""
    if mode != "3a":
        return
    combined = combined_path(root, configuration, mode)
    if combined is None:
        print("  (no run summary found — "
              "self-check skipped)")
        return
    published = json.load(open(combined)).get("recall@1")
    if published is None:
        print(f"  (no recall@1 in {combined} — self-check skipped)")
        return
    allrows = [x for rows in data.values() for x in rows]
    mine = sum(x[1] for x in allrows) / len(allrows)
    print(f"  self-check R@1: own computation {mine:.6f} against "
          f"published {published:.6f}")
    if abs(mine - published) > 1e-4:
        sys.exit("  ABORT: deviation too large — the join is wrong.")
    print("  -> matches, the join is correct.")


def pearson(rows) -> float:
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    n = len(rows)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def bin_label(lo, hi):
    return f"{lo*100:.0f}-{hi*100:.0f} %"


def summarise(rows, edges, mode):
    """Per bin: (label, n, value). 3a averages (hit rate), 3b takes the median."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r[1] for r in rows if lo <= r[0] < hi]
        if not sel:
            out.append((bin_label(lo, hi), 0, float("nan")))
            continue
        val = (sum(sel) / len(sel)) if mode == "3a" else statistics.median(sel)
        out.append((bin_label(lo, hi), len(sel), val))
    return out


def summarise_norm(rows, edges):
    """Per bin: (label, n, median in mm, median relative to the diameter).

    ``collect()`` puts ``d_sym_norm`` into the THIRD slot of every row — that is
    what the second output needs. Only 3b has the field (a retrieval rank has no
    length), in 3a the slot is ``None`` and no row is produced. ``median_mm`` is
    the same median as ``value`` in the first CSV, so that both columns can be
    read side by side.
    """
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[0] < hi]
        with_norm = [r for r in sel if r[2] is not None]
        if not with_norm:
            continue
        if len(with_norm) != len(sel):
            print(f"  WARNING {bin_label(lo, hi)}: {len(sel) - len(with_norm)} "
                  f"of {len(sel)} instances without d_sym_norm — the normalised "
                  f"median is computed over only {len(with_norm)}")
        out.append((bin_label(lo, hi), len(with_norm),
                    statistics.median([r[1] for r in with_norm]),
                    statistics.median([r[2] for r in with_norm])))
    return out


def main():
    global _PATHS
    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["3a", "3b", "both"], default="both",
                    help="3a = retrieval (R@1), 3b = pose (D_sym). Default both.")
    ap.add_argument("--results-root", default=None,
                    help="root holding the Stage-3 records (default: "
                         "<runs_root>/stage3_bop; the frozen record of the "
                         "thesis: results/stage3_bop)")
    ap.add_argument("--configuration", default=None,
                    help="configuration, release or internal name (default: "
                         "retrieval_cross_partial for 3a, "
                         "pose_proxy_cross_partial for 3b)")
    ap.add_argument("--datasets", default="ycbv,tless,lmo")
    ap.add_argument("--bins", default="0.5,0.8,0.95",
                    help="inner bin edges, comma-separated (default 0.5,0.8,0.95).")
    ap.add_argument("--csv", default=None,
                    help="result CSV (default: <runs_root>/stage3_occlusion/"
                         "occlusion_by_visibility.csv)")
    ap.add_argument("--normalized-out", default=None,
                    help="second CSV with the median relative to the object "
                         "diameter, 3b only (default: the --csv with the "
                         "suffix '_normalized', i.e. next to it)")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()
    _PATHS = _paths_mod.from_args(args)

    root = args.results_root or os.path.join(_PATHS["runs_root"], "stage3_bop")
    if not os.path.isabs(root):
        root = os.path.join(_REPO, root)
    out_csv = args.csv or os.path.join(_PATHS["runs_root"], "stage3_occlusion",
                                       "occlusion_by_visibility.csv")
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(_REPO, out_csv)
    # Default der zweiten Ausgabe: neben der ersten, gleicher Name + _normalized
    # (results/stage3_bop/occlusion_by_visibility{,_normalized}.csv).
    norm_csv = args.normalized_out
    if norm_csv is None:
        stem, ext = os.path.splitext(out_csv)
        norm_csv = f"{stem}_normalized{ext}"
    if not os.path.isabs(norm_csv):
        norm_csv = os.path.join(_REPO, norm_csv)

    edges = [0.0] + [float(x) for x in args.bins.split(",")] + [1.0001]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    modes = ["3a", "3b"] if args.mode == "both" else [args.mode]
    csv_rows = []
    norm_rows = []
    print(f"[occl] results-root={root}")

    for mode in modes:
        configuration = args.configuration or DEFAULT_CONFIGURATION[mode]
        release = to_release(to_internal(configuration, "stage3"), "stage3")
        title = ("Retrieval — does it find the exact CAD? (R@1)" if mode == "3a"
                 else "Pose with a substitute model — D_sym median in mm")
        print(f"\n=== {mode}  {title}")
        print(f"    configuration: {release}")
        data = collect(root, configuration, mode, datasets)
        if not data:
            print(f"    no records found — skipped. Expected "
                  f"per_query/{release}/<ds>.json or "
                  f"<ds>_stage3{mode[-1]}/records.json below {root}.")
            continue
        selfcheck(root, configuration, data, mode)

        allrows = [x for rows in data.values() for x in rows]
        print(f"\n  {'visibility':<16}{'n':>8}{'share':>9}{'value':>10}")
        for label, n, val in summarise(allrows, edges, mode):
            share = 100 * n / len(allrows) if allrows else 0
            print(f"  {label:<16}{n:>8}{share:>8.1f} %{val:>10.3f}")
            csv_rows.append({"mode": mode, "configuration": release,
                             "dataset": "ALL", "bin": label, "n": n,
                             "value": round(val, 4)})
        for label, n, mm, nz in summarise_norm(allrows, edges):
            norm_rows.append({"mode": mode, "configuration": release,
                              "dataset": "ALL", "bin": label, "n": n,
                              "median_mm": round(mm, 4),
                              "median_norm": round(nz, 4)})
        print(f"  {'total':<16}{len(allrows):>8}")
        print(f"  Pearson r (visibility vs metric): {pearson(allrows):+.3f}")

        # Kontrolle: haelt der Effekt INNERHALB jedes Datensatzes? Die Klassen sind
        # unterschiedlich zusammengesetzt (T-LESS stellt den Grossteil der stark
        # verdeckten Instanzen), deshalb ist diese Aufschluesselung Pflicht.
        print("\n  per dataset — does the effect hold individually?")
        head = "".join(f"{bin_label(lo, hi):>13}"
                       for lo, hi in zip(edges[:-1], edges[1:]))
        print(f"  {'':<8}{head}{'span':>10}")
        for ds, rows in data.items():
            cells, vals = "", []
            for label, n, val in summarise(rows, edges, mode):
                cells += f"{val:>8.3f} ({n:>3})" if n else f"{'—':>13}"
                if n:
                    vals.append(val)
                    csv_rows.append({"mode": mode, "configuration": release,
                                     "dataset": ds, "bin": label, "n": n,
                                     "value": round(val, 4)})
            for label, n, mm, nz in summarise_norm(rows, edges):
                norm_rows.append({"mode": mode, "configuration": release,
                                  "dataset": ds, "bin": label, "n": n,
                                  "median_mm": round(mm, 4),
                                  "median_norm": round(nz, 4)})
            span = (max(vals) - min(vals)) if vals else float("nan")
            print(f"  {ds:<8}{cells}{span:>10.3f}")

    if not csv_rows:
        sys.exit(f"[occl] no records below {root} — no CSV written. "
                 f"Either put a Stage-3 run there "
                 f"(experiments/stage3_bop.py) or read the frozen record: "
                 f"--results-root results/stage3_bop")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        # LF, like results/stage3_bop/occlusion_by_visibility.csv — csv's default
        # dialect would write CRLF and the re-run would not diff byte-identically.
        w = csv.DictWriter(fh, fieldnames=["mode", "configuration", "dataset",
                                           "bin", "n", "value"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nwritten: {out_csv} ({len(csv_rows)} rows)")

    if norm_rows:
        os.makedirs(os.path.dirname(norm_csv), exist_ok=True)
        with open(norm_csv, "w", newline="") as fh:
            # LF, wie results/stage3_bop/occlusion_by_visibility_normalized.csv.
            w = csv.DictWriter(fh, fieldnames=["mode", "configuration",
                                               "dataset", "bin", "n",
                                               "median_mm", "median_norm"],
                               lineterminator="\n")
            w.writeheader()
            w.writerows(norm_rows)
        print(f"written: {norm_csv} ({len(norm_rows)} rows)")
    else:
        print(f"not written: {norm_csv} — no d_sym_norm in the records "
              f"(only 3b has the field).")


if __name__ == "__main__":
    main()
