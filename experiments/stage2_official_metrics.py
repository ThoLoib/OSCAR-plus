#!/usr/bin/env python3
"""
stage2_official_metrics.py
==========================
Official MI3DOR metrics for all seven arms, from the eval_trace positions.

Since commit ea84ffb8 (2026-08-07) eval_common._make_per_query_record writes a
field ``eval_trace`` into every query record, with ``num_rel_true`` and, per
arm, ``len`` + ``rel_positions`` (1-based positions of ALL relevant models over
the full ranking). The binary relevance vector can be reconstructed from that
without loss — so the official metrics (cross_performance.m, port:
evaluation/mi3dor_official_metrics.py, MATLAB source commit 4325c24c) can be
recomputed for every arm WITHOUT a capture or GPU run. Pure numpy/json
arithmetic: runs directly on the host, NO container needed.

Inputs: three Stage-2 run folders (each with results_topk_15.json),
defaults below <runs_root>:

  --run-partial        <runs_root>/stage2_mi3dor/partial
                       (produce: python3 experiments/stage2_mi3dor.py --gallery partial)
  --run-fullmesh       <runs_root>/stage2_mi3dor/fullmesh
                       (produce: python3 experiments/stage2_mi3dor.py --gallery fullmesh)
  --run-legacy-views8  <runs_root>/stage2_mi3dor_views8/fullmesh
                       (produce: python3 experiments/stage2_mi3dor.py --gallery fullmesh --views 8)

The configuration column carries the release names (fused_partial,
fused_fullmesh, legacy_oscar_views8), matching the archived reference.

Output: <runs_root>/<--out>/official_metrics.csv (21 rows, six decimal places)
+ official_metrics_manifest.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(1, os.path.join(_REPO, "evaluation"))

from mi3dor_official_metrics import official_metrics  # noqa: E402

ARMS = [
    "clip_only", "dino_only_full", "ulip_only_full", "clip_dino_ulip_full",
    "oscar_maxview", "oscar_softmax", "clip_pruned_dino_ulip",
]

GALLERY_LEN = 3848
N_QUERIES = 10500
C_MIN, C_MAX = 31, 250
T_MAX_EXPECTED = 250
METRIC_COLS = ["NN", "FT", "ST", "F", "DCG", "ANMRR", "AUC"]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_integrity(records) -> dict:
    """Structure check BEFORE computing; raises on every violation."""
    assert len(records) == N_QUERIES, f"{len(records)} != {N_QUERIES} records"
    c_vals = []
    for i, rec in enumerate(records):
        et = rec.get("eval_trace")
        assert et is not None, f"Record {i}: eval_trace missing"
        c = int(et["num_rel_true"])
        assert C_MIN <= c <= C_MAX, f"Record {i}: C={c} not in {C_MIN}..{C_MAX}"
        c_vals.append(c)
        arms = et["arms"]
        missing = [a for a in ARMS if a not in arms]
        assert not missing, f"Record {i}: arms missing: {missing}"
        for a in ARMS:
            t = arms[a]
            assert int(t["len"]) == GALLERY_LEN, \
                f"Record {i}/{a}: len={t['len']} != {GALLERY_LEN}"
            pos = np.asarray(t["rel_positions"], dtype=np.int64)
            assert len(pos) == c, \
                f"Record {i}/{a}: {len(pos)} positions != num_rel_true {c}"
            assert pos[0] >= 1 and pos[-1] <= GALLERY_LEN and \
                np.all(np.diff(pos) > 0), \
                f"Record {i}/{a}: positions not strictly increasing in 1..{GALLERY_LEN}"
    t_max = max(c_vals)
    assert t_max == T_MAX_EXPECTED, f"T_max={t_max} != {T_MAX_EXPECTED}"
    return {"records": len(records), "C_min": min(c_vals), "C_max": t_max,
            "T_max": t_max, "status": "OK"}


def arm_metrics(records, arm: str, t_max: int):
    """Reconstruct the arm's binary relevance vectors and run official_metrics."""
    rres, cs, nn_count = [], [], 0
    for rec in records:
        et = rec["eval_trace"]
        t = et["arms"][arm]
        v = np.zeros(int(t["len"]), dtype=np.int64)
        v[np.asarray(t["rel_positions"], dtype=np.int64) - 1] = 1
        nn_count += int(v[0])
        rres.append(v)
        cs.append(int(et["num_rel_true"]))
    m = official_metrics(rres, cs, t_max)
    return m, nn_count


def main() -> None:
    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-partial", default=None,
                    help="run folder with results_topk_15.json (default: "
                         "<runs_root>/stage2_mi3dor/partial)")
    ap.add_argument("--run-fullmesh", default=None,
                    help="run folder with results_topk_15.json (default: "
                         "<runs_root>/stage2_mi3dor/fullmesh)")
    ap.add_argument("--run-legacy-views8", default=None,
                    help="run folder with results_topk_15.json (default: "
                         "<runs_root>/stage2_mi3dor_views8/fullmesh)")
    ap.add_argument("--out", default="stage2_official_metrics",
                    help="output folder name below runs_root (default: "
                         "stage2_official_metrics)")
    ap.add_argument("--script-commit", default="",
                    help="git commit of this script (for the manifest)")
    ap.add_argument("--port-commit", default="c94226f6",
                    help="git commit of the metric port (for the manifest)")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()
    _paths = _paths_mod.from_args(args)

    def _abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(_REPO, p)

    runs = [
        # (run_id, gallery, views, results_dir, erzeugen-mit)
        ("fused_partial", "partial", 42,
         _abs(args.run_partial or
              os.path.join(_paths["runs_root"], "stage2_mi3dor", "partial")),
         "python3 experiments/stage2_mi3dor.py --gallery partial"),
        ("fused_fullmesh", "fullmesh", 42,
         _abs(args.run_fullmesh or
              os.path.join(_paths["runs_root"], "stage2_mi3dor", "fullmesh")),
         "python3 experiments/stage2_mi3dor.py --gallery fullmesh"),
        ("legacy_oscar_views8", "fullmesh", 8,
         _abs(args.run_legacy_views8 or
              os.path.join(_paths["runs_root"], "stage2_mi3dor_views8", "fullmesh")),
         "python3 experiments/stage2_mi3dor.py --gallery fullmesh --views 8"),
    ]

    out_dir = os.path.join(_paths["runs_root"], args.out)
    missing = []
    for run_id, gallery, views, d, howto in runs:
        for fn in ("results_topk_15.json",):
            if not os.path.isfile(os.path.join(d, fn)):
                missing.append(f"  {run_id}: {os.path.join(d, fn)} is missing "
                               f"— produce with: {howto}")
                break
    if missing:
        sys.exit("Inputs missing — three Stage-2 runs are expected "
                 "(each with results_topk_15.json):\n" + "\n".join(missing))
    manifest = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "evaluator_source": ("github.com/tianbao-li/MI3DOR "
                             "Retrieval/cross_performance.m, commit "
                             "4325c24c91553283ec98d948513063f659771767"),
        "port": {"file": "evaluation/mi3dor_official_metrics.py",
                 "commit": args.port_commit, "sha256": sha256_of(
                     os.path.join(_REPO, "evaluation",
                                  "mi3dor_official_metrics.py"))},
        "script": {"file": "experiments/stage2_official_metrics.py",
                   "commit": args.script_commit or "(not provided)",
                   "sha256": sha256_of(os.path.abspath(__file__))},
        "traces_since": "eval_common commit ea84ffb8 (2026-08-07): eval_trace "
                        "with num_rel_true + len/rel_positions per arm",
        "inputs": {},
    }

    rows = []
    for run_id, gallery, views, run_dir, _howto in runs:
        res_p = os.path.join(run_dir, "results_topk_15.json")
        st = os.stat(res_p)
        print(f"[{run_id}] loading {res_p} ({st.st_size} B) ...", flush=True)
        records = json.load(open(res_p))
        integ = check_integrity(records)
        print(f"[{run_id}] integrity: {integ}", flush=True)
        manifest["inputs"][run_id] = {
            "path": os.path.relpath(res_p, _REPO), "size_bytes": st.st_size,
            "date": datetime.datetime.fromtimestamp(st.st_mtime)
            .isoformat(timespec="seconds"),
            "sha256": sha256_of(res_p), "gallery": gallery, "views": views,
            "integrity": integ,
        }
        for arm in ARMS:
            m, nn_count = arm_metrics(records, arm, integ["T_max"])
            print(f"[{run_id}] {arm:22s} NN {m['NN']:.6f} FT {m['FT']:.6f}",
                  flush=True)
            rows.append([run_id, gallery, views, arm, N_QUERIES, nn_count] +
                        [f"{m[k]:.6f}" for k in METRIC_COLS])
        del records

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "official_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        # LF, wie results/stage2_mi3dor/official_metrics.csv.
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["configuration", "gallery", "views", "variant", "N", "NN_count"] +
                   METRIC_COLS)
        w.writerows(rows)
    man_p = os.path.join(out_dir, "official_metrics_manifest.json")
    json.dump(manifest, open(man_p, "w"), indent=1)
    print(f"\n{len(rows)} rows -> {out_csv}\nmanifest -> {man_p}")


if __name__ == "__main__":
    main()
