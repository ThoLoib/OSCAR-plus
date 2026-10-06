#!/usr/bin/env python3
"""
stage5_grasping.py
==================
Stage 5 — grasping study: all GRASPABLE BOP target objects, each with a 3b proxy
AND a 3c substitute.

The only gate is graspability (smallest dimension 20-78 mm; mug as a
documented 81-mm exception). Proxy quality is NOT a criterion: per object the
MOST FREQUENT rank-1 proxy of the frozen Stage-3 run (the actual retrieval
result) and the most frequent 3c substitute (nb_id; may be a sibling object of
the same dataset). Instances: up to 10 from the 3b records in which the 3b
proxy was rank 1 (a pool < 10 is capped and logged).

The plan is built (if needed) and then computed as a series — the actual
execution is done by ``grasping/stage_5.py`` (building the PyBullet world,
FoundationPose, grasp sampling, Panda execution). This file is only the
driver: plan, paths, service check, result comparison.

Sources of the plan
-------------------
The Stage-3 records from which proxy and 3c substitute are determined per
object are the frozen results of the thesis:
    results/stage3_bop/per_query/pose_proxy_cross_partial/<ds>.json   (3b)
    results/stage3_bop/per_query/pose_decomposition_cross/<ds>.json   (3c)
Switchable to your own Stage-3 runs via ``--proxy-records`` /
``--substitute-records`` (``<runs_root>/stage3_bop/per_query/...``).

Where it runs
-------------
The run needs the container environment (PyBullet, torch, trimesh, open3d) and
the FoundationPose service. On the host the script therefore wraps itself in
``docker compose run --rm oscar-plus`` (``--no-docker`` switches that off).
FoundationPose must have been started on the host BEFOREHAND:

    docker compose up -d foundationpose

``--report`` and ``--plan-only`` need neither container nor service.

Invocations
-----------
    # Complete run (build the plan, then gt + 3b proxy + 3c substitute)
    python3 experiments/stage5_grasping.py

    # Only build the plan and look at it
    python3 experiments/stage5_grasping.py --plan-only

    # Compute with the frozen plan of the thesis
    python3 experiments/stage5_grasping.py --plan results/stage5_grasping/plan.json

    # Print the success rates of an existing trials.csv (no run)
    python3 experiments/stage5_grasping.py --report

All flags not listed are passed on unchanged to ``grasping/stage_5.py``, among
them: ``--object <ds>:<id>`` + ``--proxy <cad>`` (free mode, own series),
``--case ycbv14``, ``--inst 3``, ``--runs 10``, ``--yaw-step 36``,
``--n-tries 5``, ``--mass 0.411``, ``--no-exec``, ``--verbose``.

The result lands under ``<runs_root>/stage5_grasping/`` (``--out``): ``plan.json``
and ``trials.csv`` as in the frozen result, plus ``run_config.json``
(argv, git revision, time) and ``REPORT.md``.
"""
from __future__ import annotations

import collections
import csv as _csv
import datetime as _dt
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# CLI-Prelude — VOR den schweren Imports (grasping.* zieht PyBullet/torch),
# damit `--help`, `--report` und `--plan-only` auf dem Host laufen.
# ---------------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
sys.path.insert(0, _THIS)
for _p in (_REPO, os.path.join(_REPO, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DS_OBJS = {"ycbv": range(1, 22), "tless": range(1, 31),
           "lmo": [1, 5, 6, 8, 9, 10, 11, 12]}

CONDITION_LABEL = {"gt": "own CAD (gt)", "proxy": "3b proxy",
                   "proxy3c": "3c substitute", "gt_pose": "true pose (gt_pose)"}

# Referenz: results/stage5_grasping/trials.csv (1560 Trials, 52 Objekte x 10
# Instanzen x 3 Bedingungen). Nur bei einem VOLLEN Lauf vergleichbar.
REFERENCE = {"gt": (352, 520), "proxy": (213, 520), "proxy3c": (294, 520)}

FP_START_CMD = "docker compose up -d foundationpose"

_ARGS = None
_PATHS = None
_EXTRA: list = []


def log(msg: str) -> None:
    print(f"[stage5] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Dienste
# ---------------------------------------------------------------------------
def check_foundationpose() -> None:
    """FoundationPose must answer, otherwise every trial fails with fp_error
    (rc=0, empty success rate — exactly the class of silent failures that has
    cost this project runtime twice). Both names are tried: from the host
    'localhost', inside the container the compose service name."""
    import urllib.request
    for url in ("http://localhost:5050/health", "http://foundationpose:5050/health"):
        try:
            urllib.request.urlopen(url, timeout=5).read()
        except Exception:                                      # noqa: BLE001
            continue
        log(f"FoundationPose ok ({url})")
        return
    sys.exit(f"[stage5] FoundationPose service unreachable (neither "
             f"localhost:5050 nor foundationpose:5050). Start on the HOST:  "
             f"{FP_START_CMD}   — then start again.")


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
def _plan_helpers():
    """The plan building blocks of Stage 5. Encapsulated separately so that
    --help, --report and a run with a finished plan work without them
    (grasping.sim_scene pulls in PyBullet/trimesh)."""
    from grasping.instances import _visib
    from grasping.plan import PER_OBJECT, MIN_VISIB, draw
    from grasping.sim_scene import object_name
    return _visib, PER_OBJECT, MIN_VISIB, draw, object_name


def _records(root: str, ds: str, what: str):
    """Load the Stage-3 records of one dataset (list of query entries)."""
    p = os.path.join(root, f"{ds}.json")
    if not os.path.isfile(p):
        sys.exit(f"[stage5] {what} missing: {p}. Expected are the per-query "
                 f"records of a Stage-3 run ({ds}.json per dataset) — "
                 "either the frozen result under "
                 "results/stage3_bop/per_query/ or your own run "
                 "(<runs_root>/stage3_bop/per_query/), path via "
                 "--proxy-records / --substitute-records.")
    return json.load(open(p))


def build_plan(out_dir: str, plan_path: str) -> None:
    _visib, PER_OBJECT, MIN_VISIB, draw, object_name = _plan_helpers()
    plan, warns, rank = [], [], 0
    for ds, objs in DS_OBJS.items():
        r3b = _records(_ARGS.proxy_records, ds, "3b records (proxy per object)")
        r3c = _records(_ARGS.substitute_records, ds,
                       "3c records (substitute per object)")
        mi_path = os.path.join(_PATHS["datasets_root"], ds, "models_eval",
                               "models_info.json")
        if not os.path.isfile(mi_path):
            sys.exit(f"[stage5] {mi_path} missing — without models_info.json the "
                     "graspability gate (smallest dimension) is not decidable. "
                     "Switch the root via --datasets-root.")
        mi = json.load(open(mi_path))
        for oid in objs:
            # Gate: GREIFBAR (20-78 mm kleinste Abmessung; Mug als dokumentierte
            # 81-mm-Ausnahme) — Proxy-Qualitaet ist ausdruecklich KEIN Kriterium.
            info = mi.get(str(oid), {})
            dims = sorted([info.get("size_x", 0), info.get("size_y", 0),
                           info.get("size_z", 0)])
            if not (20 <= dims[0] <= 78 or (ds == "ycbv" and oid == 14)):
                warns.append(f"{ds}{oid}: not graspable (minDim {dims[0]:.0f} mm) "
                             f"— excluded")
                continue
            recs = [r for r in r3b if r["obj_id"] == oid and r.get("top1")]
            if not recs:
                warns.append(f"{ds}{oid}: no 3b records — skipped")
                continue
            proxy3b = collections.Counter(r["top1"] for r in recs).most_common(1)[0][0]
            c3 = collections.Counter(r["nb_id"] for r in r3c
                                     if r["obj_id"] == oid and r.get("nb_id"))
            proxy3c = c3.most_common(1)[0][0] if c3 else ""
            insts = []
            for r in recs:
                if r["top1"] != proxy3b:
                    continue
                v = _visib(ds, r["scene_id"], r["im_id"], r["gt_idx"])
                insts.append(dict(scene=r["scene_id"], im=r["im_id"],
                                  gt_idx=r["gt_idx"], visib=v,
                                  diameter=r.get("diameter")))
            want = min(PER_OBJECT, len(insts))
            good = [i for i in insts if (i["visib"] or 0) >= MIN_VISIB]
            pick = draw(good, want)
            if len(pick) < want:
                rest = sorted((i for i in insts if i not in pick),
                              key=lambda i: -(i["visib"] or 0))
                pick += rest[:want - len(pick)]
            if want < PER_OBJECT:
                warns.append(f"{ds}{oid}: pool only {len(insts)} -> {want} instances")
            rank += 1
            for i in pick:
                plan.append(dict(case=f"{ds}{oid}", rank=rank, tier=0, dataset=ds,
                                 obj_id=oid, name=object_name(ds, oid),
                                 proxy=proxy3b, proxy3c=proxy3c,
                                 pool=len(insts), n_top1=len(insts), **i))
            print(f"#{rank:>2} {ds}{oid:<3} 3b={proxy3b.split('/', 1)[1][:28]:<30} "
                  f"3c={(proxy3c.split('/', 1)[1][:28] if proxy3c else '—'):<30} "
                  f"n={len(pick)}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(ts=_dt.datetime.now().isoformat(timespec="seconds"),
                   scenario="solo-full: all graspable BOP objects (20-78 mm, mug "
                            "exception), 3b proxy + 3c substitute, most frequent "
                            "rank 1, no proxy curation", per_object=PER_OBJECT,
                   proxy_records=_ARGS.proxy_records,
                   substitute_records=_ARGS.substitute_records,
                   warns=warns, plan=plan),
              open(plan_path, "w"), indent=1)
    for w in warns:
        print("WARNING:", w)
    log(f"Plan: {rank} objects, {len(plan)} instances -> {plan_path}")


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------
def summarize(csv_path: str):
    """(n_trials, n_successes) per condition from the trial CSV."""
    out = {}
    with open(csv_path) as fh:
        for row in _csv.DictReader(fh):
            cond = row.get("condition") or ""
            n, w = out.get(cond, (0, 0))
            out[cond] = (n + 1, w + (1 if row.get("succ") == "1" else 0))
    return out


def report(csv_path: str, md_path: str = "") -> None:
    """Print the success rates per condition and (optionally) store them as
    REPORT.md. The reference comparison only runs for a FULL run (same trial
    count as in the frozen result) — a partial run is not comparable."""
    if not os.path.isfile(csv_path):
        sys.exit(f"[stage5] no trial CSV: {csv_path} — start a run "
                 "first (or point --csv at an existing file).")
    summary = summarize(csv_path)
    if not summary:
        sys.exit(f"[stage5] {csv_path} contains no trials.")
    lines = []
    for cond in ("gt_pose", "gt", "proxy", "proxy3c"):
        if cond not in summary:
            continue
        n, w = summary[cond]
        line = (f"RESULT {CONDITION_LABEL.get(cond, cond)}: "
                f"{w}/{n} = {100.0 * w / n:.1f} %")
        ref = REFERENCE.get(cond)
        if ref and n == ref[1]:
            line += (f"   (reference results/stage5_grasping/trials.csv: "
                     f"{ref[0]}/{ref[1]} = {100.0 * ref[0] / ref[1]:.0f} %)")
        elif ref:
            line += f"   [partial run: {n} of {ref[1]} trials — not comparable]"
        print(line, flush=True)
        lines.append(line)
    for cond in sorted(set(summary) - set(CONDITION_LABEL)):
        n, w = summary[cond]
        line = f"RESULT {cond}: {w}/{n} = {100.0 * w / n:.1f} %"
        print(line, flush=True)
        lines.append(line)
    if md_path:
        os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
        with open(md_path, "w") as fh:
            fh.write("# Stage 5 — grasp success per condition\n\n")
            fh.write(f"Source: `{csv_path}`  \n")
            fh.write(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}\n\n")
            fh.write("| Condition | Successes | Trials | Rate |\n|---|---|---|---|\n")
            for cond, (n, w) in summary.items():
                fh.write(f"| {CONDITION_LABEL.get(cond, cond)} | {w} | {n} | "
                         f"{100.0 * w / n:.1f} % |\n")
        log(f"REPORT.md written: {md_path}")


# ---------------------------------------------------------------------------
def main() -> None:
    plan_path = _ARGS.plan
    if _ARGS.rebuild_plan or not os.path.exists(plan_path):
        build_plan(_ARGS.out, plan_path)
    if _ARGS.plan_only:
        return
    check_foundationpose()
    cmd = [sys.executable, "-m", "grasping.stage_5", "--all", "--canonical",
           "--plan", plan_path, "--conditions", _ARGS.conditions,
           "--csv", _ARGS.csv] + _EXTRA
    log(" ".join(cmd))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_REPO, os.path.join(_REPO, "evaluation")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    rc = subprocess.call(cmd, cwd=_REPO, env=env)
    if rc != 0:
        sys.exit(rc)
    report(_ARGS.csv, os.path.join(_ARGS.out, "REPORT.md"))


if __name__ == "__main__":
    import argparse

    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", default="gt,proxy,proxy3c",
                    help="Arms of the run: gt (FoundationPose + own CAD), "
                         "proxy (3b proxy of the plan), proxy3c (3c substitute), "
                         "gt_pose (true pose). Default: gt,proxy,proxy3c")
    ap.add_argument("--out", default="",
                    help="Result folder (default: <runs_root>/stage5_grasping); "
                         "relative paths against the repo root.")
    ap.add_argument("--plan", default="",
                    help="Plan file (default: <out>/plan.json; is built "
                         "if it is missing).")
    ap.add_argument("--csv", default="",
                    help="Trial CSV (default: <out>/trials.csv; an existing "
                         "run is continued from it).")
    ap.add_argument("--rebuild-plan", action="store_true",
                    help="rebuild the plan even if it already exists")
    ap.add_argument("--plan-only", action="store_true",
                    help="only build and show the plan, do not compute")
    ap.add_argument("--report", action="store_true",
                    help="only print the success rates of the existing --csv "
                         "(no run, no container, no service)")
    ap.add_argument("--proxy-records", default="",
                    help="Folder with the 3b per-query records (<ds>.json). "
                         "Default: results/stage3_bop/per_query/"
                         "pose_proxy_cross_partial")
    ap.add_argument("--substitute-records", default="",
                    help="Folder with the 3c per-query records (<ds>.json). "
                         "Default: results/stage3_bop/per_query/"
                         "pose_decomposition_cross")
    ap.add_argument("--no-docker", action="store_true",
                    help="do not wrap into the oscar-plus container automatically")
    _paths_mod.add_path_args(ap)
    args, _EXTRA = ap.parse_known_args()

    os.environ["PYTHONHASHSEED"] = "0"
    _PATHS = _paths_mod.from_args(args)

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(_REPO, p)

    args.out = (_abs(args.out) if args.out
                else os.path.join(_PATHS["runs_root"], "stage5_grasping"))
    args.plan = _abs(args.plan) if args.plan else os.path.join(args.out, "plan.json")
    args.csv = _abs(args.csv) if args.csv else os.path.join(args.out, "trials.csv")
    args.proxy_records = (_abs(args.proxy_records) if args.proxy_records
                          else os.path.join(_REPO, "results", "stage3_bop",
                                            "per_query",
                                            "pose_proxy_cross_partial"))
    args.substitute_records = (
        _abs(args.substitute_records) if args.substitute_records
        else os.path.join(_REPO, "results", "stage3_bop", "per_query",
                          "pose_decomposition_cross"))
    _ARGS = args

    # --report wertet nur die CSV aus: kein Container, kein Dienst, keine
    # schweren Imports.
    if args.report:
        report(args.csv, os.path.join(args.out, "REPORT.md"))
        sys.exit(0)

    _IN_CONTAINER = os.path.exists("/.dockerenv")
    if not _IN_CONTAINER and not args.no_docker and not args.plan_only:
        if not os.path.isfile(os.path.join(_REPO, "docker-compose.yml")):
            sys.exit(f"[stage5] {_REPO}/docker-compose.yml missing — "
                     "incomplete repo?")
        print("[stage5] the run needs the container environment (PyBullet, "
              "torch) — wrapping into the oscar-plus container automatically.",
              flush=True)
        os.chdir(_REPO)  # docker compose needs the repo as CWD
        os.execvp("docker", ["docker", "compose", "run", "--rm", "oscar-plus",
                             "python3", "/app/experiments/stage5_grasping.py"]
                  + sys.argv[1:])

    # Volle Lauf-Provenienz im Ausgabeordner, geschrieben BEVOR gerechnet wird.
    import datetime

    os.makedirs(args.out, exist_ok=True)
    try:
        _rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:                                          # noqa: BLE001
        _rev = ""
    json.dump({"argv": sys.argv[1:], "extra_to_stage_5": _EXTRA, "git": _rev,
               "derived": {"out": args.out, "plan": args.plan, "csv": args.csv,
                           "conditions": args.conditions,
                           "proxy_records": args.proxy_records,
                           "substitute_records": args.substitute_records,
                           "datasets_root": _PATHS["datasets_root"],
                           "caches_root": _PATHS["caches_root"]},
               "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
               "time": datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(args.out, "run_config.json"), "w"), indent=1)

    main()
