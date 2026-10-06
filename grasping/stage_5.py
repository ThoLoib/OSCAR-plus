#!/usr/bin/env python3
"""Solo trial — scenario 2 (OBJEKTAUSWAHL_SOLO.md): ONE object alone on the table.

Reuses the Stage-5 building blocks unchanged; the only difference from the study
trial (grasping/trial.py, run_trial): the PyBullet world contains ONLY the
target object ("BOP frame minus clutter" — camera, conventions and initial pose
still come from the frozen frame of the plan). FoundationPose therefore sees
the SIM rendering (RGB-D + segmentation mask) instead of the real sensor image,
because the clutter stands in the real image. D_sym is computed against the
settled solo pose, not against the BOP annotation.

====================================================================
SWAPPING OBJECT AND PROXY — the free mode, no plan, no code change:

    # gt series: own CAD, 10 runs (each +36 degrees rotated), success rate at the end
    docker compose run --rm oscar-plus python3 -m grasping.stage_5 --object tless:12

    # proxy series: planned on the proxy, the real object grasped
    docker compose run --rm oscar-plus python3 -m grasping.stage_5 \
        --object tless:12 --proxy itodd/obj_000013

  Condition picked automatically: --proxy empty or equal to the object itself ->
  gt run (own CAD); --proxy set -> proxy run. At the end comes the success rate
  (X/N). Series size: --runs (default 10), rotation per run: --yaw-step (36).

  --object <ds>:<id>   the target object: ycbv | tless | lmo + BOP obj_id.
                       Scene/frame (camera + initial pose) the script picks
                       itself (most visible frame); overrides: --scene, --im.
  --proxy <id>         the CAD that is planned on: pool CAD (gso/... ,
                       housecat6d/... , itodd/...; list: --list-proxies
                       [filter]) OR a BOP object (e.g. tless/obj_000006).

  Further parameters worth adapting per object:
  --conditions gt_pose,gt,proxy   which arms run (gt_pose = true pose,
                                  gt = FoundationPose + own CAD,
                                  proxy = FoundationPose + --proxy)
  --n-tries 5                     grasp attempts per trial
  --mass 0.411                    mass of the target in kg (default: the known
                                  YCB mass from sim_scene.OBJECT_MASS_KG,
                                  else 0.2 kg)
  --csv <file>                    own result file (default solo_custom/)
  Gripper: 5-80 mm opening (PROTOCOL in grasping/trial.py).

Series mode (the prepared list): --all or --case/--inst, fed from
--plan (default <runs_root>/stage5_grasping/plan.json, built by
experiments/stage5_grasping.py: every graspable BOP object, each with a 3b proxy
and a 3c substitute). Fourth
condition in series mode: proxy3c (FoundationPose + the 3c substitute; may be a
BOP sibling object, which is resolved exactly like a target CAD).
====================================================================

    docker compose run --rm oscar-plus python3 -m grasping.stage_5 --case ycbv14
    docker compose run --rm oscar-plus python3 -m grasping.stage_5 --case tless30 --inst 2 --verbose
    docker compose run --rm oscar-plus python3 -m grasping.stage_5 --all        # whole series, resume
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import paths as _paths_mod                                      # noqa: E402

_PATHS = _paths_mod.resolve()
# Defaults der Serie: Plan und Trial-CSV liegen im Ergebnisordner von
# experiments/stage5_grasping.py, der sie auch anlegt und uebergibt.
_S5_OUT = os.path.join(_PATHS["runs_root"], "stage5_grasping")

from grasping.trial import (Ctx, PROTOCOL, _append_csv, _cad_under_test,      # noqa: E402
                            _fp_pose, _grasp_step)
from grasping.sim_scene import TabletopSim                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--object", default="",
                    help="free mode: target object as <ds>:<id>, e.g. tless:12")
    ap.add_argument("--proxy", default="",
                    help="free mode: proxy CAD as <source>/<name>, "
                         "e.g. itodd/obj_000013 (list: --list-proxies)")
    ap.add_argument("--scene", default="", help="free mode: force the scene (e.g. 000012)")
    ap.add_argument("--im", type=int, default=-1, help="free mode: force the frame")
    ap.add_argument("--list-proxies", nargs="?", const="", default=None, metavar="FILTER",
                    help="print every proxy candidate (optionally filtered) and exit")
    ap.add_argument("--runs", type=int, default=10,
                    help="free mode: number of runs in the series (default 10; "
                         "each run rotated by --yaw-step)")
    ap.add_argument("--mass", type=float, default=0.0,
                    help="mass of the target object in kg (overrides the "
                         "default mass; 0 = default)")
    ap.add_argument("--canonical", action="store_true",
                    help="stand the object up GRASPABLY instead of the BOP resting "
                         "pose: the most stable standing pose whose horizontal "
                         "extent fits the gripper (78 mm), plus a deterministic "
                         "rotation about the vertical axis per instance "
                         "(instance index x --yaw-step). Result CSV: solo_full/")
    ap.add_argument("--yaw-step", type=float, default=36.0,
                    help="degrees per instance index in --canonical mode (default 36)")
    ap.add_argument("--case", default="",
                    help="case as in plan.json (e.g. ycbv14, tless30); with --all: "
                         "run only this case")
    ap.add_argument("--inst", type=int, default=0, help="instance index in the plan (0..9)")
    ap.add_argument("--all", action="store_true",
                    help="every case x instance of the plan (resuming from the CSV)")
    ap.add_argument("--plan", default=os.path.join(_S5_OUT, "plan.json"),
                    help="plan file (default: the full plan from "
                         "experiments/stage5_grasping.py)")
    ap.add_argument("--conditions", default="gt_pose,gt,proxy",
                    help="arms: gt_pose (true pose), gt (FP + own CAD), proxy "
                         "(FP + the plan's 3b proxy), proxy3c (FP + the plan's 3c substitute)")
    ap.add_argument("--csv", default="",
                    help="default: <runs_root>/stage5_grasping/trials.csv for "
                         "--all --canonical, otherwise a separate subfolder "
                         "(custom/run/smoke)")
    ap.add_argument("--n-tries", type=int, default=PROTOCOL["executor"]["n_tries"])
    ap.add_argument("--no-exec", dest="exec", action="store_false")
    ap.add_argument("--verbose", action="store_true")
    # Felder, die Ctx/_grasp_step aus der Studie erwarten:
    ap.add_argument("--world", default="auto")
    ap.add_argument("--pose-source", default="fp", choices=["fp"])   # solo: immer frisch
    args = ap.parse_args()

    if args.list_proxies is not None:
        from grasping.proxy_gallery import proxy_pool
        for pid in proxy_pool():
            if args.list_proxies.lower() in pid.lower():
                print(pid)
        return

    if not args.csv:
        # Der Vollauf schreibt direkt nach <runs_root>/stage5_grasping/; jede
        # andere Betriebsart in einen eigenen Unterordner, damit sie die
        # Studien-CSV nicht anfasst.
        sub = ("" if (args.canonical and args.all) else
               "custom" if args.object else
               "run" if args.all else "smoke")
        args.csv = os.path.join(_S5_OUT, sub, "trials.csv")
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if args.object:
        base = _custom_tr(args)
        self_id = f"{base['dataset']}/obj_{base['obj_id']:06d}"
        if args.conditions == "gt_pose,gt,proxy":     # Default -> automatisch:
            # Proxy leer oder das Objekt selbst -> gt-Lauf (eigenes CAD);
            # Proxy gesetzt -> Proxy-Lauf (auf dem Proxy geplant).
            conds = ["gt"] if args.proxy in ("", self_id) else ["proxy"]
            print(f"[stage5] condition: {conds[0]} "
                  f"({'planned on the proxy' if conds == ['proxy'] else 'own CAD'})")
        if args.runs > 1 and not args.canonical:
            args.canonical = True                     # Serie steht immer kanonisch
            print(f"[stage5] series: {args.runs} runs, canonical standing pose, "
                  f"+{args.yaw_step:g}° rotation per run")
        todo = [dict(base, inst=i) for i in range(max(1, args.runs))]
    elif args.all:
        todo = json.load(open(args.plan))["plan"]
        if args.case:                                 # --all --case X = nur dieser Fall
            todo = [t for t in todo if t["case"] == args.case]
        if args.proxy:                                # Proxy-Override fuer Sonderlaeufe
            todo = [dict(t, proxy=args.proxy) for t in todo]
            print(f"[stage5] proxy override: {args.proxy}")
    else:
        if not args.case:
            sys.exit("a single trial needs --case (e.g. --case ycbv14), "
                     "or --all for the series, or --object for the free mode")
        plan = json.load(open(args.plan))["plan"]
        trs = [t for t in plan if t["case"] == args.case]
        if not trs:
            sys.exit(f"case {args.case} is not in the plan ({sorted({t['case'] for t in plan})})")
        todo = [trs[args.inst]]

    done = set()
    if args.object:
        pass          # freier Modus: kein Resume — jede Eingabe = frische Serie
    elif os.path.exists(args.csv):
        import csv as _csv
        for d in _csv.DictReader(open(args.csv)):
            done.add((d["case"], str(d["scene"]), str(d["im"]), str(d["gt_idx"]),
                      d["condition"]))
        if done:
            print(f"[stage5] resume: {len(done)} trials already in {args.csv}")

    ctx = Ctx(args)
    os.environ.setdefault("GRASP_QUIET", "1")
    if args.mass > 0:
        from grasping import sim_scene
        for t in todo:
            sim_scene.OBJECT_MASS_KG[(t["dataset"], t["obj_id"])] = args.mass
        print(f"[stage5] target mass from the CLI: {args.mass} kg")
    n_total = sum(1 for t in todo for c in conds
                  if (t["case"], str(t["scene"]), str(t["im"]), str(t["gt_idx"]), c)
                  not in done)
    n_run = 0
    import csv as _csv2
    n_pre = (sum(1 for _ in _csv2.DictReader(open(args.csv)))
             if args.object and os.path.exists(args.csv) else 0)
    for tr in todo:
        pend = [c for c in conds
                if (tr["case"], str(tr["scene"]), str(tr["im"]), str(tr["gt_idx"]), c)
                not in done]
        if not pend:
            continue
        print(f"[stage5] {tr['case']} ({tr['name']}) inst s{tr['scene']}/im{tr['im']} — "
              f"proxy {tr['proxy']}")
        fr, objs, cam, winfo = ctx.scene(tr["dataset"], tr["scene"], tr["im"])
        tgt = objs[tr["gt_idx"]]
        assert tgt.obj_id == tr["obj_id"]
        if args.canonical:
            idx = tr["inst"] if "inst" in tr else \
                [t for t in todo if t["case"] == tr["case"]].index(tr)
            yaw = idx * args.yaw_step
            tgt, graspable = canonical_object(ctx, tgt, yaw)
            tr = dict(tr, yaw_deg=yaw, _graspable=int(graspable), _table_z=0.0)
            if not graspable:
                print(f"[stage5]   NOTE {tr['case']}: no graspable standing pose "
                      f"(<=78 mm) — used the most stable one, marked as an exception")
        solo = [tgt]                                # <- der ganze Szenariowechsel
        n_run += run_conditions(ctx, args, tr, fr, objs, cam, winfo, tgt, solo, pend,
                                n_run, n_total)
    print(f"[stage5] done: {n_run} new trials, CSV: {args.csv}")
    if args.object:
        new = list(_csv2.DictReader(open(args.csv)))[n_pre:]
        wins = sum(1 for r in new if r["succ"] == "1")
        what = args.proxy if conds == ["proxy"] else "own CAD"
        print(f"\n[stage5] ============================================")
        print(f"[stage5] RESULT {todo[0]['name']} | {what}: "
              f"success rate {wins}/{len(new)}")
        print(f"[stage5] ============================================")


_CANON_CACHE: dict = {}


def canonical_pose(mesh_m, max_grip: float = 0.078):
    """Deterministic standing pose: among the mesh's stable resting poses
    (trimesh, sigma=0 -> no randomness) the MOST stable one whose horizontal
    extent is <= max_grip (the object is then graspable from the side).
    Returns (T_stable 4x4, graspable: bool); without a graspable standing pose
    the most stable one overall + False (applicability exception, it is logged)."""
    import numpy as np
    import trimesh
    key = id(mesh_m)
    if key in _CANON_CACHE:
        return _CANON_CACHE[key]
    Ts, probs = trimesh.poses.compute_stable_poses(mesh_m, sigma=0.0, n_samples=1)
    best, best_any = None, None
    for T, p in zip(Ts, probs):
        m2 = mesh_m.copy()
        m2.apply_transform(T)
        horiz = float(min(m2.extents[0], m2.extents[1]))
        if best_any is None or p > best_any[1]:
            best_any = (T, p)
        if horiz <= max_grip and (best is None or p > best[1]):
            best = (T, p)
    out = ((best[0], True) if best is not None else (best_any[0], False))
    _CANON_CACHE[key] = out
    return out


def canonical_object(ctx, tgt, yaw_deg: float):
    """SceneObject copy: stood up graspably at the frame's table position,
    rotated by yaw_deg about the vertical axis, bottom edge at z=0."""
    import dataclasses
    import numpy as np
    mesh_m = ctx.mesh_m(tgt.mesh_path, True)
    T_st, graspable = canonical_pose(mesh_m)
    a = np.radians(yaw_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    T = Rz @ T_st
    m2 = mesh_m.copy()
    m2.apply_transform(T)
    # Zentrum der Grundflaeche an die Tischposition des Frames, Unterkante auf z=0
    cx, cy = m2.bounds.mean(axis=0)[:2]
    T[0, 3] += tgt.T_world[0, 3] - cx
    T[1, 3] += tgt.T_world[1, 3] - cy
    T[2, 3] += -float(m2.bounds[0, 2])
    return dataclasses.replace(tgt, T_world=T), graspable


def _custom_tr(args) -> dict:
    """Free mode: build a trial entry from --object (+ optionally --scene/--im).
    Frame choice: if the object is in the full plan, the frame of its
    experiment instance 1 is used (same camera geometry as the tables — the
    merely "most visible" frame sometimes yields markedly worse FP fits);
    otherwise the most visible frame. Own frame any time via --scene/--im."""
    import glob as _glob
    from grasping.instances import _test_root
    from grasping.sim_scene import load_bop_frame, object_name
    try:
        ds, oid = args.object.replace("/", ":").split(":")
        oid = int(oid)
    except ValueError:
        sys.exit(f"--object '{args.object}' not readable — format <ds>:<id>, e.g. tless:12")
    plan_gt_idx = None
    if not args.scene and args.im < 0:
        plan_p = (getattr(args, "plan", "")
                  or os.path.join(_S5_OUT, "plan.json"))
        if os.path.isfile(plan_p):
            for t in json.load(open(plan_p))["plan"]:
                if t["dataset"] == ds and t["obj_id"] == oid:
                    args.scene = f"{int(t['scene']):06d}"
                    args.im = int(t["im"])
                    plan_gt_idx = t["gt_idx"]
                    print(f"[stage5] frame from the experiment plan: scene "
                          f"{args.scene}/im {args.im} (as table instance 1)")
                    break
    scene = args.scene
    if not scene:
        for sdir in sorted(_glob.glob(os.path.join(_test_root(ds), "*"))):
            gt = os.path.join(sdir, "scene_gt.json")
            if os.path.isfile(gt):
                first = next(iter(json.load(open(gt)).values()))
                if any(e["obj_id"] == oid for e in first):
                    scene = os.path.basename(sdir)
                    break
        if not scene:
            sys.exit(f"no test-split frame with {ds} obj {oid} found.")
    if args.im >= 0:
        im = args.im
    else:
        # sichtbarster Frame des Objekts, robust je Frame bestimmt
        sdir = os.path.join(_test_root(ds), scene)
        gt_all = json.load(open(os.path.join(sdir, "scene_gt.json")))
        info_all = json.load(open(os.path.join(sdir, "scene_gt_info.json")))
        best = None
        for k, entries in gt_all.items():
            for gi, e in enumerate(entries):
                if e["obj_id"] == oid and k in info_all and gi < len(info_all[k]):
                    v = info_all[k][gi].get("visib_fract", 0.0)
                    if best is None or v > best[0]:
                        best = (v, int(k))
        if best is None:
            sys.exit(f"{ds} obj {oid}: no annotated frame in scene {scene}.")
        im = best[1]
    fr = load_bop_frame(ds, scene, im)
    gt_idx = plan_gt_idx if plan_gt_idx is not None else fr.instances_of(oid)[0]
    import json as _json
    mi = _json.load(open(os.path.join(_PATHS["datasets_root"], ds,
                                      "models_eval", "models_info.json")))
    print(f"[stage5] free mode: {ds} obj {oid} ({object_name(ds, oid)}) — "
          f"scene {scene}/im {im} (most visible frame), proxy {args.proxy or '—'}")
    return dict(case=f"custom_{ds}{oid}", rank=0, tier=0, dataset=ds, obj_id=oid,
                name=object_name(ds, oid), proxy=args.proxy, pool="", n_top1="",
                scene=int(scene), im=int(im), gt_idx=gt_idx,
                visib=fr.visib(gt_idx), diameter=mi.get(str(oid), {}).get("diameter"))


def resolve_cad(ctx, tr, cond):
    """Resolve the CAD under test: (cad_id, path, units_m).

    proxy   = 3b proxy (tr["proxy"]) — pool CAD or BOP sibling.
    proxy3c = 3c substitute (tr["proxy3c"]) — may be a BOP sibling.
    BOP siblings are resolved EXACTLY like the object's own CAD (same file,
    same units) — bop_mesh_path returns different units per dataset,
    target_cad encapsulates that correctly. Missing id: KeyError."""
    if cond in ("proxy", "proxy3c"):
        pid = tr["proxy"] if cond == "proxy" else (tr.get("proxy3c") or "")
        if not pid:
            raise KeyError(f"{cond}: no CAD id in the plan")
        src = pid.split("/", 1)[0]
        if src in ("ycbv", "tless", "lmo"):
            cad_path, units_m = ctx.target_cad(src, int(pid.split("obj_")[1]))
        else:
            from grasping.proxy_gallery import proxy_mesh
            cad_path, units_m = proxy_mesh(pid)
        return pid, cad_path, units_m
    return _cad_under_test(ctx, tr, cond)


def run_conditions(ctx, args, tr, fr, objs, cam, winfo, tgt, solo, conds,
                   n_done, n_total) -> int:
    import numpy as np                               # noqa: F811
    ran = 0
    for cond in conds:
        t0 = time.time()
        row = dict(case=tr["case"], rank=tr["rank"], tier=tr["tier"], dataset=tr["dataset"],
                   obj_id=tr["obj_id"], name=tr["name"], scene=tr["scene"], im=tr["im"],
                   gt_idx=tr["gt_idx"], visib="", condition=cond, cad="", cad_units_m="",
                   pose_source="", fp_conf="", fp_fail="", dsym_mm="", dsym_norm="", f05="",
                   place_mm="", settle_mm="", world="solo", plane_inlier="",
                   plane_angle_deg="", bottom_gap_mm="",
                   n_cand=0, n_reach=0, n_blocked=0, n_att=0, n_succ=0, first_succ=0,
                   succ=0, lift_cm=0.0, att_seq="", fail_reason="", runtime_s=0.0,
                   ts=_dt.datetime.now().isoformat(timespec="seconds"))
        if "yaw_deg" in tr:                     # --canonical: eigene Spalten
            row["yaw_deg"] = tr["yaw_deg"]
            row["graspable_pose"] = tr["_graspable"]
        try:
            cad_id, cad_path, units_m = resolve_cad(ctx, tr, cond)
        except KeyError:
            row.update(cad="", fail_reason="cad_missing")
            _append_csv(args.csv, row)
            continue
        row.update(cad=cad_id, cad_units_m=int(units_m))
        sim = TabletopSim().connect()
        try:
            # Ohne Roboter bauen: die Kameraseiten-Platzierung stellt den Panda
            # in die Sichtlinie der BOP-Kamera — im Solo-Rendering wuerde er das
            # Ziel komplett verdecken (in der Studie sah FP das ECHTE Bild, da
            # war das egal). Der Roboter kommt nach dem Rendern dazu.
            tz = tr.get("_table_z")
            sim.build(solo, cam, target_gt_idx=tr["gt_idx"], with_robot=False,
                      table_z=tz if tz is not None else
                      min(0.0, winfo["bottom_z"].get(tr["gt_idx"], 0.0) - 0.001))
            sim.settle(PROTOCOL["physics"]["settle_steps"])
            row["settle_mm"] = round(sim.target_displacement_mm(tgt.T_world), 1)
            sim.freeze_initial()
            T_true = sim.target_pose()

            extra = 1.0
            if cond == "gt_pose":
                T_m2w, row["pose_source"] = T_true, "sim_true"
            else:
                rd = sim.render_rgbd()
                mask = sim.object_mask(tgt.obj_id, rd["seg"]).astype(np.uint8)
                if mask.sum() < 200:
                    row.update(fail_reason="fp_error", fp_fail="solo mask empty")
                    continue
                try:
                    R, t, conf = _fp_pose(cad_path, rd["rgb"], rd["depth"], mask,
                                          cam.K, units_m, extra)
                    row["pose_source"] = "fp_solo_render"
                    row["fp_conf"] = round(float(conf), 3) if conf is not None else ""
                except Exception as exc:                              # noqa: BLE001
                    row.update(fp_fail=str(exc).splitlines()[-1][:80],
                               fail_reason="fp_error")
                    continue
                T_m2c = np.eye(4)
                T_m2c[:3, :3], T_m2c[:3, 3] = R, np.asarray(t, float) / 1000.0
                T_m2w = cam.T_world @ T_m2c
                # D_sym gegen die GESETTELTE Solo-Pose (Kameraframe, mm)
                from stage3_metrics import d_sym
                T_true_c = np.linalg.inv(cam.T_world) @ T_true
                tpath, tunits = ctx.target_cad(tr["dataset"], tr["obj_id"])
                dsr = d_sym(ctx.pts_mm(tpath, tunits), T_true_c[:3, :3],
                            T_true_c[:3, 3] * 1000.0,
                            ctx.pts_mm(cad_path, units_m, extra), R, np.asarray(t, float),
                            tr.get("diameter") or 0.0)
                row.update(dsym_mm=round(dsr["d_sym"], 2),
                           dsym_norm=round(dsr["d_sym_norm"], 4)
                           if dsr["d_sym_norm"] is not None else "",
                           f05=round(dsr["fscore"]["0.05"]["f"], 4))
            sim._add_panda(solo)          # jetzt erst der Roboter (nach dem Rendern)
            sim.freeze_initial()          # Reset-Zustand inkl. Roboter einfrieren
            mesh_m = ctx.mesh_m(cad_path, units_m, extra)
            c_cad = T_m2w[:3, :3] @ mesh_m.centroid + T_m2w[:3, 3]
            c_tgt = T_true[:3, :3] @ ctx.mesh_m(tgt.mesh_path, True).centroid + T_true[:3, 3]
            row["place_mm"] = round(float(np.linalg.norm(c_cad - c_tgt) * 1000), 1)
            _grasp_step(ctx, sim, cad_path, units_m, extra, T_m2w, row)
        finally:
            sim.disconnect()
            row["runtime_s"] = round(time.time() - t0, 1)
            _append_csv(args.csv, row)
            ran += 1
            print(f"[stage5] [{n_done + ran:>3}/{n_total}] {cond:8s} "
                  f"cad={row['cad'][:36]:38s} "
                  f"D_sym={row['dsym_mm'] or '—':>6} place={row['place_mm'] or '—':>6} "
                  f"cand={row['n_cand']:>2} reach={row['n_reach']:>2} "
                  f"blocked={row['n_blocked']:>2} att={row['n_att']} "
                  f"-> {'SUCCESS' if row['succ'] else row['fail_reason'] or 'no success'} "
                  f"({row['runtime_s']} s)")
    return ran


if __name__ == "__main__":
    main()
