#!/usr/bin/env python3
"""Stage-5 · the shared runtime of one grasp trial — is a retrieved proxy CAD
good enough to grasp with?

A trial is (instance, condition).  The BOP frame is rebuilt in PyBullet [R14] —
annotated objects at their GT poses, the target dynamic, a Franka Panda — and the
SAME grasp planner and executor run per condition:

    gt        target's own CAD posed by FoundationPose [R4]  — the reference
    proxy     the retrieved proxy CAD posed by FoundationPose — the pipeline
    gt_pose   target's own CAD at the TRUE (settled) sim pose — planner ceiling (control)
    random    a hash-drawn random gallery CAD, FoundationPose — chance baseline (control)
    proxy_scaled  proxy sized to the observed depth cloud    — size ablation (optional)

FoundationPose sees RGB-D plus a target mask and is called in the Stage-3
configuration (``experiments/stage3_bop.estimate_pose``); pose quality is scored
with the Stage-3 D_sym (``evaluation/stage3_metrics.d_sym``).  A trial succeeds
when one of at most ``n_tries`` executed grasp candidates lifts the target, holds
it for 1 s and survives a shake test [R3] (``grasping/grasp_execute.py``).  Every
attempt starts from a full reset.

``PROTOCOL`` freezes every value that shapes an outcome.  Reporting helpers are
kept here as well: success rates per condition, paired as Δ plus win split — no
intervals, matching the form reported in the thesis.

This module is a library, not a CLI.  The drivers are
``experiments/stage5_grasping.py`` (study), ``grasping/stage_5.py`` (solo trial)
and ``experiments/stage5_visualize.py`` (visualisation).  Reference list:
grasping/README.md.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from typing import Optional, Tuple

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_ROOT,
           os.path.join(_ROOT, "evaluation"),      # stage3_metrics
           os.path.join(_ROOT, "experiments")):    # stage3_bop (estimate_pose, FP_URL)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import paths as _paths_mod                                  # noqa: E402

from grasping.proxy_gallery import proxy_mesh, proxy_pool, random_proxy    # noqa: E402

# Derived caches follow the central registry (config/paths.yaml), like
# sim_scene's bop_obj / vhacd caches.
GRASP_CACHE = os.path.join(_paths_mod.resolve()["caches_root"], "grasp_candidates")

# Every value that shapes an outcome, frozen here and written to the manifest.
PROTOCOL = dict(
    sampler=dict(n_samples=800, top_k=40, friction_mu=0.5, n_approach=4, seed=0,
                 gripper_min_width_m=0.005, gripper_max_width_m=0.08,
                 method="antipodal contact pairs inside the friction cone [R1, R2]"),
    executor=dict(n_tries=5, pregrasp_m=0.12, lift_m=0.15, rise_min_m=0.05,
                  hold_steps=240, shake_amp_m=0.05, in_hand_m=0.15,
                  close_force_n=(20, 120), reach_tol_mm=30, blocked_mm=30,
                  success="rose >= 5 cm AND held 1 s AND survived ±5 cm shakes [R3]; "
                          "trial = any of <= n_tries attempts, full reset before each"),
    physics=dict(engine="PyBullet [R14]", timestep_s=1 / 240, gravity=-9.81,
                 target_mass_kg="sim_scene.OBJECT_MASS_KG (YCB object list) else 0.2",
                 target_friction=1.0, finger_friction=1.0,
                 friction_combination="product (PyBullet; measured 1.0x1.0 -> 1.01, 1.6x1.5 -> 2.42)",
                 settle_steps=60, collision="V-HACD [R13] for the target / concave static clutter"),
    robot=dict(arm="Franka Panda (pybullet_data franka_panda/panda.urdf)", pedestal_m=0.35,
               placement="on the camera's side: 0.55 m from the object centroid along the "
                         "camera viewing direction projected onto the table"),
    world=dict(frame="auto: BOP extrinsics where present (YCB-V, T-LESS), else the table plane "
                     "fitted to the real depth (LM-O)",
               table_plane="RANSAC [R12], objects masked out, ±0.5 m around the object depths, "
                           "thr 6 mm, candidate planes must carry every object 0–40 cm above them",
               table_z="0 = lowest object vertex (bop) / the fitted plane; lowered to the "
                       "target's own lowest vertex if that is below"),
    pose=dict(method="FoundationPose [R4] via stage3_bop.estimate_pose", fp_refine_iter=5,
              input="real RGB-D + BOP GT mask_visib (Stage 3b)",
              d_sym="stage3_metrics.d_sym, 10 000 surface samples, seed 0"),
    sampling=dict(min_visib=0.5, per_object=6,
                  rule="round-robin over scenes, evenly spaced frames within a scene"),
)


# ===========================================================================
# 1. per-run caches: scenes, meshes, Stage-3 surface samples, grasp candidates
# ===========================================================================
class Ctx:
    def __init__(self, args):
        self.args = args
        self.scenes: dict = {}
        self.meshes: dict = {}
        self.pts: dict = {}
        self.grasps: dict = {}
        self.pool = proxy_pool()
        self.eval_fallback: set = set()

    def scene(self, ds: str, scene: int, im: int):
        """(frame, objects, camera, world-check info) for one BOP frame."""
        key = (ds, scene, im)
        if key not in self.scenes:
            from grasping.sim_scene import load_bop_frame, scene_from_frame
            fr = load_bop_frame(ds, scene, im)
            self.scenes[key] = (fr,) + scene_from_frame(fr, world=self.args.world)
        return self.scenes[key]

    def mesh_m(self, path: str, units_m: bool, extra: float = 1.0):
        """trimesh in METRES (native units × extra)."""
        key = (path, units_m, round(extra, 6))
        if key not in self.meshes:
            import trimesh
            m = trimesh.load(path, force="mesh")
            s = (1.0 if units_m else 0.001) * extra
            if abs(s - 1.0) > 1e-9:
                m.apply_scale(s)
            self.meshes[key] = m
        return self.meshes[key]

    def pts_mm(self, path: str, units_m: bool, extra: float = 1.0):
        """Stage-3 surface sample (mm) of a CAD — `stage3_metrics.sample_surface_mm`."""
        key = (path, units_m, round(extra, 6))
        if key not in self.pts:
            from stage3_metrics import sample_surface_mm
            p = sample_surface_mm(path, units_m=units_m)
            self.pts[key] = p * extra if abs(extra - 1.0) > 1e-9 else p
        return self.pts[key]

    def target_cad(self, ds: str, obj_id: int) -> Tuple[str, bool]:
        """The CAD Stage 3 posed the target with (BOP models_eval, mm) — or the
        sim mesh when that split is absent on this machine (reported)."""
        from grasping.sim_scene import eval_mesh_path
        path, units_m = eval_mesh_path(ds, obj_id)
        if units_m:
            self.eval_fallback.add(ds)
        return path, units_m

    def grasp_candidates(self, path: str, units_m: bool, extra: float = 1.0):
        """Object-frame grasps of a CAD (independent of the pose), cached on
        disk per CAD + sampler protocol."""
        P = PROTOCOL["sampler"]
        key = hashlib.md5(f"{path}|{units_m}|{extra:.6f}|{P}".encode()).hexdigest()[:16]
        if key in self.grasps:
            return self.grasps[key]
        import numpy as np
        from grasping.antipodal_grasp_sampler import Grasp, GripperConfig, sample_antipodal_grasps
        os.makedirs(GRASP_CACHE, exist_ok=True)
        f = os.path.join(GRASP_CACHE, key + ".json")
        if os.path.isfile(f) and os.path.getsize(f) > 0:
            gs = [Grasp(center=np.array(g["center"]), axis=np.array(g["axis"]),
                        approach=np.array(g["approach"]), width=g["width"], quality=g["quality"],
                        contacts=(np.array(g["contacts"][0]), np.array(g["contacts"][1])))
                  for g in json.load(open(f))["grasps"]]
        else:
            t0 = time.time()
            gs = sample_antipodal_grasps(
                self.mesh_m(path, units_m, extra),
                GripperConfig(max_width=P["gripper_max_width_m"], min_width=P["gripper_min_width_m"]),
                n_samples=P["n_samples"], friction_mu=P["friction_mu"],
                n_approach=P["n_approach"], top_k=P["top_k"], seed=P["seed"])
            tmp = f"{f}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump({"cad": path, "units_m": units_m, "extra": extra, "protocol": P,
                           "seconds": round(time.time() - t0, 1),
                           "grasps": [g.to_dict() for g in gs]}, fh)
            os.replace(tmp, f)
        self.grasps[key] = gs
        return gs


# ===========================================================================
# 2. one trial = (instance, condition): CAD -> pose -> grasps -> execution
# ===========================================================================
def _cad_under_test(ctx: Ctx, tr: dict, cond: str) -> Tuple[str, Optional[str], bool]:
    """(cad id, mesh path or None, units_m) for a condition."""
    if cond in ("gt", "gt_pose"):
        path, units_m = ctx.target_cad(tr["dataset"], tr["obj_id"])
        return "gt", path, units_m
    cid = random_proxy(tr["dataset"], tr["obj_id"], ctx.pool) if cond == "random" else tr["proxy"]
    path, units_m = proxy_mesh(cid)
    return cid, path, units_m


def _fp_pose(cad_path, rgb, depth_m, mask, K, units_m: bool, extra: float):
    """FoundationPose [R4] in the Stage-3 configuration -> (R 3x3, t mm, conf).
    `extra` != 1 only for the proxy_scaled ablation (Stage 3 never scales)."""
    from stage3_bop import estimate_pose, FP_URL
    if abs(extra - 1.0) < 1e-9:
        return estimate_pose(cad_path, rgb, depth_m, mask, K, mesh_units_m=units_m,
                             refine_iter=PROTOCOL["pose"]["fp_refine_iter"])
    from pipeline.foundationpose_bridge import call_foundationpose
    pose, conf = call_foundationpose(FP_URL, rgb=rgb, depth=depth_m, mask=mask, K=K,
                                     cad_path=cad_path, scale=(1.0 if units_m else 1e-3) * extra,
                                     refine_iter=PROTOCOL["pose"]["fp_refine_iter"])
    return pose[:3, :3], pose[:3, 3] * 1000.0, float(conf)


def _pose_step(ctx: Ctx, tr: dict, cond: str, fr, cad_path: str, units_m: bool, row: dict):
    """Camera-frame pose of the CAD under test + its Stage-3 score. Returns
    (R, t_mm, extra) or None when the pose service failed (row says why)."""
    import numpy as np
    from stage3_metrics import d_sym
    rgb, depth_m = fr.rgb(), fr.depth_m()
    mask = fr.mask_visib(tr["gt_idx"]).astype(np.uint8)
    extra = 1.0
    if cond == "proxy_scaled":                    # size the proxy to the observed depth cloud
        from grasping.perceive import observed_diag
        d_obs = observed_diag(depth_m, mask.astype(bool), fr.K)
        d_cad = float(np.linalg.norm(ctx.mesh_m(cad_path, units_m).extents))
        if d_obs and d_cad > 1e-6:
            extra = d_obs / d_cad
    stored = None
    if ctx.args.pose_source == "stage3" and cond in ("gt", "proxy"):
        stored = tr.get("fp_gt" if cond == "gt" else "fp_proxy")
    if stored is not None:                        # archived Stage-3 pose of this very instance
        R = np.asarray(stored["R"], float).reshape(3, 3)
        t = np.asarray(stored["t"], float).reshape(3)
        conf, row["pose_source"] = stored.get("conf"), "stage3"
    else:
        try:
            R, t, conf = _fp_pose(cad_path, rgb, depth_m, mask, fr.K, units_m, extra)
            row["pose_source"] = "fp"
        except Exception as exc:
            row.update(fp_fail=str(exc).splitlines()[-1][:80], fail_reason="fp_error")
            return None
    row["fp_conf"] = round(float(conf), 3) if conf is not None else ""
    # Stage-3 score: D_sym between the GT-posed target and the posed CAD (camera frame, mm)
    R_gt = np.asarray(fr.gt[tr["gt_idx"]]["cam_R_m2c"], float).reshape(3, 3)
    t_gt = np.asarray(fr.gt[tr["gt_idx"]]["cam_t_m2c"], float).reshape(3)
    tpath, tunits = ctx.target_cad(tr["dataset"], tr["obj_id"])
    dsr = d_sym(ctx.pts_mm(tpath, tunits), R_gt, t_gt,
                ctx.pts_mm(cad_path, units_m, extra), R, t, tr.get("diameter") or 0.0)
    row.update(dsym_mm=round(dsr["d_sym"], 2),
               dsym_norm=round(dsr["d_sym_norm"], 4) if dsr["d_sym_norm"] is not None else "",
               f05=round(dsr["fscore"]["0.05"]["f"], 4))
    return R, t, extra


def _grasp_step(ctx: Ctx, sim, cad_path: str, units_m: bool, extra: float, T_m2w, row: dict):
    """Plan on the posed CAD, execute on the real target: up to n_tries attempts,
    full reset before each, stop at the first success."""
    import numpy as np
    from grasping.grasp_execute import PandaGrasper, reachable_order, feasible_grasps
    from grasping.antipodal_grasp_sampler import transform_grasps, _unit
    gs = ctx.grasp_candidates(cad_path, units_m, extra)
    grasper = PandaGrasper(sim)
    grasper.reset()
    base_xy = sim._p.getBasePositionAndOrientation(sim.robot)[0][:2]
    feas = feasible_grasps(grasper, reachable_order(transform_grasps(gs, T_m2w), base_xy),
                           tol_mm=PROTOCOL["executor"]["reach_tol_mm"])
    row.update(n_cand=len(gs), n_reach=len(feas))
    seq, lifts = [], []
    for g in (feas if ctx.args.exec else []):
        if row["n_att"] >= ctx.args.n_tries:
            break
        sim.reset_objects()
        grasper.reset()
        sim.settle(30)
        r = grasper.execute(g)
        if ctx.args.verbose:
            print(f"        attempt {len(seq) + 1:>2}: q={g.quality:.2f} w={g.width * 1000:.0f}mm "
                  f"approach_z={_unit(g.approach)[2]:+.2f} -> {r}", flush=True)
        if r.get("blocked"):                        # approach corridor occupied: not an attempt
            row["n_blocked"] += 1
            seq.append("B")
            continue
        row["n_att"] += 1
        if r["success"]:
            row["n_succ"] += 1
            row["first_succ"] = int(row["n_att"] == 1)
            lifts.append(r["lift_cm"])
            seq.append("S")
            break
        seq.append("F")
    row.update(att_seq=",".join(seq), succ=int(row["n_succ"] > 0),
               lift_cm=round(float(np.mean(lifts)), 1) if lifts else 0.0)
    if not row["succ"]:
        row["fail_reason"] = ("no_candidates" if not gs else "unreachable" if not feas
                              else "all_blocked" if row["n_att"] == 0 else "grasp_failed")


def run_trial(ctx: Ctx, tr: dict, cond: str) -> dict:
    import numpy as np
    from grasping.sim_scene import TabletopSim
    t0 = time.time()
    ds, gt_idx = tr["dataset"], tr["gt_idx"]
    fr, objs, cam, winfo = ctx.scene(ds, tr["scene"], tr["im"])
    tgt = objs[gt_idx]
    assert tgt.obj_id == tr["obj_id"], f"gt_idx {gt_idx} is obj {tgt.obj_id}, not {tr['obj_id']}"
    row = dict(case=tr["case"], rank=tr["rank"], tier=tr["tier"], dataset=ds, obj_id=tr["obj_id"],
               name=tr["name"], scene=tr["scene"], im=tr["im"], gt_idx=gt_idx, visib=tr.get("visib"),
               condition=cond, cad="", cad_units_m="", pose_source="", fp_conf="", fp_fail="",
               dsym_mm="", dsym_norm="", f05="", place_mm="", settle_mm="",
               world=winfo.get("world"), plane_inlier=winfo.get("plane_inlier", ""),
               plane_angle_deg=winfo.get("plane_angle_deg", ""),
               bottom_gap_mm=winfo.get("bottom_gap_mm", ""),
               n_cand=0, n_reach=0, n_blocked=0, n_att=0, n_succ=0, first_succ=0, succ=0,
               lift_cm=0.0, att_seq="", fail_reason="", runtime_s=0.0,
               ts=_dt.datetime.now().isoformat(timespec="seconds"))

    cad_id, cad_path, units_m = _cad_under_test(ctx, tr, cond)
    row.update(cad=cad_id, cad_units_m=int(units_m))
    if not cad_path:
        row.update(fail_reason="cad_missing", runtime_s=round(time.time() - t0, 1))
        return row

    # -- world: annotated objects at their GT poses, only the target dynamic --
    os.environ.setdefault("GRASP_QUIET", "1")
    sim = TabletopSim().connect()
    # the table is z = 0 in every world; lower it only if THIS target's annotated
    # pose intersects it (a penetrating dynamic body gets kicked out otherwise)
    sim.build(objs, cam, target_gt_idx=gt_idx, with_robot=True,
              table_z=min(0.0, winfo["bottom_z"].get(gt_idx, 0.0) - 0.001))
    sim.settle(PROTOCOL["physics"]["settle_steps"])
    row["settle_mm"] = round(sim.target_displacement_mm(tgt.T_world), 1)
    sim.freeze_initial()
    T_true = sim.target_pose()                          # settled model→world pose
    try:
        # -- pose of the CAD under test ----------------------------------------
        if cond == "gt_pose":
            extra, T_m2w, row["pose_source"] = 1.0, T_true, "sim_true"
        else:
            res = _pose_step(ctx, tr, cond, fr, cad_path, units_m, row)
            if res is None:
                return row
            R, t, extra = res
            T_m2c = np.eye(4)
            T_m2c[:3, :3], T_m2c[:3, 3] = R, t / 1000.0
            T_m2w = cam.T_world @ T_m2c
        mesh_m = ctx.mesh_m(cad_path, units_m, extra)
        c_cad = T_m2w[:3, :3] @ mesh_m.centroid + T_m2w[:3, 3]
        c_tgt = T_true[:3, :3] @ ctx.mesh_m(tgt.mesh_path, True).centroid + T_true[:3, 3]
        row["place_mm"] = round(float(np.linalg.norm(c_cad - c_tgt) * 1000), 1)
        # -- grasps --------------------------------------------------------------
        _grasp_step(ctx, sim, cad_path, units_m, extra, T_m2w, row)
    finally:
        sim.disconnect()
        row["runtime_s"] = round(time.time() - t0, 1)
    return row


# ===========================================================================
# 3. results: CSV per trial (append + fsync, resume), git provenance
# ===========================================================================
def _key(r: dict) -> tuple:
    return (r["case"], int(r["scene"]), int(r["im"]), int(r["gt_idx"]), r["condition"])


def _append_csv(path: str, row: dict):
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if fresh:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())          # a hard crash otherwise leaves a torn line


def _git_state() -> dict:
    st = {"rev": "unknown", "dirty": "unknown"}
    try:
        head = open(os.path.join(_ROOT, ".git", "HEAD")).read().strip()
        ref = os.path.join(_ROOT, ".git", head[5:]) if head.startswith("ref: ") else None
        st["rev"] = (open(ref).read().strip() if ref and os.path.isfile(ref) else head)[:12]
        if ref:
            st["branch"] = head[5:].split("/")[-1]
        # the repo is bind-mounted into the container under another owner, hence safe.directory
        out = subprocess.run(["git", "-c", "safe.directory=*", "-C", _ROOT, "status", "--porcelain",
                              "--", "grasping", "evaluation", "experiments", "pipeline"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            st["dirty"] = bool(out.stdout.strip())
    except Exception:
        pass
    return st


# ===========================================================================
# 4. reporting helpers: tables from the CSV (Δ + win split, no intervals)
# ===========================================================================
def _pct(a, b):
    return f"{100.0 * a / b:.0f}%" if b else "–"


def _med(vals):
    v = [float(x) for x in vals if x not in ("", None)]
    return f"{statistics.median(v):.1f}" if v else "–"


def _headline_table(rows, conds, title):
    out = [f"**{title}**", "",
           "| condition | objects | trials | success@k | success@1 | attempts S/N | cand (med) | "
           "reach (med) | blocked | no cand | FP fail | D_sym med [mm] | place med [mm] | s/trial |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        rc = [r for r in rows if r["condition"] == c]
        if not rc:
            continue
        n, att = len(rc), sum(int(r["n_att"]) for r in rc)
        s5, s1, sa = (sum(int(r[k]) for r in rc) for k in ("succ", "first_succ", "n_succ"))
        out.append(f"| {c} | {len({r['case'] for r in rc})} | {n} | {s5}/{n} ({_pct(s5, n)}) | "
                   f"{s1}/{n} ({_pct(s1, n)}) | {sa}/{att} ({_pct(sa, att)}) | "
                   f"{_med(r['n_cand'] for r in rc)} | {_med(r['n_reach'] for r in rc)} | "
                   f"{sum(int(r['n_blocked']) for r in rc)} | "
                   f"{sum(r['fail_reason'] == 'no_candidates' for r in rc)} | "
                   f"{sum(r['fail_reason'] == 'fp_error' for r in rc)} | "
                   f"{_med(r['dsym_mm'] for r in rc)} | {_med(r['place_mm'] for r in rc)} | "
                   f"{_med(r['runtime_s'] for r in rc)} |")
    return out


def _paired_table(rows, pairs, title):
    by: dict = defaultdict(dict)
    for r in rows:
        by[_key(r)[:4]][r["condition"]] = int(r["succ"])
    out = [f"**{title}** (same instance under both conditions; Δ in percentage points)", "",
           "| A → B | n paired | A | B | Δ (B−A) | A only : B only : both : neither |",
           "|---|---|---|---|---|---|"]
    for a, b in pairs:
        ks = [k for k, d in by.items() if a in d and b in d]
        if not ks:
            continue
        sa, sb = sum(by[k][a] for k in ks), sum(by[k][b] for k in ks)
        ao = sum(1 for k in ks if by[k][a] and not by[k][b])
        bo = sum(1 for k in ks if by[k][b] and not by[k][a])
        both = sum(1 for k in ks if by[k][a] and by[k][b])
        out.append(f"| {a} → {b} | {len(ks)} | {_pct(sa, len(ks))} | {_pct(sb, len(ks))} | "
                   f"{100.0 * (sb - sa) / len(ks):+.0f} pp | {ao} : {bo} : {both} : {len(ks) - ao - bo - both} |")
    return out


def _write_report(csv_path: str, md: str) -> str:
    """REPORT.md next to the CSV; /tmp when the folder is not writable (root-owned)."""
    out = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "REPORT.md")
    try:
        with open(out, "w") as fh:
            fh.write(md + "\n")
        return out
    except PermissionError:
        alt = "/tmp/proxy_grasp_REPORT.md"
        with open(alt, "w") as fh:
            fh.write(md + "\n")
        return f"{alt} ({out} is not writable for this user)"


# ===========================================================================
# 5. host side: the oscar-plus container and the FoundationPose service
# ===========================================================================
def _docker() -> list:
    """A WORKING `docker compose` CLI. On WSL the `docker` shim can be present
    but dead after a Docker Desktop restart (I/O error), so each candidate is
    tried, not just looked up."""
    import shutil
    for c in ("docker", "docker.exe"):
        if shutil.which(c):
            try:
                if subprocess.run([c, "compose", "version"], capture_output=True, timeout=30).returncode == 0:
                    return [c, "compose"]
            except (OSError, subprocess.TimeoutExpired):
                pass
    sys.exit("no working docker CLI — start Docker Desktop (WSL: check the WSL integration) / install docker")


def _fp_healthy(dc) -> bool:
    out = subprocess.run(dc + ["ps", "--format", "{{.Name}} {{.Status}}"], capture_output=True, text=True).stdout
    return any("foundationpose" in l and "healthy" in l for l in out.splitlines())
