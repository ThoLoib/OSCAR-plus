#!/usr/bin/env python3
"""
run_pipeline_sim.py — the full OSCAR+ pipeline on a real RGB-D image, followed
by the grasp trial with the top-ranked model.

This is ``run_pipeline.py`` plus grasping: the same prompt, the same image, the
same eight steps (GroundingDINO+SAM, point cloud, CLIP, DINOv2, ULIP-2, fusion,
geometric check, FoundationPose), the same ``ranking.csv`` — and IN ADDITION the
rank-1 model is stood on the table in PyBullet and grasped by a Franka Panda. At
the end there is exactly one number: the success rate X/N.

    python3 run_pipeline_sim.py --rgb scene/rgb/000001.png \\
                                --depth scene/depth/000001.png \\
                                --intrinsics scene/scene_camera.json \\
                                --prompt "the red mug" \\
                                --gallery ycbv --top-k 5 --out ranking.csv

The arguments are exactly those of ``run_pipeline.py`` (--rgb, --depth,
--intrinsics, --prompt, --gallery, --top-k, --out, plus --geometry-reranking,
--pose-method, --no-docker and the path overrides — one shared definition in
``pipeline/demo_cli.py``). Only the flags of the grasp trial come on top
(--runs, --yaw-step, --n-tries, --cad-units, --no-grasp, --headless). Anyone who
needs one of the remaining pipeline flags calls ``python3 -m
pipeline.run_pipeline`` directly.

Besides ``ranking.csv`` + ``run_config.json`` (identical to ``run_pipeline.py``)
the run writes ``pipeline_sim_result.json`` next to them: retrieval, pose and
the grasp tally in one file.

What the grasp trial shows — and what it does not
-------------------------------------------------
A robot's table plane cannot be reconstructed from a photo; the estimated 6D
pose holds in the camera of the CAPTURE, not in the simulation. The grasp trial
therefore stands the retrieved CAD up by the Stage-5 rule (the most stable
standing pose whose horizontal extent fits the 78 mm gripper —
``grasping.stage_5.canonical_pose``), rotated further by ``--yaw-step`` each
run, and plans/executes the grasps on it. What is measured is therefore: is the
RETRIEVED model graspable? The estimated pose is reported and recorded, but not
used for standing the object up. The paired proxy-vs-original comparison of the
thesis is in ``experiments/stage5_grasping.py``.

Display (GUI mode, the default): a PyBullet window, the observer camera freely
rotatable/zoomable with the mouse. The retrieved CAD floats as a green ghost in
its standing pose, grasp candidates as yellow lines, the active attempt blue,
success green / failure red. ``--headless`` runs without a window.

Services and environment
------------------------
Runs in the oscar-plus container (GUI via X forwarding, DISPLAY is set in
docker-compose.yml). FoundationPose has to be started on the HOST:

    docker compose up -d foundationpose
    python3 run_pipeline_sim.py \
        --rgb <image> --depth <depth> --intrinsics <scene_camera.json> \
        --prompt "the red mug" --gallery gso

Outside a container the script wraps itself into ``docker compose run --rm
oscar python3 /app/run_pipeline_sim.py ...`` (``--no-docker`` switches that
off). With ``--geometry-reranking`` the dGeDi service is needed as well
(``docker compose up -d dgedi``).

Paths come from ``config/paths.yaml`` (overridable per flag, see ``--help``).
"""
from __future__ import annotations

import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# CLI-Prelude — VOR den schweren Imports (torch, PyBullet, open3d), damit
# `--help` auf dem Host ohne die Container-Abhaengigkeiten laeuft.
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO, os.path.join(_REPO, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TAG = "sim"
DEFAULT_RUN_DIR = "run_pipeline_sim"     # <runs_root>/run_pipeline_sim/

_ARGS = None
_PATHS = None
_RESOLVED = None

if __name__ == "__main__":
    import argparse

    from pipeline import demo_cli as _demo
    from pipeline import paths as _paths_mod

    _ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- Argumente von run_pipeline.py (eine gemeinsame Definition) ---
    _demo.add_demo_args(_ap)
    # --- Greifversuch ---
    _grp = _ap.add_argument_group("grasp trial (PyBullet + Franka Panda)")
    _grp.add_argument("--runs", type=int, default=10,
                      help="runs in the series (default 10)")
    _grp.add_argument("--yaw-step", type=float, default=36.0,
                      help="rotation of the object per run in degrees (default 36)")
    _grp.add_argument("--n-tries", type=int, default=5,
                      help="grasp attempts per run (default 5)")
    _grp.add_argument("--cad-units", choices=["auto", "m", "mm"], default="auto",
                      help="unit of the retrieved CAD. 'auto' decides from "
                           "the extent (> 10 -> mm) and logs the "
                           "decision.")
    _grp.add_argument("--no-grasp", action="store_true",
                      help="only run the pipeline, do not grasp")
    _grp.add_argument("--headless", action="store_true",
                      help="without a GUI window (console only)")
    _paths_mod.add_path_args(_ap)
    _ARGS = _ap.parse_args()
    _PATHS = _paths_mod.from_args(_ARGS)
    _ARGS.gui = not _ARGS.headless
    _RESOLVED = _demo.resolve_demo_args(
        _ARGS, _PATHS, repo_root=_REPO, default_run_dir=DEFAULT_RUN_DIR, tag=TAG)
    _demo.wrap_into_container(_ARGS, repo_root=_REPO,
                              script="/app/run_pipeline_sim.py", tag=TAG)

if _PATHS is None:  # als Modul importiert (keine CLI): paths.yaml-Defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()

import numpy as np                                                  # noqa: E402


def log(msg: str) -> None:
    print(f"[{TAG}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Schritt A: die Pipeline auf dem echten Bild
# ---------------------------------------------------------------------------
def run_pipeline(args, resolved: dict, paths: dict) -> dict:
    """Run the eight steps unchanged from pipeline/run_pipeline.py.

    Configuration, input loading and the ranking.csv export are the shared ones
    of ``run_pipeline.py`` (pipeline/demo_cli.py), so both entry points agree
    bit for bit on what they run and what they write.

    Returns the JSON-serialisable part ({'top1', 'cad_path', 'pose',
    'pose_conf', 'pose_method', 'ranking', 'summary', 'ranking_csv'}) plus the
    three '_'-prefixed keys '_result'/'_rows'/'_notes', which main() pops back
    out before writing pipeline_sim_result.json.
    """
    from pipeline import demo_cli
    from pipeline.run_pipeline import OSCARPlusPipeline

    config = demo_cli.build_pipeline_config(args, resolved, paths)
    log(f"RGB: {args.rgb}")
    log(f"Depth: {args.depth}")
    rgb_image, depth_image, camera_intrinsics, depth_note = demo_cli.load_inputs(
        args, resolved, config)
    log(f"depth conversion: {depth_note}")

    pipeline = OSCARPlusPipeline(config)
    pipeline.initialize()
    result = pipeline.run(rgb_image=rgb_image, depth_image=depth_image,
                          prompt=args.prompt,
                          camera_intrinsics=camera_intrinsics)

    if result.get("error"):
        sys.exit(f"[sim] the pipeline stopped: {result['error']} — check the "
                 "prompt and the image.")
    fusion = result.get("fusion")
    best = getattr(fusion, "best_match", None) if fusion else None
    if best is None:
        sys.exit("[sim] the pipeline retrieved no model (fusion without "
                 "best_match) — check the prompt, the mask and the gallery.")
    geo = result.get("geometry_reranking")
    if geo is not None and getattr(geo, "best_candidate", None):
        best = geo.best_candidate
    cad_path = pipeline._resolve_mesh_path_for_candidate(best)
    if not cad_path or not os.path.isfile(cad_path):
        sys.exit(f"[sim] no CAD mesh found for '{best.object_id}' "
                 f"(gallery {resolved['cad_models']}).")

    # ranking.csv — dieselbe Funktion und dasselbe Schema wie run_pipeline.py
    rows, notes = demo_cli.write_ranking_csv(
        result, resolved["out_csv"], args.top_k,
        mesh_resolver=pipeline._resolve_mesh_path_for_candidate)

    pose = result.get("pose_estimation")
    ranking = [(c.object_id, float(getattr(c, "fused_score", 0.0)))
               for c in demo_cli.final_candidates(result)[0][:args.top_k]]
    return {"top1": best.object_id, "cad_path": cad_path, "ranking": ranking,
            "pose": (pose.pose_matrix.tolist() if pose is not None else None),
            "pose_conf": (float(pose.confidence) if pose is not None else None),
            "pose_method": (pose.method if pose is not None else ""),
            "summary": result.get("summary", {}),
            "ranking_csv": resolved["out_csv"],
            "_result": result, "_rows": rows, "_notes": notes}


# ---------------------------------------------------------------------------
# Schritt B: Greifversuch auf dem abgerufenen Modell
# ---------------------------------------------------------------------------
def cad_units_m(cad_path: str, choice: str) -> bool:
    """True when the CAD is in METRES. 'auto' decides from the
    extent: a household object is < 10 in metres and > 10 in millimetres."""
    if choice in ("m", "mm"):
        return choice == "m"
    import trimesh
    ext = float(np.max(trimesh.load(cad_path, force="mesh").extents))
    units_m = ext <= 10.0
    log(f"--cad-units auto: largest extent {ext:.4g} -> "
        f"{'METRES' if units_m else 'MILLIMETRES'} (set --cad-units if in doubt)")
    return units_m


def fixed_camera(w: int = 640, h: int = 480,
                 eye=(0.55, 0.0, 0.45), look=(0.0, 0.0, 0.05)):
    """Fixed sim camera (OpenCV convention: x right, y down, z forward). It
    only serves to build the PyBullet world, not perception."""
    from grasping.sim_scene import SceneCamera
    eye = np.asarray(eye, float)
    fwd = np.asarray(look, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, eye
    K = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1]])
    return SceneCamera(K=K, T_world=T, width=w, height=h)


def _canonical_pose():
    """The placement rule of Stage 5 — one source, no reimplementation."""
    from grasping.stage_5 import canonical_pose
    return canonical_pose


def standing_object(mesh_obj: str, yaw_deg: float):
    """Stand the retrieved CAD graspably on the table: most stable standing pose
    <= 78 mm horizontally, rotated by yaw_deg about the vertical axis, footprint
    centre at the origin, bottom edge at z=0. (Like the solo setup of Stage 5.)"""
    import trimesh
    from grasping.sim_scene import SceneObject
    canonical_pose = _canonical_pose()
    mesh = trimesh.load(mesh_obj, force="mesh")
    T_st, graspable = canonical_pose(mesh)
    a = np.radians(yaw_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    T = Rz @ T_st
    m2 = mesh.copy()
    m2.apply_transform(T)
    cx, cy = m2.bounds.mean(axis=0)[:2]
    T[0, 3] -= cx
    T[1, 3] -= cy
    T[2, 3] -= float(m2.bounds[0, 2])
    return SceneObject(obj_id=0, T_world=T, mesh_path=mesh_obj), graspable


class Overlay:
    """Ghost (the retrieved CAD) + grasp lines in the GUI window."""

    def __init__(self, p):
        self.p = p
        self.ghost = None
        self.lines = []

    def show_ghost(self, mesh_obj: str, T_w):
        import trimesh
        q = trimesh.transformations.quaternion_from_matrix(T_w)
        vis = self.p.createVisualShape(self.p.GEOM_MESH, fileName=mesh_obj,
                                       meshScale=[1.0] * 3,
                                       rgbaColor=(0.15, 0.95, 0.35, 0.5))
        self.ghost = self.p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis,
            basePosition=T_w[:3, 3].tolist(),
            baseOrientation=[q[1], q[2], q[3], q[0]])

    def draw_grasps(self, grasps, color=(1.0, 0.85, 0.1), width=2.0):
        from grasping.grasp_execute import _unit
        for g in grasps:
            half = _unit(g.axis) * g.width / 2
            a = _unit(g.approach)
            self.lines.append(self.p.addUserDebugLine(
                (g.center - half).tolist(), (g.center + half).tolist(),
                color, lineWidth=width))
            self.lines.append(self.p.addUserDebugLine(
                (g.center - a * 0.06).tolist(), (g.center - a * 0.01).tolist(),
                color, lineWidth=max(1.0, width - 1)))

    def mark(self, g, color, width=4.0):
        self.draw_grasps([g], color=color, width=width)

    def clear(self):
        for lid in self.lines:
            try:
                self.p.removeUserDebugItem(lid)
            except Exception:                              # noqa: BLE001
                pass
        self.lines = []
        if self.ghost is not None:
            try:
                self.p.removeBody(self.ghost)
            except Exception:                              # noqa: BLE001
                pass
            self.ghost = None


def grasp_once(sim, args, mesh_obj: str, run_idx: int) -> bool:
    """One run: stand up -> plan grasps -> execute. True on success."""
    import trimesh
    from grasping.antipodal_grasp_sampler import (GripperConfig,
                                                  sample_antipodal_grasps,
                                                  transform_grasps)
    from grasping.grasp_execute import (PandaGrasper, feasible_grasps,
                                        reachable_order)
    p = sim._p
    yaw = run_idx * args.yaw_step
    label = f"run {run_idx + 1}/{args.runs} ({yaw:g} deg)"

    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    sim.body = {}
    sim.robot = None
    obj, graspable = standing_object(mesh_obj, yaw)
    if not graspable and run_idx == 0:
        log(f"NOTE: no standing pose <= 78 mm — used the most stable one")
    sim.build([obj], fixed_camera(), target_gt_idx=obj.gt_idx,
              with_robot=False, table_z=0.0)
    sim.settle(120)
    sim.freeze_initial()
    T_w = sim.target_pose()

    sim._add_panda([obj])
    sim.freeze_initial()
    grasper = PandaGrasper(sim)
    grasper.reset()
    base_xy = p.getBasePositionAndOrientation(sim.robot)[0][:2]

    mesh = trimesh.load(mesh_obj, force="mesh")
    gs = sample_antipodal_grasps(mesh, GripperConfig(), n_samples=800, top_k=40)
    feas = feasible_grasps(grasper, reachable_order(
        transform_grasps(gs, T_w), base_xy))
    log(f"{label}: {len(gs)} candidates, {len(feas)} reachable")

    ov = Overlay(p)
    if args.gui:
        ov.show_ghost(mesh_obj, T_w)
        ov.draw_grasps(feas[:20])

    ok, n_att = False, 0
    for g in feas:
        if n_att >= args.n_tries:
            break
        sim.reset_objects()
        grasper.reset()
        sim.settle(30)
        if args.gui:
            ov.mark(g, color=(0.2, 0.55, 1.0))
        r = grasper.execute(g)
        if r.get("blocked"):
            if args.gui:
                ov.mark(g, color=(0.5, 0.5, 0.5), width=2.5)
            continue
        n_att += 1
        if args.gui:
            ov.mark(g, color=((0.15, 0.9, 0.3) if r["success"] else (0.95, 0.2, 0.2)))
        log(f"{label}: attempt {n_att} -> "
            f"{'SUCCESS' if r['success'] else 'failure'}")
        if r["success"]:
            ok = True
            break
    if args.gui:
        time.sleep(1.5)                                 # Endzustand kurz zeigen
    ov.clear()
    return ok


def grasp_series(args, cad_path: str) -> dict:
    """The series over --runs runs; returns the success tally."""
    from grasping.sim_scene import TabletopSim, _as_metre_obj
    units_m = cad_units_m(cad_path, args.cad_units)
    mesh_obj = _as_metre_obj(cad_path, 1.0 if units_m else 0.001)

    sim = TabletopSim(gui=args.gui).connect()
    p = sim._p
    if args.gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=55,
                                     cameraPitch=-28,
                                     cameraTargetPosition=[0.12, 0.0, 0.12])
        # Echtzeit-Tempo: jeder Physikschritt schlaeft 1/240 s
        real_step = p.stepSimulation

        def _paced(*a, **kw):
            out = real_step(*a, **kw)
            time.sleep(1.0 / 240.0)
            return out

        p.stepSimulation = _paced

    wins, per_run = 0, []
    try:
        for run_idx in range(max(1, args.runs)):
            try:
                ok = grasp_once(sim, args, mesh_obj, run_idx)
            except Exception as exc:                       # noqa: BLE001
                log(f"run {run_idx + 1}: ERROR {exc} — counts as a failure")
                ok = False
            per_run.append(bool(ok))
            wins += int(ok)
    finally:
        sim.disconnect()
    return {"runs": len(per_run), "successes": wins, "per_run": per_run,
            "cad_units_m": bool(units_m), "sim_mesh": mesh_obj}


# ---------------------------------------------------------------------------
def main(args, paths: dict, resolved: dict) -> None:
    from pipeline import demo_cli

    os.environ.setdefault("GRASP_QUIET", "1")
    os.makedirs(resolved["out_dir"], exist_ok=True)
    demo_cli.write_run_config(
        resolved["out_dir"], sys.argv[1:], resolved, repo_root=_REPO,
        extra={"top_k": args.top_k,
               "geometry_reranking": args.geometry_reranking,
               "pose_method": args.pose_method,
               "grasp": {"runs": args.runs, "yaw_step": args.yaw_step,
                         "n_tries": args.n_tries, "cad_units": args.cad_units,
                         "no_grasp": args.no_grasp, "gui": args.gui}})
    log(resolved["intrinsics_note"])
    if args.pose_method == "foundationpose":
        demo_cli.check_foundationpose(args.foundationpose_url, tag=TAG)

    log(f"prompt: {args.prompt!r}  |  gallery: {resolved['gallery']}  |  "
        f"top-k: {args.top_k}  |  {args.runs} runs x max. {args.n_tries} grasps")
    out = run_pipeline(args, resolved, paths)
    log(f"retrieved (rank 1) = {out['top1']}")
    log(f"CAD under test = {out['cad_path']}")
    if out["pose"] is not None:
        log(f"pose ({out['pose_method']}) confidence {out['pose_conf']:.3f} — "
            "holds in the camera of the capture, not in the simulation")

    grasp = None
    if not args.no_grasp:
        grasp = grasp_series(args, out["cad_path"])

    result = out.pop("_result")
    rows = out.pop("_rows")
    notes = out.pop("_notes")
    payload = dict(out, grasp=grasp, argv=sys.argv[1:])
    res_path = os.path.join(resolved["out_dir"], "pipeline_sim_result.json")
    with open(res_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    extra = [("prompt", repr(args.prompt))]
    if grasp is not None:
        extra.append(("grasp success rate",
                      f"{grasp['successes']}/{grasp['runs']}"))
    extra.append(("sim result json", res_path))
    demo_cli.print_demo_summary(result, rows, out["ranking_csv"], notes,
                                extra=extra)


if __name__ == "__main__":
    main(_ARGS, _PATHS, _RESOLVED)
