#!/usr/bin/env python3
"""stage5_visualize.py — visualisation of a Stage-5 series.

Observer camera (robot + table + object in frame), and with it:
  grasp.mp4     THE VIDEO: all runs of the series (default 10, the object
                rotated a further 36 degrees per run) back to back. Per run: a
                short pose insert (the CAD under test as a green silhouette in
                the FoundationPose pose, the planned grasps in yellow), then the
                grasp execution at 3x slow motion. Ghost and grasps stay
                visible during the APPROACH (active attempt blue, failed red,
                success green) and disappear at the moment of the grip.
                Closing frame: success-rate card (X/N).
  overlay.png   high-resolution still of run 1 (pose + grasps including the
                execution markers) — the figure for the thesis.

Operated like the free mode of grasping/stage_5.py. The run needs the
container environment (PyBullet, FoundationPose service, imageio-ffmpeg):

    # proxy series: planned on the proxy, the real object grasped
    docker compose run --rm oscar-plus python3 /app/experiments/stage5_visualize.py \
        --object ycbv:14 --proxy housecat6d/cup-red_heart

    # gt series (proxy empty or equal to the object itself): own CAD
    docker compose run --rm oscar-plus python3 /app/experiments/stage5_visualize.py \
        --object ycbv:14

  --runs 10       number of runs; --yaw-step 36 degrees per run; --yaw start angle
  --out DIR       target folder (default
                  <runs_root>/stage5_grasping/viz/<object>_<condition>/)
  --scene/--im/--mass as in the free mode (otherwise the frame comes from the
                  experiment plan, so the video reproduces the tables).

Paths come from ``config/paths.yaml`` (overridable per flag, see ``--help``).
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# CLI-Prelude — VOR den schweren Imports (numpy, PyBullet, trimesh), damit
# `--help` auf dem Host ohne die Container-Abhaengigkeiten laeuft.
# ---------------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
sys.path.insert(0, _THIS)
for _p in (_REPO, os.path.join(_REPO, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ARGS = None
_PATHS = None

if __name__ == "__main__":
    import argparse

    from pipeline import paths as _paths_mod

    _ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ap.add_argument("--object", required=True, help="<ds>:<id>, e.g. ycbv:14")
    _ap.add_argument("--proxy", default="", help="CAD the grasps are planned on "
                     "(empty/own object = gt run); pool or BOP id")
    _ap.add_argument("--runs", type=int, default=10, help="runs in the series")
    _ap.add_argument("--yaw", type=float, default=0.0, help="start angle")
    _ap.add_argument("--yaw-step", type=float, default=36.0, help="degrees per run")
    _ap.add_argument("--scene", default="")
    _ap.add_argument("--im", type=int, default=-1)
    _ap.add_argument("--mass", type=float, default=0.0)
    _ap.add_argument("--n-tries", type=int, default=0,
                     help="grasp attempts per run (0 = protocol default from "
                          "PROTOCOL['executor']['n_tries'])")
    _ap.add_argument("--out", default="",
                     help="target folder (default: <runs_root>/stage5_grasping/viz/"
                          "<object>_<condition>); relative paths resolve "
                          "against the repo root.")
    # Felder, die Ctx erwartet:
    _ap.add_argument("--world", default="auto")
    _ap.add_argument("--pose-source", default="fp", choices=["fp"])
    _paths_mod.add_path_args(_ap)
    _ARGS = _ap.parse_args()
    _ARGS.exec, _ARGS.verbose = True, False
    _PATHS = _paths_mod.from_args(_ARGS)

if _PATHS is None:  # als Modul importiert (keine CLI): paths.yaml-Defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()

import numpy as np                                                  # noqa: E402

from grasping.trial import Ctx, PROTOCOL, _fp_pose                    # noqa: E402
from grasping.sim_scene import TabletopSim, _as_metre_obj            # noqa: E402
from grasping.stage_5 import _custom_tr, canonical_object, resolve_cad  # noqa: E402

W, H = 1600, 1200          # Standbild
GW, GH = 1280, 960         # Video-Frames (durch 16 teilbar fuer den Encoder)
FARBE_KANDIDAT = ((255, 210, 40), 2)     # geplant (gelb)
FARBE_AKTIV = ((80, 160, 255), 4)        # gerade ausgefuehrt (blau)
FARBE_FEHL = ((235, 60, 60), 4)          # ausgefuehrt, gescheitert (rot)
FARBE_ERFOLG = ((60, 220, 90), 5)        # ausgefuehrt, Erfolg (gruen)


def observer(p, target_xyz, robot_xy, dist=0.88, pitch=-22.0, w=W, h=H):
    """Observer camera: oblique three-quarter view, robot arm AND object."""
    t, r = np.asarray(target_xyz, float), np.asarray(robot_xy, float)
    v = t[:2] - r
    yaw = np.degrees(np.arctan2(v[1], v[0])) + 78.0      # schraeg hinter dem Arm
    look = [0.5 * (t[0] + r[0]), 0.5 * (t[1] + r[1]), 0.26]
    view = p.computeViewMatrixFromYawPitchRoll(look, dist, yaw, pitch, 0, 2)
    proj = p.computeProjectionMatrixFOV(55, w / h, 0.05, 4.0)
    return view, proj


def snap(p, view, proj, w=W, h=H):
    out = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(out[2], (h, w, 4))[:, :, :3].astype(np.uint8)
    seg = np.reshape(out[4], (h, w))
    return rgb, seg


def to_px(view, proj, pts, w, h):
    """World points (N,3) -> pixels (N,2)."""
    V = np.array(view).reshape(4, 4, order="F")
    P = np.array(proj).reshape(4, 4, order="F")
    q = np.c_[np.atleast_2d(pts), np.ones(len(np.atleast_2d(pts)))] @ (P @ V).T
    q = q[:, :3] / np.clip(q[:, 3:4], 1e-9, None)
    return np.c_[(q[:, 0] + 1) * 0.5 * w, (1 - q[:, 1]) * 0.5 * h]


def draw_grasps(img, view, proj, feas, marks):
    """Draw the grasps as a finger line + approach arrow (in image
    resolution!).  marks: index -> (colour, width); everything else is a
    yellow candidate."""
    from PIL import Image, ImageDraw
    h, w = img.shape[:2]
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for i, g in enumerate(feas):
        col, wd = marks.get(i, FARBE_KANDIDAT)
        half = g.axis / max(np.linalg.norm(g.axis), 1e-9) * g.width / 2
        f1, f2 = to_px(view, proj, np.stack([g.center - half, g.center + half]), w, h)
        a0, a1 = to_px(view, proj, np.stack(
            [g.center - g.approach * 0.07, g.center - g.approach * 0.015]), w, h)
        d.line([tuple(f1), tuple(f2)], fill=col, width=wd)
        d.line([tuple(a0), tuple(a1)], fill=col, width=max(1, wd - 1))
        d.ellipse([a1[0] - wd, a1[1] - wd, a1[0] + wd, a1[1] + wd], fill=col)
    return np.asarray(im)


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:                                  # noqa: BLE001
        return ImageFont.load_default()


def stamp(img, text):
    """Write the caption line into the top of the image (scales with the
    resolution)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im)
    strip = max(18, img.shape[0] // 30)
    d.rectangle([0, 0, img.shape[1], strip], fill=(0, 0, 0))
    d.text((8, strip // 6), text, fill=(255, 255, 255), font=_font(strip - strip // 3))
    return np.asarray(im)


def karte(text_lines, w=GW, h=GH):
    """Closing card (black, centred text)."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (10, 10, 10))
    d = ImageDraw.Draw(im)
    f = _font(h // 16)
    y = h // 2 - (h // 12) * len(text_lines) // 2 - h // 24
    for ln in text_lines:
        tw = d.textlength(ln, font=f)
        d.text(((w - tw) / 2, y), ln, fill=(240, 240, 240), font=f)
        y += h // 12
    return np.asarray(im)


def ghost_tint(img, seg, gb):
    """Tint the ghost pixels green (occlusion-correct via the seg mask)."""
    m = seg == gb
    if m.any():
        sh = img[m].astype(float).mean(axis=1, keepdims=True) / 255.0
        img[m] = (np.array([60, 235, 110]) * (0.35 + 0.65 * sh)).astype(np.uint8)
    return img


def ghost_overlay(p, view, proj, w, h, gb, feas, marks):
    """Current scene + ghost (must be standing visible) + grasps."""
    img, seg = snap(p, view, proj, w, h)
    return draw_grasps(ghost_tint(img, seg, gb), view, proj, feas, marks)


def run_once(ctx, args, tr, cam, tgt_base, cad_path, units_m, yaw, label,
             frames, want_hires):
    """One run of the series: stand up, FP pose, grasps, execution — with a
    frame capture on every 4th simulation step (240 Hz / 4 at 20 fps = 3x slow
    motion).  Returns (success: bool, hires_overlay | None)."""
    from grasping.antipodal_grasp_sampler import transform_grasps
    from grasping.grasp_execute import (PandaGrasper, feasible_grasps,
                                        reachable_order)
    tgt, _ = canonical_object(ctx, tgt_base, yaw)
    sim = TabletopSim().connect()
    hires = None
    try:
        p = sim._p
        sim.build([tgt], cam, target_gt_idx=tr["gt_idx"], with_robot=False,
                  table_z=0.0)
        sim.settle(PROTOCOL["physics"]["settle_steps"])
        sim.freeze_initial()
        T_true = sim.target_pose()

        rd = sim.render_rgbd()
        mask = sim.object_mask(tgt.obj_id, rd["seg"]).astype(np.uint8)
        if mask.sum() < 200:
            print(f"[viz] {label}: solo mask empty — run skipped")
            return False, None
        try:
            R, t, _conf = _fp_pose(cad_path, rd["rgb"], rd["depth"], mask,
                                   cam.K, units_m, 1.0)
        except Exception as exc:                              # noqa: BLE001
            print(f"[viz] {label}: FP error ({str(exc).splitlines()[-1][:60]})")
            return False, None
        T_m2c = np.eye(4)
        T_m2c[:3, :3], T_m2c[:3, 3] = R, np.asarray(t, float) / 1000.0
        T_m2w = cam.T_world @ T_m2c

        sim._add_panda([tgt])
        sim.freeze_initial()
        gs = ctx.grasp_candidates(cad_path, units_m, 1.0)
        grasper = PandaGrasper(sim)
        grasper.reset()
        base_xy = p.getBasePositionAndOrientation(sim.robot)[0][:2]
        feas = feasible_grasps(grasper, reachable_order(
            transform_grasps(gs, T_m2w), base_xy),
            tol_mm=PROTOCOL["executor"]["reach_tol_mm"])[:20]

        view, proj = observer(p, T_true[:3, 3], base_xy)
        gview, gproj = observer(p, T_true[:3, 3], base_xy, w=GW, h=GH)
        quat = __import__("trimesh").transformations.quaternion_from_matrix(T_m2w)
        ghost_obj = _as_metre_obj(cad_path, 1.0 if units_m else 0.001)
        # Geist als dauerhaften (rein visuellen) Body anlegen: sichtbar waehrend
        # Intro UND Anfahrt, versteckt ab dem Zugriff. Leicht aufgeblasen: liegt
        # das CAD unter Test VOLLSTAENDIG im Zielobjekt (kleineres Substitut),
        # waere die verdeckungs-korrekte Silhouette sonst unsichtbar.
        gvis = p.createVisualShape(p.GEOM_MESH, fileName=ghost_obj,
                                   meshScale=[1.05] * 3,
                                   rgbaColor=(0.15, 0.95, 0.35, 1.0))
        gpos = T_m2w[:3, 3].tolist()
        gorn = [quat[1], quat[2], quat[3], quat[0]]
        gb = p.createMultiBody(baseMass=0, baseVisualShapeIndex=gvis,
                               basePosition=gpos, baseOrientation=gorn)

        def ghost_show(on):
            p.resetBasePositionAndOrientation(gb, gpos if on else [0, 0, -5], gorn)

        marks, cur = {}, [None]
        hide, setg_n = [False], [0]
        real_setg = grasper.set_gripper

        def _setg(width, force=40):
            # 1. Aufruf je Versuch = Oeffnen (Geist+Pfeile sichtbar); ab dem 2.
            # beginnt das Greifen -> beide ausblenden, freie Sicht auf den Griff
            if setg_n[0] > 0 and not hide[0]:
                hide[0] = True
                ghost_show(False)
            setg_n[0] += 1
            return real_setg(width, force)

        grasper.set_gripper = _setg
        intro = ghost_overlay(p, gview, gproj, GW, GH, gb, feas, marks)
        frames.extend([stamp(intro, f"{label} | pose + planned grasps")] * 40)

        real_step = p.stepSimulation
        nstep = [0]

        def dyn():
            d = dict(marks)
            if cur[0] is not None and cur[0] not in d:
                d[cur[0]] = FARBE_AKTIV
            return d

        def _step(*a, **kw):
            out = real_step(*a, **kw)
            nstep[0] += 1
            if nstep[0] % 4 == 0:
                img, seg = snap(p, gview, gproj, GW, GH)
                if not hide[0]:
                    img = draw_grasps(ghost_tint(img, seg, gb), gview, gproj,
                                      feas, dyn())
                frames.append(stamp(img, label))
            return out

        succ_i, n_att = None, 0
        for i, g in enumerate(feas):
            if n_att >= args.n_tries:
                break
            sim.reset_objects()                       # ohne Aufnahme (Teleport)
            grasper.reset()
            sim.settle(30)
            hide[0], setg_n[0] = False, 0             # Geist+Pfeile wieder an
            ghost_show(True)
            p.stepSimulation = _step
            cur[0] = i
            r = grasper.execute(g)
            cur[0] = None
            p.stepSimulation = real_step
            if r.get("blocked"):
                continue
            n_att += 1
            marks[i] = FARBE_FEHL
            if r["success"]:
                marks[i] = FARBE_ERFOLG
                succ_i = i
                break
        if frames:
            frames.extend([frames[-1]] * 20)          # 1 s Endzustand halten
        print(f"[viz] {label}: "
              f"{'SUCCESS (attempt ' + str(n_att) + ')' if succ_i is not None else 'no success (' + str(n_att) + ' attempts)'}")

        if want_hires:
            sim.reset_objects()
            grasper.reset()
            ghost_show(True)
            hires = ghost_overlay(p, view, proj, W, H, gb, feas, marks)
        return succ_i is not None, hires
    finally:
        sim.disconnect()


def main(args):
    if args.n_tries <= 0:
        args.n_tries = PROTOCOL["executor"]["n_tries"]

    ctx = Ctx(args)
    os.environ.setdefault("GRASP_QUIET", "1")
    tr = _custom_tr(args)
    self_id = f"{tr['dataset']}/obj_{tr['obj_id']:06d}"
    cond = "gt" if args.proxy in ("", self_id) else "proxy"
    out = args.out or os.path.join(_PATHS["runs_root"], "stage5_grasping", "viz",
                                   f"{tr['dataset']}{tr['obj_id']}_{cond}")
    if not os.path.isabs(out):
        out = os.path.join(_REPO, out)
    os.makedirs(out, exist_ok=True)
    if args.mass > 0:
        from grasping import sim_scene
        sim_scene.OBJECT_MASS_KG[(tr["dataset"], tr["obj_id"])] = args.mass

    cad_id, cad_path, units_m = resolve_cad(ctx, tr, cond)
    print(f"[viz] {tr['name']} | condition {cond} | CAD under test: {cad_id} | "
          f"{args.runs} runs")

    fr, objs, cam, winfo = ctx.scene(tr["dataset"], tr["scene"], tr["im"])
    tgt_base = objs[tr["gt_idx"]]

    frames, wins = [], 0
    png = os.path.join(out, "overlay.png")
    for ridx in range(max(1, args.runs)):
        yaw = args.yaw + ridx * args.yaw_step
        label = f"{tr['name']} on {cad_id.split('/')[-1][:24]} | run {ridx + 1}/{args.runs} ({yaw:g} deg)"
        ok, hires = run_once(ctx, args, tr, cam, tgt_base, cad_path, units_m,
                             yaw, label, frames, want_hires=(ridx == 0))
        wins += int(ok)
        if hires is not None:
            from PIL import Image
            Image.fromarray(stamp(hires, label)).save(png)

    frames.extend([karte([f"{tr['name']}  |  planned on: {cad_id}",
                          f"success rate: {wins}/{args.runs}"])] * 60)
    import imageio
    try:
        vid = os.path.join(out, "grasp.mp4")
        imageio.mimsave(vid, frames, fps=20, macro_block_size=16)
    except Exception:                                  # ohne ffmpeg: GIF
        vid = os.path.join(out, "grasp.gif")
        imageio.mimsave(vid, frames, fps=20, loop=0)
    print(f"[viz] RESULT {tr['name']} | "
          f"{cad_id if cond == 'proxy' else 'own CAD'}: "
          f"success rate {wins}/{args.runs}")
    print(f"[viz] written: {vid} ({len(frames)} frames) + {png}")


if __name__ == "__main__":
    main(_ARGS)
