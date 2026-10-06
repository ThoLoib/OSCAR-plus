#!/usr/bin/env python3
"""Stage-5 · 5.3 — Grasp execution with a Franka Panda (PyBullet [R14]).

Reference list: grasping/README.md. The IK is PyBullet's damped-least-squares
solver with joint limits, joint ranges and rest poses (null-space variant of
`calculateInverseKinematics`, PyBullet Quickstart Guide); the arm model is the
`franka_panda/panda.urdf` shipped with pybullet_data.

Take grasp candidates (from `antipodal_grasp_sampler`, in the object frame),
transform them by the object's pose into the world, and drive the Panda to
execute the best **reachable** one: pre-grasp → approach → close → lift →
success check. CPU-only (no GPU) so it runs alongside anything else.

Frame convention: a `Grasp` has x = closing axis, z = approach. The Panda's
grasp-target link is aligned so its **+z = approach** (into the object) and the
fingers close along the grasp's closing axis.

CLI (standalone test — grasps the target's OWN mesh at GT pose, to validate the
mechanics without the perception stage):
    python -m grasping.grasp_execute --scene 000048 --target 6 --gif /tmp/grasp.gif
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)

from grasping.antipodal_grasp_sampler import Grasp, _unit          # noqa: E402


# Panda joint layout (pybullet franka_panda/panda.urdf)
PANDA_ARM = list(range(7))            # 7 revolute arm joints
PANDA_FINGERS = [9, 10]               # prismatic finger joints (0=closed, 0.04=open)
PANDA_EE = 11                         # panda_grasptarget link (TCP)
FINGER_OPEN = 0.04
ARM_REST = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]   # neutral "ready" pose

# Success protocol of one attempt (all values reported in the experiment manifest):
LIFT_M = 0.15          # lift the closed gripper this high above the grasp
RISE_MIN_M = 0.05      # the object must have risen at least this much …
HOLD_STEPS = 240       # … stay in the hand for this many steps (1 s at 240 Hz) …
SHAKE_AMP_M = 0.05     # … and survive ±x/±y/±z jerks of this amplitude
IN_HAND_M = 0.15       # object further than this from the TCP == dropped
FINGER_FRICTION = 1.0  # x sim_scene.TARGET_FRICTION (PyBullet multiplies) = 1.0 effective


class PandaGrasper:
    def __init__(self, sim):
        self.sim = sim
        self.p = sim._p
        self.robot = sim.robot
        assert self.robot is not None, "scene built without a robot"
        # Movable joints + limits, for LIMIT-AWARE (null-space) IK. Unconstrained
        # calculateInverseKinematics returns configurations that violate the
        # Panda joint limits (notably joints 1 & 3); position control then clamps
        # them and the EE lands ~20 cm off target. Feeding the limits back into
        # the solver keeps the solution reachable.
        p = self.p
        self.movable, self.ll, self.ul = [], [], []
        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            if info[2] != p.JOINT_FIXED:
                self.movable.append(j); self.ll.append(info[8]); self.ul.append(info[9])
        self.jr = [u - l for l, u in zip(self.ll, self.ul)]
        self.rp = ARM_REST + [FINGER_OPEN] * (len(self.movable) - len(PANDA_ARM))
        self.jd = [0.05] * len(self.movable)
        for j in PANDA_FINGERS:
            p.changeDynamics(self.robot, j, lateralFriction=FINGER_FRICTION, spinningFriction=0.005)

    # ---- IK ----------------------------------------------------------------
    def _ik(self, pos, orn):
        return self.p.calculateInverseKinematics(
            self.robot, PANDA_EE, pos, orn,
            lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr,
            restPoses=self.rp, jointDamping=self.jd,
            maxNumIterations=200, residualThreshold=1e-4)

    def reach_error(self, pos, orn) -> float:
        """mm error of the limit-clamped IK solution at (pos, orn) — no motion.
        Used to check a grasp is reachable *before* committing to executing it."""
        p = self.p
        q = self._ik(pos, orn)
        arm_ll, arm_ul = self.ll[:len(PANDA_ARM)], self.ul[:len(PANDA_ARM)]
        saved = [p.getJointState(self.robot, j)[0] for j in PANDA_ARM]
        for j, qi, lo, hi in zip(PANDA_ARM, q[:7], arm_ll, arm_ul):
            p.resetJointState(self.robot, j, min(max(qi, lo), hi))
        st = p.getLinkState(self.robot, PANDA_EE, computeForwardKinematics=True)
        err = float(np.linalg.norm(np.array(st[4]) - np.asarray(pos)) * 1000)
        for j, qi in zip(PANDA_ARM, saved):                # restore
            p.resetJointState(self.robot, j, qi)
        return err

    def home(self):
        """Snap the arm to the neutral rest pose (reset between tries)."""
        for j, qi in zip(PANDA_ARM, ARM_REST):
            self.p.resetJointState(self.robot, j, qi, targetVelocity=0.0)
        self._drive_arm(ARM_REST)          # motors must not chase the previous target

    def reset(self):
        """Full robot reset between attempts: arm at rest, fingers open, zero
        velocities — an instantaneous teleport, so every attempt starts from
        the same state (the scene objects are reset by the sim)."""
        self.home()
        for j in PANDA_FINGERS:
            self.p.resetJointState(self.robot, j, FINGER_OPEN, targetVelocity=0.0)
        self.set_gripper(FINGER_OPEN * 2)

    # ---- low-level control -------------------------------------------------
    def _drive_arm(self, q: List[float], force: float = 400):
        for j, qi in zip(PANDA_ARM, q):
            self.p.setJointMotorControl2(self.robot, j, self.p.POSITION_CONTROL,
                                         qi, force=force)

    def set_gripper(self, width: float, force: float = 40):
        w = float(np.clip(width / 2, 0.0, FINGER_OPEN))
        for j in PANDA_FINGERS:
            self.p.setJointMotorControl2(self.robot, j, self.p.POSITION_CONTROL,
                                         w, force=force)

    def move_to(self, pos, orn_quat, steps: int = 400, cap=None, force: float = 400):
        """Limit-aware IK to an EE pose, drive there, step until the arm settles.

        Position control leaves a small steady-state joint residual that compounds
        to several cm at the gripper, so we settle until the max joint error is
        tiny (<0.005 rad ≈ ~1 cm at the TCP) rather than a fixed step count."""
        q = self._ik(pos, orn_quat)
        self._drive_arm(q[:7], force=force)
        for i in range(steps):
            self.p.stepSimulation()
            if cap is not None and i % 12 == 0:
                cap()
            if i > 40 and i % 20 == 0:
                jerr = max(abs(self.p.getJointState(self.robot, j)[0] - qi)
                           for j, qi in zip(PANDA_ARM, q[:7]))
                if jerr < 0.005:
                    break

    # ---- one grasp attempt -------------------------------------------------
    def execute(self, grasp_world: Grasp, cap=None) -> dict:
        """Pre-grasp → approach → close → lift; return success + diagnostics."""
        p = self.p
        z = _unit(grasp_world.approach)               # approach (into object)
        orn = _hand_quat(grasp_world)
        center = grasp_world.center
        pregrasp = center - z * 0.12                  # 12 cm back along approach

        self.set_gripper(FINGER_OPEN * 2)             # open
        self.move_to(pregrasp, orn, 400, cap)
        self.move_to(center - z * 0.01, orn, 400, cap)   # approach to contacts

        # Verify the gripper actually ARRIVED. A straight-line approach can be
        # blocked by clutter (the arm rams a neighbouring object) or fail to
        # converge; closing then just grasps air. Bail out fast so the caller can
        # try the next candidate — this is the clutter-avoidance mechanism (pick a
        # grasp whose approach corridor happens to be clear), short of full
        # collision-aware motion planning.
        tcp = np.array(p.getLinkState(self.robot, PANDA_EE,
                                      computeForwardKinematics=True)[4])
        approach_err = float(np.linalg.norm(tcp - (center - z * 0.01)))
        if approach_err > 0.03:                       # >3 cm short == blocked
            return {"success": False, "lift_cm": 0.0, "rose": False,
                    "held": False, "blocked": True,
                    "approach_err_mm": round(approach_err * 1000)}

        # Two-stage close: a GENTLE pre-close establishes light contact without
        # shoving the (unfixtured) object away, then a firm clamp secures it.
        # (A single hard close ejects thin/off-centre objects before it grips.)
        self.set_gripper(grasp_world.width * 0.85, force=20)
        for _ in range(80):
            p.stepSimulation()
            if cap:
                cap()
        self.set_gripper(max(grasp_world.width * 0.4, 0.0), force=120)
        for _ in range(200):                          # let the grip settle firmly
            p.stepSimulation()
            if cap:
                cap()
        # lift
        target_body = None
        z0 = None
        for oid, b in self.sim.body.items():
            # the dynamic (grasped) object is the one with mass>0
            if p.getDynamicsInfo(b, -1)[0] > 0:
                target_body = b
                z0 = p.getBasePositionAndOrientation(b)[0][2]
        lift_pose = center - z * 0.01 + np.array([0, 0, LIFT_M])
        self.move_to(lift_pose, orn, 240, cap)        # lift
        lifted = 0.0
        if target_body is not None:
            z1 = p.getBasePositionAndOrientation(target_body)[0][2]
            lifted = z1 - z0
        rose = lifted > RISE_MIN_M                    # object rose enough

        # HOLD phase: keep the lift pose for HOLD_STEPS and require the object
        # to stay with the gripper — a grasp that merely flicks the object
        # upward passes the rise check but fails here.
        hold = False
        if rose and target_body is not None:
            hold = self._hold_test(target_body, HOLD_STEPS, cap=cap)

        # Shake test in the spirit of ACRONYM [R3] (Eppner et al. 2021 label a
        # simulated grasp successful only if the object stays in the hand while
        # the gripper is shaken): a real grasp must survive perturbation, not
        # merely rise. Jerk the gripper along ±x/±y/±z and require the object to
        # stay in the hand — this rejects marginal grips that a lift-only check
        # would pass. Amplitude/direction set are this project's choice.
        held = False
        if hold:
            held = self._shake_test(lift_pose, orn, target_body, cap=cap)

        success = rose and hold and held
        return {"success": bool(success), "lift_cm": round(100 * lifted, 1),
                "rose": bool(rose), "hold": bool(hold), "held": bool(held)}

    def _hold_test(self, target_body, steps: int, cap=None) -> bool:
        """Hold still for `steps`; True if the object never leaves the hand."""
        p = self.p
        for i in range(steps):
            p.stepSimulation()
            if cap and i % 12 == 0:
                cap()
            if i % 40 == 0 or i == steps - 1:
                tcp = np.array(p.getLinkState(self.robot, PANDA_EE,
                                              computeForwardKinematics=True)[4])
                obj = np.array(p.getBasePositionAndOrientation(target_body)[0])
                if np.linalg.norm(obj - tcp) > IN_HAND_M:
                    return False
        return True

    def _shake_test(self, base_pose, orn, target_body, amp: float = SHAKE_AMP_M,
                    cap=None) -> bool:
        """Perturb the held object along ±x/±y/±z; return True if it stays in the
        gripper (object–TCP distance never blows past the gripper's reach)."""
        p = self.p
        dirs = [(amp, 0, 0), (-amp, 0, 0), (0, amp, 0), (0, -amp, 0),
                (0, 0, amp), (0, 0, -amp)]
        for d in dirs:
            self.move_to(base_pose + np.array(d), orn, steps=60, cap=cap)  # fast jerk
            tcp = np.array(p.getLinkState(self.robot, PANDA_EE,
                                          computeForwardKinematics=True)[4])
            obj = np.array(p.getBasePositionAndOrientation(target_body)[0])
            if np.linalg.norm(obj - tcp) > IN_HAND_M: # object slipped out of the hand
                return False
        return True


# ---------------------------------------------------------------------------
def _hand_quat(g: Grasp):
    """Panda grasp-target orientation. The Franka fingers translate along the
    hand's **local y-axis**, so hand-y := the grasp closing axis, and
    hand-z := approach (into the object). Returns a pybullet quat [x,y,z,w]."""
    from scipy.spatial.transform import Rotation
    z = _unit(g.approach)             # hand +z = approach
    y = _unit(g.axis)                 # hand +y = closing axis (finger travel)
    x = _unit(np.cross(y, z))
    y = np.cross(z, x)                # re-orthogonalise
    R = np.column_stack([x, y, z])
    return Rotation.from_matrix(R).as_quat()


def grasps_to_world(grasps: List[Grasp], T_obj2world: np.ndarray) -> List[Grasp]:
    """Convenience re-export (transform object-frame grasps to world)."""
    from grasping.antipodal_grasp_sampler import transform_grasps
    return transform_grasps(grasps, T_obj2world)


def reachable_order(grasps: List[Grasp], base_xy) -> List[Grasp]:
    """Heuristic: strongly prefer top-down grasps near the robot. Top-down
    approaches are both easier for the arm to reach and far less likely to sweep
    the object off the table than tilted/side approaches."""
    def key(g):
        downness = -_unit(g.approach)[2]          # approach pointing downward = good
        near = -np.linalg.norm(g.center[:2] - np.asarray(base_xy))
        return 0.5 * g.quality + 0.4 * downness + 0.1 * near
    return sorted(grasps, key=key, reverse=True)


def feasible_grasps(grasper: "PandaGrasper", grasps_world: List[Grasp],
                    pre: float = 0.12, touch: float = 0.01,
                    tol_mm: float = 30.0) -> List[Grasp]:
    """Keep only grasps whose pre-grasp AND approach poses are reachable within
    the Panda's joint limits (checked with `grasper.reach_error`, no motion),
    preserving input order. Most sampled grasps are unreachable from a fixed base
    — executing an unreachable one just knocks the object, so filter first."""
    out = []
    for g in grasps_world:
        z = _unit(g.approach)
        orn = _hand_quat(g)
        if (grasper.reach_error(g.center - z * pre, orn) < tol_mm and
                grasper.reach_error(g.center - z * touch, orn) < tol_mm):
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# CLI — standalone mechanics test on the target's own mesh + GT pose.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage-5 grasp execution (5.3)")
    ap.add_argument("--scene", default="000048")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--n-tries", type=int, default=5)
    ap.add_argument("--gif", default=None, help="save a GIF of the attempt")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    import trimesh
    from grasping.sim_scene import TabletopSim, load_ycbv_scene, YCBV_NAMES
    from grasping.antipodal_grasp_sampler import (sample_antipodal_grasps,
                                                  GripperConfig, transform_grasps)

    objs, cam = load_ycbv_scene(args.scene, args.frame)
    tgt = next(o for o in objs if o.obj_id == args.target)
    sim = TabletopSim(gui=args.gui).connect()
    sim.build(objs, cam, target_id=args.target, with_robot=True)
    sim.settle(120)

    # sample grasps on the target's OWN mesh (object frame) — stand-in for the
    # proxy in this mechanics test — then place them by the object's world pose.
    mesh = trimesh.load(tgt.mesh_path, force="mesh")
    grasps = sample_antipodal_grasps(mesh, GripperConfig(), n_samples=800, top_k=40)
    grasps_w = reachable_order(transform_grasps(grasps, tgt.T_world),
                               sim._p.getBasePositionAndOrientation(sim.robot)[0][:2])
    grasper = PandaGrasper(sim)
    grasper.home()
    grasps_w = feasible_grasps(grasper, grasps_w)             # keep only reachable
    print(f"[grasp-exec] target {args.target}:{YCBV_NAMES.get(args.target,'?')} — "
          f"{len(grasps_w)} reachable candidates, trying top {args.n_tries}")

    frames = []
    cap = (lambda: frames.append(sim.render_rgbd()["rgb"])) if args.gif else None
    result = {"success": False}
    attempts = 0
    for i, g in enumerate(grasps_w):                          # blocked ones are cheap
        if attempts >= args.n_tries:
            break
        sim.reset_objects(); grasper.reset(); sim.settle(30)  # clean slate per try
        if cap:
            frames.clear()                               # keep only this attempt's frames
        result = grasper.execute(g, cap=cap)
        if result.get("blocked"):                            # clutter — doesn't count
            print(f"  cand {i}: approach blocked ({result['approach_err_mm']}mm), skip")
            continue
        attempts += 1
        print(f"  try {attempts} (cand {i}): q={g.quality:.3f} width={g.width*1000:.0f}mm "
              f"approach_z={_unit(g.approach)[2]:+.2f} -> {result}")
        if result["success"]:
            break

    print(f"[grasp-exec] RESULT: {result}")
    if args.gif and frames:
        try:
            import imageio
            imageio.mimsave(args.gif, frames, fps=20)
            print(f"[grasp-exec] wrote {len(frames)} frames -> {args.gif}")
        except Exception as e:
            print(f"[grasp-exec] gif save failed ({e})")
    sim.disconnect()


if __name__ == "__main__":
    main()
