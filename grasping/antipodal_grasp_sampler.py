#!/usr/bin/env python3
"""Antipodal grasp sampler — Stage-5 (grasping demo) sub-step E1.

Given the *retrieved proxy CAD mesh* (the output of Stage-3 retrieval), sample
parallel-jaw **antipodal** grasp candidates on its surface: pairs of contact
points whose surface normals both lie within the friction cone of the line
joining them, so the two-finger grip is force-closure under Coulomb friction —
the antipodal condition of Nguyen [R1] and the antipodal-point-grasp search of
Chen & Burdick [R2] (references: grasping/README.md). Each surviving pair is
turned into one or more 6-DoF gripper poses, collision-checked against the
mesh, scored, and de-duplicated. Grasps are produced in the **mesh (object)
frame**; applying the estimated pose T_obj→world then places them in the scene
for execution with the Panda in PyBullet (`grasp_execute`).

This module has **no simulator dependency** — it needs only trimesh + numpy
(+ scipy for the KD-tree). It is deliberately analytic (training-free), matching
the rest of OSCAR+.

Pipeline:
    mesh ──sample surface (p1,n1)──▶ ray-cast −n1 ▶ antipodal partner (p2,n2)
         ──friction-cone + width test──▶ candidate contact pair
         ──build gripper frame(s) (approach sweep)──▶ collision reject
         ──score (antipodal · centrality · clearance)──▶ NMS ──▶ top-K grasps
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np

try:
    import trimesh
except Exception as exc:  # pragma: no cover
    trimesh = None
    _TRIMESH_ERR = exc


# ---------------------------------------------------------------------------
# Gripper model (parallel-jaw). Defaults ≈ a Franka-Panda-scale gripper (metres).
# ---------------------------------------------------------------------------
@dataclass
class GripperConfig:
    max_width: float = 0.08          # max opening between the two fingers (m)
    min_width: float = 0.005         # ignore contacts closer than this
    finger_length: float = 0.045     # how far the fingers extend along −approach
    finger_thickness: float = 0.012  # finger size across the closing axis
    finger_width: float = 0.020      # finger size across the approach-⊥ axis
    palm_depth: float = 0.02         # palm block behind the fingers
    collision_margin: float = 0.002  # inflate the gripper by this for the check


# ---------------------------------------------------------------------------
# A single grasp candidate (mesh frame).
# ---------------------------------------------------------------------------
@dataclass
class Grasp:
    center: np.ndarray               # (3,) midpoint of the two contacts
    axis: np.ndarray                 # (3,) closing direction (between fingers)
    approach: np.ndarray             # (3,) approach direction (palm → object)
    width: float                     # contact separation (m)
    quality: float                   # combined score in [0, 1]-ish
    contacts: Tuple[np.ndarray, np.ndarray]   # (p1, p2)

    def pose(self) -> np.ndarray:
        """4×4 SE(3) gripper pose in the mesh frame.

        Columns: x = closing axis, z = approach, y = z×x; origin = grasp center
        pulled back along −approach by ``finger_length`` is left to the caller
        (this returns the TCP-at-contacts convention)."""
        x = _unit(self.axis)
        z = _unit(self.approach)
        y = _unit(np.cross(z, x))
        x = np.cross(y, z)           # re-orthogonalise
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
        T[:3, 3] = self.center
        return T

    def to_dict(self) -> dict:
        d = asdict(self)
        d["center"] = self.center.tolist()
        d["axis"] = self.axis.tolist()
        d["approach"] = self.approach.tolist()
        d["contacts"] = [self.contacts[0].tolist(), self.contacts[1].tolist()]
        d["pose"] = self.pose().tolist()
        return d


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------
def sample_antipodal_grasps(
    mesh: "trimesh.Trimesh",
    gripper: GripperConfig = GripperConfig(),
    n_samples: int = 2000,
    friction_mu: float = 0.5,
    n_approach: int = 4,
    top_k: int = 100,
    nms_pos: float = 0.01,           # NMS position radius (m)
    nms_ang: float = 0.35,           # NMS angular radius (rad, ≈20°)
    seed: int = 0,
) -> List[Grasp]:
    """Sample up to ``top_k`` scored antipodal grasps on ``mesh``.

    friction_mu → cone half-angle θ = atan(μ); a pair (p1,p2) is antipodal iff
    the grasp axis a=(p2−p1)/‖·‖ satisfies  (−n1)·a ≥ cosθ  and  n2·a ≥ cosθ
    (two opposing contacts inside their friction cones: force closure for a
    parallel-jaw gripper [R1]). The scoring (cone alignment, centrality, width)
    is a plain heuristic of this project, not taken from a publication.
    """
    if trimesh is None:  # pragma: no cover
        raise ImportError(f"trimesh is required: {_TRIMESH_ERR}")
    rng = np.random.default_rng(seed)
    mesh = _prep_mesh(mesh)
    cos_theta = np.cos(np.arctan(friction_mu))
    centroid = mesh.center_mass if mesh.is_watertight else mesh.centroid
    diag = float(np.linalg.norm(mesh.extents)) + 1e-9

    # 1) sample surface contacts + outward normals
    pts, face_idx = trimesh.sample.sample_surface_even(mesh, n_samples, seed=seed)
    nrm = mesh.face_normals[face_idx]

    raw: List[Grasp] = []
    for p1, n1 in zip(pts, nrm):
        p2, n2 = _antipodal_partner(mesh, p1, n1, gripper)
        if p2 is None:
            continue
        a = _unit(p2 - p1)
        # friction-cone (antipodal) test at both contacts
        if (-n1) @ a < cos_theta or n2 @ a < cos_theta:
            continue
        width = float(np.linalg.norm(p2 - p1))
        center = 0.5 * (p1 + p2)
        antip = float(min((-n1) @ a, n2 @ a))          # cone alignment ∈ [cosθ,1]
        # 2) sweep the approach direction (1-DoF freedom around the closing axis)
        for approach in _approach_dirs(a, n_approach, rng):
            g = Grasp(center=center, axis=a, approach=approach, width=width,
                      quality=0.0, contacts=(p1, p2))
            if _gripper_collides(mesh, g, gripper):
                continue
            g.quality = _score(g, antip, cos_theta, center, centroid, diag)
            raw.append(g)

    raw.sort(key=lambda g: g.quality, reverse=True)
    return _nms(raw, nms_pos, nms_ang)[:top_k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prep_mesh(mesh):
    m = mesh.copy()
    # trimesh cleanup — API differs across versions (nondegenerate_faces() is the
    # newer name; older exposes remove_degenerate_faces()).
    try:
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
    except Exception:
        try:
            m.remove_degenerate_faces(); m.remove_duplicate_faces()
        except Exception:
            pass
    try:
        m.fix_normals()
    except Exception:
        pass
    return m


def _antipodal_partner(mesh, p1, n1, gr: GripperConfig):
    """Ray-cast from p1 into the object (−n1) to the opposite wall."""
    origin = p1 - 1e-4 * n1                      # nudge just inside the surface
    locs, _, tri = mesh.ray.intersects_location([origin], [-n1],
                                                multiple_hits=True)
    if len(locs) == 0:
        return None, None
    d = np.linalg.norm(locs - p1, axis=1)
    ok = (d > gr.min_width) & (d <= gr.max_width)
    if not ok.any():
        return None, None
    j = np.where(ok)[0][np.argmin(d[ok])]        # nearest valid far wall
    return locs[j], mesh.face_normals[tri[j]]


def _approach_dirs(axis, n, rng):
    """n unit approach directions ⊥ to the closing axis, evenly in angle."""
    # any vector ⊥ axis:
    ref = np.array([1.0, 0.0, 0.0])
    if abs(axis @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(axis, ref))
    v = _unit(np.cross(axis, u))
    phase = rng.uniform(0, np.pi / max(n, 1))
    out = []
    for k in range(n):
        t = phase + np.pi * k / max(n, 1)        # sweep 0..π (approach is a line)
        out.append(_unit(np.cos(t) * u + np.sin(t) * v))
    return out


def _gripper_finger_points(g: Grasp, gr: GripperConfig, per_axis=4):
    """Sample points inside the two finger volumes + palm, in the mesh frame."""
    x = _unit(g.axis); z = _unit(g.approach); y = _unit(np.cross(z, x))
    half_open = g.width / 2 + gr.finger_thickness / 2 + gr.collision_margin
    ls = np.linspace(0.002, gr.finger_length, per_axis)              # along −z
    ws = np.linspace(-gr.finger_width / 2, gr.finger_width / 2, per_axis)
    pts = []
    for side in (+half_open, -half_open):                            # two fingers
        for l in ls:
            for w in ws:
                pts.append(g.center + side * x - l * z + w * y)
    # palm block behind the fingers
    for w in ws:
        for s in np.linspace(-half_open, half_open, per_axis):
            pts.append(g.center - (gr.finger_length + gr.palm_depth / 2) * z + w * y + s * x)
    return np.asarray(pts)


def _gripper_collides(mesh, g: Grasp, gr: GripperConfig) -> bool:
    """True if any gripper-body point lies inside the mesh (fingers must not
    penetrate the object; the volume *between* the fingers may)."""
    pts = _gripper_finger_points(g, gr)
    try:
        inside = mesh.contains(pts)
        return bool(inside.any())
    except Exception:
        # non-watertight fallback: proximity test
        d = trimesh.proximity.signed_distance(mesh, pts)
        return bool((d > gr.collision_margin).any())


def _score(g, antip, cos_theta, center, centroid, diag) -> float:
    """Combine antipodal cone alignment, centrality, and width preference."""
    # antipodal ∈ [cosθ,1] → normalise to [0,1]
    a = (antip - cos_theta) / (1.0 - cos_theta + 1e-9)
    # centrality: closer to the CoM = more stable moment arm
    c = 1.0 - min(np.linalg.norm(center - centroid) / (0.5 * diag), 1.0)
    # width: mildly prefer mid-range (not maximally open)
    return float(0.6 * a + 0.3 * c + 0.1 * (1.0 - g.width / (0.5 * diag)))


def _nms(grasps: List[Grasp], pos_r: float, ang_r: float) -> List[Grasp]:
    kept: List[Grasp] = []
    for g in grasps:                              # already score-sorted desc
        dup = False
        for k in kept:
            if (np.linalg.norm(g.center - k.center) < pos_r and
                    abs(_unit(g.axis) @ _unit(k.axis)) > np.cos(ang_r)):
                dup = True
                break
        if not dup:
            kept.append(g)
    return kept


def render_grasps(mesh, grasps: List[Grasp], path: str, top_n: int = 15):
    """Matplotlib 3-D render: object surface (grey) + gripper 'claws' coloured
    by quality. No GPU / offscreen-GL needed (Agg backend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    pts = mesh.sample(4000)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="0.8", alpha=0.35, linewidths=0)

    segs, cols = [], []
    cmap = plt.cm.viridis
    fl = 0.04
    for g in grasps[:top_n]:
        x, z = _unit(g.axis), _unit(g.approach)
        f1, f2 = g.center + x * g.width / 2, g.center - x * g.width / 2
        b1, b2 = f1 - z * fl, f2 - z * fl
        base = g.center - z * fl
        c = cmap(float(np.clip(g.quality, 0, 1)))
        segs += [[f1, f2], [f1, b1], [f2, b2], [b1, base], [b2, base]]  # gripper claw
        cols += [c] * 5
    ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=2.2))

    ctr = mesh.centroid
    r = float(max(mesh.extents)) * 0.62
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_title(f"Antipodal grasps — top {min(top_n, len(grasps))} of {len(grasps)} "
                 f"(colour = quality)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def transform_grasps(grasps: List[Grasp], T: np.ndarray) -> List[Grasp]:
    """Apply a 4×4 pose (e.g. the Stage-3 T_obj→cam) to a list of grasps."""
    R, t = T[:3, :3], T[:3, 3]
    out = []
    for g in grasps:
        out.append(Grasp(center=R @ g.center + t, axis=R @ g.axis,
                         approach=R @ g.approach, width=g.width,
                         quality=g.quality,
                         contacts=(R @ g.contacts[0] + t, R @ g.contacts[1] + t)))
    return out


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Antipodal grasp sampler (Stage-5 E1)")
    ap.add_argument("mesh", nargs="?", help="path to a CAD mesh (.obj/.ply/.glb); "
                    "omit for a synthetic-box self-test")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-width", type=float, default=0.08)
    ap.add_argument("--out", default=None, help="write grasps to this JSON")
    ap.add_argument("--scene", default=None, help="export a trimesh scene (.glb)")
    ap.add_argument("--render", default=None, help="matplotlib PNG of mesh + grasps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if trimesh is None:
        raise SystemExit(f"trimesh required: {_TRIMESH_ERR}")
    if args.mesh:
        mesh = trimesh.load(args.mesh, force="mesh")
    else:
        mesh = trimesh.creation.box(extents=(0.06, 0.04, 0.10))  # self-test box
        print("[grasp] no mesh given — self-test on a 6×4×10 cm box")
    # auto-scale to metres: BOP/YCB CADs are in mm (extents ~100s), GSO in m.
    if float(max(mesh.extents)) > 1.5:
        mesh.apply_scale(0.001)
        print(f"[grasp] mesh looked like mm — scaled ×0.001 to metres "
              f"(extents now {np.round(mesh.extents, 3)} m)")

    gr = GripperConfig(max_width=args.max_width)
    grasps = sample_antipodal_grasps(mesh, gr, n_samples=args.n_samples,
                                     friction_mu=args.mu, top_k=args.top_k,
                                     seed=args.seed)
    print(f"[grasp] mesh extents (m): {np.round(mesh.extents, 3)}  "
          f"watertight={mesh.is_watertight}")
    print(f"[grasp] {len(grasps)} grasps after collision + NMS "
          f"(top score {grasps[0].quality:.3f})" if grasps else "[grasp] no grasps found")
    for i, g in enumerate(grasps[:5]):
        print(f"  #{i}: q={g.quality:.3f} width={g.width*1000:.1f}mm "
              f"center={np.round(g.center,3)}")

    if args.out:
        json.dump([g.to_dict() for g in grasps], open(args.out, "w"), indent=2)
        print(f"[grasp] wrote {len(grasps)} grasps -> {args.out}")
    if args.scene:
        scene = trimesh.Scene([mesh])
        for g in grasps[:min(len(grasps), 30)]:
            for c in g.contacts:
                m = trimesh.creation.uv_sphere(radius=0.003); m.apply_translation(c)
                m.visual.face_colors = [255, 40, 40, 255]; scene.add_geometry(m)
        scene.export(args.scene)
        print(f"[grasp] wrote viz scene -> {args.scene}")
    if args.render and grasps:
        render_grasps(mesh, grasps, args.render)
        print(f"[grasp] wrote render -> {args.render}")


if __name__ == "__main__":
    main()
