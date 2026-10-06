#!/usr/bin/env python3
"""Stage-5 · 5.1 — Tabletop sim scene from a BOP capture (PyBullet [R14]).

Reference list: grasping/README.md. BOP datasets and file formats [R5];
YCB-V [R6], T-LESS [R7], LM-O [R8]; YCB meshes [R16].

Reconstructs a real BOP test frame (YCB-V, T-LESS Primesense or LM-O) inside
PyBullet: the annotated objects at their ground-truth poses on a table, a Franka
Panda, and an RGB-D camera at the capture's camera pose so a rendered view
matches the real one. Produces RGB + depth + segmentation for the perception
stage (5.2) and the physics world for grasp execution (5.3).

Coordinate handling (explicit on purpose):
  * BOP poses are in **mm**, camera convention **OpenCV** (x-right, y-down, z-fwd).
  * `scene_gt.json`     gives T_model→cam (cam_R_m2c, cam_t_m2c) per instance.
  * `scene_camera.json` gives K, depth_scale and — for YCB-V and T-LESS only —
    T_world→cam. LM-O has no world frame at all.
  * The sim world is z-up with the TABLE at z = 0. Two ways to get there:
      world="plane"  fit the table plane to the frame's real depth image
                     (RANSAC, object pixels masked out) — works for every
                     dataset and is what a robot with a depth camera would do;
      world="bop"    use the BOP extrinsics (legacy; YCB-V/T-LESS only).
    Where BOP extrinsics exist, the fitted plane is checked against them
    (`plane_angle_deg`), and the lowest object vertex is checked against the
    table (`bottom_gap_mm`) — both are recorded, never silently trusted.
  * Instances are keyed by their BOP `gt_idx` (a scene can hold several copies
    of one obj_id); `sim.body_inst[gt_idx]` is the bullet body of an instance,
    `sim.body[obj_id]` the first instance of an object (kept for the demos).

CLI (run standalone):
    python -m grasping.sim_scene --dataset tless --scene 000008 --frame 120 --out /tmp/s
    python -m grasping.sim_scene --scene 000048 --gui            # interactive
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from pipeline import paths as _paths_mod                          # noqa: E402

# All data locations come from the central registry (config/paths.yaml), so no
# particular working directory and no machine-specific path is needed.
_P = _paths_mod.resolve()
_DATA, _CAD, _CACHES = _P["datasets_root"], _P["cad_root"], _P["caches_root"]


def _first_dir(*cands: str) -> str:
    """First candidate directory that exists and is non-empty (else the first)."""
    for c in cands:
        if os.path.isdir(c) and os.listdir(c):
            return c
    return cands[0]


YCBV_TEST = _first_dir(os.path.join(_DATA, "ycbv/test"))
YCBV_MESHES = os.path.join(_CAD, "ycbv")      # <catalog>_<name>/textured_simple.obj
TLESS_TEST = os.path.join(_DATA, "tless/test_primesense")
TLESS_MESHES = os.path.join(_DATA, "tless/models_cad")   # obj_0000NN.ply, mm
LMO_TEST = os.path.join(_DATA, "lmo/test")
LMO_MESHES = os.path.join(_DATA, "lmo/models")           # obj_0000NN.ply, mm


def _ycbv_mesh_map() -> Dict[int, str]:
    """BOP obj_id → YCB mesh path.

    The meshes live in catalog-named folders (`002_master_chef_can/`, …), not
    BOP `obj_000001/` folders. The 21 folders sort in exact BOP-obj_id order
    (zero-padded catalog numbers → lexical == the canonical YCB-V ordering), so
    the 1-based sort index is the BOP obj_id."""
    dirs = sorted(d for d in glob.glob(os.path.join(YCBV_MESHES, "*"))
                  if os.path.isdir(d))
    return {i + 1: os.path.join(d, "textured_simple.obj")
            for i, d in enumerate(dirs)}


_YCBV_MESH_BY_ID = _ycbv_mesh_map()


# BOP dataset registry. `models` is what the SIM spawns (full-detail CAD),
# `models_eval` the BOP evaluation mesh Stage 3 posed and scored with. Every
# sim mesh resolves to a **metre-scale .obj** (see bop_mesh_path), so the rest of
# the module can assume meshScale = 1.0 regardless of dataset.
BOP_DATASETS: Dict[str, dict] = {
    "ycbv":  dict(test=YCBV_TEST,  models=YCBV_MESHES,  units=1.0,
                  models_eval=os.path.join(_DATA, "ycbv/models_eval")),
    "tless": dict(test=TLESS_TEST, models=TLESS_MESHES, units=0.001,
                  models_eval=os.path.join(_DATA, "tless/models_eval")),
    "lmo":   dict(test=LMO_TEST,   models=LMO_MESHES,   units=0.001,
                  models_eval=os.path.join(_DATA, "lmo/models_eval")),
}

_OBJ_CACHE = os.path.join(_CACHES, "bop_obj")


def _as_metre_obj(src: str, units: float) -> str:
    """Return a metre-scale **.obj** for `src` (converting + caching if needed).

    PyBullet's GEOM_MESH only loads .obj, but BOP ships its CAD models as .ply in
    millimetres. Converting once (rather than carrying a per-dataset meshScale)
    keeps a single unit convention through the sim, the grasp sampler and the
    FoundationPose call."""
    if src.lower().endswith(".obj") and abs(units - 1.0) < 1e-9:
        return src
    os.makedirs(_OBJ_CACHE, exist_ok=True)
    key = hashlib.md5(f"{src}|{units}".encode("utf-8")).hexdigest()[:10]
    stem = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(_OBJ_CACHE, f"{stem}_{key}.obj")
    if not _cached(out):
        import trimesh
        m = trimesh.load(src, force="mesh")
        if abs(units - 1.0) > 1e-9:
            m.apply_scale(units)
        _atomic_export(m.export, out)
    return out


def _cached(path: str) -> bool:
    """A cache entry counts only if it is non-empty: a machine crash mid-write
    leaves a 0-byte file that `os.path.isfile` happily accepts, and the truncated
    mesh then fails far away (FoundationPose 500 on an empty vertex array)."""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _atomic_export(export_fn, out: str):
    """Write via a temp file + rename, so an interrupted run never leaves a
    partial file behind in the cache."""
    # keep the real extension — trimesh picks the exporter from it
    stem, ext = os.path.splitext(out)
    tmp = f"{stem}.{os.getpid()}.tmp{ext}"
    try:
        export_fn(tmp)
        if not _cached(tmp):
            raise IOError(f"export produced no data: {tmp}")
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def bop_mesh_path(dataset: str, obj_id: int) -> str:
    """BOP obj_id → a metre-scale mesh both PyBullet and FoundationPose can load
    (the full-detail CAD the sim spawns)."""
    ds = BOP_DATASETS[dataset]
    if dataset == "ycbv":
        # textured YCB meshes from object_database (already metres)
        return _YCBV_MESH_BY_ID.get(
            obj_id, os.path.join(YCBV_MESHES, f"obj_{obj_id:06d}", "textured_simple.obj"))
    return _as_metre_obj(os.path.join(ds["models"], f"obj_{obj_id:06d}.ply"), ds["units"])


def eval_mesh_path(dataset: str, obj_id: int) -> Tuple[str, bool]:
    """(path, units_m) of the mesh Stage 3 posed + scored with: BOP `models_eval`
    (mm) when the split is present, else the sim mesh (metres). The choice is
    reported by the experiment so a fallback is never invisible."""
    p = os.path.join(BOP_DATASETS[dataset]["models_eval"], f"obj_{obj_id:06d}.ply")
    if os.path.isfile(p):
        return p, False
    return bop_mesh_path(dataset, obj_id), True


def _image_size(scene_dir: str) -> Tuple[int, int]:
    """(W, H) of the scene's colour frames — YCB-V/LM-O 640x480, T-LESS 720x540."""
    for pat in ("rgb/*.png", "rgb/*.jpg", "gray/*.png"):
        files = sorted(glob.glob(os.path.join(scene_dir, pat)))
        if files:
            from PIL import Image
            with Image.open(files[0]) as im:
                return im.size
    return 640, 480


def best_frame(dataset: str, scene_id: str, obj_id: int) -> Optional[int]:
    """Frame in which `obj_id` (first instance) is most visible (`scene_gt_info`)."""
    sdir = os.path.join(BOP_DATASETS[dataset]["test"], scene_id)
    try:
        gt = json.load(open(os.path.join(sdir, "scene_gt.json")))
        info = json.load(open(os.path.join(sdir, "scene_gt_info.json")))
    except (OSError, ValueError):
        return None
    keys = sorted(gt, key=int)
    idx = [i for i, o in enumerate(gt[keys[0]]) if o["obj_id"] == obj_id]
    if not idx or keys[0] not in info:
        return None
    i = idx[0]
    scored = [(info[k][i].get("visib_fract", 0.0), int(k)) for k in keys if k in info]
    return max(scored)[1] if scored else None


LMO_NAMES = {1: "ape", 5: "can", 6: "cat", 8: "driller", 9: "duck", 10: "eggbox",
             11: "glue", 12: "holepuncher"}


def object_name(dataset: str, obj_id: int) -> str:
    if dataset == "ycbv":
        return YCBV_NAMES.get(obj_id, "?")
    if dataset == "lmo":
        return LMO_NAMES.get(obj_id, f"obj_{obj_id:02d}")
    return f"obj_{obj_id:02d}"


# ---------------------------------------------------------------------------
# One BOP frame, parsed (lazy image access)
# ---------------------------------------------------------------------------
def _pose(R, t_mm) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float).reshape(3, 3)
    T[:3, 3] = np.asarray(t_mm, float) / 1000.0        # mm → m
    return T


@dataclass
class BopFrame:
    dataset: str
    scene_id: int
    im_id: int
    scene_dir: str
    K: np.ndarray
    width: int
    height: int
    depth_scale: float
    gt: list                                 # scene_gt entries for this frame
    info: list                               # scene_gt_info entries (visib_fract …)
    T_w2c: Optional[np.ndarray] = None       # BOP world→camera (m), if the split has one

    # -- images (loaded on demand; the sim only needs poses) -----------------
    def rgb(self) -> np.ndarray:
        from PIL import Image
        p = os.path.join(self.scene_dir, "rgb", f"{self.im_id:06d}.png")
        if not os.path.isfile(p):
            p = p[:-4] + ".jpg"
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)

    def depth_m(self) -> np.ndarray:
        """Sensor depth in metres — identical to Stage 3's `_pose_inputs`."""
        from PIL import Image
        raw = np.array(Image.open(os.path.join(self.scene_dir, "depth", f"{self.im_id:06d}.png")))
        return raw.astype(np.float32) * float(self.depth_scale) / 1000.0

    def mask_visib(self, gt_idx: int) -> np.ndarray:
        from PIL import Image
        p = os.path.join(self.scene_dir, "mask_visib", f"{self.im_id:06d}_{gt_idx:06d}.png")
        return np.array(Image.open(p)) > 0

    def all_masks(self) -> np.ndarray:
        m = np.zeros((self.height, self.width), bool)
        for i in range(len(self.gt)):
            try:
                m |= self.mask_visib(i)
            except OSError:
                pass
        return m

    def visib(self, gt_idx: int) -> Optional[float]:
        try:
            return float(self.info[gt_idx]["visib_fract"])
        except (IndexError, KeyError, TypeError):
            return None

    def T_m2c(self, gt_idx: int) -> np.ndarray:
        o = self.gt[gt_idx]
        return _pose(o["cam_R_m2c"], o["cam_t_m2c"])

    def instances_of(self, obj_id: int) -> List[int]:
        return [i for i, o in enumerate(self.gt) if o["obj_id"] == obj_id]


def load_bop_frame(dataset: str, scene_id, im_id) -> BopFrame:
    """Parse one BOP test frame (poses, camera, annotation) without touching
    the images. `scene_id`/`im_id` accept ints or zero-padded strings."""
    sid, iid = int(scene_id), int(im_id)
    sdir = os.path.join(BOP_DATASETS[dataset]["test"], f"{sid:06d}")
    gt = json.load(open(os.path.join(sdir, "scene_gt.json")))
    cam = json.load(open(os.path.join(sdir, "scene_camera.json")))
    try:
        info = json.load(open(os.path.join(sdir, "scene_gt_info.json")))
    except OSError:
        info = {}
    k = str(iid)
    if k not in gt:
        raise KeyError(f"{dataset} scene {sid:06d} has no frame {iid}")
    c = cam[k]
    W, H = _image_size(sdir)
    T_w2c = (_pose(c["cam_R_w2c"], c["cam_t_w2c"])
             if "cam_R_w2c" in c and "cam_t_w2c" in c else None)
    return BopFrame(dataset, sid, iid, sdir, np.asarray(c["cam_K"], float).reshape(3, 3),
                    W, H, float(c.get("depth_scale", 1.0)), gt[k], info.get(k, []), T_w2c)


# ---------------------------------------------------------------------------
# Scene description in the sim world frame
# ---------------------------------------------------------------------------
@dataclass
class SceneObject:
    obj_id: int
    T_world: np.ndarray                 # 4×4, model→world (metres)
    mesh_path: str
    gt_idx: int = 0                     # BOP instance index within the frame
    T_cam: Optional[np.ndarray] = None  # 4×4, model→camera (metres)
    dataset: str = ""                   # for the per-object mass lookup


@dataclass
class SceneCamera:
    K: np.ndarray                       # 3×3 intrinsics (px)
    T_world: np.ndarray                 # 4×4 camera→world (metres)
    width: int
    height: int


def fit_table_plane(depth_m: np.ndarray, K: np.ndarray, exclude: Optional[np.ndarray] = None,
                    z_range: Optional[Tuple[float, float]] = None, stride: int = 3,
                    n_iter: int = 400, thr: float = 0.006, seed: int = 0,
                    anchors: Optional[np.ndarray] = None, band: Tuple[float, float] = (-0.02, 0.40),
                    touch: float = 0.20):
    """RANSAC [R12] plane through the back-projected depth (n·p + d = 0, n unit
    and pointing "up"). Returns (n, d, inlier_fraction).

    `exclude` masks the annotated objects out; `z_range` keeps only depths
    around the objects. `anchors` (N,3, camera frame) are the object centres:
    a candidate plane is accepted only if every anchor lies between band[0]
    and band[1] ABOVE it and the lowest anchor is within `touch` of it — the
    objects rest on the table, so a wall or the floor (both large planes in a
    cluttered depth image, and both won the vote in some YCB-V frames) cannot
    be chosen."""
    K = np.asarray(K, float)
    H, W = depth_m.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_m[ys, xs]
    ok = (z > 0.15) & (z < 3.0)
    if z_range is not None:
        ok &= (z > z_range[0]) & (z < z_range[1])
    if exclude is not None:
        ok &= ~exclude[ys, xs]
    xs, ys, z = xs[ok].astype(float), ys[ok].astype(float), z[ok].astype(float)
    if z.size < 200:
        raise ValueError(f"too few depth points for a plane fit ({z.size})")
    P = np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], 1)
    rng = np.random.default_rng(seed)
    A = None if anchors is None else np.asarray(anchors, float).reshape(-1, 3)

    def _orient(n, d):
        """Flip so the objects (or, without anchors, the camera) are above."""
        if A is not None:
            if (A @ n + d).mean() < 0:
                return -n, -d
            return n, d
        return (n, d) if d < 0 else (-n, -d)          # camera origin: n·0 + d > 0 ⇔ above

    def _ok(n, d):
        if A is None:
            return True
        h = A @ n + d
        return bool(h.min() >= band[0] and h.max() <= band[1] and h.min() <= touch)

    best_cnt, best_inl = 0, None
    for _ in range(n_iter):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        n, d = _orient(n, float(-n @ a))
        if not _ok(n, d):
            continue
        inl = np.abs(P @ n + d) < thr
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
    if best_inl is None:
        raise ValueError("no plane satisfies the object-height constraints")
    Q = P[best_inl]
    c0 = Q.mean(0)
    n = np.linalg.svd(Q - c0, full_matrices=False)[2][2]      # least-squares refit
    n, d = _orient(n, float(-n @ c0))
    if not _ok(n, d):                                           # refit drifted out of the band
        n, d = _orient(*_plane_from_inliers_fallback(P, best_inl))
    return n, d, float(best_inl.mean())


def _plane_from_inliers_fallback(P, inl):
    """Median-based plane through the inliers (used if the SVD refit violates
    the object-height band): normal from SVD, offset from the median point."""
    Q = P[inl]
    n = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2][2]
    return n, float(-n @ np.median(Q, axis=0))


def world_from_plane(n: np.ndarray, d: float, anchor: np.ndarray) -> np.ndarray:
    """Camera→world transform whose z-axis is the plane normal `n` (up), whose
    x-axis is the camera x-axis projected onto the plane, and whose origin is
    `anchor` (camera frame) dropped onto the plane. The table is z = 0."""
    z = n / np.linalg.norm(n)
    x = np.array([1.0, 0.0, 0.0])
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])                                      # rows: world axes in cam coords
    p0 = anchor - (n @ anchor + d) * n                           # foot point on the plane
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ p0
    return T


def scene_from_frame(fr: BopFrame, world: str = "auto", exclude_dilate: int = 6
                     ) -> Tuple[List[SceneObject], SceneCamera, dict]:
    """Objects + camera in a z-up sim world with the TABLE at z = 0, plus the
    validity checks.

    world="auto":  BOP extrinsics where the split has them (YCB-V, T-LESS),
                   else the depth-fitted plane (LM-O).
    world="bop":   BOP extrinsics (falls back to "plane" without them). The BOP
                   world height is arbitrary, so it is shifted such that the
                   lowest object vertex sits at z = 0 (something touches the table).
    world="plane": table plane fitted to the frame's real depth (all datasets),
                   constrained so the annotated objects rest on it.
    The returned `info` carries `world`, `plane_inlier`, `plane_angle_deg` (the
    fitted plane vs the BOP up-vector, computed whenever both exist) and
    `bottom_gap_mm` (lowest object vertex vs the table) — the experiment
    records them per trial."""
    T_m2c = [fr.T_m2c(i) for i in range(len(fr.gt))]
    centres = np.array([T[:3, 3] for T in T_m2c])
    if world == "auto":
        world = "bop" if fr.T_w2c is not None else "plane"
    if world == "bop" and fr.T_w2c is None:
        world = "plane"
    info: dict = {"world": world}
    # depth-fitted plane: the world for "plane", the self-check for "bop"
    plane = None
    try:
        from scipy import ndimage
        excl = ndimage.binary_dilation(fr.all_masks(), iterations=exclude_dilate)
        zr = (float(centres[:, 2].min()) - 0.5, float(centres[:, 2].max()) + 0.5)
        plane = fit_table_plane(fr.depth_m(), fr.K, exclude=excl, z_range=zr, anchors=centres)
        info["plane_inlier"] = round(plane[2], 3)
    except Exception as exc:
        info["plane_error"] = str(exc)[:60]
        if world == "plane":
            raise
    if fr.T_w2c is not None and plane is not None:
        up_c = fr.T_w2c[:3, :3] @ np.array([0.0, 0.0, 1.0])
        info["plane_angle_deg"] = round(float(np.degrees(np.arccos(
            np.clip(abs(up_c @ plane[0]), -1.0, 1.0)))), 2)
    if world == "bop":
        T_c2w = np.linalg.inv(fr.T_w2c)
    else:
        T_c2w = world_from_plane(plane[0], plane[1], centres.mean(0))

    def _objs(T):
        return [SceneObject(o["obj_id"], T @ T_m2c[i], bop_mesh_path(fr.dataset, o["obj_id"]),
                            gt_idx=i, T_cam=T_m2c[i], dataset=fr.dataset) for i, o in enumerate(fr.gt)]
    objs = _objs(T_c2w)
    # lowest vertex of every object in the world: objects rest on the table, so
    # the minimum over the scene should sit at z ≈ 0 (a bad plane shows here).
    bottoms = [_bottom_z(o) for o in objs]
    if world == "bop":
        shift = min(bottoms)                     # BOP height is arbitrary: table := lowest vertex
        T_c2w = T_c2w.copy()
        T_c2w[2, 3] -= shift
        objs = _objs(T_c2w)
        bottoms = [b - shift for b in bottoms]
        info["bop_shift_mm"] = round(1000.0 * shift, 1)
    info["bottom_gap_mm"] = round(1000.0 * min(bottoms), 1)
    info["bottom_z"] = {o.gt_idx: b for o, b in zip(objs, bottoms)}
    return objs, SceneCamera(fr.K, T_c2w, fr.width, fr.height), info


_VERT_CACHE: Dict[str, np.ndarray] = {}


def _bottom_z(o: SceneObject) -> float:
    """World z of the object's lowest mesh vertex (vertices cached per mesh)."""
    v = _VERT_CACHE.get(o.mesh_path)
    if v is None:
        import trimesh
        v = np.asarray(trimesh.load(o.mesh_path, force="mesh").vertices, float)
        _VERT_CACHE[o.mesh_path] = v
    return float((v @ o.T_world[:3, :3].T + o.T_world[:3, 3])[:, 2].min())


def load_bop_scene(dataset: str, scene_id: str, frame: Optional[int] = None
                   ) -> Tuple[List[SceneObject], SceneCamera]:
    """Legacy entry (demos): objects + camera in the BOP world frame.
    `frame=None` takes the lowest frame id."""
    sdir = os.path.join(BOP_DATASETS[dataset]["test"], f"{int(scene_id):06d}")
    if frame is None or str(frame) not in json.load(open(os.path.join(sdir, "scene_gt.json"))):
        frame = int(sorted(json.load(open(os.path.join(sdir, "scene_gt.json"))), key=int)[0])
    fr = load_bop_frame(dataset, scene_id, frame)
    objs, cam, _ = scene_from_frame(fr, world="bop")
    return objs, cam


def load_ycbv_scene(scene_id: str, frame: int = 1
                    ) -> Tuple[List[SceneObject], SceneCamera]:
    """YCB-V shorthand for :func:`load_bop_scene` (kept for existing callers)."""
    return load_bop_scene("ycbv", scene_id, frame)


# ---------------------------------------------------------------------------
# PyBullet world
# ---------------------------------------------------------------------------
@dataclass
class TabletopSim:
    gui: bool = False
    _p: object = field(default=None, repr=False)
    body: Dict[int, int] = field(default_factory=dict)       # obj_id → bullet body (first instance)
    body_inst: Dict[int, int] = field(default_factory=dict)  # gt_idx → bullet body
    table_z: float = 0.0
    robot: Optional[int] = None
    cam: Optional[SceneCamera] = None
    _egl: bool = field(default=False, repr=False)            # EGL plugin loaded?

    def connect(self):
        import pybullet as p
        import pybullet_data
        self._p = p
        p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        # OPT-IN EGL rasteriser (GRASP_EGL=1). It is worth it only for capturing
        # many frames (GIFs): ~2x faster than TINY_RENDERER. But in this container
        # EGL resolves to Mesa **llvmpipe** — a SOFTWARE rasteriser that saturates
        # every CPU core, not the NVIDIA GPU. Sustained all-core load is implicated
        # in the CLOCK_WATCHDOG_TIMEOUT bugchecks on this machine (2026-09-06), so
        # it stays off unless a caller explicitly asks for the throughput.
        self._egl = False
        if not self.gui and os.environ.get("GRASP_EGL") == "1":
            try:
                import pkgutil
                loader = pkgutil.get_loader("eglRenderer")
                pid = (p.loadPlugin(loader.get_filename(), "_eglRendererPlugin")
                       if loader is not None else p.loadPlugin("eglRendererPlugin"))
                self._egl = pid >= 0
            except Exception as exc:                       # pragma: no cover
                print(f"[sim] EGL renderer unavailable ({exc}); using CPU renderer")
        if os.environ.get("GRASP_QUIET") != "1":
            print(f"[sim] renderer = {'EGL/llvmpipe' if self._egl else 'tiny (single-thread)'}")
        return self

    # ---- world construction ------------------------------------------------
    def build(self, objects: List[SceneObject], camera: SceneCamera,
              target_id: Optional[int] = None, with_robot: bool = True,
              target_gt_idx: Optional[int] = None, table_z: Optional[float] = None):
        """Static clutter + ONE dynamic target + table + (optional) Panda.

        The target is `target_gt_idx` (instance) or else the first instance of
        `target_id` (obj_id). `table_z=None` puts the plane just under the
        lowest object origin (legacy); pass 0.0 for a plane-fitted world."""
        p = self._p
        self.cam = camera
        if target_gt_idx is None and target_id is not None:
            cands = [o.gt_idx for o in objects if o.obj_id == target_id]
            target_gt_idx = cands[0] if cands else None
        if table_z is None:
            table_z = min(o.T_world[2, 3] for o in objects) - 0.005
        self.table_z = table_z
        p.loadURDF("plane.urdf", [0, 0, self.table_z])

        self.body, self.body_inst = {}, {}
        for o in objects:
            b = self._spawn_mesh(o, dynamic=(o.gt_idx == target_gt_idx))
            self.body_inst[o.gt_idx] = b
            self.body.setdefault(o.obj_id, b)

        # remember the as-built pose of every object, so a failed grasp attempt
        # can be undone before the next try (see reset_objects()).
        self.target_gt_idx = target_gt_idx
        self.target_id = next((o.obj_id for o in objects if o.gt_idx == target_gt_idx), None)
        self.freeze_initial()
        if with_robot:
            self._add_panda(objects)

    def freeze_initial(self):
        """Snapshot every body's current pose as the reset state (call after a
        settle so resets return to the settled, not the annotated, pose)."""
        p = self._p
        self.init_pose = {g: p.getBasePositionAndOrientation(b)
                          for g, b in self.body_inst.items()}

    def reset_objects(self):
        """Restore every object to its snapshot pose + zero velocity."""
        p = self._p
        for g, (pos, quat) in self.init_pose.items():
            b = self.body_inst[g]
            p.resetBasePositionAndOrientation(b, pos, quat)
            p.resetBaseVelocity(b, [0, 0, 0], [0, 0, 0])

    def target_body(self) -> Optional[int]:
        return self.body_inst.get(self.target_gt_idx)

    def target_pose(self) -> np.ndarray:
        """Current 4×4 model→world pose of the dynamic target (its body origin
        IS the model origin — createMultiBody was given the model pose)."""
        pos, quat = self._p.getBasePositionAndOrientation(self.target_body())
        T = np.eye(4)
        T[:3, :3] = _quat2mat(quat)
        T[:3, 3] = pos
        return T

    def target_displacement_mm(self, T_ref: np.ndarray) -> float:
        """How far the target's origin moved from `T_ref` (e.g. the annotated
        pose) — settling drift, a validity check of the reconstruction."""
        return float(np.linalg.norm(self.target_pose()[:3, 3] - T_ref[:3, 3]) * 1000)

    def _spawn_mesh(self, o: SceneObject, dynamic: bool) -> int:
        """Load a mesh at its world pose. Static clutter uses concave
        collision; the dynamic target uses a V-HACD decomposition."""
        p = self._p
        pos = o.T_world[:3, 3]
        quat = _mat2quat(o.T_world[:3, :3])
        # bop_mesh_path() guarantees a METRE-scale mesh for every dataset, so
        # meshScale = 1.0. (Only the BOP *poses* are mm and get ÷1000 in _pose.)
        S = [1.0, 1.0, 1.0]
        vis = p.createVisualShape(p.GEOM_MESH, fileName=o.mesh_path, meshScale=S)
        if dynamic:
            # V-HACD convex decomposition so the dynamic target keeps its CONCAVE
            # geometry (mug handle/cavity, bowl interior). A plain GEOM_MESH
            # collision is a single convex hull — a mug becomes a solid blob and
            # can't be grasped realistically. (Concave trimesh collision can't be
            # dynamic in PyBullet, hence the decomposition into convex parts.)
            col = self._vhacd_collision(o.mesh_path, S)
            mass = object_mass(o.dataset, o.obj_id)
        else:
            col = p.createCollisionShape(p.GEOM_MESH, fileName=o.mesh_path,
                                         meshScale=S,
                                         flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            mass = 0.0                                              # static clutter
        body = p.createMultiBody(mass, col, vis, pos, quat)
        if dynamic:
            # high friction so a parallel-jaw grip can hold the object during lift
            p.changeDynamics(body, -1, lateralFriction=TARGET_FRICTION,
                             spinningFriction=0.005, rollingFriction=0.001)
        return body

    def _vhacd_collision(self, mesh_path: str, scale) -> int:
        """Convex-decomposition collision shape (cached), falling back to a plain
        convex hull if V-HACD is unavailable or fails. V-HACD is PyBullet's
        bundled implementation of approximate convex decomposition [R13]."""
        p = self._p
        cache_dir = os.path.join(_CACHES, "vhacd")
        os.makedirs(cache_dir, exist_ok=True)
        key = hashlib.md5(mesh_path.encode("utf-8")).hexdigest()[:12]
        out = os.path.join(cache_dir, f"{key}.obj")
        if not _cached(out):
            log = os.path.join(cache_dir, f"{key}.log")
            try:
                _atomic_export(
                    lambda dst: p.vhacd(mesh_path, dst, log, resolution=200000,
                                        maxNumVerticesPerCH=64), out)
            except Exception as exc:                      # pragma: no cover
                print(f"[sim] V-HACD unavailable ({exc}); using convex hull")
                return p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path,
                                              meshScale=scale)
        if not _cached(out):                              # vhacd ran but wrote nothing
            print("[sim] V-HACD produced no output; using convex hull")
            return p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path,
                                          meshScale=scale)
        return p.createCollisionShape(p.GEOM_MESH, fileName=out, meshScale=scale)

    def _add_panda(self, objects: List[SceneObject], pedestal_h: float = 0.35):
        """Mount a Panda behind the objects, on a pedestal that lifts the base
        above the tabletop.

        A Panda based *at* table height jams against the infinite ground plane
        (its lower links collide with the plane), so the arm can never execute a
        reach. Raising the base ~0.35 m onto a support box gives the arm a clear
        workspace over the objects — the standard tabletop-manipulation setup."""
        p = self._p
        cx = np.mean([o.T_world[0, 3] for o in objects])
        cy = np.mean([o.T_world[1, 3] for o in objects])
        # The robot stands on the CAMERA's side of the table, 0.55 m back from the
        # objects' centroid along the camera's viewing direction projected onto
        # the table — as if the camera were mounted on the robot. This makes the
        # reachable side a property of the frame, not of an arbitrary world axis
        # (with a fixed world −x the same grasp set was reachable from one BOP
        # world convention and unreachable from another).
        d = np.array([1.0, 0.0, 0.0])
        if self.cam is not None:
            v = self.cam.T_world[:3, 2].copy()                     # camera +z (viewing dir) in world
            v[2] = 0.0
            if np.linalg.norm(v) > 0.05:                           # not looking straight down
                d = v / np.linalg.norm(v)
        bx, by = cx - 0.55 * d[0], cy - 0.55 * d[1]
        base_z = self.table_z + pedestal_h

        # visible support pedestal (static) from the table up to the base
        half = [0.06, 0.06, pedestal_h / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                  rgbaColor=[0.30, 0.30, 0.33, 1.0])
        p.createMultiBody(0, col, vis, [bx, by, self.table_z + pedestal_h / 2])

        self.robot_base_z = base_z
        self.robot = p.loadURDF("franka_panda/panda.urdf", [bx, by, base_z],
                                useFixedBase=True)

    def settle(self, steps: int = 240):
        for _ in range(steps):
            self._p.stepSimulation()

    # ---- sensing -----------------------------------------------------------
    def render_rgbd(self) -> Dict[str, np.ndarray]:
        """RGB (H,W,3 uint8), depth (H,W float m), seg (H,W int body id)."""
        p, cam = self._p, self.cam
        eye = cam.T_world[:3, 3]
        R = cam.T_world[:3, :3]
        target = eye + R @ np.array([0, 0, 1.0])          # OpenCV: look along +z
        up = R @ np.array([0, -1.0, 0])                   # OpenCV y is down
        view = p.computeViewMatrix(eye, target, up)
        fx, fy = cam.K[0, 0], cam.K[1, 1]
        near, far = 0.01, 3.0
        fov_y = 2 * np.degrees(np.arctan(cam.height / (2 * fy)))
        proj = p.computeProjectionMatrixFOV(fov_y, cam.width / cam.height, near, far)
        renderer = (p.ER_BULLET_HARDWARE_OPENGL if getattr(self, "_egl", False)
                    else p.ER_TINY_RENDERER)
        w, h, rgba, depth_buf, seg = p.getCameraImage(
            cam.width, cam.height, view, proj, renderer=renderer)
        rgb = np.reshape(rgba, (h, w, 4))[:, :, :3].astype(np.uint8)
        depth_buf = np.reshape(depth_buf, (h, w))
        depth = far * near / (far - (far - near) * depth_buf)   # buffer → metric depth
        seg = np.reshape(seg, (h, w)).astype(np.int32)
        return {"rgb": rgb, "depth": depth, "seg": seg}

    def object_mask(self, obj_id: int, seg: np.ndarray) -> np.ndarray:
        """Boolean mask of one object (first instance) in a seg image."""
        return seg == self.body.get(obj_id, -999)

    def disconnect(self):
        if self._p is not None:
            self._p.disconnect()


# Physics constants of the dynamic target (reported in the experiment manifest).
# PyBullet combines the lateral friction of two bodies MULTIPLICATIVELY (measured
# 2026-09-11 with a box on a plane: 1.0 x 1.0 -> 1.01, 1.6 x 1.5 -> 2.42), so the
# finger-object coefficient is TARGET_FRICTION x grasp_execute.FINGER_FRICTION.
# 1.0 x 1.0 = 1.0 is a plain, physically plausible value (rubber pad on plastic);
# the pilot's 1.6 x 1.5 = 2.4 was unrealistically high.
TARGET_FRICTION = 1.0
TARGET_MASS_KG = 0.2                 # fallback where no measured mass is known
# Measured object masses in kg, keyed by (dataset, obj_id). YCB objects: take the
# values from the YCB object set list (Calli et al. 2015, ycbbenchmarks.com) —
# fill in before a run and keep the source in the commit message. T-LESS and
# LM-O publish no masses: fallback.
OBJECT_MASS_KG: Dict[Tuple[str, int], float] = {
    # Quelle: YCB object list (Calli et al. 2015), ycbbenchmarks.com,
    # object-list-Sheet1.pdf, Spalte "Mass" — abgelesen 2026-09-11.
    ("ycbv", 2):  0.411,   # Cheez-it Cracker box        411 g
    ("ycbv", 3):  0.514,   # Domino Sugar box            514 g
    ("ycbv", 5):  0.603,   # French's Mustard bottle     603 g
    ("ycbv", 14): 0.118,   # Mug                         118 g
    ("ycbv", 17): 0.082,   # Scissors                     82 g
    # T-LESS / LM-O publizieren keine Massen -> TARGET_MASS_KG-Fallback.
}


def object_mass(dataset: str, obj_id: int) -> float:
    return OBJECT_MASS_KG.get((dataset, obj_id), TARGET_MASS_KG)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mat2quat(R: np.ndarray):
    """3×3 rotation → PyBullet quaternion [x,y,z,w]."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()             # [x,y,z,w]


def _quat2mat(q) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q).as_matrix()


# YCB-V obj_id → readable name (for the CLI / prompts).
YCBV_NAMES = {
    1: "master chef can", 2: "cracker box", 3: "sugar box", 4: "tomato soup can",
    5: "mustard bottle", 6: "tuna fish can", 7: "pudding box", 8: "gelatin box",
    9: "potted meat can", 10: "banana", 11: "pitcher base", 12: "bleach cleanser",
    13: "bowl", 14: "mug", 15: "power drill", 16: "wood block", 17: "scissors",
    18: "large marker", 19: "large clamp", 20: "extra large clamp", 21: "foam brick",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage-5 tabletop sim scene (5.1)")
    ap.add_argument("--dataset", default="ycbv", choices=sorted(BOP_DATASETS))
    ap.add_argument("--scene", default="000048", help="BOP test scene id")
    ap.add_argument("--frame", type=int, default=None, help="BOP im_id (default: first)")
    ap.add_argument("--target", type=int, default=None,
                    help="obj_id of the dynamic (graspable) target; default = none dynamic")
    ap.add_argument("--world", default="auto", choices=("auto", "plane", "bop"),
                    help="auto = BOP extrinsics where present else the depth-fitted table plane")
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--settle", type=int, default=0, help="physics settle steps")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", default=None, help="save <out>_rgb.png/_depth.png/_seg.png")
    args = ap.parse_args()

    sdir = os.path.join(BOP_DATASETS[args.dataset]["test"], f"{int(args.scene):06d}")
    frame = args.frame
    if frame is None:
        frame = int(sorted(json.load(open(os.path.join(sdir, "scene_gt.json"))), key=int)[0])
    fr = load_bop_frame(args.dataset, args.scene, frame)
    objs, cam, info = scene_from_frame(fr, world=args.world)
    print(f"[scene] {args.dataset} {int(args.scene):06d} frame {frame}: "
          f"{len(objs)} objects -> " +
          ", ".join(f"{o.gt_idx}:{o.obj_id}:{object_name(args.dataset, o.obj_id)}" for o in objs))
    print(f"[scene] world={info['world']} plane_inlier={info.get('plane_inlier')} "
          f"plane_angle_deg={info.get('plane_angle_deg')} bottom_gap_mm={info['bottom_gap_mm']} "
          f"bop_shift_mm={info.get('bop_shift_mm')}")

    sim = TabletopSim(gui=args.gui).connect()
    sim.build(objs, cam, target_id=args.target, with_robot=not args.no_robot, table_z=0.0)
    if args.settle:
        sim.settle(args.settle)
    out = sim.render_rgbd()
    print(f"[scene] rendered RGB {out['rgb'].shape}, depth range "
          f"[{out['depth'][out['depth']<2.9].min():.2f},{out['depth'][out['depth']<2.9].max():.2f}] m, "
          f"{len(np.unique(out['seg']))} seg ids")

    if args.out:
        from PIL import Image
        Image.fromarray(out["rgb"]).save(f"{args.out}_rgb.png")
        d = out["depth"].copy(); d[d > 2.9] = 0
        Image.fromarray((255 * d / (d.max() + 1e-9)).astype(np.uint8)).save(f"{args.out}_depth.png")
        segv = (out["seg"] - out["seg"].min())
        Image.fromarray((255 * segv / (segv.max() + 1e-9)).astype(np.uint8)).save(f"{args.out}_seg.png")
        print(f"[scene] wrote {args.out}_{{rgb,depth,seg}}.png")
    if args.gui:
        print("[scene] GUI up — Ctrl-C to exit"); import time
        while True:
            sim._p.stepSimulation(); time.sleep(1 / 240)
    sim.disconnect()


if __name__ == "__main__":
    main()
