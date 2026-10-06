#!/usr/bin/env python3
"""Stage-5 · 5.2 — Perception: segment → retrieve PROXY → pose.

Given the sim's RGB-D + a target object, run the **actual OSCAR+ pipeline**:
  1. segment the target (here: the sim's GT mask, standing in for Step-1 / SAM);
  2. retrieve the best CAD from **G_proxy only** (`stage3_gallery.assemble_gallery`
     with no target datasets) — the target's own CAD is *not* in the gallery, so
     the top-1 is necessarily a PROXY (the open-set case, = Stage-3 setting 3b);
  3. estimate the 6-DoF pose of that proxy with **FoundationPose**
     (`pipeline.foundationpose_bridge.call_foundationpose`).

Nothing here is re-implemented — it wires the existing modules:
    assemble_gallery · run_query · fusion_ranking · crop_by_bbox · call_foundationpose

CLI:
    python -m grasping.perceive --scene 000048 --target 6 --topk 10
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)                                # pipeline.*
sys.path.insert(0, os.path.join(_ROOT, "evaluation"))    # stage3_gallery, eval_common

FP_URL = os.environ.get("FP_URL", "http://foundationpose:5050")


# ---------------------------------------------------------------------------
@dataclass
class Perception:
    gallery: object            # stage3_gallery.UnionGallery (proxy-only)
    components: tuple          # (pcfg, clip_retr, dino_rer, fusion_mod, shape_m)
    cfg: object                # EvalConfig

    @classmethod
    def load(cls, use_uni3d: bool = False, proxy_ds=None) -> "Perception":
        """Assemble G_proxy (default GSO ∪ HouseCat6D ∪ ITODD) — targets EXCLUDED.

        ``proxy_ds`` overrides the proxy dataset set (e.g. ("gso",) to use only
        the datasets actually present on this machine)."""
        from stage3_gallery import assemble_gallery, UNI3D_OVERRIDES, PROXY_DATASETS
        kw = {} if proxy_ds is None else {"proxy_ds": tuple(proxy_ds)}
        gal = assemble_gallery(target_datasets=(),                 # proxy-only
                               extra_overrides=(UNI3D_OVERRIDES if use_uni3d else None),
                               **kw)
        comp = gal.components()
        print(f"[perceive] proxy gallery |G_proxy| = {len(gal.gallery_ids)}")
        return cls(gallery=gal, components=comp, cfg=gal.eval_cfg)

    # ---- 1. segment (GT mask stand-in for SAM) ----------------------------
    @staticmethod
    def segment(seg_img: np.ndarray, body_id: int) -> np.ndarray:
        """Boolean mask of the target in the sim segmentation image."""
        return seg_img == body_id

    # ---- 2. retrieve the top-1 PROXY --------------------------------------
    def retrieve_proxy(self, rgb: np.ndarray, mask: np.ndarray
                       ) -> Tuple[str, str, List[Tuple[str, float]]]:
        """Crop by the mask, score against G_proxy, return (proxy_id, cad_path, ranking)."""
        from PIL import Image
        from eval_common import run_query, fusion_ranking, crop_by_bbox
        pcfg, clip_retr, dino_rer, fusion_mod, shape_m = self.components
        bbox = _mask_bbox(mask)
        roi = crop_by_bbox(Image.fromarray(rgb), bbox)
        top_k = len(self.gallery.gallery_ids) + 5
        out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m,
                        roi, self.cfg, ulip_query_emb=None,
                        dino_full_top_k=top_k, ulip_full_top_k=top_k,
                        clip_full_top_k=len(clip_retr._desc_labels))
        ranking = fusion_ranking(out["fused_full"])
        proxy_id = ranking[0][0]
        # UnionGallery keeps the native-scale pose mesh as id_to_pose_mesh:
        #   namespaced_id -> (path, units_m)   (units_m: True == metres, e.g. GSO)
        cad_path, self._proxy_units_m = self.gallery.id_to_pose_mesh[proxy_id]
        return proxy_id, cad_path, ranking

    # ---- 3. pose the proxy with FoundationPose ----------------------------
    def _units(self) -> float:
        """Native-mesh -> metres factor for the retrieved proxy."""
        return 1.0 if getattr(self, "_proxy_units_m", True) else 0.001

    def observed_scale(self, depth, mask, K, cad_path) -> float:
        """Factor that resizes the proxy (in METRES) to the target's OBSERVED
        metric size. A retrieved proxy is an arbitrary-sized CAD — a 16 cm plant
        pot standing in for a 9 cm mug must be shrunk or its grasps miss. Uses the
        back-projected masked-depth cloud diagonal vs the proxy's diagonal."""
        import trimesh
        obs_diag = observed_diag(depth, mask, K)
        if obs_diag is None:
            return 1.0
        proxy = trimesh.load(cad_path, force="mesh")
        proxy_diag = float(np.linalg.norm(np.asarray(proxy.extents) * self._units()))
        return obs_diag / proxy_diag if proxy_diag > 1e-6 else 1.0

    def estimate_pose(self, rgb, depth, mask, cad_path, K,
                      refine_iter: int = 5, size_scale: float = 1.0
                      ) -> Tuple[np.ndarray, float]:
        from pipeline.foundationpose_bridge import call_foundationpose
        # FP server route is /estimate_pose (mirror step8); FP_URL is the base.
        url = FP_URL.rstrip("/") + "/estimate_pose"
        # scale takes the native mesh to metres AND resizes it to the observed
        # target size (metre meshes -> units 1.0, mm meshes -> 0.001).
        scale = self._units() * float(size_scale)
        return call_foundationpose(url, rgb, depth, mask.astype(np.uint8),
                                   np.asarray(K), cad_path, scale=scale,
                                   refine_iter=refine_iter)


def observed_diag(depth, mask, K, min_px: int = 50) -> Optional[float]:
    """Metric diagonal (m) of the target's back-projected masked-depth cloud.

    The measured size of the object as the camera sees it — used to size an
    arbitrary retrieved CAD to the real thing. Returns None if the mask carries
    too few valid depth pixels to be trusted."""
    K = np.asarray(K, float)
    ys, xs = np.where(mask)
    z = depth[ys, xs]
    ok = (z > 0.05) & (z < 2.9)
    xs, ys, z = xs[ok], ys[ok], z[ok]
    if z.size < min_px:
        return None
    X = (xs - K[0, 2]) * z / K[0, 0]
    Y = (ys - K[1, 2]) * z / K[1, 1]
    P = np.stack([X, Y, z], axis=1)
    lo, hi = np.percentile(P, [2, 98], axis=0)         # trim depth outliers
    return float(np.linalg.norm(hi - lo))


def _mask_bbox(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    return [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1]


# ---------------------------------------------------------------------------
# CLI — runs the sim (5.1) then perception (5.2) end-to-end for one target.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage-5 perception (5.2)")
    ap.add_argument("--scene", default="000048")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--target", type=int, required=True, help="YCB obj_id to grasp")
    ap.add_argument("--uni3d", action="store_true", help="Uni3D shape arm")
    ap.add_argument("--no-pose", action="store_true", help="retrieval only (skip FoundationPose)")
    ap.add_argument("--proxy", default=None,
                    help="comma-separated proxy datasets (default: all; e.g. 'gso')")
    ap.add_argument("--topk", type=int, default=10, help="print this many ranked proxies")
    args = ap.parse_args()
    proxy_ds = ([p.strip() for p in args.proxy.split(",") if p.strip()]
                if args.proxy else None)

    from grasping.sim_scene import TabletopSim, load_ycbv_scene, YCBV_NAMES
    objs, cam = load_ycbv_scene(args.scene, args.frame)
    sim = TabletopSim().connect()
    sim.build(objs, cam, target_id=args.target, with_robot=False)
    obs = sim.render_rgbd()
    body = sim.body[args.target]
    mask = Perception.segment(obs["seg"], body)
    print(f"[perceive] target = {args.target}:{YCBV_NAMES.get(args.target,'?')}  "
          f"mask px = {int(mask.sum())}")

    per = Perception.load(use_uni3d=args.uni3d, proxy_ds=proxy_ds)
    proxy_id, cad_path, ranking = per.retrieve_proxy(obs["rgb"], mask)
    print(f"[perceive] TOP-1 PROXY = {proxy_id}\n           cad = {cad_path}")
    for i, (oid, s) in enumerate(ranking[:args.topk]):
        print(f"   {i:2d}. {oid:32s} {s:.4f}")

    if not args.no_pose:
        pose, conf = per.estimate_pose(obs["rgb"], obs["depth"], mask, cad_path, cam.K)
        print(f"[perceive] FoundationPose pose (proxy):\n{np.round(pose,4)}\n"
              f"           confidence {conf:.3f}")
    sim.disconnect()


if __name__ == "__main__":
    main()
