#!/usr/bin/env python3
"""
Compute each gallery object's normalization diameter (in METRES) for dGeDi
co-scaling — the SAME quantity the gallery descriptor self-normalized by.

dGeDi co-scales query and candidate by ONE common divisor so their local
descriptors sit at a consistent physical scale. The gallery descriptor
(server.compute_feats -> prep_cloud) self-normalizes each candidate by
``_diameter(FPS(6000)(sample_cloud(mesh, 10000)))`` — the max pairwise distance
of the down-sampled surface cloud. At query time the server divides the query
by ``diameters.json[cid]``, so that value MUST be the *identical* measure, in
the query's units (metres). Reconstructing it from a different point set (raw
mesh vertices, a 2-pass approximation) leaves query and gallery at slightly
different scales and changes the meaning of the fixed RANSAC threshold
(audit P0.4).

This reproduces the gallery pipeline exactly — same seeded surface sampling
(`preprocessing/precompute_dgedi.py`, ``sample_cloud``), same FPS budget and
`_diameter` (`server`) — then converts native units -> metres. No descriptor
regen needed: the descriptors are already self-normalized; only the query
divisor is fixed.

Run (dgedi container; this repo is mounted at /oscar):
    docker compose run --rm --no-deps dgedi python3 \
        /oscar/services/dgedi/compute_diameters.py \
        --manifest /oscar/caches/dgedi/manifest_bop.json \
        --out      /oscar/caches/dgedi/bop/diameters.json
"""

import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, _THIS)                                    # server (this folder)
sys.path.insert(0, os.path.join(_REPO, "preprocessing"))     # precompute_dgedi
import server                       # _diameter, FPS_POINTS         # noqa: E402
from precompute_dgedi import sample_cloud   # identical seeded sampling  # noqa: E402

# Native units per dataset (mirrors stage3_gallery.DATASET_LAYOUT units_m).
UNITS_M = {"gso": True, "housecat6d": True, "itodd": False,
           "ycbv": False, "tless": False, "lmo": False}


def gallery_diameter_native(mesh_path: str, n_points: int = 10000) -> float:
    """The exact divisor prep_cloud used: max pairwise distance of the FPS'd
    surface cloud, in the mesh's native units."""
    pts = sample_cloud(mesh_path, n_points)          # seeded, == gallery sample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    if len(pcd.points) > server.FPS_POINTS:
        pcd = pcd.farthest_point_down_sample(server.FPS_POINTS)
    return server._diameter(np.asarray(pcd.points))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default="/oscar")
    ap.add_argument("--n-points", type=int, default=10000)
    ap.add_argument("--diam-scale", type=float, default=None,
                    help="Override the native->query-units factor. Default uses the "
                         "UNITS_M mm/m table (BOP query=metres, CAD=mm -> 0.001).")
    ap.add_argument("--ones", action="store_true",
                    help="Write 1.0 for every manifest entry instead of measuring. "
                         "SHREC'18 matches scale-invariantly (both sides unit "
                         "diameter), so the server's per-candidate co-scaling "
                         "must be a no-op.")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    if args.ones:
        json.dump({nsid: 1.0 for nsid in manifest}, open(args.out, "w"))
        print(f"[diam] wrote {len(manifest)} unit diameters -> {args.out}",
              flush=True)
        return
    diam, missing = {}, 0
    for i, (nsid, rel) in enumerate(sorted(manifest.items())):
        ds = nsid.split("/")[0]
        path = os.path.join(args.repo_root, rel)
        try:
            d_native = gallery_diameter_native(path, args.n_points)
        except Exception as exc:
            print(f"[diam] FAIL {nsid}: {exc}", flush=True)
            missing += 1
            continue
        if not (d_native > 0):
            missing += 1
            continue
        factor = (args.diam_scale if args.diam_scale is not None
                  else (1.0 if UNITS_M.get(ds, False) else 0.001))  # -> query units
        diam[nsid] = d_native * factor
        if (i + 1) % 200 == 0:
            print(f"[diam] {i+1}/{len(manifest)}", flush=True)

    json.dump(diam, open(args.out, "w"))
    print(f"[diam] wrote {len(diam)} diameters ({missing} skipped) -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
