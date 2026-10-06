#!/usr/bin/env python3
"""
Precompute dGeDi descriptors for every gallery object (runs in oscar-dgedi).

For each ``{id: mesh}`` in the manifest: load the mesh, uniformly sample a
point cloud (deterministic, seeded from the id — same reproducibility rule as
``step_b2._load_cad_pointcloud``), self-normalize + extract dGeDi features, and
save ``<out>/<id>.npz`` (arrays ``points`` MxD self-normalized, ``feats`` MxD).
The service (``services/dgedi/server.py``) loads these at startup for /rerank.

Normally driven by ``preprocess_gallery.py --step dgedi``, which builds the
manifest and starts the container.  Directly (host; this repo is mounted at
/oscar inside the dgedi container):

    docker compose run --rm --no-deps dgedi python3 \
        /oscar/preprocessing/precompute_dgedi.py \
        --manifest /oscar/caches/dgedi/manifest_<gallery>.json \
        --out      /oscar/caches/dgedi/<gallery> \
        --n-points 10000 --mode multi_scale
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import open3d as o3d

# server.py provides the model loader + self-normalizing feature extractor;
# reuse them so precompute and serving are bit-identical. It lives with the
# service (services/dgedi/), not next to this script.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "services", "dgedi"))
import server  # noqa: E402


def sample_cloud(mesh_path: str, n_points: int) -> np.ndarray:
    """Deterministic uniform surface sample -> (n,3). Empty on failure."""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        # Some 'meshes' may already be point clouds (.ply); try that.
        pcd = o3d.io.read_point_cloud(mesh_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        return pts
    mesh.compute_vertex_normals()
    # o3d's uniform sampler draws from the GLOBAL RNG (no seed arg) — seed from
    # the id so the cloud is a pure function of the model, not of call order.
    o3d.utility.random.seed(
        int(hashlib.sha1(os.path.basename(mesh_path).encode()).hexdigest()[:8], 16)
        % (2 ** 31 - 1))
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return np.asarray(pcd.points, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default="/oscar",
                    help="prefix for the manifest's repo-relative mesh paths")
    ap.add_argument("--config", default="/dgedi/config_dgedi.yaml")
    ap.add_argument("--mode", default="multi_scale")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-points", type=int, default=10000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--timing-json", default="",
                    help="optional: write per-object wall times + model-load "
                         "time here (measurement only, results unchanged)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    print(f"[precompute] loading dGeDi ({args.mode}) ...", flush=True)
    t0 = time.perf_counter()
    server._STATE["model"] = server.load_model(args.config, args.mode, args.device)
    server._STATE["device"] = args.device
    t_model_load = time.perf_counter() - t0

    done = skipped = failed = 0
    per_object = {}
    for i, (nsid, rel) in enumerate(sorted(manifest.items())):
        out_path = os.path.join(args.out, server.id_to_fname(nsid))
        if os.path.isfile(out_path) and not args.overwrite:
            skipped += 1
            continue
        mesh_path = os.path.join(args.repo_root, rel)
        try:
            t1 = time.perf_counter()
            pts = sample_cloud(mesh_path, args.n_points)
            if pts.shape[0] < 4:
                raise ValueError(f"cloud too small ({pts.shape[0]} pts)")
            pcd, feats = server.compute_feats(pts)     # self-normalized
            np.savez_compressed(
                out_path,
                points=np.asarray(pcd.points, dtype=np.float32),
                feats=feats.astype(np.float32))
            per_object[nsid] = time.perf_counter() - t1
            done += 1
        except Exception as exc:
            print(f"[precompute] FAIL {nsid} ({mesh_path}): {exc}", flush=True)
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"[precompute] {i+1}/{len(manifest)} "
                  f"(done={done} skip={skipped} fail={failed})", flush=True)

    print(f"[precompute] DONE: {done} written, {skipped} skipped, "
          f"{failed} failed -> {args.out}", flush=True)
    if args.timing_json:
        with open(args.timing_json, "w") as f:
            json.dump({"model_load_s": t_model_load,
                       "per_object_s": per_object}, f, indent=1)
        print(f"[precompute] timings -> {args.timing_json}", flush=True)


if __name__ == "__main__":
    main()
