#!/usr/bin/env python3
"""
run_pipeline.py — the demo entry point of OSCAR+: one RGB-D frame plus a
natural-language prompt in, a ranked list of CAD models out (``ranking.csv``),
with the 6D pose of the rank-1 model.

    python3 run_pipeline.py --rgb scene/rgb/000001.png \\
                            --depth scene/depth/000001.png \\
                            --intrinsics scene/scene_camera.json \\
                            --prompt "the red mug" \\
                            --gallery ycbv --top-k 5 --out ranking.csv

This is a thin wrapper: the eight steps themselves live in
``pipeline/run_pipeline.py`` (GroundingDINO+SAM, point cloud, CLIP, DINOv2,
ULIP-2, fusion, geometric check, FoundationPose) and are executed here through
``OSCARPlusPipeline`` — no logic is reimplemented.  What this file adds is the
readable interface, the gallery lookup and the CSV export.

The gallery
-----------
``--gallery <name>`` is resolved through the path registry
(``config/paths.yaml``): CAD models ``<cad_root>/<name>/``, rendered views
``<gallery_root>/<name>/``, captions
``<cad_root>/<name>/descriptions_attributes.json``.  If one of them is missing
the run aborts and prints the command that builds it:

    python3 preprocess_gallery.py --dataset <name> --step all

What ranking.csv contains
-------------------------
One row per candidate, ``--top-k`` rows, best first:

    rank,object_id,mesh_path,fused_score,clip_score,dino_score,shape_score,
    dgedi_fitness,trimmed_distance,foundationpose_confidence,pose_00..pose_33

``pose_00..pose_33`` is the 4x4 camera-from-object matrix written out row by
row.  Pose and confidence exist for the rank-1 candidate only — the pipeline
estimates a pose for that one model.  The two geometry columns are filled only
with ``--geometry-reranking``.  Every column that stays empty is reported as a
``note:`` line at the end of the run, so an absent value is never mistaken for
a zero.  ``run_config.json`` (argv, git revision, ISO time, resolved paths)
is written next to the CSV.

``--top-k`` sets the depth of the CSV, i.e. the output depth of the fusion
(``PipelineConfig.fusion_top_k``).  The per-channel top-k values keep their
defaults (CLIP 20, DINOv2 5, ULIP-2 5); a ``--top-k`` above that simply yields
fewer rows.

Services and environment
------------------------
Outside a container the script wraps itself into ``docker compose run --rm
oscar-plus python3 /app/run_pipeline.py ...`` (``--no-docker`` switches that off).
FoundationPose has to be running on the HOST:

    docker compose up -d foundationpose

With ``--geometry-reranking`` the dGeDi service is needed as well
(``docker compose up -d dgedi``).

The grasp trial of Stage 5 on the retrieved model is one file over:
``run_pipeline_sim.py`` — same arguments, same ranking.csv.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# CLI prelude — BEFORE the heavy imports (torch, open3d, PIL), so that
# `--help` and the "gallery is missing" abort work on the host without the
# container dependencies.
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO, os.path.join(_REPO, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TAG = "demo"
DEFAULT_RUN_DIR = "demo"          # <runs_root>/demo/ranking.csv

_ARGS = None
_PATHS = None
_RESOLVED = None

if __name__ == "__main__":
    import argparse

    from pipeline import demo_cli
    from pipeline import paths as _paths_mod

    _ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    demo_cli.add_demo_args(_ap)
    _paths_mod.add_path_args(_ap)
    _ARGS = _ap.parse_args()
    _PATHS = _paths_mod.from_args(_ARGS)
    _RESOLVED = demo_cli.resolve_demo_args(
        _ARGS, _PATHS, repo_root=_REPO, default_run_dir=DEFAULT_RUN_DIR, tag=TAG)
    demo_cli.wrap_into_container(_ARGS, repo_root=_REPO,
                                 script="/app/run_pipeline.py", tag=TAG)


# ---------------------------------------------------------------------------
def main(args, paths, resolved) -> int:
    """Run the eight steps on one frame and write ranking.csv + run_config.json."""
    from pipeline import demo_cli

    def log(message: str) -> None:
        print(f"[{TAG}] {message}", flush=True)

    os.makedirs(resolved["out_dir"], exist_ok=True)
    demo_cli.write_run_config(resolved["out_dir"], sys.argv[1:], resolved,
                              repo_root=_REPO,
                              extra={"top_k": args.top_k,
                                     "geometry_reranking": args.geometry_reranking,
                                     "pose_method": args.pose_method})

    log(f"prompt: {args.prompt!r}  |  gallery: {resolved['gallery']}  "
        f"|  top-k: {args.top_k}")
    log(resolved["intrinsics_note"])
    if args.pose_method == "foundationpose":
        demo_cli.check_foundationpose(args.foundationpose_url, tag=TAG)

    # Ab hier die schweren Importe (torch, open3d) — erst nach der Provenienz
    # und dem Dienst-Check, damit ein Abbruch eine Spur hinterlaesst.
    from pipeline.run_pipeline import OSCARPlusPipeline

    config = demo_cli.build_pipeline_config(args, resolved, paths)
    rgb_image, depth_image, intrinsics, depth_note = demo_cli.load_inputs(
        args, resolved, config)
    log(f"depth: {depth_note}")

    pipeline = OSCARPlusPipeline(config)
    pipeline.initialize()
    result = pipeline.run(rgb_image=rgb_image, depth_image=depth_image,
                          prompt=args.prompt, camera_intrinsics=intrinsics)

    if result.get("error"):
        log(f"the pipeline stopped: {result['error']} — no ranking written.")
        return 2

    rows, notes = demo_cli.write_ranking_csv(
        result, resolved["out_csv"], args.top_k,
        mesh_resolver=pipeline._resolve_mesh_path_for_candidate)
    demo_cli.print_demo_summary(result, rows, resolved["out_csv"], notes)
    return 0


if __name__ == "__main__":
    sys.exit(main(_ARGS, _PATHS, _RESOLVED))
