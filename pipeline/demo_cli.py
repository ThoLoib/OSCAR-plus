"""Shared command line and result reporting for the two demo entry points.

``run_pipeline.py`` (retrieval + pose) and ``run_pipeline_sim.py`` (the same,
followed by the Stage-5 grasp trial) have to expose the SAME flags and write
the SAME ``ranking.csv``.  Keeping both in this one module is what guarantees
it: the argument names, the ``--intrinsics`` parsing, the gallery lookup, the
depth-unit convention, the ``PipelineConfig`` translation and the CSV schema
exist exactly once.  The root scripts stay thin readable wrappers around
:class:`pipeline.run_pipeline.OSCARPlusPipeline`.

It lives in ``pipeline/`` because that is the only import path both root
scripts already have, and it deliberately imports nothing heavy (no torch, no
numpy, no PIL) at module level: the root scripts call ``add_demo_args`` /
``resolve_demo_args`` / ``wrap_into_container`` in their CLI prelude, so
``--help`` and the "gallery is missing" abort work on the HOST, outside the
container.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

FP_START_CMD = "docker compose up -d foundationpose"
DGEDI_START_CMD = "docker compose up -d dgedi"

# ---------------------------------------------------------------------------
# CSV schema
#
# Where every column comes from (see pipeline/run_pipeline.py -> run()):
#   rank, object_id          order/ids of the FINAL ranking — step 7 candidates
#                            when the geometric check ran, otherwise the
#                            step-6 candidates (descending fused_score)
#   mesh_path                OSCARPlusPipeline._resolve_mesh_path_for_candidate
#                            (the very mesh step 8 would pose), else the
#                            candidate's cad_model_path
#   fused_score              FusedCandidate.fused_score / GeometryCandidate.fused_score
#   clip_score               .clip_score   (S_text, min-max normalised in step 6)
#   dino_score               .dino_score   (S_view, min-max normalised in step 6)
#   shape_score              .ulip_score   (S_shape, min-max normalised in step 6)
#   dgedi_fitness            GeometryCandidate.ransac_fitness   (step 7 only)
#   trimmed_distance         GeometryCandidate.d_ransac in mm    (step 7 only)
#   foundationpose_confidence PoseEstimationResult.confidence    (rank 1 only)
#   pose_00..pose_33         PoseEstimationResult.pose_matrix, row-major
#                            (rank 1 only — the pipeline poses one candidate)
# ---------------------------------------------------------------------------

POSE_FIELDS = ["pose_%d%d" % (r, c) for r in range(4) for c in range(4)]
RANKING_FIELDS = [
    "rank", "object_id", "mesh_path", "fused_score", "clip_score", "dino_score",
    "shape_score", "dgedi_fitness", "trimmed_distance",
    "foundationpose_confidence",
] + POSE_FIELDS


# ---------------------------------------------------------------------------
# Argumente
# ---------------------------------------------------------------------------

def add_demo_args(parser) -> None:
    """Attach the demo interface — byte-identical in both root entry points."""
    grp = parser.add_argument_group("demo interface")
    grp.add_argument("--rgb", required=True,
                     help="RGB image of the scene (PNG/JPG).")
    grp.add_argument("--depth", required=True,
                     help="depth image registered to --rgb (16-bit PNG). It is "
                          "converted to metres with the BOP depth_scale when "
                          "--intrinsics is a scene_camera.json, otherwise with "
                          "PipelineConfig.depth_scale (see --depth-scale).")
    grp.add_argument("--intrinsics", default=None,
                     help="camera intrinsics, EITHER the path to a BOP "
                          "scene_camera.json (cam_K + depth_scale; the frame is "
                          "picked by the numeric stem of --rgb, else the first "
                          "entry) OR four numbers 'fx,fy,cx,cy' in pixels. "
                          "Without the flag the defaults of pipeline/config.py "
                          "are used and a warning is printed.")
    grp.add_argument("--prompt", required=True,
                     help="natural-language request, e.g. \"the red mug\".")
    grp.add_argument("--gallery", required=True,
                     help="name of a PREPROCESSED gallery. Resolved through the "
                          "path registry (config/paths.yaml): CAD models "
                          "<cad_root>/<gallery>/, rendered views "
                          "<gallery_root>/<gallery>/, captions "
                          "<cad_root>/<gallery>/descriptions_attributes.json. "
                          "Build one with: python3 preprocess_gallery.py "
                          "--dataset <gallery> --step all")
    grp.add_argument("--top-k", type=int, default=5, dest="top_k",
                     help="number of rows in the CSV = depth of the fused "
                          "ranking (PipelineConfig.fusion_top_k). The per-channel "
                          "top-k values stay at their defaults (CLIP 20, DINOv2 "
                          "5, ULIP-2 5), so a --top-k above ~20 simply yields "
                          "fewer rows. Default: 5.")
    grp.add_argument("--out", default="",
                     help="CSV to write. A path that does not end in .csv is "
                          "read as the folder to write ranking.csv into. "
                          "run_config.json and the pipeline's own artefacts "
                          "land in the same folder. Default: "
                          "<runs_root>/<demo folder>/ranking.csv.")

    adv = parser.add_argument_group(
        "advanced (defaults are the configuration reported in the thesis)")
    adv.add_argument("--geometry-reranking", action="store_true",
                     dest="geometry_reranking",
                     help="enable step 7 (dGeDi geometric check of the "
                          "shortlist). Needs the dgedi service: "
                          + DGEDI_START_CMD)
    adv.add_argument("--geometry-reranking-top-k", type=int, default=5,
                     dest="geometry_reranking_top_k",
                     help="shortlist length of step 7 (default 5).")
    adv.add_argument("--pose-method", default="foundationpose",
                     choices=["foundationpose"], dest="pose_method",
                     help="pose backend (FoundationPose is the only one; a "
                          "failed call counts as a failure).")
    adv.add_argument("--foundationpose-url", default="http://foundationpose:5050",
                     dest="foundationpose_url",
                     help="URL of the FoundationPose HTTP service.")
    adv.add_argument("--depth-scale", type=float, default=None,
                     dest="depth_scale",
                     help="override PipelineConfig.depth_scale (raw / scale = "
                          "metres). Ignored when --intrinsics is a "
                          "scene_camera.json carrying a depth_scale.")
    adv.add_argument("--no-docker", action="store_true",
                     help="do not wrap into the oscar-plus container automatically.")


# ---------------------------------------------------------------------------
# Intrinsics: scene_camera.json ODER fx,fy,cx,cy
# ---------------------------------------------------------------------------

def classify_intrinsics(spec: Optional[str], tag: str = "demo") -> Tuple[str, Any]:
    """Decide what --intrinsics is, without loading anything heavy.

    Returns ``("defaults", None)``, ``("numbers", (fx, fy, cx, cy))`` or
    ``("file", abs path)``.  Aborts when the spec is neither.
    """
    if not spec:
        return "defaults", None
    parts = [s.strip() for s in spec.replace(";", ",").split(",") if s.strip()]
    if len(parts) == 4:
        try:
            return "numbers", tuple(float(x) for x in parts)
        except ValueError:
            pass
    if not os.path.isfile(spec):
        sys.exit(f"[{tag}] --intrinsics {spec!r}: neither an existing "
                 "scene_camera.json nor four numbers 'fx,fy,cx,cy'.")
    return "file", os.path.abspath(spec)


def load_intrinsics(kind: str, payload: Any, rgb_path: str) -> Optional[dict]:
    """Turn the classified spec into the dict the pipeline expects.

    ``None`` means "no intrinsics" — every consumer then falls back to
    ``PipelineConfig.camera_fx/fy/cx/cy``.
    """
    if kind == "numbers":
        fx, fy, cx, cy = payload
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    if kind == "file":
        from .utils import load_camera_intrinsics
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        try:
            image_id = int(stem)
        except ValueError:
            image_id = 0          # non-BOP file name: take the first frame
        return load_camera_intrinsics(payload, image_id=image_id)
    return None


def intrinsics_note(kind: str, payload: Any) -> str:
    """One line describing where the intrinsics came from (printed + stored)."""
    if kind == "numbers":
        return "--intrinsics fx,fy,cx,cy = %g,%g,%g,%g" % payload
    if kind == "file":
        return f"--intrinsics {payload} (BOP scene_camera.json)"
    return ("NO --intrinsics given: falling back to the defaults of "
            "pipeline/config.py (camera_fx/fy/cx/cy and depth_scale). The point "
            "cloud and the pose are only as good as those numbers.")


# ---------------------------------------------------------------------------
# Galerie + Ausgabepfade auflösen
# ---------------------------------------------------------------------------

def resolve_demo_args(args, paths: Dict[str, str], repo_root: str,
                      default_run_dir: str, tag: str = "demo") -> Dict[str, Any]:
    """Validate the inputs and resolve every path; abort with the fix command.

    Runs in the CLI prelude on the HOST (before the container wrap), so a
    missing gallery is reported in one second instead of after the image start.
    """
    def _die(msg: str) -> None:
        sys.exit(f"[{tag}] {msg}")

    for label, path in (("--rgb", args.rgb), ("--depth", args.depth)):
        if not os.path.isfile(path):
            _die(f"{label}: {path} does not exist.")

    gallery = args.gallery
    cad_models = os.path.join(paths["cad_root"], gallery)
    reference_images = os.path.join(paths["gallery_root"], gallery)
    descriptions = os.path.join(cad_models, "descriptions_attributes.json")
    build_cmd = (f"python3 preprocess_gallery.py --dataset {gallery} --step all")
    for what, path in (("CAD models", cad_models),
                       ("rendered views", reference_images),
                       ("captions", descriptions)):
        if not os.path.exists(path):
            _die(f"gallery {gallery!r}: {what} missing at {path}\n"
                 f"       Prepare the gallery first:\n"
                 f"         {build_cmd}")

    if args.top_k < 1:
        _die("--top-k must be >= 1.")

    kind, payload = classify_intrinsics(args.intrinsics, tag=tag)

    out = args.out or os.path.join(paths["runs_root"], default_run_dir,
                                   "ranking.csv")
    if not os.path.isabs(out):
        out = os.path.join(repo_root, out)
    # A path that does not name a .csv file is read as the FOLDER to write into.
    if not out.lower().endswith(".csv"):
        out = os.path.join(out, "ranking.csv")
    out_dir = os.path.dirname(out) or repo_root

    return {
        "gallery": gallery,
        "cad_models": cad_models,
        "reference_images": reference_images,
        "descriptions": descriptions,
        "rgb": os.path.abspath(args.rgb),
        "depth": os.path.abspath(args.depth),
        "out_csv": out,
        "out_dir": out_dir,
        "intrinsics_kind": kind,
        "intrinsics_value": list(payload) if kind == "numbers" else payload,
        "intrinsics_note": intrinsics_note(kind, payload),
        "runs_root": paths["runs_root"],
    }


# ---------------------------------------------------------------------------
# Container + Dienste
# ---------------------------------------------------------------------------

def wrap_into_container(args, repo_root: str, script: str,
                        tag: str = "demo") -> None:
    """Re-exec inside the oscar-plus container unless we are already in one.

    Same mechanism as experiments/stage3_bop.py and stage5_grasping.py:
    ``docker compose run --rm oscar-plus python3 <script> <original argv>``.
    ``--no-docker`` switches it off (for an environment that already has
    torch/open3d/PyBullet).
    """
    if os.path.exists("/.dockerenv") or getattr(args, "no_docker", False):
        return
    if not os.path.isfile(os.path.join(repo_root, "docker-compose.yml")):
        sys.exit(f"[{tag}] {repo_root}/docker-compose.yml is missing — "
                 "incomplete repo?")
    print(f"[{tag}] the run needs the container environment (torch, open3d) — "
          "wrapping into the oscar-plus container automatically.", flush=True)
    os.chdir(repo_root)               # docker compose needs the repo as CWD
    os.execvp("docker", ["docker", "compose", "run", "--rm", "oscar-plus",
                         "python3", script] + sys.argv[1:])


def check_foundationpose(url: str, tag: str = "demo") -> None:
    """Fail fast: without the service step 8 silently returns an identity pose.

    Both names are tried — the compose service name inside the container and
    localhost from the host.
    """
    import urllib.request
    base = url.rstrip("/")
    for candidate in (base + "/health", "http://localhost:5050/health"):
        try:
            urllib.request.urlopen(candidate, timeout=5).read()
        except Exception:                                          # noqa: BLE001
            continue
        print(f"[{tag}] FoundationPose ok ({candidate})", flush=True)
        return
    sys.exit(f"[{tag}] FoundationPose service unreachable ({base}). Start it on "
             f"the HOST:  {FP_START_CMD}   — then run again.")


# ---------------------------------------------------------------------------
# Pipeline-Konfiguration + Eingaben (identisch in beiden Einstiegen)
# ---------------------------------------------------------------------------

def build_pipeline_config(args, resolved: Dict[str, Any],
                          paths: Dict[str, str]):
    """Translate the demo flags onto ``PipelineConfig``.

    ``--top-k`` becomes ``fusion_top_k`` — the OUTPUT depth of step 6 and
    therefore of the CSV.  Everything else keeps the class defaults, in
    particular the per-channel top-k values (clip_top_k=20, dino_top_k=5,
    ulip2_top_k=5) and the fusion weights.
    """
    from .config import PipelineConfig

    geo = bool(getattr(args, "geometry_reranking", False))
    geo_k = int(getattr(args, "geometry_reranking_top_k", 5))
    config = PipelineConfig(
        description_file=resolved["descriptions"],
        reference_images_dir=resolved["reference_images"],
        cad_models_dir=resolved["cad_models"],
        output_dir=resolved["out_dir"],
        pose_method=args.pose_method,
        foundationpose_url=args.foundationpose_url,
        geometry_reranking_enabled=geo,
        geometry_reranking_top_k=geo_k,
        # Fused ranking has to be at least as deep as the CSV and, with the
        # geometric check on, at least as deep as its shortlist.
        fusion_top_k=max(int(args.top_k), geo_k if geo else 1),
    )
    if getattr(args, "depth_scale", None):
        config.depth_scale = float(args.depth_scale)
    # ULIP-2 defaults of the container image.
    if os.path.isdir("/ulip"):
        config.ulip_repo_path = config.ulip_repo_path or "/ulip"
        config.ulip2_checkpoint = (config.ulip2_checkpoint
                                   or paths["checkpoint_ulip2"])
    return config


def load_inputs(args, resolved: Dict[str, Any], config):
    """RGB (PIL), depth in METRES, intrinsics dict — one convention, both entries.

    Depth: the BOP ``depth_scale`` of a scene_camera.json wins
    (raw * depth_scale / 1000 = m), otherwise ``config.depth_scale``
    (raw / depth_scale = m) — exactly as pipeline/run_pipeline.py does it.
    """
    import numpy as np
    from PIL import Image

    rgb_image = Image.open(args.rgb).convert("RGB")
    intrinsics = load_intrinsics(resolved["intrinsics_kind"],
                                 resolved["intrinsics_value"], args.rgb)
    depth = np.array(Image.open(args.depth)).astype(np.float32)
    if intrinsics and intrinsics.get("depth_scale", 0) > 0:
        scale = float(intrinsics["depth_scale"])
        depth = depth * scale / 1000.0
        note = f"BOP depth_scale={scale:g} -> raw * {scale:g} / 1000 = metres"
    else:
        depth = depth / config.depth_scale
        note = (f"config depth_scale={config.depth_scale:g} -> raw / "
                f"{config.depth_scale:g} = metres")
    return rgb_image, depth, intrinsics, note


# ---------------------------------------------------------------------------
# ranking.csv
# ---------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 6) -> str:
    """Number formatter: non-numeric / None / inf / NaN become an empty cell."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def final_candidates(result: dict) -> Tuple[List[Any], str]:
    """The pipeline's FINAL ranking and the step it came from.

    Step 7 reorders the head of the fused list, so its candidate list — not
    the fused one — is what rank 1 refers to and what step 8 posed.  Without
    the geometric check the fused list is already the final one (descending
    ``fused_score``).
    """
    geo = result.get("geometry_reranking")
    candidates = list(getattr(geo, "candidates", None) or []) if geo else []
    if candidates:
        return candidates, "step7_geometry_reranking"
    fusion = result.get("fusion")
    return list(getattr(fusion, "candidates", None) or []), "step6_fusion"


def posed_object_id(result: dict) -> Optional[str]:
    """Object step 8 estimated the pose for (mirrors run() exactly)."""
    geo = result.get("geometry_reranking")
    best_geo = getattr(geo, "best_candidate", None) if geo else None
    if best_geo is not None:
        return best_geo.object_id
    fusion = result.get("fusion")
    best = getattr(fusion, "best_match", None) if fusion else None
    return getattr(best, "object_id", None)


def build_ranking_rows(result: dict, top_k: int, mesh_resolver=None
                       ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build the ``ranking.csv`` rows; returns ``(rows, notes)``.

    ``notes`` lists every column that stayed empty and why — the caller prints
    them, so an absent value is never silently mistaken for a zero.
    ``mesh_resolver`` is ``OSCARPlusPipeline._resolve_mesh_path_for_candidate``.
    """
    candidates, order_source = final_candidates(result)
    notes: List[str] = []
    rows: List[Dict[str, Any]] = []

    pose = result.get("pose_estimation")
    pose_failed = bool(pose is not None
                       and str(getattr(pose, "method", "")).endswith("_failed"))
    target_id = posed_object_id(result)

    n_geo = n_mesh_missing = 0
    pose_written = False

    for index, cand in enumerate(candidates[:max(1, int(top_k))]):
        mesh_path = ""
        if mesh_resolver is not None:
            mesh_path = mesh_resolver(cand) or ""
        if not mesh_path:
            mesh_path = getattr(cand, "cad_model_path", "") or ""
        if not mesh_path:
            n_mesh_missing += 1

        row = {name: "" for name in RANKING_FIELDS}
        row.update({
            "rank": index + 1,
            "object_id": getattr(cand, "object_id", ""),
            "mesh_path": mesh_path,
            "fused_score": _fmt(getattr(cand, "fused_score", None)),
            "clip_score": _fmt(getattr(cand, "clip_score", None)),
            "dino_score": _fmt(getattr(cand, "dino_score", None)),
            # Step 6 calls the shape channel ulip_score; the CSV calls it
            # shape_score because the encoder is swappable (ULIP-2 / Uni3D).
            "shape_score": _fmt(getattr(cand, "ulip_score", None)),
        })

        # --- Schritt 7: nur die Kandidaten mit gelungener Registrierung ---
        distance = getattr(cand, "d_ransac", None)
        if distance is None:
            distance = getattr(cand, "chamfer_score", None)
        if distance is not None and math.isfinite(float(distance)):
            row["trimmed_distance"] = _fmt(distance)
            row["dgedi_fitness"] = _fmt(getattr(cand, "ransac_fitness", None))
            n_geo += 1

        # --- Schritt 8: nur der Rang-1-Kandidat wird gestellt ---
        if (pose is not None and not pose_failed and not pose_written
                and row["object_id"] == target_id):
            row["foundationpose_confidence"] = _fmt(
                getattr(pose, "confidence", None))
            matrix = getattr(pose, "pose_matrix", None)
            if matrix is not None:
                for r in range(4):
                    for c in range(4):
                        row["pose_%d%d" % (r, c)] = _fmt(matrix[r][c])
                pose_written = True

        rows.append(row)

    # ---------------- Notizen zu leer gebliebenen Spalten ----------------
    if not rows:
        notes.append("no candidates at all — the pipeline produced neither a "
                     "step-6 nor a step-7 ranking.")
    if len(candidates) < int(top_k):
        notes.append(f"--top-k {int(top_k)} but only {len(candidates)} "
                     f"candidate(s) in the {order_source} ranking: the "
                     "per-channel top-k defaults (CLIP 20 / DINOv2 5 / ULIP-2 "
                     "5) bound the fused list.")
    if n_geo == 0:
        notes.append("dgedi_fitness + trimmed_distance are EMPTY: the "
                     "geometric check (step 7) did not run — pass "
                     "--geometry-reranking (needs " + DGEDI_START_CMD + ").")
    elif n_geo < len(rows):
        notes.append(f"dgedi_fitness + trimmed_distance filled for {n_geo} of "
                     f"{len(rows)} rows: ranks beyond the step-7 shortlist and "
                     "failed registrations have no geometry values.")
    if pose is None:
        notes.append("foundationpose_confidence + pose_00..pose_33 are EMPTY: "
                     "step 8 did not run (no best match, or the step was "
                     "skipped).")
    elif pose_failed:
        notes.append("foundationpose_confidence + pose_00..pose_33 are EMPTY: "
                     f"step 8 failed (method={getattr(pose, 'method', '?')}, "
                     "identity pose with confidence 0) — start the service: "
                     + FP_START_CMD)
    elif not pose_written and rows:
        notes.append("pose columns are EMPTY: the posed object "
                     f"{target_id!r} is not among the {len(rows)} CSV rows.")
    if n_mesh_missing:
        notes.append(f"mesh_path is empty for {n_mesh_missing} row(s): no mesh "
                     "found below the gallery's CAD folder for those ids.")
    return rows, notes


def write_ranking_csv(result: dict, out_csv: str, top_k: int, mesh_resolver=None
                      ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Write ``ranking.csv`` (schema: :data:`RANKING_FIELDS`)."""
    rows, notes = build_ranking_rows(result, top_k, mesh_resolver=mesh_resolver)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, notes


# ---------------------------------------------------------------------------
# Provenienz + Kurzfassung
# ---------------------------------------------------------------------------

def write_run_config(out_dir: str, argv: Sequence[str], resolved: Dict[str, Any],
                     repo_root: str, extra: Optional[dict] = None) -> str:
    """Write ``run_config.json`` next to the CSV (argv, git rev, ISO time, paths)."""
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                  capture_output=True, text=True).stdout.strip()
    except Exception:                                              # noqa: BLE001
        revision = ""
    payload = {
        "argv": list(argv),
        "git": revision,
        "time": datetime.now().isoformat(timespec="seconds"),
        "in_container": os.path.exists("/.dockerenv"),
        "resolved": resolved,
    }
    if extra:
        payload.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "run_config.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, default=str)
    return path


def print_demo_summary(result: dict, rows: Sequence[dict], out_csv: str,
                       notes: Sequence[str] = (),
                       extra: Sequence[Tuple[str, Any]] = ()) -> None:
    """Print the short version: rank 1, its scores, the pose, the runtime."""
    summary = result.get("summary", {}) or {}
    timing = (result.get("timing", {}) or {}).get("total")
    top = rows[0] if rows else {}
    print("\n" + "=" * 64)
    print("RESULT")
    print("=" * 64)
    print(f"  rank 1 object_id     {top.get('object_id', '-')}")
    print(f"  rank 1 fused_score   {top.get('fused_score', '-')}")
    print(f"  rank 1 mesh_path     {top.get('mesh_path', '-') or '-'}")
    pose_conf = top.get("foundationpose_confidence", "")
    method = summary.get("pose_method", "")
    print(f"  pose confidence      {pose_conf or '-'}"
          + (f"  ({method})" if method else ""))
    for label, value in extra:
        print(f"  {label:<20} {value}")
    if timing is not None:
        print(f"  runtime              {float(timing):.2f} s")
    print(f"  ranking.csv          {out_csv}  ({len(rows)} rows)")
    print("=" * 64)
    for note in notes:
        print(f"  note: {note}")
    if notes:
        print("=" * 64)
