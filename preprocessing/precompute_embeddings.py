#!/usr/bin/env python3
# =============================================================================
# preprocessing/precompute_embeddings.py — build gallery embedding caches for ANY
# dataset, by hand, no queue/systemd machinery required.
# =============================================================================
#
# WHAT THIS DOES
# ---------------
# Given a rendered + described dataset (see preprocess_gallery.py for that
# earlier stage), this builds every embedding "channel" the retrieval
# pipeline needs and caches each one to disk:
#
#   pass          builds                                    cache file
#   ----          ------                                    ----------
#   base          CLIP-text (descriptions)                  object_database/<ds>/.clip_text_cache_*.pt
#                 DINOv2 (rendered views)                   object_images/<ds>/.dino_cache_*.pt
#                 ULIP-2 colored, partial-view point clouds object_images/<ds>/.ulip_partial_cache_*.pt
#   siglip        SigLIP (rendered views)                   object_images/<ds>/.siglip_cache_*.pt
#   ulip_fullmesh ULIP-2 colored, full CAD mesh              <data_root>/cad/.ulip_cache_*.pt
#   ulip_pc_rgb   ULIP-2 colored, partial-view (pc-mode tag) same cache as `base` — near-instant, no rebuild
#   ulip_pc_xyz   ULIP-2 XYZ-only (8k pts), partial-view     object_images/<ds>/.ulip_partial_cache_*.pt (different digest)
#   uni3d         Uni3D-g, partial-view                      object_images/<ds>/.ulip_partial_cache_*.pt (different digest)
#
# All caches are content-fingerprinted (model config + source data, never
# absolute paths or mtimes), so they can be copied to another machine and
# stay valid there.
#
# Every pass is resumable: rerun the same command and finished passes are
# skipped (cache hit), only missing/changed ones recompute.
#
# WHY "shape" ≠ "images": the `shape` channel embeds the object's 3D point
# cloud (geometry only) via ULIP-2/Uni3D's point-cloud encoder. `clip` and
# `dino`/`siglip` embed the *rendered images* of the object. All three live
# in the same joint ULIP-2 space, but only `shape` ever looks at geometry.
#
# USAGE
# -----
#   Normally you do not call this directly — use
#       python3 preprocess_gallery.py --dataset MI3DOR --step embed --passes base
#   which resolves every location from config/paths.yaml. Called by hand:
#
#   python3 preprocessing/precompute_embeddings.py \
#       --dataset MI3DOR \
#       --data-root <cad_root>/MI3DOR/model/test \
#       --images-dir <gallery_root>/MI3DOR \
#       --desc-file <cad_root>/MI3DOR/descriptions_attributes.json \
#       --results-root <caches_root>/precompute_MI3DOR
#
#   # Only rebuild specific passes (comma-separated, see --list):
#   python3 preprocessing/precompute_embeddings.py --dataset MI3DOR ... --passes uni3d,siglip
#
#   # See what a dataset already has without building anything:
#   python3 preprocessing/precompute_embeddings.py --dataset MI3DOR ... --dry-run
#
# Run this from the repo root, inside the `oscar` container (it needs the
# same torch/CLIP/ULIP/Uni3D stack the rest of the pipeline uses):
#   docker compose run --rm oscar-plus bash -lc \
#       "python3 preprocessing/precompute_embeddings.py --dataset MI3DOR ..."
#
# Requires, before running:
#   - Renders + partial point clouds:  preprocess_gallery.py --dataset <ds> --step render,partial
#   - Descriptions (VLM captions):     preprocessing/generate_descriptions.py
#   - <data_root>/cad/*.obj            the CAD meshes for this dataset
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. The six embedding passes.
#
# Each entry says which channel(s) to build and which encoder config to use.
# This list is intentionally dataset-agnostic — dataset choice is entirely a
# matter of the --data-root/--images-dir/--desc-file paths below, nothing
# here changes per dataset.
# ---------------------------------------------------------------------------

ULIP_CKPT_DEFAULT = "/ulip/checkpoints/ulip2_pointbert_10k.pt"
# XYZ-only arm needs the released ULIP-2 *xyz* PointBERT (8192 pts, no RGB,
# input_dim=3, SLIP ViT-B tower / 512-d) — the colored 10k checkpoint has a
# 6-channel input conv and cannot encode xyz-only point clouds. The basename
# is part of the cache fingerprint (step5's _get_partial_cache_path), so any
# machine reusing these caches must stage the same file at the same name.
ULIP_CKPT_XYZ = "/ulip/checkpoints/ulip2_pointbert_8k_xyz.pt"

PASS_DEFS: "OrderedDict[str, dict]" = OrderedDict([
    ("base", dict(
        description="CLIP-text + DINOv2 + ULIP-2 colored partial-view shape",
        channels=("clip", "dino", "shape"), ulip2_mode="cross",
        partial=True, overrides={})),
    ("siglip", dict(
        description="SigLIP image embeddings (replaces DINOv2 for O3)",
        channels=("dino",), ulip2_mode="cross", partial=True, no_shape=True,
        overrides={"appearance_encoder": "siglip"})),
    ("ulip_fullmesh", dict(
        description="ULIP-2 colored, full CAD mesh (no partial views)",
        channels=("shape",), ulip2_mode="cross", partial=False,
        overrides={})),
    ("ulip_pc_rgb", dict(
        description="ULIP-2 colored, partial-view, PC-mode query tag "
                     "(reuses the `base` cache — same config, no rebuild)",
        channels=("shape",), ulip2_mode="pc", partial=True, overrides={})),
    ("ulip_pc_xyz", dict(
        description="ULIP-2 XYZ-only (8k pts, no color), partial-view (O5)",
        channels=("shape",), ulip2_mode="pc", partial=True,
        overrides={"ulip2_use_colors": False,
                   "ulip2_backbone": "pointbert",
                   "ulip2_checkpoint": ULIP_CKPT_XYZ,
                   "ulip2_num_points": 8192,
                   "ulip2_embed_dim": 512})),  # SLIP ViT-B tower -> 512-d
    ("uni3d", dict(
        description="Uni3D-g, partial-view point clouds (E7)",
        channels=("shape",), ulip2_mode="pc", partial=True,
        overrides={"shape_encoder": "uni3d"})),
])


# ---------------------------------------------------------------------------
# 2. Path / input validation
# ---------------------------------------------------------------------------

def resolve_paths(args: argparse.Namespace) -> Dict[str, str]:
    # mesh_glob defaults to the shrec18-style <data_root>/cad/*.obj layout,
    # but any dataset with a different CAD layout (MI3DOR's nested
    # model/test/*/*.obj, housecat6d's */*.obj, BOP's prepared object_database)
    # passes its own glob via --mesh-glob. Only the ulip_fullmesh pass reads
    # meshes at all; the other five read renders/partial-PCs/descriptions.
    mesh_glob = args.mesh_glob or os.path.join(args.data_root or "", "cad", "*.obj")
    return {
        "data_root": args.data_root or "",
        "images_dir": args.images_dir,
        "desc_file": args.desc_file,
        "results_root": args.results_root,
        "mesh_glob": mesh_glob,
        "mesh_id_mode": args.mesh_id_mode,
    }


def build_mesh_items(mesh_glob: str, mode: str):
    """(obj_id, path) pairs for the ulip_fullmesh pass, or None for the default.

    eval_common.build_pipeline derives the full-mesh obj_id from the FILENAME
    STEM. That is right for flat layouts (MI3DOR's `<id>.obj`, BOP's
    `obj_000001.ply`) but silently wrong for nested ones where every object's
    mesh has the SAME name: GSO's `<id>/meshes/model.obj` and YCB-V's
    `<id>/textured_simple.obj` would map all 1030 / 21 objects onto the single
    id "model" / "textured_simple", keeping one arbitrary mesh and discarding
    the rest. build_pipeline takes an explicit `cad_mesh_items` for exactly this
    case, so pick the id off the right path component instead of copying meshes
    into a flat staging directory.

      stem        <id>.obj                 -> "<id>"   (default; MI3DOR, BOP)
      parent      <id>/textured_simple.obj -> "<id>"   (YCB-V)
      grandparent <id>/meshes/model.obj    -> "<id>"   (GSO)
    """
    import glob as _glob

    if mode == "stem":
        return None                      # let build_pipeline do its default
    paths = sorted(_glob.glob(mesh_glob))
    if not paths:
        return None
    if mode == "parent":
        pick = lambda p: os.path.basename(os.path.dirname(p))
    elif mode == "grandparent":
        pick = lambda p: os.path.basename(os.path.dirname(os.path.dirname(p)))
    else:
        raise ValueError(f"unknown --mesh-id-mode {mode!r}")

    items = [(pick(p), p) for p in paths]
    ids = [i for i, _ in items]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise SystemExit(
            f"[mesh-items] --mesh-id-mode {mode} still yields duplicate object "
            f"ids (e.g. {dupes}) for glob {mesh_glob!r}. Fix the mode or the "
            f"glob — duplicates silently drop meshes from the fullmesh cache.")
    return items


def validate_inputs(paths: Dict[str, str]) -> List[str]:
    """Confirm the gallery is ready and return the object-id list to build.

    The authoritative gallery = objects that have BOTH renders and a
    description (that's what every pass needs). CAD meshes are only consumed
    by the ulip_fullmesh pass, so a missing/partial mesh set is a warning,
    not a hard error. Raw query/GT folders are never needed here (they matter
    for evaluation, not for building gallery embeddings).
    """
    import glob as _glob

    imgs_dir = paths["images_dir"]
    rendered = set()
    if os.path.isdir(imgs_dir):
        for d in os.listdir(imgs_dir):
            full = os.path.join(imgs_dir, d)
            if os.path.isdir(full) and any(
                    f.endswith(".png") and not f.endswith("_bg.png")
                    for f in os.listdir(full)):
                rendered.add(d)

    desc_ids = set()
    if os.path.isfile(paths["desc_file"]):
        with open(paths["desc_file"]) as f:
            desc_ids = set(json.load(f).keys())

    object_ids = sorted(rendered & desc_ids)
    n_meshes = len(_glob.glob(paths["mesh_glob"]))
    print(f"[validate] rendered: {len(rendered)}  described: {len(desc_ids)}  "
          f"meshes(glob): {n_meshes}  -> gallery: {len(object_ids)}")

    if not object_ids:
        raise SystemExit(
            "[validate] empty gallery (no object has BOTH renders AND a "
            "description). Run, in order:\n"
            "  1) python3 preprocess_gallery.py --dataset <name> "
            "--step render   (then --step partial)\n"
            "  2) python3 preprocess_gallery.py --dataset <name> "
            "--step describe")

    with_partial = sum(
        1 for d in object_ids
        if any(f.endswith("_partial.npz")
               for f in os.listdir(os.path.join(imgs_dir, d))))
    if with_partial < len(object_ids):
        print(f"[validate] WARNING: only {with_partial}/{len(object_ids)} "
              f"objects have *_partial.npz point clouds — the ULIP/Uni3D "
              f"partial-view passes need them; objects without fall back to "
              f"full-mesh shape encoding.")

    if n_meshes == 0:
        print(f"[validate] WARNING: mesh glob matched 0 files "
              f"({paths['mesh_glob']}) — the ulip_fullmesh pass will be "
              f"empty. The other five passes (renders/partial-PCs/text) are "
              f"unaffected.")

    return object_ids


# ---------------------------------------------------------------------------
# 3. Running one pass (build-only: writes gallery caches, scores nothing)
# ---------------------------------------------------------------------------

def _git_commit(root: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _pass_provenance(pass_key: str, pipe_cfg, need, root: str) -> dict:
    """Encoder identity for this pass, so another machine can confirm its
    shipped caches were built with a matching encoder (same weights+code)."""
    return {
        "pass": pass_key,
        "channels": sorted(need),
        "appearance_encoder": pipe_cfg.appearance_encoder,
        "shape_encoder": pipe_cfg.shape_encoder,
        "dino_model": pipe_cfg.dino_model_name,
        "siglip_model": pipe_cfg.siglip_model_name,
        "ulip2_checkpoint": os.path.basename(pipe_cfg.ulip2_checkpoint or ""),
        "ulip2_backbone": pipe_cfg.ulip2_backbone,
        "ulip2_use_colors": bool(pipe_cfg.ulip2_use_colors),
        "ulip2_mode": pipe_cfg.ulip2_mode,
        "ulip2_num_points": pipe_cfg.ulip2_num_points,
        "ulip2_embed_dim": pipe_cfg.ulip2_embed_dim,
        "code_commit": _git_commit(root),
    }


def run_pass(pass_key: str, paths: Dict[str, str], root: str) -> dict:
    """Build (or load-from-cache) the gallery reference caches for one pass.

    Delegates the actual encoding to evaluation/eval_common.py's
    build_pipeline(), the same function the rest of the pipeline uses — this
    script only supplies dataset paths and pass configuration, it does not
    reimplement any encoder logic.
    """
    import evaluation.eval_common as ec

    pdef = PASS_DEFS[pass_key]
    need = set(pdef["channels"])
    tqdm.write(f"[pass:{pass_key}] {pdef['description']}")
    tqdm.write(f"[pass:{pass_key}] channels={sorted(need)}")

    cfg = ec.EvalConfig(
        ref_dir=paths["images_dir"],
        desc_file=paths["desc_file"],
        cad_mesh_glob=("" if pdef.get("no_shape") else paths["mesh_glob"]),
        result_folder=os.path.join(paths["results_root"], "_cache"),
        clip_top_k=10 ** 6, dino_top_k=10 ** 6, fusion_top_k=10 ** 6,
        weight_clip=0.3, weight_dino=0.4, weight_ulip=0.3,
        ulip2_mode=pdef["ulip2_mode"],
        ulip2_use_partial_views=pdef["partial"],
        ulip2_checkpoint=ULIP_CKPT_DEFAULT,
        # Encode/keep ALL views here; any view-count trimming happens at
        # retrieval time, not during precompute.
        pipeline_overrides={"num_views": None, **pdef["overrides"]},
    )
    if "shape" not in need:
        cfg.cad_mesh_glob = ""

    mesh_items = (None if "shape" not in need or not cfg.cad_mesh_glob
                  else build_mesh_items(paths["mesh_glob"],
                                        paths.get("mesh_id_mode", "stem")))
    pipe_cfg, _clip, _dino, _fusion, shape_m = ec.build_pipeline(
        cfg, cad_mesh_items=mesh_items)
    if "shape" in need and shape_m is None:
        raise RuntimeError(
            f"[pass:{pass_key}] shape encoder failed to load "
            f"(checkpoint/repo missing?) — cannot build the shape channel.")

    prov = _pass_provenance(pass_key, pipe_cfg, need, root)

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tqdm.write(f"[pass:{pass_key}] done.")
    return prov


# ---------------------------------------------------------------------------
# 4. Driver
# ---------------------------------------------------------------------------

def precompute(paths: Dict[str, str], passes: List[str], root: str) -> None:
    manifest = {"code_commit": _git_commit(root), "passes": []}
    bar = tqdm(passes, desc="precompute", unit="pass")
    for pkey in bar:
        bar.set_postfix_str(pkey)
        try:
            prov = run_pass(pkey, paths, root)
            manifest["passes"].append(prov)
        except Exception as exc:
            tqdm.write(f"[pass:{pkey}] FAILED: {exc}")
            manifest["passes"].append({"pass": pkey, "error": str(exc)})

    out = os.path.join(paths["images_dir"], "precompute_manifest.json")
    os.makedirs(paths["images_dir"], exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    n_ok = sum(1 for p in manifest["passes"] if "error" not in p)
    print(f"\n[precompute] {n_ok}/{len(passes)} passes built.")
    print(f"[precompute] manifest -> {out}")
    if n_ok < len(passes):
        failed = [p["pass"] for p in manifest["passes"] if "error" in p]
        print(f"[precompute] FAILED passes: {failed} — see messages above.")
        sys.exit(1)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="Dataset name, used only for log "
                                       "messages (paths are set explicitly "
                                       "below).")
    ap.add_argument("--data-root", help="Dataset root. Only used to default "
                                         "--mesh-glob to <data_root>/cad/*.obj "
                                         "when --mesh-glob is not given.")
    ap.add_argument("--mesh-id-mode", default="stem",
                    choices=("stem", "parent", "grandparent"),
                    help="Which path component names the object for the "
                         "ulip_fullmesh pass. 'stem' (default) = filename, for "
                         "flat layouts (MI3DOR, BOP obj_000001.ply). 'parent' "
                         "for YCB-V's <id>/textured_simple.obj. 'grandparent' "
                         "for GSO's <id>/meshes/model.obj. Using 'stem' on a "
                         "nested layout maps every object to the same id.")
    ap.add_argument("--mesh-glob", default="",
                     help="Glob for this dataset's CAD meshes (e.g. "
                          "'object_database/MI3DOR/model/test/*/*.obj'). Only "
                          "the ulip_fullmesh pass reads these. Defaults to "
                          "<data_root>/cad/*.obj (shrec18 layout).")
    ap.add_argument("--images-dir", help="object_images/<dataset> — "
                                          "renders + partial point clouds.")
    ap.add_argument("--desc-file", help="object_database/<dataset>/"
                                         "descriptions_attributes.json")
    ap.add_argument("--results-root",
                     help="Where to put the small _cache/ bookkeeping dir "
                          "for this run (not the embedding caches "
                          "themselves — those live next to the source "
                          "data, see the module docstring).")
    ap.add_argument("--passes", default="all",
                     help="Comma-separated subset of passes to (re)build, "
                          "or 'all' (default). See --list for names.")
    ap.add_argument("--list", action="store_true",
                     help="Print the available passes and exit (no paths "
                          "needed).")
    ap.add_argument("--dry-run", action="store_true",
                     help="Only run --validate and print the gallery size, "
                          "build nothing.")
    args = ap.parse_args(argv)

    if args.list:
        for key, pdef in PASS_DEFS.items():
            print(f"  {key:15s} {pdef['description']}")
        return

    missing = [f"--{name.replace('_', '-')}" for name in
               ("dataset", "images_dir", "desc_file", "results_root")
               if getattr(args, name) is None]
    if missing:
        ap.error(f"the following arguments are required: {', '.join(missing)}")
    if not args.data_root and not args.mesh_glob:
        ap.error("provide --mesh-glob (or --data-root to default it to "
                 "<data_root>/cad/*.obj)")

    passes = (list(PASS_DEFS) if args.passes == "all"
              else [p.strip() for p in args.passes.split(",")])
    unknown = [p for p in passes if p not in PASS_DEFS]
    if unknown:
        ap.error(f"unknown pass(es) {unknown}; --list for valid names")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = resolve_paths(args)
    os.makedirs(os.path.join(paths["results_root"], "_cache"), exist_ok=True)

    print(f"[precompute] dataset={args.dataset}")
    object_ids = validate_inputs(paths)
    if args.dry_run:
        print(f"[precompute] dry-run: would build {passes} for "
              f"{len(object_ids)} objects. Nothing written.")
        return

    precompute(paths, passes, root)


if __name__ == "__main__":
    main()
