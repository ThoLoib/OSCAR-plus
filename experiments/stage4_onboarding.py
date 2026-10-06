#!/usr/bin/env python3
"""
Stage 4a — onboarding latency: what does ONE new CAD model cost?

Question
--------
A user has a CAD file and wants to make the object findable. How long does
that take, and which step dominates?

Setup
-----
The basis is the **3b database** (G_proxy = gso + housecat6d + itodd, 1257
objects). Every target CAD from ycbv/tless/lmo (59 of them) is onboarded and
measured INDIVIDUALLY; what is evaluated is the distribution over these 59
cases. Taking real CADs of differing complexity is the reason for the
spread — vertex count and file size are recorded alongside, so that the
variance stays explainable.

Measured steps (in execution order)
    mesh        load mesh, weld vertices, normals, diameter
    render      Blender, 42 views from icosphere vertices, FPS-ordered
    partial     partial point clouds per view (hidden point removal)
    describe    LLaVA description per view
    embed_dino  DINOv2 over the 42 renderings
    embed_clip  CLIP text over the 42 descriptions
    embed_ulip  ULIP-2 over the 42 partial clouds
    dgedi       GeDi descriptors (only with --dgedi; only needed for geometry)

The embed steps measure the **incremental** cost: only the views of the new
object, with the models already loaded. That is the work an appending cache
would have to do — not simulated, but measured directly.

Important: the current cache CANNOT do this
-------------------------------------------
The cache key is a fingerprint over the entire inventory
(step5_shape_matching.py, `_get_partial_cache_path`: one line per object per
view). A new object changes the hash and invalidates everything — onboarding
really costs O(gallery), not O(1). `--measure-invalidation` measures this
surcharge once, so that both numbers stand side by side.

Host or container?
------------------
The steps ``render`` (Blender) and ``dgedi`` (docker client) run on the HOST,
all the others in the oscar-plus container. The script enforces that: a selection
of container steps wraps itself in ``docker compose run``, a host step inside
the container aborts, and a mixture of the two is rejected (call them
separately then).

Examples
--------
    # Quick test: 3 objects, without Blender and LLaVA (container steps)
    python3 experiments/stage4_onboarding.py --max-objects 3 \\
        --stages mesh,embed

    # Complete container chain, all 59 target CADs
    python3 experiments/stage4_onboarding.py \\
        --stages mesh,partial,describe,embed

    # Host steps separately (Blender resp. dGeDi container)
    python3 experiments/stage4_onboarding.py --stages render
    python3 experiments/stage4_onboarding.py --stages dgedi

    # Plus the surcharge of the cache invalidation
    python3 experiments/stage4_onboarding.py --stages embed \\
        --measure-invalidation

    # Shape channel from the FULL MESH instead of from N partial clouds.
    # 'partial' drops out; in exchange the step 'mesh_sample' is added.
    python3 experiments/stage4_onboarding.py --stages mesh,describe,embed \\
        --shape-source fullmesh --reuse-renders

Paths come from ``config/paths.yaml`` (overridable per flag, see
``--help``). The result lands under ``<runs_root>/stage4_onboarding/``; a
``run_config.json`` with argv, git revision and timestamp is written next to
it beforehand.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# CLI-Prelude — VOR den schweren Imports und vor dem os.chdir(), damit
# `--help` auf dem Host ohne die Container-Abhaengigkeiten laeuft.
# ---------------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
sys.path.insert(0, _THIS)
for p in (_REPO, os.path.join(_REPO, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

ALL_STAGES = ["mesh", "render", "partial", "describe", "embed", "dgedi"]
# 'render' braucht das Blender-Binary, 'dgedi' den docker-Client — beides gibt
# es nur auf dem HOST. Alles andere braucht die Container-Umgebung (torch,
# LLaVA, ULIP-Checkpoints).
HOST_STAGES = ("render", "dgedi")

_ARGS = None
_PATHS = None

# Ziel-CADs: genau die Objekte, die in der 3b-Datenbank FEHLEN und deshalb
# onboardet werden muessten. Layout je Datensatz wie in stage3_gallery,
# relativ zu _PATHS["cad_root"].
TARGET_LAYOUT = {
    "ycbv":  ("ycbv/*/textured_simple.obj", "parent"),
    "tless": ("tless/*/model.ply", "parent"),
    "lmo":   ("lmo/*/model.ply", "parent"),
}


if __name__ == "__main__":
    import argparse

    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="ycbv,tless,lmo",
                    help="Datasets whose CADs are onboarded (default: all 59).")
    ap.add_argument("--shape-source", choices=["partial", "fullmesh"],
                    default="partial",
                    help="Source of the shape channel. 'partial' (default) encodes "
                         "N partial clouds per object; 'fullmesh' samples the mesh "
                         "ONCE and encodes it once. Affects only the shape channel — "
                         "DINOv2 and CLIP text need the renderings either way. "
                         "With 'fullmesh' the step 'partial' drops out as well.")
    ap.add_argument("--stages", default="mesh,partial,describe,embed",
                    help=f"Comma list out of {ALL_STAGES}. The default is the "
                         "complete CONTAINER chain; "
                         f"{' and '.join(HOST_STAGES)} run on the HOST and "
                         "have to be called each on their own ('all' would be "
                         "a host/container mixture and is rejected).")
    ap.add_argument("--max-objects", type=int, default=0,
                    help="Only the first N CADs (0 = all).")
    ap.add_argument("--num-views", default="42",
                    help="Comma list, e.g. '16,42'. Every number is measured as "
                         "its own pass. The views are FPS-ordered, so the "
                         "first 16 of 42 are a valid 16-view set "
                         "(Stage-1 O4: 0.5820 at V16 vs 0.5868 at V42).")
    ap.add_argument("--num-points", type=int, default=8192)
    ap.add_argument("--work-dir", default=None,
                    help="Renders/clouds land HERE, not in the gallery "
                         "(default: <caches_root>/stage4_work).")
    ap.add_argument("--reuse-renders", action="store_true",
                    help="Copy existing renderings/clouds into the work "
                         "directory instead of generating them. Needed where no "
                         "Blender is installed — describe and embed are then "
                         "really measured, only the render time is missing.")
    ap.add_argument("--inv-sample", type=int, default=15,
                    help="How many real gallery objects are encoded for the "
                         "per-item invalidation cost.")
    ap.add_argument("--measure-invalidation", action="store_true",
                    help="Also measure the surcharge of the cache invalidation.")
    ap.add_argument("--out", default="",
                    help="Result file (default: <runs_root>/stage4_onboarding/"
                         "onboarding.json resp. onboarding_fullmesh.json with "
                         "--shape-source fullmesh); relative paths against the "
                         "repo root.")
    ap.add_argument("--no-docker", action="store_true",
                    help="do not wrap into the oscar-plus container automatically")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = "0"
    _PATHS = _paths_mod.from_args(args)

    # Blender-Binary: explizites --blender (Pfadregistrierung) gewinnt, dann der
    # bisherige Env-Schalter BLENDER_BIN der Mess-Skripte, sonst tools.blender
    # aus config/paths.yaml. Kein Host-Pfad mehr im Code.
    # HINWEIS Provenienz: die FINALEN Galerien wurden IM oscar-plus-Container
    # gerendert (/blender/blender-3.4.1-linux-x64); diese Stage-4-MESSUNG
    # rendert per Default mit dem HOST-Blender DERSELBEN Version 3.4.1 — die
    # 3.3.1-Installation hat KEIN PIL und scheitert still.
    args.blender = (args.blender or os.environ.get("BLENDER_BIN")
                    or _PATHS["blender"])
    if args.work_dir is None:
        args.work_dir = os.path.join(_PATHS["caches_root"], "stage4_work")
    if not args.out:
        args.out = os.path.join(
            _PATHS["runs_root"], "stage4_onboarding",
            "onboarding_fullmesh.json" if args.shape_source == "fullmesh"
            else "onboarding.json")
    elif not os.path.isabs(args.out):
        args.out = os.path.join(_REPO, args.out)

    args.stage_list = (list(ALL_STAGES) if args.stages == "all"
                       else [s.strip() for s in args.stages.split(",") if s.strip()])
    _unknown = [s for s in args.stage_list if s not in ALL_STAGES]
    if _unknown:
        ap.error(f"unknown stage(s): {_unknown}; allowed: {ALL_STAGES}")

    # --- Host/Container-Regel (siehe Docstring) ---
    _host = [s for s in args.stage_list if s in HOST_STAGES]
    _cont = [s for s in args.stage_list if s not in HOST_STAGES]
    _in_container = os.path.exists("/.dockerenv")
    if _host and _cont:
        sys.exit(f"[stage4] host stages {_host} and container stages {_cont} "
                 "cannot be measured in ONE call: 'render' needs "
                 "the Blender binary, 'dgedi' the docker client (both only on "
                 "the host), the rest needs the container environment. Call "
                 f"separately, e.g.  --stages {','.join(_cont)}  and  "
                 f"--stages {','.join(_host)}  (set different --out).")
    if _host and _in_container:
        sys.exit(f"[stage4] stage(s) {_host} run on the HOST, but this process "
                 "runs INSIDE the container. Start again on the host "
                 "(without docker compose run).")
    if _cont and not _in_container and not args.no_docker:
        if not os.path.isfile(os.path.join(_REPO, "docker-compose.yml")):
            sys.exit(f"[stage4] {_REPO}/docker-compose.yml missing — "
                     "incomplete repo?")
        print("[stage4] runs in the oscar-plus container — wrapping automatically.",
              flush=True)
        os.chdir(_REPO)  # docker compose needs the repo as CWD
        os.execvp("docker", ["docker", "compose", "run", "--rm", "oscar-plus",
                             "python3", "/app/experiments/stage4_onboarding.py"]
                  + sys.argv[1:])

    _ARGS = args

if _PATHS is None:  # als Modul importiert (keine CLI): paths.yaml-Defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()

from stage4_common import (Timings, aggregate, host_provenance,  # noqa: E402
                           print_table, summarize, write_results)


def _repo_rel(path, what):
    """Repo-relative path — or abort. The dGeDi container mounts ONLY the repo
    (``.:/oscar``); a folder next to it is invisible to it and the run fails
    only inside the container with FileNotFoundError (2026-09-04)."""
    rel = os.path.relpath(os.path.abspath(path), _REPO)
    if rel.startswith(".."):
        sys.exit(f"[stage4] {what} lies outside the repo ({path}). The "
                 "dGeDi container sees only the repo — pick a path under "
                 f"{_REPO}.")
    return rel


def target_meshes(datasets):
    """(dataset, obj_id, mesh_path) for all CADs to be onboarded."""
    out = []
    for ds in datasets:
        pattern, mode = TARGET_LAYOUT[ds]
        for p in sorted(glob.glob(os.path.join(_PATHS["cad_root"], pattern))):
            oid = (os.path.basename(os.path.dirname(p)) if mode == "parent"
                   else os.path.splitext(os.path.basename(p))[0])
            out.append((ds, oid, p))
    return out


# --------------------------------------------------------------------------
# Einzelschritte. Jeder gibt zusaetzlich Kennzahlen zurueck, die die Streuung
# erklaeren (Vertexzahl, Dateigroesse) — ohne die ist die Verteilung nicht
# interpretierbar.
# --------------------------------------------------------------------------
def stage_mesh(mesh_path, t: Timings) -> dict:
    import trimesh
    with t.measure("mesh"):
        m = trimesh.load(mesh_path, force="mesh", process=True)
        m.merge_vertices()
        m.fix_normals()
        extents = m.bounding_box.extents
        diameter = float((extents ** 2).sum() ** 0.5)
    return {"vertices": int(len(m.vertices)), "faces": int(len(m.faces)),
            "diameter": diameter,
            "file_mb": round(os.path.getsize(mesh_path) / 1e6, 3)}


def stage_render(mesh_path, out_dir, obj_id, num_views, blender, t: Timings) -> dict:
    """Blender is a foreign process; RENDER_ONLY restricts it to one object.

    Output deliberately goes into a work directory, NOT into object_images/ —
    the experiment must not overwrite the existing gallery.
    """
    env = dict(os.environ,
               OBJECT_FOLDER=os.path.dirname(os.path.dirname(mesh_path)),
               OBJECT_IMAGES=out_dir + "/",
               RENDER_ONLY=obj_id,
               NUM_VIEWS=str(num_views),
               OVERWRITE_EXISTING="1")
    cmd = [blender, "-b", "-P",
           os.path.join(_REPO, "preprocessing", "render_views.py")]
    with t.measure("render"):
        # Ein- und Ausgabeordner kommen ueber die Env-Variablen, beide absolut;
        # render_views.py bricht ab, wenn sie fehlen.
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           cwd=os.path.join(_REPO, "preprocessing"))
    n = len(glob.glob(os.path.join(out_dir, obj_id, "*.png")))
    # Blender beendet sich bei einem Python-Fehler mit rc=0 (beobachtet
    # 2026-08-31: fehlendes PIL -> Traceback, rc=0, 250 ms, null Bilder).
    # Der Rueckgabewert taugt hier also nicht als Erfolgspruefung; die Zahl der
    # erzeugten Bilder tut es.
    out = {"render_rc": r.returncode, "render_views": n}
    if n == 0:
        out["render_error"] = (r.stderr or r.stdout or "")[-300:]
    return out


def stage_partial(mesh_path, out_dir, num_points, obj_id, t: Timings) -> dict:
    """Partial clouds per view from mesh + stored camera matrix (HPR)."""
    # KEIN --mesh-glob: das ist ein eigenstaendiges Muster mit obj_id =
    # Dateistamm, was fuer ycbv jedes Objekt auf "textured_simple" abbilden
    # wuerde (dieselbe ID-Kollision wie im Full-Mesh-Pfad). Stattdessen die
    # <cad_dir>/<obj_id>/-Konvention: das Skript entdeckt die Objekte ueber
    # --images_dir, und obj_root enthaelt genau eines.
    cad_dir = os.path.dirname(os.path.dirname(mesh_path))
    cmd = [sys.executable, os.path.join(_REPO, "preprocessing",
                                        "generate_partial_pointclouds.py"),
           "--cad_dir", cad_dir,
           "--images_dir", out_dir,
           "--num_points", str(num_points), "--overwrite"]
    with t.measure("partial"):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO)
    n = len(glob.glob(os.path.join(out_dir, obj_id, "*_partial.npz")))
    out = {"partial_rc": r.returncode, "partial_clouds": n}
    if n == 0:
        out["partial_error"] = (r.stderr or r.stdout or "")[-300:]
    return out


def reuse_renders(ds, oid, obj_dir, num_views, gen_partial=False) -> dict:
    """Copy existing renderings into the work directory.

    Blender is not installed on this machine (the gallery was rendered on the
    second PC), so the render step drops out — and with it the input for
    describe and embed. The cost of LLaVA and the encoders does not depend on
    WHERE an image comes from, though, only on how many there are. Copying the
    first V renderings therefore makes these steps honestly measurable; only
    the render time itself is missing and is reported as such.
    """
    import shutil
    src = os.path.join(_PATHS["gallery_root"], ds, oid)
    if not os.path.isdir(src):
        return {"reuse_renders": f"no renderings under {src}"}
    imgs = [p for p in sorted(glob.glob(os.path.join(src, "*.png")))
            if not p.endswith("_bg.png")][:num_views]
    # Kameramatrizen MUESSEN mit: generate_partial_pointclouds.py entdeckt seine
    # Views ueber <obj>_viewN_CamMatrix.npy. Ohne sie findet es null Views und
    # die partial-Stufe laesst sich gar nicht messen — der Grund, warum sie in
    # allen Laeufen bis 2026-09-01 fehlte.
    cams = sorted(glob.glob(os.path.join(src, "*_CamMatrix.npy")))
    cams = _first_n_views(cams, num_views)
    files = imgs + cams
    npz = []
    if not gen_partial:
        # Wolken nur uebernehmen, wenn sie NICHT erzeugt werden sollen; sonst
        # wuerde die Messung ueberschriebene Dateien zaehlen statt neue.
        npz = sorted(glob.glob(os.path.join(src, "*_partial.npz")))[:num_views]
        files += npz
    for p in files:
        dst = os.path.join(obj_dir, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy2(p, dst)
    return {"reused_images": len(imgs), "reused_cams": len(cams),
            "reused_clouds": len(npz)}


def _first_n_views(paths, n):
    """The first n views in FPS order, sorted by view index.

    Lexicographically view10 < view2 — the cut would then hit a different
    subset than the one Stage 1 evaluated as V16.
    """
    import re
    keyed = []
    for p in paths:
        m = re.search(r"_view(\d+)_", os.path.basename(p))
        keyed.append((int(m.group(1)) if m else 10 ** 9, p))
    return [p for _, p in sorted(keyed)[:n]]


def stage_describe(obj_root, t: Timings) -> dict:
    """LLaVA description per view.

    ``--images_dir`` must be the folder that contains the OBJECT FOLDERS, not
    the object folder itself — otherwise the script reports "Total objects: 0"
    and returns within milliseconds with rc=0 (bug of 2026-08-31). That is why
    every object gets its own root folder with exactly one subfolder in it.
    """
    out_json = os.path.join(obj_root, "descriptions.json")
    cmd = [sys.executable, os.path.join(_REPO, "preprocessing",
                                        "generate_descriptions.py"),
           "--images_dir", obj_root, "--output", out_json, "--overwrite"]
    with t.measure("describe"):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO)
    info = {"describe_rc": r.returncode,
            "describe_json": os.path.isfile(out_json)}
    if not info["describe_json"]:
        info["describe_error"] = (r.stderr or r.stdout or "")[-300:]
    return info


class Encoders:
    """Load the models ONCE, after that only encode.

    Exactly this separation is what makes the number meaningful: the load time
    is a system startup cost, not an onboarding cost. It is reported separately.
    """

    def __init__(self, t: Timings, fullmesh_shape: bool = False):
        # fullmesh_shape schaltet den Shape-Kanal von N Teilwolken auf EINE
        # Mesh-Abtastung um (--shape-source fullmesh). Betrifft NUR den
        # Shape-Kanal; DINOv2 und CLIP-Text brauchen die Renderings unabhaengig
        # davon und bleiben unveraendert.
        self.fullmesh_shape = fullmesh_shape
        self.mesh_path_for_obj = None      # je Objekt vom Aufrufer gesetzt
        # Ueber build_pipeline, NICHT ueber ein blankes PipelineConfig(): dessen
        # ulip_repo_path ist "" und der Encoder bricht ab. Wichtiger noch — nur
        # so sind Backbone, Checkpoint, Punktzahl und Farbmodus identisch mit der
        # Pipeline, die in Stage 1–3 gemessen wurde. Eine Latenzzahl aus einer
        # anders konfigurierten Encoder-Instanz waere nicht vergleichbar.
        from eval_common import build_pipeline
        from stage3_gallery import _base_cfg
        with t.measure("load_encoders"):
            cfg = _base_cfg("ycbv")
            _, self.clip, self.dino, _, self.ulip = build_pipeline(cfg)
        if self.ulip is None:
            raise RuntimeError("ShapeMatcher not available — "
                               "check the ULIP repo/checkpoint.")

    def embed_object(self, obj_dir, num_views, t: Timings) -> dict:
        """Incremental cost: only the views of THIS object.

        Restricted to the first ``num_views``. That is admissible because the
        renderings are stored FPS-ordered — the first V are exactly the V-view
        set that the O4 ablation in Stage 1 evaluated.

        Every channel is measured INDIVIDUALLY, and within the channels the
        loading is separated from the computation: otherwise `embed_dino` would
        contain the JPEG decoding, which has nothing to do with the encoder and
        scales entirely differently on other hardware.
        """
        import json as _json

        import numpy as np
        from PIL import Image
        info = {}

        imgs = [p for p in sorted(glob.glob(os.path.join(obj_dir, "*.png")))
                if not p.endswith("_bg.png")][:num_views]
        if imgs:
            loaded = []
            with t.measure("io_load_images"):
                for p in imgs:
                    loaded.append(Image.open(p).convert("RGB"))
            with t.measure("embed_dino"):
                for im in loaded:
                    self.dino.encode_image(im)
            info["n_views_dino"] = len(imgs)

        desc = os.path.join(os.path.dirname(obj_dir), "descriptions.json")
        if os.path.isfile(desc):
            try:
                texts = _json.load(open(desc))
                texts = (list(texts.values()) if isinstance(texts, dict)
                         else list(texts))[:num_views]
                # _encode_texts_batch ist der Pfad, den load_descriptions selbst
                # nimmt (step3_clip_retrieval.py:212) — gebatcht, nicht je String.
                # Ein oeffentliches encode_text gibt es nicht und wird auch nicht
                # gebraucht; eine Schleife ueber Einzelstrings haette zudem etwas
                # anderes gemessen als die Pipeline tut.
                with t.measure("embed_clip"):
                    self.clip._encode_texts_batch([str(s) for s in texts])
                info["n_texts"] = len(texts)
            except Exception as exc:            # nicht abbrechen, nur vermerken
                info["embed_clip_skipped"] = str(exc)

        if self.fullmesh_shape:
            # --- Full-Mesh-Variante des Shape-Kanals (--shape-source fullmesh) --
            # Statt N Teilwolken wird das Mesh EINMAL abgetastet und EINMAL
            # encodiert. Beide Schritte kommen aus der Pipeline selbst
            # (step5.sample_pointcloud_from_mesh / ShapeMatcher.encode_pointcloud),
            # damit hier nicht etwas anderes gemessen wird als die Pipeline tut.
            # Getrennt gemessen, weil das Abtasten CPU- und das Encodieren
            # GPU-Arbeit ist und beide ganz anders skalieren.
            if self.mesh_path_for_obj:
                from pipeline.step5_shape_matching import (
                    sample_pointcloud_from_mesh)
                cfg = self.ulip.config
                use_colors = (cfg.ulip2_use_colors
                              and cfg.ulip2_backbone == "pointbert_colored")
                with t.measure("mesh_sample"):
                    pts, col = sample_pointcloud_from_mesh(
                        self.mesh_path_for_obj,
                        num_points=cfg.ulip2_num_points,
                        with_colors=use_colors)
                with t.measure("embed_ulip"):
                    self.ulip.encode_pointcloud(pts, colors=col)
                info["n_clouds"] = 1
                info["shape_source"] = "fullmesh"
            else:
                raise RuntimeError(
                    "--shape-source fullmesh, but no mesh path set. "
                    "Without a mesh there would silently be no shape channel — "
                    "and the onboarding total would be too low unnoticed.")
        else:
            npz = sorted(glob.glob(os.path.join(obj_dir, "*_partial.npz")))[:num_views]
            if npz:
                clouds = []
                with t.measure("io_load_clouds"):
                    for p in npz:
                        d = np.load(p)
                        clouds.append((d["points"], d.get("colors")))
                with t.measure("embed_ulip"):
                    for pts, col in clouds:
                        self.ulip.encode_pointcloud(pts, colors=col)
                info["n_clouds"] = len(npz)
                info["shape_source"] = "partial"

        info.update(self.append_to_cache(num_views, t))
        return info

    @staticmethod
    def append_to_cache(num_views, t, work=None):
        """The simulated INCREMENTAL append — real, not a dummy.

        This is exactly what an appending cache would have to do: load the
        existing file, insert the one new object entry, write it back. The
        largest real gallery cache serves as a stand-in (gso, 1030 of the 1257
        proxy objects).

        Keeping the measurement separated into load / insert / save is the
        point: the insert is O(1), but loading and writing a monolithic .pt
        file are O(gallery). So even an "appending" cache in this storage form
        pays for the whole gallery per new object — just serialisation instead
        of encoding.
        """
        import torch
        cands = sorted(glob.glob(os.path.join(
            _PATHS["gallery_root"], "gso", ".ulip_partial_cache_*.pt")))
        if not cands:
            return {"cache_append_skipped": "no gso partial cache found"}
        with t.measure("cache_load"):
            blob = torch.load(cands[0], map_location="cpu", weights_only=False)
        emb = blob.get("embeddings", blob) if isinstance(blob, dict) else blob
        n_before = len(emb)
        with t.measure("cache_insert"):
            any_v = next(iter(emb.values()))
            emb["__stage4_probe__"] = torch.zeros(
                num_views, any_v.shape[-1], dtype=any_v.dtype)
        tmp = os.path.join(work or "/tmp", ".stage4_cache_append.pt")
        with t.measure("cache_save"):
            torch.save(blob, tmp)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return {"cache_entries": n_before}


def stage_dgedi(mesh_path, obj_root, obj_id, ds, t: Timings) -> dict:
    """GeDi descriptors for the new object, so that it becomes geometrically
    searchable.

    Via `preprocessing/precompute_dgedi.py` in the dgedi container with a
    one-entry manifest — that is the route the dGeDi chain takes (the same
    model loader and feature extractor as services/dgedi/server.py, so that
    precompute and service are bit-identical). NOT via an HTTP call to the OLD
    `gedi` service (default `http://gedi:5060`), which is not running and would
    fail silently; the same mix-up has already cost the Stage-1 geometry run
    (AI_LOG 2026-08-26 and 2026-09-04).

    Only relevant if geometric re-ranking is used — which Stage 3 refuted for
    BOP. Hence optional, not the default.
    """
    import json as _json
    # Der dGeDi-Container mountet NUR das Repo (`.:/oscar`). Manifest, Mesh und
    # Ausgabe muessen deshalb INNERHALB des Repos liegen — ein Arbeitsordner
    # unter $HOME ist fuer ihn unsichtbar (FileNotFoundError, 2026-09-04).
    work = os.path.join(_PATHS["caches_root"], "dgedi", "stage4_onboarding",
                        ds, obj_id)
    cache = os.path.join(work, "gedi_cache")
    man = os.path.join(work, "manifest.json")
    os.makedirs(cache, exist_ok=True)
    with open(man, "w") as fh:
        _json.dump({f"{ds}/{obj_id}": _repo_rel(mesh_path, "target CAD")}, fh)
    cmd = ["docker", "compose", "run", "--rm", "--no-deps", "dgedi",
           "python3", "/oscar/preprocessing/precompute_dgedi.py",
           "--manifest", "/oscar/" + _repo_rel(man, "dGeDi manifest"),
           "--out", "/oscar/" + _repo_rel(cache, "dGeDi descriptor folder"),
           "--overwrite"]
    with t.measure("dgedi"):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO)
    n = len(glob.glob(os.path.join(cache, "*.npz")))
    out = {"dgedi_rc": r.returncode, "dgedi_descriptors": n}
    if n == 0:
        out["dgedi_error"] = (r.stderr or r.stdout or "")[-300:]
    return out


# --------------------------------------------------------------------------
def measure_invalidation(enc, num_views, sample, t: Timings) -> dict:
    """The surcharge the CURRENT cache enforces — measured, not guessed.

    When an object is added, the inventory fingerprint changes and the whole
    cache expires: every gallery object has to be re-encoded. What is measured
    is therefore the encoding of a sample of real gallery objects WITHOUT a
    cache, extrapolated to the gallery size.

    The first attempt (2026-09-01) timed `assemble_gallery` here and reported
    6.1 s — that was an assembly with WARM caches, i.e. precisely not the
    quantity at issue. Hence now via the per-item cost.
    """
    GALLERY = 1257                      # G_proxy, die 3b-Datenbank
    src_root = os.path.join(_PATHS["gallery_root"], "gso")
    objs = [d for d in sorted(os.listdir(src_root))
            if os.path.isdir(os.path.join(src_root, d))][:sample]
    per = Timings()
    for oid in objs:
        enc.embed_object(os.path.join(src_root, oid), num_views, per)
    enc_keys = [k for k in per.as_dict() if k.startswith("embed_")]
    per_obj = sum(sum(per.runs[k]) for k in enc_keys) / max(len(objs), 1)
    t.add("invalidation_total_extrapolated", per_obj * GALLERY)
    return {"gallery_size": GALLERY, "sampled_objects": len(objs),
            "per_object_encode_s": round(per_obj, 3),
            "extrapolated_full_reencode_s": round(per_obj * GALLERY, 1),
            "extrapolated_full_reencode_min": round(per_obj * GALLERY / 60, 1)}


def main():
    # Alle Flags, die Default-Ableitung und die Host/Container-Regel stehen im
    # Prelude am Kopf der Datei (damit `--help` ohne die schweren Imports geht).
    args = _ARGS
    stages = list(args.stage_list)

    if "render" in stages:
        from shutil import which
        if which(args.blender) is None:
            print(f"[stage4] WARNING: '{args.blender}' not found — "
                  f"stage 'render' is skipped (set --blender).")
            stages = [s for s in stages if s != "render"]

    view_counts = [int(v) for v in str(args.num_views).split(",") if v.strip()]
    meshes = target_meshes([d.strip() for d in args.targets.split(",")])
    if args.max_objects:
        meshes = meshes[:args.max_objects]
    print(f"[stage4] {len(meshes)} target CADs, stages: {stages}, "
          f"view counts: {view_counts}")
    os.makedirs(args.work_dir, exist_ok=True)

    setup = Timings()
    enc = (Encoders(setup, fullmesh_shape=(args.shape_source == "fullmesh"))
           if "embed" in stages else None)

    by_views, records = {}, []
    for V in view_counts:
        per_object = []
        print(f"\n[stage4] --- {V} views ---")
        for i, (ds, oid, mesh) in enumerate(meshes, 1):
            t = Timings()
            rec = {"dataset": ds, "object_id": oid, "num_views": V,
                   "mesh": os.path.relpath(mesh, _REPO)}
            # Zwei Ebenen: obj_root enthaelt genau EINEN Objektordner, weil
            # generate_descriptions.py ueber Unterordner iteriert. Ein flaches
            # Layout wuerde bei jedem Objekt alle vorherigen mitbeschreiben.
            obj_root = os.path.join(args.work_dir, f"v{V}", ds, oid)
            obj_dir = os.path.join(obj_root, oid)
            os.makedirs(obj_dir, exist_ok=True)
            try:
                if "mesh" in stages:
                    rec.update(stage_mesh(mesh, t))
                if args.reuse_renders:
                    rec.update(reuse_renders(ds, oid, obj_dir, V,
                                             gen_partial="partial" in stages))
                if "render" in stages:
                    rec.update(stage_render(mesh, obj_root, oid, V,
                                            args.blender, t))
                if "partial" in stages:
                    rec.update(stage_partial(mesh, obj_root, args.num_points,
                                             oid, t))
                if "describe" in stages:
                    rec.update(stage_describe(obj_root, t))
                if "embed" in stages and enc is not None:
                    src = obj_dir if os.path.isdir(obj_dir) else None
                    # Ohne render/describe liegen die Views in der bestehenden
                    # Gallery — von dort lesen, statt die Stufe stillschweigend
                    # zu ueberspringen.
                    if not glob.glob(os.path.join(obj_dir, "*.png")):
                        src = os.path.join(_PATHS["gallery_root"], ds, oid)
                    # Der Full-Mesh-Zweig braucht das Mesh selbst, nicht die Views.
                    enc.mesh_path_for_obj = mesh
                    rec.update(enc.embed_object(src, V, t))
                if "dgedi" in stages:
                    rec.update(stage_dgedi(mesh, obj_root, oid, ds, t))
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  [{i}/{len(meshes)}] {ds}/{oid}  ERROR: {rec['error']}")

            rec["timings"] = t.as_dict()
            rec["total_s"] = t.total()
            per_object.append(t.as_dict())
            records.append(rec)
            print(f"  [{i}/{len(meshes)}] {ds}/{oid:<16} {t.total():7.2f} s")
        by_views[V] = {
            "per_step": aggregate(per_object),
            "per_object_total_s": summarize(
                [r["total_s"] for r in records if r["num_views"] == V]),
        }

    # Einzelne Objektfehler sind Messdaten (und stehen in den records) — aber
    # wenn JEDES Objekt scheiterte, gibt es nichts zu berichten, und ein
    # rc=0 mit Null-Medianen taeuschte eine Messung vor (so geschehen mit
    # einem nicht beschreibbaren Cache-Verzeichnis).
    _failed = [r for r in records if "error" in r]
    if records and len(_failed) == len(records):
        sys.exit(f"[stage4] ABORT: all {len(records)} object measurements "
                 f"failed — first error: {_failed[0]['error']}")

    payload = {
        "experiment": "stage4a_onboarding",
        "base_gallery": "G_proxy (3b) = gso + housecat6d + itodd = 1257",
        "stages": stages,
        "view_counts": view_counts,
        "n_objects": len(meshes),
        "provenance": host_provenance(),
        "model_load_once_s": {k: v[0] for k, v in setup.as_dict().items()},
        "by_views": by_views,
        "records": records,
        # Qualitaetsseite aus Stage 1 (SHREC'18, nDCG), damit die Kostenzahlen
        # unmittelbar gegen den Nutzen gestellt werden koennen.
        "stage1_quality_ndcg": {"8": 0.5714, "16": 0.5820,
                                "32": 0.5800, "42": 0.5868},
    }

    if args.measure_invalidation:
        if enc is None:
            payload["invalidation"] = {"skipped": "needs --stages embed"}
        else:
            inv = Timings()
            payload["invalidation"] = {
                **measure_invalidation(enc, max(view_counts), args.inv_sample, inv),
                **{k: round(sum(v), 2) for k, v in inv.as_dict().items()}}

    for V in view_counts:
        print_table(f"Onboarding per object — {V} views", by_views[V]["per_step"])
        tot = by_views[V]["per_object_total_s"]
        if tot.get("n"):
            print(f"\n  Total per object: median {tot['median']:.2f} s, "
                  f"IQR {tot['iqr']:.2f} s, p95 {tot['p95']:.2f} s  (n={tot['n']})")

    if len(view_counts) > 1:
        # Nur view-abhaengige Stufen duerfen in den Vergleich: `mesh` kostet bei
        # 16 und 42 Views dasselbe, die Differenz waere reines Messrauschen und
        # wuerde als Ergebnis missverstanden.
        view_dependent = {"render", "partial", "describe", "embed", "dgedi"}
        print_view_tradeoff(by_views, payload["stage1_quality_ndcg"],
                            has_view_stage=bool(view_dependent & set(stages)))

    if payload["model_load_once_s"]:
        print("\n  One-off model load time (NOT an onboarding cost):")
        for k, v in payload["model_load_once_s"].items():
            print(f"    {k:<24}{v:8.2f} s")
    write_results(args.out, payload)


def print_view_tradeoff(by_views, quality, has_view_stage=True):
    """Cost against benefit — the actual purpose of the view sweep.

    Stage 1 shows that the quality runs flat from 16 views on (V32 is even
    below V16). If the onboarding cost rises linearly with the view count, 42
    cannot be justified — this table puts both side by side.
    """
    if not has_view_stage:
        print("\n[stage4] view comparison skipped: none of the active "
              "stages depends on the view count (mesh costs the same at 16 and "
              "42). Measure with --stages render,partial,describe,embed.")
        return
    ref = max(by_views)
    ref_cost = by_views[ref]["per_object_total_s"].get("median", 0.0)
    ref_q = quality.get(str(ref))
    print("\n=== Cost-benefit: views ===")
    print(f"  {'Views':>6}{'Onboarding (median)':>22}{'Cost':>10}"
          f"{'nDCG (Stage 1)':>16}{'Benefit':>10}")
    for V in sorted(by_views):
        c = by_views[V]["per_object_total_s"].get("median", 0.0)
        q = quality.get(str(V))
        cs = f"{100 * c / ref_cost:6.0f}%" if ref_cost else "     —"
        qs = f"{q - ref_q:+.4f}" if (q is not None and ref_q is not None) else "    —"
        print(f"  {V:>6}{c:>19.2f} s{cs:>10}"
              f"{(f'{q:.4f}' if q is not None else '—'):>16}{qs:>10}")
    print(f"  (cost and benefit each relative to {ref} views)")


if __name__ == "__main__":
    import datetime
    import json

    # Volle Lauf-Provenienz im Ausgabeordner (argv + git-Revision + Zeit),
    # geschrieben BEVOR die Messung startet.
    _outdir = os.path.dirname(os.path.abspath(_ARGS.out))
    os.makedirs(_outdir, exist_ok=True)
    try:
        _rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        _rev = ""
    json.dump({"argv": sys.argv[1:], "git": _rev,
               "stages": _ARGS.stage_list, "shape_source": _ARGS.shape_source,
               "blender": _ARGS.blender, "work_dir": _ARGS.work_dir,
               "out": _ARGS.out,
               "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
               "time": datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(_outdir, "run_config.json"), "w"), indent=1)

    main()
