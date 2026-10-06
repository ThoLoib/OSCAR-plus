#!/usr/bin/env python3
"""
preprocess_gallery.py — generic preprocessing for every OSCAR+ gallery.

One stage per invocation, flat and flag-based. The defaults are exactly the
values the evaluation galleries were built with. Foreign datasets can be added
via --cad-dir/--mesh-glob/--id-mode without changing this script.

The script downloads NOTHING. Before every stage it checks that the inputs are
there, and AFTER every stage that the result is plausible — return codes alone
are demonstrably unreliable here (Blender rc=0 on failure, "0 objects" on a
wrong path).

Paths come from ``config/paths.yaml`` (overridable per flag, see
``--help``): the CADs under ``<cad_root>``/``<datasets_root>``, the generated
gallery under ``<gallery_root>/<dataset>`` (redirect with ``--cache-dir``), the
dGeDi descriptors under ``<caches_root>/dgedi/<dataset>``.

Stages and where they run:
    render    host  (Blender 3.4.1 + CUDA; --blender or $BLENDER_BIN)
    partial   container (docker compose run oscar-plus) — wrapped automatically
    describe  container — wrapped automatically
    embed     container — wrapped automatically
    dgedi     host (starts the dgedi compose service itself)
    check     anywhere — only inspects what exists, computes nothing

Host/container rule (as in experiments/stage4_onboarding.py): a host stage in
the container aborts, a container stage on the host wraps itself in
``docker compose run --rm --no-deps oscar-plus``. Because ``--step`` takes exactly
ONE stage, a host/container mix can only arise via ``--step all`` — and ``all``
resolves it into separate invocations: the host stage runs directly, every
container stage gets its own ``docker compose run``. ``all`` in the container
therefore aborts and points at the host.

Examples (ONE terminal line each, from the repo root on the host):
    python3 preprocess_gallery.py --dataset shrec18 --step check
    python3 preprocess_gallery.py --dataset ycbv --step render --views 42
    python3 preprocess_gallery.py --dataset ycbv --step partial
    python3 preprocess_gallery.py --dataset ycbv --step describe
    python3 preprocess_gallery.py --dataset ycbv --step embed --passes base
    python3 preprocess_gallery.py --dataset MI3DOR --step partial --hpr-param 3.2 --jitter-std 0
    python3 preprocess_gallery.py --dataset ycbv --step all          # render->embed
    python3 preprocess_gallery.py --dataset foreign --cad-dir path/to/cads --id-mode stem --step render
"""
from __future__ import annotations

import glob
import json
import os
import shlex
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# CLI-Prelude — VOR den schweren Imports, damit `--help` auf dem Host ohne die
# Container-Abhaengigkeiten laeuft.
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO, os.path.join(_REPO, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

IN_CONTAINER = os.path.exists("/.dockerenv")

# 'render' braucht das Blender-Binary, 'dgedi' den docker-Client — beides gibt
# es nur auf dem HOST. partial/describe/embed brauchen die Container-Umgebung
# (torch, LLaVA, ULIP-/Uni3D-Checkpoints).
HOST_STEPS = ("render", "dgedi")
CONTAINER_STEPS = ("partial", "describe", "embed")
CHAIN = ["render", "partial", "describe", "embed"]      # was '--step all' faehrt

_PATHS = None
_ARGS = None

# ---------------------------------------------------------------------------
# Datensatz-Tabelle. "root" waehlt die Registry-Wurzel des CAD-Ordners
# ("cad" = cad_root, "datasets" = datasets_root), "cad_dir" ist der Pfad DARIN.
# "id_mode" bestimmt, wie aus dem Mesh-Pfad die Objekt-ID wird; "mesh_glob" nur,
# wo das Layout vom Standard <cad_dir>/<obj_id>/ abweicht.
# ---------------------------------------------------------------------------
DATASETS = {
    "shrec18":     dict(root="datasets", cad_dir="shrec18/shrec18_full/cad",
                        mesh_glob="*.obj", id_mode="stem", n_objects=3308,
                        note="colour: texture (~70 % readable)"),
    "MI3DOR":      dict(root="cad", cad_dir="MI3DOR/model/test",
                        mesh_glob="*/*.obj", id_mode="stem", n_objects=3848,
                        note="NO mesh colour (uniform 0.4). Partial clouds of the "
                             "evaluation: --hpr-param 3.2 --jitter-std 0"),
    "ycbv":        dict(root="cad", cad_dir="ycbv",
                        mesh_glob="*/textured_simple.obj", id_mode="parent",
                        n_objects=21, note="texture; mm"),
    "tless":       dict(root="cad", cad_dir="tless",
                        mesh_glob="*/model.ply", id_mode="parent",
                        n_objects=30, note="no colour; mm"),
    "lmo":         dict(root="cad", cad_dir="lmo",
                        mesh_glob="*/model.ply", id_mode="parent",
                        n_objects=8, note="vertex colours; mm"),
    "gso":         dict(root="cad", cad_dir="gso",
                        mesh_glob="*/meshes/model.obj", id_mode="grandparent",
                        n_objects=1030, note="texture; unit METRES"),
    "housecat6d":  dict(root="cad", cad_dir="housecat6d",
                        mesh_glob="*/*.obj", id_mode="stem",
                        n_objects=199, note="category folders, id = file stem"),
    "itodd":       dict(root="cad", cad_dir="itodd",
                        mesh_glob="*/model.ply", id_mode="parent",
                        n_objects=28, note="no colour; mm"),
}

EMBED_PASSES = ["base", "siglip", "ulip_fullmesh", "ulip_pc_rgb",
                "ulip_pc_xyz", "uni3d", "all"]

# Die Arbeitsskripte. Alle Pfade absolut — im Container ist _REPO == /app.
RENDER_SCRIPT = os.path.join("preprocessing", "render_views.py")
PARTIAL_SCRIPT = os.path.join("preprocessing", "generate_partial_pointclouds.py")
DESCRIBE_SCRIPT = os.path.join("preprocessing", "generate_descriptions.py")
EMBED_SCRIPT = os.path.join("preprocessing", "precompute_embeddings.py")
DGEDI_SCRIPT = "preprocessing/precompute_dgedi.py"      # im dgedi-Container


def log(msg: str) -> None:
    print(f"[preprocess] {msg}", flush=True)


def die(msg: str) -> None:
    sys.exit(f"[preprocess] ABORT: {msg}")


def run(cmd, env_extra=None, cwd=None) -> None:
    env = dict(os.environ)
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    log("$ " + " ".join(shlex.quote(str(c)) for c in cmd)
        + ("" if not env_extra else "   [env: "
           + " ".join(f"{k}={v}" for k, v in env_extra.items()) + "]"))
    rc = subprocess.call([str(c) for c in cmd], env=env, cwd=cwd or _REPO)
    if rc != 0:
        die(f"subprocess exited with rc={rc}")


def reexec_in_container(argv) -> None:
    """partial/describe/embed need the oscar-plus container — wrap ourselves."""
    cmd = ["docker", "compose", "run", "--rm", "--no-deps", "oscar-plus",
           "python3", "/app/preprocess_gallery.py"] + list(argv)
    log("stage runs in the oscar-plus container — wrapping automatically.")
    os.chdir(_REPO)                     # docker compose braucht das Repo als CWD
    os.execvp("docker", cmd)


def _abs(path: str) -> str:
    """Resolve relative user paths against the repo root (the way the
    experiment drivers resolve their --out paths)."""
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def _repo_rel(path: str, what: str) -> str:
    """Repo-relative path — or abort. The dGeDi container mounts ONLY the
    repo (``.:/oscar``); a folder beside it is invisible to it."""
    rel = os.path.relpath(os.path.abspath(path), _REPO)
    if rel.startswith(".."):
        die(f"{what} lies outside the repo ({path}). The dGeDi container "
            f"only sees the repo — pick a path under {_REPO}.")
    return rel


def mesh_list(ds: dict):
    return sorted(glob.glob(os.path.join(ds["cad_dir"], ds["mesh_glob"])))


def gallery_dir() -> str:
    """Output location of the gallery: renderings, partial clouds, embedding caches."""
    return _ARGS.cache_dir


def desc_file(name: str) -> str:
    """Descriptions live next to the CADs — that is where the drivers expect them
    (evaluation/stage3_gallery.py: <cad_root>/<ds>/descriptions_attributes.json)."""
    return os.path.join(_PATHS["cad_root"], name, "descriptions_attributes.json")


# ---------------------------------------------------------------------------
# Verifikation je Stufe: Ergebnis pruefen, nicht den Rueckgabewert
# ---------------------------------------------------------------------------
def verify_render(name, ds, views) -> None:
    pngs = glob.glob(os.path.join(gallery_dir(), "*", "*_[0-9]*.png"))
    mats = glob.glob(os.path.join(gallery_dir(), "*", "*_CamMatrix.npy"))
    want = ds["n_objects"] * views
    log(f"render: {len(pngs)} views, {len(mats)} camera matrices "
        f"(~{want} expected each)")
    if len(mats) < want * 0.98:
        die(f"only {len(mats)}/{want} camera matrices — Blender probably "
            "failed silently (known for Blender != 3.4.1: rc=0 without PIL).")


def verify_partial(name, ds, views) -> None:
    npz = glob.glob(os.path.join(gallery_dir(), "*", "*_partial.npz"))
    want = ds["n_objects"] * views
    log(f"partial: {len(npz)} partial clouds (~{want} expected)")
    if len(npz) < want * 0.98:
        die(f"only {len(npz)}/{want} partial clouds.")


def verify_describe(name, ds) -> None:
    p = desc_file(name)
    if not os.path.isfile(p):
        die(f"{p} is missing.")
    d = json.load(open(p))
    n_caps = sum(len(v.get("image_descriptions", {})) for v in d.values())
    empty = [k for k, v in d.items() if not v.get("image_descriptions")]
    log(f"describe: {len(d)} objects, {n_caps} image descriptions "
        f"({ds['n_objects']} objects expected)")
    if len(d) < ds["n_objects"]:
        die("incomplete — most common cause: --images_dir pointed at the "
            "object folder instead of the folder WITH the object subfolders.")
    if empty:
        die(f"{len(empty)} objects WITHOUT descriptions (e.g. {empty[:3]}). "
            "Most common cause: CUDA OOM per batch (running services occupy "
            "memory) — rerun with --batch-size 2.")


def verify_embed(name) -> None:
    caches = [os.path.basename(p) for p in
              glob.glob(os.path.join(gallery_dir(), ".*cache*.pt"))]
    log(f"embed: {len(caches)} cache files in {gallery_dir()}: "
        + (", ".join(sorted(caches)) or "NONE"))
    if not caches:
        die("no embedding cache was produced.")


# ---------------------------------------------------------------------------
def _write_run_config(outdir: str, step: str, name: str, ds: dict) -> None:
    """Run provenance next to the result (argv + git revision + time + the
    resolved locations). Every stage writes its own file so that a later
    stage does not overwrite the provenance of an earlier one."""
    import datetime
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                             capture_output=True, text=True).stdout.strip()
    except Exception:                                          # noqa: BLE001
        rev = ""
    os.makedirs(outdir, exist_ok=True)
    json.dump({"argv": sys.argv[1:], "step": step, "dataset": name,
               "git": rev,
               "cad_dir": ds["cad_dir"], "mesh_glob": ds["mesh_glob"],
               "id_mode": ds["id_mode"], "n_objects": ds["n_objects"],
               "cache_dir": gallery_dir(), "desc_file": desc_file(name),
               "in_container": IN_CONTAINER,
               "time": datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(outdir, f"run_config_{step}.json"), "w"),
              indent=1)


def build_argparser():
    import argparse

    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="name from the built-in table OR free for foreign "
                         "datasets (then pass --cad-dir/--id-mode). "
                         "Known: " + ", ".join(DATASETS))
    ap.add_argument("--step", required=True,
                    choices=["render", "partial", "describe", "embed", "dgedi",
                             "check", "all"],
                    help="'all' = render -> partial -> describe -> embed in "
                         "one go (start it on the host; exactly what a new "
                         "gallery for pipeline.run_pipeline needs). The "
                         "container stages each get their own "
                         "'docker compose run'.")
    ap.add_argument("--cache-dir", default="",
                    help="output location of the gallery (renderings, partial "
                         "clouds, embedding caches). Default: <gallery_root>/<dataset>. "
                         "Careful: the eval drivers look for the gallery in "
                         "the default layout — another location only makes "
                         "sense for your own runs.")
    # Render
    ap.add_argument("--views", type=int, default=42,
                    help="number of views (icosphere, FPS-ordered). Default 42 = eval.")
    # Das Blender-Binary kommt als --blender aus der Pfadregistrierung
    # (add_path_args, Eintrag tools.blender); $BLENDER_BIN bleibt als
    # Vorrang-Env. Es MUSS 3.4.1 sein — 3.3.x scheitert still (rc=0 ohne PIL).
    ap.add_argument("--shard", default="0/1",
                    help="i/n to parallelise the rendering, result-neutral.")
    ap.add_argument("--overwrite", action="store_true")
    # Partial
    ap.add_argument("--num-points", type=int, default=10000)
    ap.add_argument("--hpr-param", type=float, default=2.8,
                    help="hidden-point-removal parameter. Eval: 2.8 — EXCEPT "
                         "MI3DOR (3.2, with --jitter-std 0).")
    ap.add_argument("--jitter-std", type=float, default=0.001)
    # Describe
    ap.add_argument("--batch-size", type=int, default=8)
    # Embed
    ap.add_argument("--passes", default="base",
                    help="comma list from " + "|".join(EMBED_PASSES) +
                         ". The eval used: base, siglip, ulip_fullmesh, "
                         "ulip_pc_rgb, ulip_pc_xyz, uni3d.")
    # dGeDi
    ap.add_argument("--dgedi-out", default="",
                    help="target folder for the descriptors "
                         "(default <caches_root>/dgedi/<dataset>).")
    # Fremde Datensaetze
    ap.add_argument("--cad-dir", default="",
                    help="CAD root (foreign dataset; relative = against the "
                         "repo root)")
    ap.add_argument("--mesh-glob", default="", help="glob relative to --cad-dir")
    ap.add_argument("--id-mode", default="stem",
                    choices=["stem", "parent", "grandparent"])
    ap.add_argument("--n-objects", type=int, default=0,
                    help="expected object count (for the verification, foreign dataset)")
    ap.add_argument("--no-docker", action="store_true",
                    help="do not wrap into the oscar-plus container automatically")
    _paths_mod.add_path_args(ap)
    return ap, _paths_mod


def dispatch_all(ds: dict) -> None:
    """'all' as a chain of separate invocations: every stage its own child process,
    which applies the host/container rule to ITSELF. The host stage (render)
    therefore runs directly here, every container stage wraps itself in its
    own `docker compose run` — exactly the mechanism the script already uses
    for single stages."""
    if IN_CONTAINER:
        die("start --step all on the HOST: 'render' needs Blender (host), "
            "partial/describe/embed need the container. The host run "
            "distributes that automatically; in the container invoke them "
            "separately (--step partial, --step describe, --step embed).")
    argv, skip_next = [], False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a == "--step":
            skip_next = True
            continue
        if a.startswith("--step="):
            continue
        argv.append(a)
    for st in CHAIN:
        log(f"===== stage {st} =====")
        rc = subprocess.call([sys.executable, os.path.abspath(__file__)]
                             + argv + ["--step", st], cwd=_REPO)
        if rc != 0:
            die(f"stage {st} exited with rc={rc}")
    log("all stages done — the gallery is ready to use "
        "(pipeline.run_pipeline --gallery <name>).")


def step_check(name: str) -> None:
    for label, fn in [("render (PNGs)", lambda: len(glob.glob(os.path.join(
                          gallery_dir(), "*", "*_[0-9]*.png")))),
                      ("cam matrices", lambda: len(glob.glob(os.path.join(
                          gallery_dir(), "*", "*_CamMatrix.npy")))),
                      ("partial (npz)", lambda: len(glob.glob(os.path.join(
                          gallery_dir(), "*", "*_partial.npz")))),
                      ("describe", lambda: len(json.load(open(desc_file(name))))
                          if os.path.isfile(desc_file(name)) else 0),
                      ("embed-caches", lambda: len(glob.glob(os.path.join(
                          gallery_dir(), ".*cache*.pt"))))]:
        try:
            log(f"  {label:14s}: {fn()}")
        except Exception as e:                                    # noqa: BLE001
            log(f"  {label:14s}: ERROR {e}")
    log(f"gallery: {gallery_dir()}")
    log(f"descriptions: {desc_file(name)}")
    log("note: partial(npz)=0 and cam matrices=0 are fine when "
        "a .ulip_partial_cache_*.pt exists — the embedding cache "
        "replaces the raw files.")


def step_render(name: str, ds: dict) -> None:
    blender = _ARGS.blender
    if not (os.path.isfile(blender) or shutil.which(blender)):
        die(f"Blender not found: {blender} (set --blender or "
            "tools.blender in config/paths.yaml). It MUST be 3.4.1 — 3.3.x "
            "fails silently with rc=0.")
    si, st = _ARGS.shard.split("/")
    _write_run_config(gallery_dir(), "render", name, ds)
    run([blender, "-b", "-P", os.path.join(_REPO, RENDER_SCRIPT)],
        env_extra={"OBJECT_FOLDER": ds["cad_dir"],
                   "OBJECT_IMAGES": gallery_dir() + "/",
                   "NUM_VIEWS": _ARGS.views,
                   "OVERWRITE_EXISTING": "1" if _ARGS.overwrite else "0",
                   "SHARD_INDEX": si, "SHARD_TOTAL": st})
    verify_render(name, ds, _ARGS.views)


def step_partial(name: str, ds: dict) -> None:
    cmd = ["python3", os.path.join(_REPO, PARTIAL_SCRIPT),
           "--cad_dir", ds["cad_dir"], "--images_dir", gallery_dir(),
           "--num_points", _ARGS.num_points, "--hpr-param", _ARGS.hpr_param,
           "--jitter-std", _ARGS.jitter_std]
    if ds["mesh_glob"] and ds["id_mode"] == "stem":
        cmd += ["--mesh-glob", ds["mesh_glob"]]
    if _ARGS.overwrite:
        cmd += ["--overwrite"]
    _write_run_config(gallery_dir(), "partial", name, ds)
    run(cmd)
    verify_partial(name, ds, _ARGS.views)


def step_describe(name: str, ds: dict) -> None:
    cmd = ["python3", os.path.join(_REPO, DESCRIBE_SCRIPT),
           "--images_dir", gallery_dir(),
           "--output", desc_file(name),
           "--batch-size", _ARGS.batch_size]
    if _ARGS.overwrite:
        cmd += ["--overwrite"]
    os.makedirs(os.path.dirname(desc_file(name)), exist_ok=True)
    _write_run_config(gallery_dir(), "describe", name, ds)
    run(cmd)
    verify_describe(name, ds)


def step_embed(name: str, ds: dict) -> None:
    bad = [p for p in _ARGS.passes.split(",") if p not in EMBED_PASSES]
    if bad:
        die(f"unknown passes: {bad}")
    _write_run_config(gallery_dir(), "embed", name, ds)
    run(["python3", os.path.join(_REPO, EMBED_SCRIPT),
         "--dataset", name,
         "--data-root", ds["cad_dir"],
         # precompute_embeddings erwartet den VOLLEN Glob, nicht relativ
         "--mesh-glob", os.path.join(ds["cad_dir"], ds["mesh_glob"]),
         "--mesh-id-mode", ds["id_mode"],
         "--images-dir", gallery_dir(),
         "--desc-file", desc_file(name),
         "--results-root", os.path.join(_PATHS["caches_root"],
                                        f"precompute_{name}"),
         "--passes", _ARGS.passes])
    verify_embed(name)


def step_dgedi(name: str, ds: dict, meshes) -> None:
    out = _abs(_ARGS.dgedi_out) if _ARGS.dgedi_out else os.path.join(
        _PATHS["caches_root"], "dgedi", name)
    manifest = os.path.join(_PATHS["caches_root"], "dgedi",
                            f"manifest_{name}.json")
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    json.dump({(os.path.splitext(os.path.basename(m))[0]
                if ds["id_mode"] == "stem" else
                os.path.basename(os.path.dirname(m))
                if ds["id_mode"] == "parent" else
                os.path.basename(os.path.dirname(os.path.dirname(m)))):
               os.path.relpath(m, _REPO) for m in meshes},
              open(manifest, "w"), indent=1)
    log(f"manifest: {manifest} ({len(meshes)} objects)")
    _write_run_config(out, "dgedi", name, ds)
    run(["docker", "compose", "run", "--rm", "--no-deps", "dgedi",
         "python3", "/oscar/" + DGEDI_SCRIPT,
         "--manifest", "/oscar/" + _repo_rel(manifest, "dGeDi manifest"),
         "--out", "/oscar/" + _repo_rel(out, "dGeDi descriptor folder"),
         "--n-points", 10000, "--mode", "multi_scale"])
    n = len(glob.glob(os.path.join(out, "*.npz")))
    log(f"dgedi: {n} descriptor entries in {out}")
    if n < len(meshes):
        die(f"only {n}/{len(meshes)} descriptors.")


def main() -> None:
    global _ARGS, _PATHS
    ap, _paths_mod = build_argparser()
    args = ap.parse_args()
    _ARGS = args
    _PATHS = _paths_mod.from_args(args)

    name = args.dataset
    if name in DATASETS:
        ds = dict(DATASETS[name])
    else:
        if not args.cad_dir:
            die(f"unknown dataset '{name}' — for foreign datasets pass "
                "--cad-dir (and if needed --mesh-glob/--id-mode/--n-objects).")
        ds = dict(root="cad", cad_dir="", mesh_glob=args.mesh_glob or "*/*.obj",
                  id_mode=args.id_mode, n_objects=args.n_objects or 0,
                  note="foreign")
    if args.mesh_glob:
        ds["mesh_glob"] = args.mesh_glob

    # CAD-Wurzel aufloesen: --cad-dir gewinnt, sonst die Registry-Wurzel des
    # Datensatzes (cad_root bzw. datasets_root) + der Pfad darin.
    if args.cad_dir:
        ds["cad_dir"] = _abs(args.cad_dir)
    else:
        ds["cad_dir"] = os.path.join(
            _PATHS["cad_root"] if ds.get("root", "cad") == "cad"
            else _PATHS["datasets_root"], ds["cad_dir"])

    # Ausgabeort der Galerie
    args.cache_dir = (_abs(args.cache_dir) if args.cache_dir
                      else os.path.join(_PATHS["gallery_root"], name))

    # Blender: explizites --blender gewinnt, dann $BLENDER_BIN, sonst
    # tools.blender aus config/paths.yaml. Kein Host-Pfad im Code.
    args.blender = (args.blender or os.environ.get("BLENDER_BIN")
                    or _PATHS["blender"])

    # Die gewrappten Stufen sehen nur das Repo (`.:/app` bzw. `.:/oscar`).
    for label, path in (("--cad-dir", args.cad_dir),
                        ("--cache-dir", args.cache_dir)):
        if os.path.relpath(os.path.abspath(path), _REPO).startswith(".."):
            log(f"WARNING: {label} lies outside the repo ({path}) — the "
                "container stages will not see the folder. Either put it under "
                f"{_REPO} or start the stages inside the container yourself.")

    meshes = mesh_list(ds)
    if ds["n_objects"] == 0:
        ds["n_objects"] = len(meshes)
    log(f"dataset {name}: {len(meshes)} meshes under "
        f"{ds['cad_dir']}/{ds['mesh_glob']} ({ds['n_objects']} expected) — "
        f"{ds['note']}")
    if args.step != "check" and len(meshes) == 0:
        die("no meshes found — dataset not at the expected location? "
            "Switch the root per flag (--cad-root/--datasets-root/--cad-dir).")
    if ds["n_objects"] and len(meshes) != ds["n_objects"] and args.step != "check":
        die(f"mesh count {len(meshes)} != expected {ds['n_objects']} — "
            "dataset incomplete?")

    if args.step == "all":
        dispatch_all(ds)
        return

    if args.step == "check":
        step_check(name)
        return

    # --- Host/Container-Regel (siehe Docstring) ---
    if args.step in HOST_STEPS and IN_CONTAINER:
        die(f"stage '{args.step}' runs on the HOST "
            f"({'Blender' if args.step == 'render' else 'docker client'}), "
            "but this process is running IN the container. Start it again "
            "on the host (without docker compose run).")
    if args.step in CONTAINER_STEPS and not IN_CONTAINER and not args.no_docker:
        if not os.path.isfile(os.path.join(_REPO, "docker-compose.yml")):
            die(f"{_REPO}/docker-compose.yml is missing — repo incomplete?")
        reexec_in_container(sys.argv[1:])

    if args.step == "render":
        step_render(name, ds)
    elif args.step == "partial":
        step_partial(name, ds)
    elif args.step == "describe":
        step_describe(name, ds)
    elif args.step == "embed":
        step_embed(name, ds)
    elif args.step == "dgedi":
        step_dgedi(name, ds, meshes)


if __name__ == "__main__":
    main()
