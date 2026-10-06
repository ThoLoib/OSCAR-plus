#!/usr/bin/env python3
"""
Stage 4a — WARM onboarding times: what does a step cost WITHOUT process start?

Question
--------
``stage4_onboarding.py`` starts a separate process per object (Blender, the
LLaVA script, the dGeDi container) and thereby also measures binary startup and
model loading PER OBJECT. That is the honest number for a single new CAD, but
not the per-item cost a service with loaded models would have. This script
measures exactly that warm side — one session for ALL objects, the startup
measured once and reported separately.

Three steps, one entry point (``--step``)
    render    One Blender session per dataset x view count.
              ``render_views.py`` loops over all objects of an OBJECT_FOLDER
              anyway; ``RENDER_TIMING_JSON`` delivers the wall time per
              object (import + all view renders + camera matrices), and the
              difference between the outer wall time and ``script_total_s`` is
              the Blender binary start (incl. bpy import). -> HOST (Blender)
    describe  Load LLaVA ONCE (load time separately), then measure per object
              only the describing of its views. The batching function is the
              PRODUCTION FUNCTION ``generate_captions`` from
              ``preprocessing/generate_descriptions.py`` — no re-implementation.
              Batch 8 = previous default; batch 1 = single-image calls like the
              original OSCAR scripts, to quantify the batching effect per
              view.                                    -> CONTAINER (oscar)
    dgedi     ONE container run of ``preprocessing/precompute_dgedi.py`` over
              all 59 target CADs with ``--timing-json``: per object mesh loading
              + sampling + features + npz save, model loading once.
                                                       -> HOST (docker client)

Output goes into a work directory, NOT into the gallery — the existing
renderings and descriptors stay untouched (same rule as
``stage4_onboarding.stage_render``).

Guard
-----
Every step checks the object count per dataset (ycbv 21, tless 30, lmo 8 = 59);
``describe`` additionally compares the batch-8 texts against the existing
``descriptions_attributes.json`` (LLaVA runs greedily) and reports the share of
identical texts — nothing is replaced.

Invocation
----------
    python3 experiments/stage4_onboarding_warm.py --step render
    python3 experiments/stage4_onboarding_warm.py --step describe
    python3 experiments/stage4_onboarding_warm.py --step dgedi

Paths come from ``config/paths.yaml`` (overridable per flag, see
``--help``). ``--step describe`` wraps itself into the oscar-plus container;
``render`` and ``dgedi`` have to run on the host. The result lands under
``<runs_root>/stage4_onboarding_warm/<step>_timings.json``; a
``run_config_<step>.json`` with argv, git revision and timestamp sits next to
it (one per step, because the three steps share one output folder).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# CLI-Prelude — VOR den schweren Imports (torch/transformers), damit `--help`
# auf dem Host ohne die Container-Abhaengigkeiten laeuft.
# ---------------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
sys.path.insert(0, _THIS)
for p in (_REPO, os.path.join(_REPO, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Ziel-CADs je Datensatz — Riegel gegen eine halb gerenderte Gallery oder ein
# unvollstaendiges Manifest. Summe 59, die Basis der Onboarding-Tabelle.
TARGETS = {"ycbv": 21, "tless": 30, "lmo": 8}

# Mesh-Layout der Ziel-CADs, relativ zu _PATHS["cad_root"] — dasselbe wie
# stage4_onboarding.TARGET_LAYOUT (und damit wie stage3_gallery).
TARGET_LAYOUT = {
    "ycbv":  ("ycbv/*/textured_simple.obj", "parent"),
    "tless": ("tless/*/model.ply", "parent"),
    "lmo":   ("lmo/*/model.ply", "parent"),
}

# 'render' braucht das Blender-Binary, 'dgedi' den docker-Client — beides gibt
# es nur auf dem HOST. 'describe' braucht LLaVA/torch, also den Container.
HOST_STEPS = ("render", "dgedi")

_ARGS = None
_PATHS = None

_PROMPT_DEFAULT = ("Extract visual attributes of the object in the image: "
                   "object type, brand name, color, material, and label text.")


if __name__ == "__main__":
    import argparse

    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True,
                    choices=["render", "describe", "dgedi"],
                    help="which warm step is measured: render (Blender, "
                         "HOST), describe (LLaVA, container), dgedi "
                         "(dGeDi container, HOST)")
    ap.add_argument("--targets", default="ycbv,tless,lmo",
                    help="Target datasets (default: all 59 CADs).")
    ap.add_argument("--views", default="16,42",
                    help="Comma list of the view counts, e.g. '16,42'. No "
                         "effect for --step dgedi (descriptors come from "
                         "the mesh, not from views).")
    ap.add_argument("--batch-sizes", default="8,1",
                    help="only --step describe: LLaVA batch sizes, each its "
                         "own measurement series (default: 8,1).")
    ap.add_argument("--prompt", default=_PROMPT_DEFAULT,
                    help="only --step describe: LLaVA prompt (default = the "
                         "production prompt).")
    ap.add_argument("--work-dir", default=None,
                    help="Renders/manifests/descriptors land HERE, not in "
                         "the gallery (default: <caches_root>/stage4_warm). For "
                         "--step dgedi the folder must lie IN the repo: the "
                         "dGeDi container mounts only the repo.")
    ap.add_argument("--out", default="",
                    help="Result file (default: <runs_root>/"
                         "stage4_onboarding_warm/<step>_timings.json); relative "
                         "paths against the repo root.")
    ap.add_argument("--no-docker", action="store_true",
                    help="do not wrap into the oscar-plus container automatically")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = "0"
    _PATHS = _paths_mod.from_args(args)

    # Blender-Binary: explizites --blender (Pfadregistrierung) gewinnt, dann der
    # Env-Schalter BLENDER_BIN der Mess-Skripte, sonst tools.blender aus
    # config/paths.yaml. Genau 3.4.1 — die 3.3.1-Installation hat kein PIL und
    # scheitert still (rc=0, null Bilder).
    args.blender = (args.blender or os.environ.get("BLENDER_BIN")
                    or _PATHS["blender"])
    if args.work_dir is None:
        args.work_dir = os.path.join(_PATHS["caches_root"], "stage4_warm")
    if not args.out:
        args.out = os.path.join(_PATHS["runs_root"], "stage4_onboarding_warm",
                                f"{args.step}_timings.json")
    elif not os.path.isabs(args.out):
        args.out = os.path.join(_REPO, args.out)

    # --- Host/Container-Regel ---
    _in_container = os.path.exists("/.dockerenv")
    if args.step in HOST_STEPS and _in_container:
        sys.exit(f"[warm-{args.step}] step '{args.step}' runs on the HOST "
                 "(Blender resp. docker client), but this process runs INSIDE "
                 "the container. Start again on the host (without docker "
                 "compose run).")
    if args.step not in HOST_STEPS and not _in_container and not args.no_docker:
        if not os.path.isfile(os.path.join(_REPO, "docker-compose.yml")):
            sys.exit(f"[warm-{args.step}] {_REPO}/docker-compose.yml missing — "
                     "incomplete repo?")
        print(f"[warm-{args.step}] runs in the oscar-plus container — wrapping "
              "automatically.", flush=True)
        os.chdir(_REPO)  # docker compose needs the repo as CWD
        os.execvp("docker",
                  ["docker", "compose", "run", "--rm", "oscar-plus", "python3",
                   "/app/experiments/stage4_onboarding_warm.py"] + sys.argv[1:])

    _ARGS = args

if _PATHS is None:  # als Modul importiert (keine CLI): paths.yaml-Defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()

from stage4_common import host_provenance, summarize  # noqa: E402


# ---------------------------------------------------------------------------
# Gemeinsame Kleinteile
# ---------------------------------------------------------------------------
def _csv(text):
    return [s.strip() for s in str(text).split(",") if s.strip()]


def _ints(text):
    return [int(s) for s in _csv(text)]


def _write(path, payload, indent=1):
    """Intermediate state straight to disk. An abort must not cost measured
    hours (lesson from sweep v2)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=indent)


def _repo_rel(path, what):
    """Repo-relative path — or abort. The dGeDi container mounts ONLY the repo
    (``.:/oscar``); a folder next to it is invisible to it and the run fails
    only inside the container with FileNotFoundError (2026-09-04)."""
    rel = os.path.relpath(os.path.abspath(path), _REPO)
    if rel.startswith(".."):
        sys.exit(f"[warm-dgedi] {what} lies outside the repo ({path}). The "
                 f"dGeDi container sees only the repo — pick a path under {_REPO} "
                 "(e.g. --work-dir, --caches-root).")
    return rel


def _check_counts(tag, counts, targets):
    for ds in targets:
        n = counts.get(ds, 0)
        assert n == TARGETS[ds], (f"{tag}: {ds} has {n} instead of "
                                  f"{TARGETS[ds]} objects")


# ---------------------------------------------------------------------------
# --step render — eine Blender-Sitzung je Datensatz x View-Zahl
# ---------------------------------------------------------------------------
def step_render(args) -> dict:
    from shutil import which
    targets, views = _csv(args.targets), _ints(args.views)
    # which() statt os.path.isfile(): der paths.yaml-Default ist der BLOSSE Name
    # "blender" (auf dem PATH), ein --blender-Wert dagegen ein absoluter Pfad.
    blender = which(args.blender)
    if blender is None:
        sys.exit(f"[warm-render] Blender not found: {args.blender} — "
                 "set --blender or export BLENDER_BIN (exactly 3.4.1; "
                 "3.3.1 has no PIL and fails silently).")
    args.blender = blender
    for ds in targets:
        cad_dir = os.path.join(_PATHS["cad_root"], ds)
        if not os.path.isdir(cad_dir):
            sys.exit(f"[warm-render] CAD folder missing: {cad_dir}")

    out = {"step": "render", "blender": args.blender,
           "provenance": host_provenance(), "sessions": {}}
    for V in views:
        for ds in targets:
            render_dir = os.path.join(args.work_dir, "render", f"v{V}", ds)
            timing = os.path.join(args.work_dir, "render",
                                  f"v{V}_{ds}_timing.json")
            os.makedirs(render_dir, exist_ok=True)
            env = dict(os.environ,
                       OBJECT_FOLDER=os.path.join(_PATHS["cad_root"], ds),
                       OBJECT_IMAGES=render_dir + "/",
                       NUM_VIEWS=str(V),
                       OVERWRITE_EXISTING="1",
                       RENDER_TIMING_JSON=timing)
            cmd = [args.blender, "-b", "-P",
                   os.path.join(_REPO, "preprocessing", "render_views.py")]
            print(f"[warm-render] v{V}/{ds} ...", flush=True)
            t0 = time.perf_counter()
            # Ein- und Ausgabeordner kommen ueber die Env-Variablen oben, beide
            # absolut; render_views.py bricht ab, wenn sie fehlen.
            r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               cwd=os.path.join(_REPO, "preprocessing"))
            wall = time.perf_counter() - t0
            # Blender beendet sich bei einem Python-Fehler mit rc=0; die
            # Timing-Datei ist der belastbare Erfolgsnachweis, nicht rc.
            if not os.path.isfile(timing):
                print(f"[warm-render] FAILED v{V}/{ds} "
                      f"(rc={r.returncode}):")
                print((r.stderr or r.stdout or "")[-500:])
                raise SystemExit(1)
            tj = json.load(open(timing))
            n = len(tj["per_object_s"])
            assert n == TARGETS[ds], (f"v{V}/{ds}: {n} instead of "
                                      f"{TARGETS[ds]} objects")
            sess = {"wall_s": wall, "script_total_s": tj["script_total_s"],
                    "blender_start_s": wall - tj["script_total_s"],
                    "per_object_s": tj["per_object_s"]}
            out["sessions"][f"v{V}_{ds}"] = sess
            print(f"[warm-render] v{V}/{ds}: {n} objects, wall {wall:.1f} s, "
                  f"start {sess['blender_start_s']:.2f} s", flush=True)
            _write(args.out, out)          # Persistenz nach jeder Sitzung
    return out


# ---------------------------------------------------------------------------
# --step describe — LLaVA einmal laden, dann je Objekt nur beschreiben
# ---------------------------------------------------------------------------
def _object_ids(ds):
    root = os.path.join(_PATHS["gallery_root"], ds)
    if not os.path.isdir(root):
        sys.exit(f"[warm-describe] gallery folder missing: {root}")
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d))
                  and not d.startswith("."))


def _view_files(ds, oid, num_views):
    """The first V views (FPS prefix = indices 0..V-1), then sorted
    lexicographically — exactly the pending order of generate_descriptions.py."""
    files = [f"{oid}_{i}.png" for i in range(num_views)]
    root = os.path.join(_PATHS["gallery_root"], ds, oid)
    missing = [f for f in files if not os.path.isfile(os.path.join(root, f))]
    if missing:
        raise FileNotFoundError(f"{ds}/{oid}: {len(missing)} views missing "
                                f"(e.g. {missing[0]})")
    return sorted(files)


def step_describe(args) -> dict:
    targets, views = _csv(args.targets), _ints(args.views)
    batch_sizes = _ints(args.batch_sizes)

    objs = [(ds, oid) for ds in targets for oid in _object_ids(ds)]
    counts = {ds: sum(1 for d, _ in objs if d == ds) for ds in targets}
    _check_counts("warm-describe", counts, targets)
    print(f"[warm-describe] {len(objs)} objects, views {views}, "
          f"batches {batch_sizes}", flush=True)

    # generate_captions ist die Produktionsfunktion — kein Nachbau.
    _prep = os.path.join(_REPO, "preprocessing")
    if _prep not in sys.path:
        sys.path.insert(0, _prep)

    import torch
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from generate_descriptions import generate_captions

    t0 = time.perf_counter()
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16,
        device_map="auto")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model_load_s = time.perf_counter() - t0
    print(f"[warm-describe] model loaded in {model_load_s:.2f} s", flush=True)

    # Ein Warm-up-Batch (2 Bilder), damit cuDNN-Autotuning nicht die erste
    # Objektmessung verfaelscht; nicht gewertet.
    ds0, oid0 = objs[0]
    root0 = os.path.join(_PATHS["gallery_root"], ds0, oid0)
    warm_imgs = [Image.open(os.path.join(root0, f)).convert("RGB")
                 for f in _view_files(ds0, oid0, 16)[:2]]
    generate_captions(model, processor, warm_imgs, args.prompt)
    print("[warm-describe] warm-up done", flush=True)

    out = {"step": "describe", "model_load_s": model_load_s,
           "prompt": args.prompt, "provenance": host_provenance(),
           "runs": {}, "captions": {}}
    for V in views:
        for bs in batch_sizes:
            key = f"v{V}_b{bs}"
            per_obj, caps = {}, {}
            for i, (ds, oid) in enumerate(objs):
                root = os.path.join(_PATHS["gallery_root"], ds, oid)
                files = _view_files(ds, oid, V)
                t1 = time.perf_counter()
                images = [Image.open(os.path.join(root, f)).convert("RGB")
                          for f in files]
                captions = []
                for k in range(0, len(images), bs):
                    captions += generate_captions(
                        model, processor, images[k:k + bs], args.prompt)
                per_obj[f"{ds}/{oid}"] = time.perf_counter() - t1
                caps[f"{ds}/{oid}"] = dict(zip(files, captions))
                if (i + 1) % 10 == 0 or i == len(objs) - 1:
                    print(f"[warm-describe] {key}: {i + 1}/{len(objs)} "
                          f"(last {per_obj[f'{ds}/{oid}']:.2f} s)",
                          flush=True)
                # Persistenz VOR jedem weiteren Schritt — ein Abbruch darf
                # keine gemessenen Stunden kosten (Lektion Sweep v2).
                out["runs"][key] = per_obj
                out["captions"][key] = caps
                _write(args.out, out, indent=None)

    _write(args.out, out, indent=None)
    return out


# ---------------------------------------------------------------------------
# --step dgedi — EIN Container-Lauf ueber alle 59 Ziel-CADs
# ---------------------------------------------------------------------------
def _target_manifest(targets):
    """``{"<ds>/<obj_id>": <repo-relative mesh path>}`` for the target CADs.

    Format and relativisation as in preprocessing/build_dgedi_manifest.py:
    precompute_dgedi.py prepends ``--repo-root`` (default ``/oscar``), so the
    manifest stays container-portable.
    """
    import glob as _glob
    manifest, counts = {}, {}
    for ds in targets:
        if ds not in TARGET_LAYOUT:
            sys.exit(f"[warm-dgedi] unknown target dataset {ds!r}; "
                     f"allowed: {sorted(TARGET_LAYOUT)}")
        pattern, mode = TARGET_LAYOUT[ds]
        hits = sorted(_glob.glob(os.path.join(_PATHS["cad_root"], pattern)))
        for p in hits:
            oid = (os.path.basename(os.path.dirname(p)) if mode == "parent"
                   else os.path.splitext(os.path.basename(p))[0])
            manifest[f"{ds}/{oid}"] = _repo_rel(p, f"target CAD {ds}/{oid}")
        counts[ds] = len(hits)
    return manifest, counts


def step_dgedi(args) -> dict:
    """Warm dGeDi descriptor time per object — ONE container run.

    Procedure taken over from the archived measurement
    (``results/stage4_latency/onboarding_dgedi_warm{,_manifest}.json``):
    build the manifest of the 59 target CADs,
    ONE ``docker compose run --rm --no-deps dgedi`` on precompute_dgedi.py with
    ``--overwrite --timing-json``, measure the wall time from the outside.
    ``model_load_s`` and ``per_object_s`` come from the timing json; the rest of
    the wall time is container startup + Python imports + teardown and is
    reported as ``container_python_overhead``, not spread over the objects.
    """
    targets = _csv(args.targets)
    manifest, counts = _target_manifest(targets)
    _check_counts("warm-dgedi", counts, targets)
    print(f"[warm-dgedi] manifest: {len(manifest)} target CADs {counts}",
          flush=True)

    work = os.path.join(args.work_dir, "dgedi")
    cache = os.path.join(work, "gedi_cache")
    man = os.path.join(work, "manifest.json")
    timing = os.path.join(work, "timings.json")
    os.makedirs(cache, exist_ok=True)
    _write(man, manifest, indent=1)

    # Alle drei Pfade muessen IM Repo liegen: der dgedi-Dienst mountet nur
    # `.:/oscar`, ein Ordner daneben ist fuer ihn unsichtbar.
    cmd = ["docker", "compose", "run", "--rm", "--no-deps", "dgedi",
           "python3", "/oscar/preprocessing/precompute_dgedi.py",
           "--manifest", "/oscar/" + _repo_rel(man, "dGeDi manifest"),
           "--out", "/oscar/" + _repo_rel(cache, "dGeDi descriptor folder"),
           "--timing-json", "/oscar/" + _repo_rel(timing, "dGeDi timing JSON"),
           "--overwrite"]
    print("[warm-dgedi] " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=_REPO)
    wall = time.perf_counter() - t0
    if not os.path.isfile(timing):
        sys.exit(f"[warm-dgedi] FAILED (rc={r.returncode}): no "
                 f"{timing}. Is the dgedi image (oscar-dgedi) running and is "
                 "../dGeDi present?")

    tj = json.load(open(timing))
    per_object = tj["per_object_s"]
    n_written = len([f for f in os.listdir(cache) if f.endswith(".npz")])
    all_s = list(per_object.values())
    per_ds = {}
    for ds in targets:
        vals = [v for k, v in per_object.items() if k.startswith(ds + "/")]
        if vals:
            per_ds[ds] = summarize(vals)

    payload = {
        "step": "dgedi",
        "description": "Warm dGeDi descriptor time per object across the "
                       "target CADs. ONE container run of "
                       "preprocessing/precompute_dgedi.py with --timing-json; "
                       "per object mesh loading + sampling + features + "
                       "npz save.",
        "n_objects": len(manifest),
        "per_dataset_counts": counts,
        "provenance": host_provenance(),
        "warm_per_object_s": summarize(all_s),
        "per_dataset_s": per_ds,
        "one_off_startup_s": {
            "model_load": tj.get("model_load_s"),
            "container_python_overhead": (wall - (tj.get("model_load_s") or 0.0)
                                          - sum(all_s)),
            "wall_total": wall,
            "description": "model_load = perf_counter around server.load_model; "
                           "container_python_overhead = wall time minus "
                           "model loading minus sum of the object times "
                           "(container startup, Python imports, teardown).",
        },
        "per_object_s": per_object,
        "work_dir": work,
    }
    if n_written != len(manifest) or len(per_object) != len(manifest):
        sys.exit(f"[warm-dgedi] ABORT: {len(manifest)} target CADs, but "
                 f"{n_written} descriptors / {len(per_object)} object times — "
                 f"incomplete run, nothing written.")
    _write(args.out, payload)
    return payload


# ---------------------------------------------------------------------------
def main():
    args = _ARGS
    steps = {"render": step_render, "describe": step_describe,
             "dgedi": step_dgedi}
    # Jede Stufe schreibt --out selbst und tut das fortlaufend (nach jeder
    # Blender-Sitzung bzw. nach jedem Objekt), damit ein Abbruch die bereits
    # gemessenen Stunden nicht kostet.
    result = steps[args.step](args)
    print(f"[warm-{args.step}] done -> {args.out}", flush=True)

    # Kurzer Kopf ins Log, damit ein Lauf ohne JSON-Lesen beurteilbar ist.
    if args.step == "dgedi":
        s = result["warm_per_object_s"]
        print(f"[warm-dgedi] per object: median {s['median']:.2f} s, "
              f"IQR {s['iqr']:.2f} s, p95 {s['p95']:.2f} s (n={s['n']})",
              flush=True)
    elif args.step == "render":
        for key, sess in result["sessions"].items():
            vals = list(sess["per_object_s"].values())
            s = summarize(vals)
            print(f"[warm-render] {key}: per object median {s['median']:.2f} s "
                  f"(n={s['n']}), Blender start {sess['blender_start_s']:.2f} s",
                  flush=True)
    else:
        for key, per_obj in result["runs"].items():
            s = summarize(list(per_obj.values()))
            print(f"[warm-describe] {key}: per object median "
                  f"{s['median']:.2f} s (n={s['n']})", flush=True)


if __name__ == "__main__":
    import datetime

    # Volle Lauf-Provenienz im Ausgabeordner (argv + git-Revision + Zeit),
    # geschrieben BEVOR die Messung startet.
    _outdir = os.path.dirname(os.path.abspath(_ARGS.out))
    os.makedirs(_outdir, exist_ok=True)
    try:
        _rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        _rev = ""
    json.dump({"argv": sys.argv[1:], "git": _rev, "step": _ARGS.step,
               "targets": _ARGS.targets, "views": _ARGS.views,
               "batch_sizes": _ARGS.batch_sizes, "blender": _ARGS.blender,
               "work_dir": _ARGS.work_dir, "out": _ARGS.out,
               "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
               "time": datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(_outdir, f"run_config_{_ARGS.step}.json"), "w"),
              indent=1)

    main()
