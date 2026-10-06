#!/usr/bin/env python3
"""
Stage 4b — query latency: how long does a request take, up to the pose?

Question
--------
A user names an object ("the mayonnaise tube"), the camera delivers RGB-D.
How long does it take until a usable 6D pose is out, and which step costs?

Measured chain — every step on its own
    io_load        read RGB + depth from disk, decode
    segment_box    GroundingDINO: box from the language prompt
    segment_mask   SAM2.1: mask from the box (incl. post-processing)
    pointcloud     backproject the depth and cut it with the mask
    encode_query   ULIP-2 over the query point cloud
    clip           S_text  against the gallery descriptions
    dino           S_view  against the renderings per gallery object
    ulip           S_shape against the partial clouds
    fusion         weighted sum of the three channels
    geometry       GeDi descriptors + RANSAC over the top K  (only --geometry)
    pose           FoundationPose on the top-1 CAD           (unless --no-pose)

Cold and warm are reported SEPARATELY. Loading the models once costs a multiple
of one query; a number that mixes both only depends on how many queries were
averaged and says nothing about the system.

View count
----------
``--views 16,42`` measures the same chain with different numbers of views per
gallery object. That is cheap because the embeddings are always cached for all
42 views and ``_apply_view_limit()`` only filters — nothing is re-encoded.
Stage 1 (SHREC'18, O4) measures the quality side for it:
V8 0.5714 | V16 0.5820 | V32 0.5800 | V42 0.5868 nDCG.

Gallery
-------
As in Stage 3: G_proxy + target CADs = 1316 objects. With ``--proxy-only`` it
runs against the pure 3b database (1257) — the case where the exact model is
missing and a proxy has to be found.

Paths come from ``config/paths.yaml`` (overridable per flag, see
``--help``). Outside the container the script wraps itself in
``docker compose run``; with ``--geometry`` it is checked beforehand that the
dgedi service is running with the BOP gallery.

Examples
--------
    # Quick test without pose
    python3 experiments/stage4_latency.py --dataset ycbv \\
        --n-queries 5 --no-pose

    # Complete, 16 against 42 views
    python3 experiments/stage4_latency.py --dataset ycbv \\
        --n-queries 50 --views 16,42

    # With geometric re-ranking at K=5
    python3 experiments/stage4_latency.py --dataset lmo \\
        --n-queries 30 --geometry --geo-k 5

The result lands under ``<runs_root>/stage4_latency/`` (name derived from
--dataset and the switches that were set, unless --out is given); a
``run_config.json`` with argv, git revision and timestamp is written next to
it beforehand.
"""
from __future__ import annotations

import json
import os
import sys

# ---------------------------------------------------------------------------
# CLI-Prelude — VOR den schweren Imports (numpy/PIL/eval_common -> torch,
# open3d) und vor dem os.chdir(), damit `--help` auf dem Host ohne die
# Container-Abhaengigkeiten laeuft.
# ---------------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
sys.path.insert(0, _THIS)
for p in (_REPO, os.path.join(_REPO, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

STAGE1_QUALITY = {8: 0.5714, 16: 0.5820, 32: 0.5800, 42: 0.5868}

_ARGS = None
_PATHS = None


def _check_dgedi(expected_n, cache_hint):
    """The dGeDi service must be running AND have the right gallery loaded
    (identical to stage3_bop._check_dgedi)."""
    import urllib.request
    start_cmd = (f"DGEDI_CACHE_DIR={cache_hint} docker compose up -d "
                 "--force-recreate dgedi")
    for url in ("http://localhost:5061/health", "http://dgedi:5061/health"):
        try:
            h = json.load(urllib.request.urlopen(url, timeout=5))
        except Exception:
            continue
        n = h.get("n_gallery", -1)
        if n == expected_n:
            print(f"[stage4] dGeDi ok: n_gallery={n}", flush=True)
            return
        sys.exit(f"[stage4] dGeDi is running, but has n_gallery={n} instead of "
                 f"{expected_n}. Switch over with: {start_cmd}   — then start "
                 "again.")
    sys.exit(f"[stage4] dGeDi service unreachable. Start: {start_cmd}")


if __name__ == "__main__":
    import argparse

    from pipeline import paths as _paths_mod

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ycbv", choices=["ycbv", "tless", "lmo"])
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2,
                    help="Discarded warmup runs (CUDA kernels, allocator).")
    ap.add_argument("--views", default="42",
                    help="Comma list, e.g. '16,42'. Only filters the cache, "
                         "re-encodes nothing.")
    ap.add_argument("--shape-source", choices=["partial", "fullmesh"],
                    default="partial",
                    help="Gallery representation of the shape channel. 'partial' "
                         "(default) compares against N view embeddings per object, "
                         "'fullmesh' against ONE. Careful when interpreting this: "
                         "the two branches are implemented differently "
                         "(step5_shape_matching.py:1490 loop vs :1513 "
                         "vectorised), so the time difference is partly "
                         "implementation and not representation.")
    ap.add_argument("--proxy-only", action="store_true",
                    help="Gallery without the target CADs (3b case, proxy needed).")
    ap.add_argument("--geometry", action="store_true",
                    help="Also measure geometric re-ranking.")
    ap.add_argument("--geo-k", type=int, default=5)
    ap.add_argument("--no-pose", action="store_true",
                    help="Without FoundationPose (retrieval latency only).")
    ap.add_argument("--refine-iter", type=int, default=5,
                    help="FoundationPose refinement steps (Stage-3 default: 5).")
    ap.add_argument("--out", default="",
                    help="Result file (default: <runs_root>/stage4_latency/"
                         "query_latency_<dataset>[_fullmesh][_geo|_geo_pose]"
                         ".json); relative paths against the repo root.")
    ap.add_argument("--no-docker", action="store_true",
                    help="do not wrap into the oscar-plus container automatically")
    _paths_mod.add_path_args(ap)
    args = ap.parse_args()

    # Wie in stage2/stage3: Hash-Seed unabhaengig von --seed auf 0 festnageln.
    os.environ["PYTHONHASHSEED"] = "0"
    _PATHS = _paths_mod.from_args(args)

    if not args.out:
        # Schema: query_latency[_<dataset>][_fullmesh][_geo|_geo_pose].json —
        # genau die Namen der eingefrorenen Stage-4-Messungen, damit ein neuer
        # Lauf unmittelbar neben seinem Vorbild liegt. ycbv ist der berichtete
        # Datensatz und bleibt deshalb ohne Suffix (wie results/).
        _name = ("query_latency" if args.dataset == "ycbv"
                 else f"query_latency_{args.dataset}")
        if args.shape_source == "fullmesh":
            _name += "_fullmesh"
        if args.geometry:
            _name += "_geo" if args.no_pose else "_geo_pose"
        args.out = os.path.join(_PATHS["runs_root"], "stage4_latency",
                                _name + ".json")
    elif not os.path.isabs(args.out):
        # Relative --out gegen die Repo-Wurzel aufloesen, unabhaengig davon,
        # von wo aus der Aufruf kommt.
        args.out = os.path.join(_REPO, args.out)

    # --geometry: sofort abbrechen, wenn der dgedi-Dienst fehlt oder die falsche
    # Gallery haelt (laeuft auf dem Host VOR dem Wrappen und im Container
    # nochmals; die Union-Gallery G_proxy ∪ G_target hat 1316 Eintraege).
    if args.geometry:
        _check_dgedi(1316, "caches/dgedi/bop")

    if not os.path.exists("/.dockerenv") and not args.no_docker:
        if not os.path.isfile(os.path.join(_REPO, "docker-compose.yml")):
            sys.exit(f"[stage4] {_REPO}/docker-compose.yml missing — "
                     "incomplete repo?")
        print("[stage4] runs in the oscar-plus container — wrapping automatically.",
              flush=True)
        os.chdir(_REPO)  # docker compose needs the repo as CWD
        os.execvp("docker", ["docker", "compose", "run", "--rm", "oscar-plus",
                             "python3", "/app/experiments/stage4_latency.py"]
                  + sys.argv[1:])

    _ARGS = args

if _PATHS is None:  # als Modul importiert (keine CLI): paths.yaml-Defaults
    from pipeline import paths as _paths_mod
    _PATHS = _paths_mod.resolve()

from stage4_common import (Timings, aggregate, host_provenance,  # noqa: E402
                           print_table, summarize, write_results)


def wrap_timed(obj, method_name: str, label: str, sink: dict) -> bool:
    """Wrap a method so that its duration lands in `sink['timings']`.

    Deliberately here instead of in the pipeline code: ``run_query`` runs all
    channels in one pass, and the modules shared by all stages should not be
    touched for a measurement experiment.
    """
    fn = getattr(obj, method_name, None)
    if fn is None or getattr(fn, "_stage4_wrapped", False):
        return False

    def wrapper(*a, **kw):
        with sink["timings"].measure(label):
            return fn(*a, **kw)

    wrapper._stage4_wrapped = True                    # type: ignore[attr-defined]
    setattr(obj, method_name, wrapper)
    return True


def scene_camera(test_root, scene, im_id):
    """K and depth_scale per shot — BOP stores both per scene."""
    with open(os.path.join(test_root, scene, "scene_camera.json")) as fh:
        cams = json.load(fh)
    cam = cams[str(int(im_id))]
    import numpy as np
    return (np.array(cam["cam_K"], float).reshape(3, 3),
            float(cam.get("depth_scale", 1.0)))


# LLaVA beginnt praktisch jede Beschreibung mit derselben Floskel. Fuer CLIP ist
# das harmlos, fuer eine Detektion nicht: GroundingDINO sucht nach dem Substantiv
# im Prompt, und "image" ist im Bild nun mal nicht zu finden.
_LLAVA_OPENERS = (
    "the image features a ", "the image features an ", "the image features ",
    "the image shows a ", "the image shows an ", "the image shows ",
    "the image depicts a ", "the image depicts an ", "the image depicts ",
    "this image features ", "in the image, there is a ", "in the image, there is ",
    "the object in the image is a ", "the object in the image is an ",
    "the object in the image is ",
)


def prompt_for(dataset, obj_id, mode="phrase", _cache={}):
    """Language prompt from the stored description of the target object.

    Deliberately the same source as the text channel: a hand-written prompt
    would make the segmentation better or worse than the system could in
    operation, and would mix the latency measurement with a quality question.

    The file comes as {obj_id: {"image_descriptions": {image: text}}}.
    ``mode="phrase"`` takes the FIRST SENTENCE without the LLaVA boilerplate —
    GroundingDINO expects a short noun phrase, a 300-character paragraph runs
    into the token limit of the text encoder and detects worse. ``mode="full"``
    passes the whole description through, as a control.
    """
    if dataset not in _cache:
        f = os.path.join(_PATHS["cad_root"], dataset,
                         "descriptions_attributes.json")
        _cache[dataset] = json.load(open(f)) if os.path.isfile(f) else {}
    entry = _cache[dataset].get(obj_id) or {}

    text = ""
    if isinstance(entry, dict):
        imgs = entry.get("image_descriptions")
        if isinstance(imgs, dict) and imgs:
            text = str(next(iter(imgs.values())))
        else:
            for k in ("description", "descriptions", "caption", "text"):
                v = entry.get(k)
                if isinstance(v, str) and v:
                    text = v
                    break
                if isinstance(v, list) and v:
                    text = str(v[0])
                    break
    elif isinstance(entry, list) and entry:
        text = str(entry[0])

    if not text:
        return f"object {obj_id}"
    if mode == "full":
        return text

    sentence = text.split(".")[0].strip()
    low = sentence.lower()
    for opener in _LLAVA_OPENERS:
        if low.startswith(opener):
            sentence = sentence[len(opener):].strip()
            break
    return sentence or text


def run_one(tgt, dataset, test_root, gallery, localizer, args, sink) -> dict:
    """One query from disk to the pose. Raises nothing — errors land in the
    record, so a single failure does not end the measurement series."""
    import numpy as np
    from PIL import Image

    from stage3_bop import FP_URL
    from query_cloud import backproject_masked
    from eval_common import run_query
    from pipeline.foundationpose_bridge import call_foundationpose

    t = sink["timings"]
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = gallery.components()
    rec = {"scene": tgt["scene_id"], "im": tgt["im_id"], "obj": tgt["obj_id"]}

    scene, im = f"{tgt['scene_id']:06d}", f"{tgt['im_id']:06d}"
    with t.measure("io_load"):
        rgb_p = os.path.join(test_root, scene, "rgb", im + ".png")
        if not os.path.isfile(rgb_p):
            rgb_p = os.path.join(test_root, scene, "rgb", im + ".jpg")
        rgb = Image.open(rgb_p).convert("RGB")
        depth_raw = np.array(Image.open(os.path.join(
            test_root, scene, "depth", im + ".png")), dtype=np.float32)
        K, dscale = scene_camera(test_root, scene, im)

    prompt = prompt_for(dataset, f"obj_{tgt['obj_id']:06d}")
    rec["prompt"] = prompt[:80]

    with t.measure("segment"):
        loc = localizer.localize(rgb, prompt, top_k=1)
    if loc is None:
        rec["error"] = "no detection"
        return rec

    with t.measure("pointcloud"):
        depth_m = depth_raw * dscale / 1000.0
        cloud, colors = backproject_masked(
            depth_m, np.asarray(loc.mask), K, rgb=np.asarray(rgb))

    ulip_q = None
    if cloud is not None and len(cloud):
        with t.measure("encode_query"):
            ulip_q = shape_m.encode_pointcloud(cloud, colors=colors)

    with t.measure("retrieval_total"):
        out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m,
                        loc.roi_image, gallery.eval_cfg, ulip_query_emb=ulip_q)

    if args.geometry and cloud is not None and len(cloud):
        # GENAU der Pfad aus stage3_bop (3a/3b-Retrieval): der dGeDi-DIENST ueber
        # dgedi_rerank, mit denselben Repo-Parametern. Der erste Entwurf rief
        # GeometryReRanker(pcfg).rerank(out, ...) — eine andere Implementierung
        # (lokales GeDi), mit falschen Argumenten (erwartet List[FusedCandidate]
        # und eine Open3D-Wolke) und pro Query neu konstruiert. Das waere sofort
        # gescheitert und haette, wenn nicht, eine Latenz fuer einen Pfad
        # gemessen, den die Evaluation nie benutzt.
        from dgedi_bridge import dgedi_rerank
        from eval_common import fusion_ranking as _fr
        cand_ids = [oid for oid, _ in _fr(out["fused_full"])[:args.geo_k]]
        with t.measure("geometry"):
            geo = dgedi_rerank(cloud, cand_ids, ransac_keypoints=6000,
                               ransac_max_iter=10000, use_icp=True)
        rec["geo_ok"] = sum(1 for v in (geo or {}).values() if v.get("ok"))
        rec["geo_requested"] = len(cand_ids)

    if not args.no_pose:
        top1 = _top1_id(out)
        rec["top1"] = top1
        entry = (gallery.id_to_pose_mesh or {}).get(top1)
        if entry:
            path, units_m = (entry if isinstance(entry, (tuple, list))
                             else (entry, False))
            # Genau wie Stage 3: FoundationPose rechnet in Metern. Meshes in
            # Millimetern (BOP, ITODD) bekommen scale=0.001, Meshes in Metern
            # (GSO, HouseCat6D) scale=1.0.
            with t.measure("pose"):
                call_foundationpose(FP_URL, rgb=np.asarray(rgb), depth=depth_m,
                                    mask=np.asarray(loc.mask), K=K,
                                    cad_path=path,
                                    scale=1.0 if units_m else 0.001,
                                    refine_iter=args.refine_iter)
        else:
            rec["pose_skipped"] = f"no pose mesh for {top1}"
    return rec


# `retrieval_total` umschliesst clip/dino/ulip/fusion — es ist eine Klammer um
# bereits gemessene Schritte, keine eigene Arbeit. Waere es in der Summe, zaehlte
# die Retrieval-Zeit doppelt (im Smoke-Test 1.10 s echte Arbeit vs 1.90 s Summe).
# Es bleibt als eigene Zeile stehen: die Differenz zu den vier Kanaelen ist der
# Overhead von run_query selbst (~1 ms, also praktisch keiner).
_CONTAINER_STEPS = {"retrieval_total"}


def _wall(timings: dict) -> float:
    """True end-to-end time: do not count the bracket measurements."""
    return sum(sum(v) for k, v in timings.items() if k not in _CONTAINER_STEPS)


def _top1_id(out):
    """Top-1 of the full fusion — the same path as stage3_bop.

    `run_query` does not return a flat ranking but the raw results of all arms;
    Stage 3 derives `fusion_ranking(out["fused_full"])` from them. The first
    draft here guessed key names and got None throughout, whereupon the pose
    step was silently skipped.
    """
    from eval_common import fusion_ranking
    ranking = fusion_ranking(out["fused_full"])
    return ranking[0][0] if ranking else None


def _bop_query_paths(dataset):
    """(test_root, targets_json) of ONE BOP dataset, anchored to the
    ``--datasets-root`` of THIS run.

    stage3_bop.DATASET_TEST remains the only source for the layout (ycbv/test,
    tless/test_primesense, lmo/test + test_targets_bop19.json); stage3_bop
    resolves the paths.yaml defaults at import time, however, and does not know
    the path flags of this driver — which is why only the root is reset and the
    layout is not copied.
    """
    from stage3_bop import DATASET_TEST, _PATHS as _S3_PATHS
    cfg = DATASET_TEST[dataset]
    return tuple(
        os.path.join(_PATHS["datasets_root"],
                     os.path.relpath(p, _S3_PATHS["datasets_root"]))
        for p in (cfg["test_root"], cfg["targets"]))


def main():
    args = _ARGS

    import random

    from stage3_bop import load_bop_targets
    from stage3_gallery import PROXY_DATASETS, assemble_gallery

    views = [int(v) for v in args.views.split(",") if v.strip()]
    cold = Timings()

    print("[stage4] loading gallery and models ...")
    with cold.measure("gallery_assembly"):
        gallery = assemble_gallery(
            target_datasets=() if args.proxy_only else (args.dataset,),
            proxy_ds=PROXY_DATASETS,
            use_partial=(args.shape_source == "partial"))
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = gallery.components()

    with cold.measure("load_groundingdino_sam"):
        from pipeline.step1_localization import ObjectLocalizer
        localizer = ObjectLocalizer(pcfg)
        # ObjectLocalizer laedt LAZY (erst in localize()). Ohne diesen Aufruf
        # landen ~25 s Modell-Ladezeit in der ERSTEN Query und verfaelschen
        # sowohl den Kaltstart- als auch den Warm-Median.
        localizer._load_model()

    if args.geometry:
        # dGeDi laeuft ebenfalls als Dienst. Erreichbarkeit und Gallery-Groesse
        # VOR der Messung pruefen: eine falsche Gallery hat am 2026-08-28 einen
        # 17-Stunden-Leerlauf verursacht, in dem keine Registrierung gelang.
        with cold.measure("check_dgedi"):
            from dgedi_bridge import dgedi_health
            h = dgedi_health()
            # dgedi_health() liefert das Health-Dict ODER None bei
            # Unerreichbarkeit — es gibt KEINEN "ok"-Schluessel. Ein
            # h.get("ok") schlaegt deshalb auch dann Alarm, wenn der Dienst
            # laeuft (beobachtet 2026-09-01). Massgeblich ist n_gallery.
            n_gal = (h or {}).get("n_gallery", 0)
            print(f"[stage4] dGeDi n_gallery={n_gal}")
            if not n_gal:
                print("[stage4] WARNING: dGeDi unreachable or empty "
                      "gallery — the geometry step will measure nothing.")

    if not args.no_pose:
        # FoundationPose laeuft als eigener Dienst; "laden" heisst hier
        # Erreichbarkeit pruefen. Ohne den Test faellt ein toter Dienst erst in
        # der ersten Query auf, und zwar als Latenz statt als Fehler.
        with cold.measure("check_foundationpose"):
            import urllib.request
            try:
                urllib.request.urlopen("http://foundationpose:5050/health",
                                       timeout=10).read()
            except Exception as exc:
                print(f"[stage4] WARNING: FoundationPose unreachable "
                      f"({exc}) — the pose step will fail.")

    print(f"[stage4] |Gallery| = {len(gallery.gallery_ids)}, Views: {views}")

    sink = {"timings": Timings()}
    instrumented = [lbl for obj, meth, lbl in [
        (clip_retr, "retrieve", "clip"), (dino_rer, "rerank", "dino"),
        (shape_m, "match", "ulip"), (fusion_mod, "fuse", "fusion")]
        if wrap_timed(obj, meth, lbl, sink)]
    print(f"[stage4] instrumented channels: {instrumented}")

    test_root, targets_json = _bop_query_paths(args.dataset)
    targets = load_bop_targets(targets_json)
    random.Random(args.seed).shuffle(targets)
    targets = targets[:args.n_queries + args.warmup]

    by_views, records = {}, []
    # ABSTEIGEND. _apply_view_limit() ERSETZT self._ref_embeddings durch die
    # getrimmte Fassung — der Schnitt ist destruktiv und nicht umkehrbar. In
    # aufsteigender Reihenfolge liefe der 42er-Durchgang auf den 16 Views, die
    # der vorige uebrig gelassen hat, und beide Zeilen waeren dieselbe Messung
    # (im ersten Lauf: 0.866 s gegen 0.856 s, also scheinbar kein Unterschied).
    for V in sorted(views, reverse=True):
        print(f"\n[stage4] --- {V} views ---")
        pcfg.num_views = V
        if hasattr(dino_rer, "_apply_view_limit"):
            dino_rer.config.num_views = V
            dino_rer._apply_view_limit()
        # Der Shape-Kanal hat kein Gegenstueck zu _apply_view_limit: seine
        # Gallery-Embeddings sind (42, D) je Objekt und die top-k-softmax laeuft
        # ueber alle. Ohne diesen Schnitt bliebe ULIP bei 42 Views, waehrend DINO
        # auf V faellt — der Vergleich waere nur halb durchgefuehrt.
        if shape_m is not None and getattr(shape_m, "_cad_embeddings", None):
            shape_m._cad_embeddings = {
                oid: (emb[:V] if getattr(emb, "ndim", 1) == 2 else emb)
                for oid, emb in shape_m._cad_embeddings.items()}
        per_query = []
        for i, tgt in enumerate(targets):
            is_warmup = i < args.warmup
            sink["timings"] = Timings()
            try:
                rec = run_one(tgt, args.dataset, test_root, gallery, localizer,
                              args, sink)
            except Exception as exc:
                rec = {"error": f"{type(exc).__name__}: {exc}"}
            # total_s ueber _wall, nicht Timings.total(): letzteres zaehlte die
            # Klammer retrieval_total DOPPELT (Records des Laufs vom 04.09.:
            # 16-V-Median 2.70 s statt 2.18 s; die Zusammenfassung
            # per_query_total_s war davon nie betroffen).
            rec.update(num_views=V, warmup=is_warmup,
                       timings=sink["timings"].as_dict(),
                       total_s=_wall(sink["timings"].as_dict()))
            records.append(rec)
            if not is_warmup and "error" not in rec:
                per_query.append(rec["timings"])
            note = "  (warmup)" if is_warmup else ""
            note += "  " + rec["error"] if "error" in rec else ""
            print(f"  [{i + 1}/{len(targets)}] {rec['total_s']:6.2f} s{note}")
        by_views[V] = {
            "per_step": aggregate(per_query),
            "per_query_total_s": summarize([_wall(q) for q in per_query]),
            "n_no_detection": sum(1 for r in records
                                  if r.get("num_views") == V
                                  and not r.get("warmup")
                                  and r.get("error") == "no detection"),
        }

    payload = {
        "experiment": "stage4b_query_latency",
        "dataset": args.dataset,
        "gallery_size": len(gallery.gallery_ids),
        "proxy_only": args.proxy_only,
        "geometry": args.geometry, "geo_k": args.geo_k,
        "pose": not args.no_pose,
        "views": views, "n_warmup": args.warmup,
        "provenance": host_provenance(),
        "cold_start_s": {k: v[0] for k, v in cold.as_dict().items()},
        "by_views": by_views,
        "stage1_quality_ndcg": {str(k): v for k, v in STAGE1_QUALITY.items()},
        "records": records,
    }

    for V in views:
        print_table(f"Query latency (warm) — {V} views", by_views[V]["per_step"],
                    total_key="retrieval_total")
        tot = by_views[V]["per_query_total_s"]
        if tot.get("n"):
            print(f"\n  End to end: median {tot['median']:.2f} s, "
                  f"IQR {tot['iqr']:.2f} s, p95 {tot['p95']:.2f} s  (n={tot['n']})")

    if len(views) > 1:
        ref = max(views)
        ref_c = by_views[ref]["per_query_total_s"].get("median", 0.0)
        print("\n=== Cost-benefit: views (query side) ===")
        print(f"  {'Views':>6}{'Latency (median)':>19}{'Cost':>10}"
              f"{'nDCG (Stage 1)':>16}")
        for V in sorted(views):
            c = by_views[V]["per_query_total_s"].get("median", 0.0)
            q = STAGE1_QUALITY.get(V)
            cs = f"{100 * c / ref_c:6.0f}%" if ref_c else "     —"
            print(f"  {V:>6}{c:>16.3f} s{cs:>10}"
                  f"{(f'{q:.4f}' if q else '—'):>16}")

    print("\n  Cold start, one-off (NOT part of the query latency):")
    for k, v in payload["cold_start_s"].items():
        print(f"    {k:<28}{v:8.2f} s")
    write_results(args.out, payload)


if __name__ == "__main__":
    import datetime
    import subprocess

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
               "out": _ARGS.out,
               "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
               "time": datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(_outdir, "run_config.json"), "w"), indent=1)

    main()
