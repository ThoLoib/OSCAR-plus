"""
stage3_gallery.py
=================
Assemble a multi-dataset *union* gallery for the Stage-3 BOP evaluation.

Stage 3 retrieves against a gallery that spans several preprocessed datasets:

    3a (exact CAD available):  G_proxy  ∪  G_target,d
    3b (proxy only):           G_proxy

where ``G_proxy = GSO ∪ HouseCat6D ∪ ITODD`` (no curation) and
``G_target,d`` is the set of
evaluated target CADs of BOP dataset ``d`` (ycbv | tless | lmo).

``eval_common.build_pipeline`` only loads a *single* ``ref_dir``; the gallery
ids come from ``DINOReRanker._ref_embeddings`` (one dataset).  This module reuses
the already-loaded encoders and the per-dataset embedding caches written by
``preprocessing/precompute_embeddings.py`` — it loads each dataset in turn and merges the
three reference stores under **namespaced instance ids** ``"<ds>/<obj_id>"`` so
ids never collide across datasets.

Nothing is re-encoded: every dataset's DINO / CLIP-text / ULIP-partial embeddings
are pulled straight from disk cache (the loaders fall back to encoding only if a
cache is missing, which should not happen post-preprocessing).

Base retrieval channel only (``base`` pass = CLIP-text + DINOv2@42 + ULIP-2
partial), matching the frozen config; the ``uni3d`` / ``ulip_fullmesh`` ablation
channels are not merged here.
"""

import copy
import os
import sys

import torch

try:
    from eval_common import EvalConfig, build_pipeline
except ImportError:  # pragma: no cover
    from .eval_common import EvalConfig, build_pipeline


_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, ".."))


# ---------------------------------------------------------------------------
# Per-dataset layout. All paths are absolute, resolved from config/paths.yaml,
# so callers need no particular working directory.
# ---------------------------------------------------------------------------
# - ref_dir     : object_images/<ds>        (renders + partial .npz + caches)
# - desc_file   : object_database/<ds>/descriptions_attributes.json
# - mesh_glob   : the ulip_fullmesh glob (only used for _cad_paths fallback)
# - id_mode     : how the gallery obj_id maps from the mesh path (see
#                 preprocessing/precompute_embeddings.build_mesh_items)
# - pose_mesh   : native-scale mesh used at POSE time (Phase B/C):
#                   targets -> BOP models_eval (mm, BOP frame)
#                   proxies -> the original CAD (native units; see units_m)
# - units_m     : True if pose_mesh is in METRES (needs *1000 -> mm for BOP)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from pipeline import paths as _paths_mod   # noqa: E402

_P = _paths_mod.resolve()
_GALLERY, _CAD, _DATA = (_P["gallery_root"], _P["cad_root"], _P["datasets_root"])


def _desc(ds: str) -> str:
    return os.path.join(_CAD, ds, "descriptions_attributes.json")


DATASET_LAYOUT = {
    "ycbv": dict(
        ref_dir=os.path.join(_GALLERY, "ycbv"),
        desc_file=_desc("ycbv"),
        mesh_glob=os.path.join(_CAD, "ycbv/*/textured_simple.obj"),
        id_mode="parent",
        pose_mesh_dir=os.path.join(_DATA, "ycbv/models_eval"),  # obj_0000NN.ply, mm
        pose_mesh_pat="obj_{id6}.ply",
        units_m=False,
    ),
    "tless": dict(
        ref_dir=os.path.join(_GALLERY, "tless"),
        desc_file=_desc("tless"),
        mesh_glob=os.path.join(_CAD, "tless/*/model.ply"),
        id_mode="stem",
        pose_mesh_dir=os.path.join(_DATA, "tless/models_eval"),
        pose_mesh_pat="obj_{id6}.ply",
        units_m=False,
    ),
    "lmo": dict(
        ref_dir=os.path.join(_GALLERY, "lmo"),
        desc_file=_desc("lmo"),
        mesh_glob=os.path.join(_CAD, "lmo/*/model.ply"),
        id_mode="stem",
        pose_mesh_dir=os.path.join(_DATA, "lmo/models_eval"),
        pose_mesh_pat="obj_{id6}.ply",
        units_m=False,
    ),
    "gso": dict(
        ref_dir=os.path.join(_GALLERY, "gso"),
        desc_file=_desc("gso"),
        mesh_glob=os.path.join(_CAD, "gso/*/meshes/model.obj"),
        id_mode="grandparent",
        pose_mesh_dir=os.path.join(_CAD, "gso"),
        pose_mesh_pat="{oid}/meshes/model.obj",
        units_m=True,       # GSO meshes are in metres
    ),
    "housecat6d": dict(
        ref_dir=os.path.join(_GALLERY, "housecat6d"),
        desc_file=_desc("housecat6d"),
        mesh_glob=os.path.join(_CAD, "housecat6d/*/*.obj"),
        id_mode="stem",
        pose_mesh_dir=os.path.join(_CAD, "housecat6d"),
        pose_mesh_pat=None,   # resolved by glob (category subdir unknown from id)
        units_m=True,         # HouseCat6D meshes are in metres
    ),
    "itodd": dict(
        ref_dir=os.path.join(_GALLERY, "itodd"),
        desc_file=_desc("itodd"),
        mesh_glob=os.path.join(_CAD, "itodd/*/model.ply"),
        id_mode="stem",
        pose_mesh_dir=os.path.join(_CAD, "itodd"),
        pose_mesh_pat="{oid}/model.ply",
        units_m=False,        # ITODD (BOP) meshes are in millimetres
    ),
}

PROXY_DATASETS = ("gso", "housecat6d", "itodd")
TARGET_DATASETS = ("ycbv", "tless", "lmo")

# Shape-arm ablation: swap ULIP-2 -> Uni3D. Passed as ``extra_overrides`` to
# assemble_gallery so it flows into pipeline_overrides. The partial cache path
# is encoder-keyed (step5 _get_partial_cache_path), so the Uni3D partial gallery
# embeddings live in their OWN .ulip_partial_cache_* file and never collide with
# the ULIP-2 caches. The Uni3D checkpoint (uni3d-g) is mounted at /uni3d and its
# defaults come from PipelineConfig (uni3d_model_name / embed_dim / num_points).
# View aggregation (topk_softmax/5/0.5) is unchanged — only the encoder swaps.
UNI3D_OVERRIDES = {"shape_encoder": "uni3d"}


def namespaced_id(ds: str, obj_id: str) -> str:
    return f"{ds}/{obj_id}"


def split_id(nsid: str):
    ds, _, obj_id = nsid.partition("/")
    return ds, obj_id


# ---------------------------------------------------------------------------
# Gallery assembly
# ---------------------------------------------------------------------------

class UnionGallery:
    """Holds the assembled pipeline components plus id bookkeeping."""

    def __init__(self, config, clip_retr, dino_rer, fusion_mod, shape_m,
                 gallery_ids, id_to_pose_mesh, target_ds, proxy_ds,
                 eval_cfg=None):
        self.config = config          # PipelineConfig
        self.eval_cfg = eval_cfg      # EvalConfig (run_query needs it for S')
        self.clip_retr = clip_retr
        self.dino_rer = dino_rer
        self.fusion_mod = fusion_mod
        self.shape_m = shape_m
        self.gallery_ids = gallery_ids            # set of namespaced ids
        self.id_to_pose_mesh = id_to_pose_mesh    # nsid -> (path, units_m)
        self.target_ds = target_ds                # str or None
        self.proxy_ds = proxy_ds                  # tuple

    def components(self):
        """Return the 5-tuple that run_evaluation / run_query expect."""
        return (self.config, self.clip_retr, self.dino_rer,
                self.fusion_mod, self.shape_m)


def _pose_mesh_path(ds: str, obj_id: str):
    """Resolve the native-scale mesh used at pose time for a gallery id."""
    import glob as _glob
    lay = DATASET_LAYOUT[ds]
    base = os.path.join(_THIS, lay["pose_mesh_dir"])
    pat = lay["pose_mesh_pat"]
    if pat is None:
        # HouseCat6D: <cat>/<oid>.obj — category unknown from id, glob for it.
        hits = _glob.glob(os.path.join(base, "*", f"{obj_id}.obj"))
        path = hits[0] if hits else ""
    else:
        # obj ids like obj_000001 -> id6 = "000001"
        id6 = obj_id.replace("obj_", "") if obj_id.startswith("obj_") else obj_id
        path = os.path.join(base, pat.format(id6=id6, oid=obj_id))
    return (os.path.abspath(path), lay["units_m"])


def _base_cfg(ds: str, extra_overrides=None, weights=None,
              use_partial: bool = True, oscar_cascade: bool = False) -> EvalConfig:
    """EvalConfig seeded to one dataset, with the frozen base-pass overrides.

    ``weights`` = (w_clip, w_dino, w_ulip); default is the BASE (0.3,0.4,0.3).
    ``use_partial`` toggles partial-view vs full-mesh shape references
    (A4-transfer). ``oscar_cascade`` reproduces OSCAR's *actual* mechanism for
    the E5 baseline: CLIP-text threshold pruning (τ=0.37, top-20 fallback) then
    DINOv2 best-view re-rank within the shortlist, **no shape** — the driver
    ranks this by the ``oscar_maxview`` arm, so weights are unused here.
    """
    lay = DATASET_LAYOUT[ds]
    wc, wd, wu = weights if weights is not None else (0.3, 0.4, 0.3)
    prune = (dict(clip_prune_mode="threshold", clip_tau=0.37, clip_fallback_k=20)
             if oscar_cascade else {})
    overrides = {
        "num_views": 42,
        "dino_view_aggregation": "topk_softmax",
        "dino_view_topk": 5,
        "dino_view_temperature": 0.5,
        "ulip_view_aggregation": "topk_softmax",
        "ulip_view_topk": 5,
        "ulip_view_temperature": 0.5,
        # Mean-patch DINO pooling (Pulli), the MI3DOR-proven default — user
        # decision for Stage-3 too. NOTE the preprocessed gallery .dino_cache_*
        # are CLS-pooled (precompute used the global "cls" default; the "mean"
        # default was MI3DOR-driver scoped), so the FIRST assembly cache-misses
        # and re-encodes DINO from the renders, writing new *_mean caches that
        # then persist. Query DINO uses mean too, so gallery/query stay
        # consistent. (One-time ~15-20 min GPU encode, gso dominating.)
        "dino_pooling": "mean",
    }
    if extra_overrides:
        overrides.update(extra_overrides)
    return EvalConfig(
        ref_dir=lay["ref_dir"],
        desc_file=lay["desc_file"],
        cad_mesh_glob=lay["mesh_glob"],
        result_folder="results_stage3_tmp",
        clip_top_k=10 ** 6, dino_top_k=10 ** 6,
        ulip2_top_k=10 ** 6, fusion_top_k=10 ** 6,
        # Frozen Stage-1 E2 full-fusion weights (best_config.json). MUST be set
        # explicitly: EvalConfig defaults to 0/0.5/0.5 (CLIP off), which would
        # silently make Stage-3 a DINO+ULIP run, not full fusion (audit P0.1).
        weight_clip=wc, weight_dino=wd, weight_ulip=wu,
        ulip2_use_partial_views=use_partial,   # partial views (base) / full mesh (A4)
        pipeline_overrides=overrides,
        **prune,                               # OSCAR-cascade τ-pruning (E5 baseline)
    )


def _mesh_items(ds: str):
    """(obj_id, mesh_path) list for a dataset, using the correct id mode."""
    import glob as _glob
    lay = DATASET_LAYOUT[ds]
    paths = sorted(_glob.glob(os.path.join(_THIS, lay["mesh_glob"])))
    mode = lay["id_mode"]
    items = []
    for p in paths:
        if mode == "stem":
            oid = os.path.splitext(os.path.basename(p))[0]
        elif mode == "parent":
            oid = os.path.basename(os.path.dirname(p))
        elif mode == "grandparent":
            oid = os.path.basename(os.path.dirname(os.path.dirname(p)))
        else:
            raise ValueError(mode)
        items.append((oid, p))
    return items


# Wie die full-mesh-ID aus dem Mesh-Pfad zu bilden ist, damit sie mit den
# Gallery-IDs (= Render-Ordnernamen unter object_images/<ds>/) zusammenfaellt.
# BEWUSST getrennt von DATASET_LAYOUT["id_mode"]: das steuert _mesh_items und
# damit den cad_mesh_items-Fallback von build_pipeline — daran wird hier nichts
# geaendert, um die laufenden partial-Arme nicht zu stoeren.
#
# Warum ueberhaupt noetig (Bug, gefunden 2026-08-29 vor dem 3a_fullmesh-Lauf):
# load_cad_models -> _collect_mesh_items bildet die ID aus dem ORDNERNAMEN.
# Fuer obj_XXXXXX/model.ply stimmt das. HouseCat6D liegt aber als
# <kategorie>/<objekt>.obj vor, d.h. alle 199 Objekte kollabierten auf 12
# Kategorie-IDs, die auf KEINE Gallery-ID passen -> 199/1316 (15 %) der Gallery
# waeren ohne Shape-Embedding in den Lauf gegangen, ohne Fehlermeldung.
_FULLMESH_ID_MODE = {
    "ycbv": "parent",        # <obj_id>/textured_simple.obj
    "tless": "parent",       # <obj_id>/model.ply   (NICHT stem -> waere "model")
    "lmo": "parent",
    "itodd": "parent",
    "gso": "grandparent",    # <obj_id>/meshes/model.obj
    "housecat6d": "stem",    # <kategorie>/<obj_id>.obj
}


def _fullmesh_items(ds: str):
    """(gallery_id, mesh_path) for the full-mesh arm, ids = render dir names."""
    import glob as _glob
    lay = DATASET_LAYOUT[ds]
    mode = _FULLMESH_ID_MODE[ds]
    items = []
    for p in sorted(_glob.glob(os.path.join(_THIS, lay["mesh_glob"]))):
        if mode == "stem":
            oid = os.path.splitext(os.path.basename(p))[0]
        elif mode == "parent":
            oid = os.path.basename(os.path.dirname(p))
        else:
            oid = os.path.basename(os.path.dirname(os.path.dirname(p)))
        items.append((oid, p))
    return items


def _absorb_dataset(ds, clip_retr, dino_rer, shape_m, master, use_partial=True):
    """Load one dataset's cached stores and merge into the master dicts
    under namespaced ids. Reuses the already-loaded encoders.

    ``use_partial=False`` loads whole-mesh ULIP embeddings (A4 full-mesh
    transfer) via ``load_cad_models`` instead of the per-view partial cache.
    """
    lay = DATASET_LAYOUT[ds]
    ref_dir = os.path.join(_THIS, lay["ref_dir"])
    desc_file = os.path.join(_THIS, lay["desc_file"])

    # --- DINO (per-object per-view embeddings) ---
    dino_rer.load_reference_images(ref_dir=ref_dir)   # model reused, cache hit
    ds_gallery_ids = list(dino_rer._ref_embeddings.keys())
    for oid, views in dino_rer._ref_embeddings.items():
        master["dino"][namespaced_id(ds, oid)] = views

    # --- CLIP text (per-view description rows) ---
    clip_retr.load_descriptions(desc_file=desc_file)  # no id_to_label -> obj ids
    embs = clip_retr._desc_embeddings                 # (M, D) on device
    for i, (txt, lbl) in enumerate(zip(clip_retr._desc_texts,
                                       clip_retr._desc_labels)):
        master["clip_emb"].append(embs[i].detach().cpu())
        master["clip_txt"].append(txt)
        master["clip_lbl"].append(namespaced_id(ds, lbl))

    # --- ULIP-2 shape embeddings (partial-view base, or full-mesh for A4) ---
    if shape_m is not None:
        if use_partial:
            partial_items = shape_m._collect_partial_items(ref_dir)
            if partial_items:
                shape_m._partial_view_paths = dict(partial_items)
                cache_path = shape_m._get_partial_cache_path(ref_dir, partial_items)
                if shape_m._try_load_partial_cache(cache_path):
                    for oid, emb in shape_m._cad_embeddings.items():
                        nsid = namespaced_id(ds, oid)
                        master["cad_emb"][nsid] = emb.detach().cpu()
                        master["cad_path"][nsid] = shape_m._cad_paths.get(oid, "")
        else:
            # A4 full-mesh transfer: whole-mesh ULIP embeddings (cache or sample).
            # IDs kommen explizit aus _fullmesh_items, NICHT aus dem
            # ordnernamen-basierten _collect_mesh_items — siehe den Kommentar
            # bei _FULLMESH_ID_MODE (HouseCat6D-Kollaps, 2026-08-29).
            # cad_dir wird trotzdem gebraucht: er geht in den Cache-Pfad ein.
            # Basisordner = Glob bis zum ersten Wildcard; os.path.dirname() waere
            # falsch, bei ".../ycbv/*/textured_simple.obj" bliebe das "*" stehen
            # (FileNotFoundError, Fehler vom 2026-08-28).
            cad_dir = os.path.join(_THIS, lay["mesh_glob"].split("*")[0]).rstrip("/")
            items = _fullmesh_items(ds)
            shape_m.config.ulip2_use_partial_views = False
            shape_m.load_cad_models(cad_dir=cad_dir, mesh_items=items)

            # --- Deckungspruefung: JEDE Gallery-ID braucht ein Embedding. ---
            # Ohne diesen Riegel laeuft eine ID-Fehlzuordnung stumm durch: der
            # Arm liefert plausible Zahlen, denen ein Teil des Shape-Kanals
            # fehlt. Lieber hier hart abbrechen als 3 h spaeter Muell auswerten.
            gal = set(ds_gallery_ids)
            hit = gal & set(shape_m._cad_embeddings)
            n0 = len(master["cad_emb"])
            for oid, emb in shape_m._cad_embeddings.items():
                if oid not in gal:
                    continue            # bg/collision/models_orig etc.
                nsid = namespaced_id(ds, oid)
                master["cad_emb"][nsid] = emb.detach().cpu()
                master["cad_path"][nsid] = shape_m._cad_paths.get(oid, "")
            cov = len(hit) / max(len(gal), 1)
            print(f"[stage3][fullmesh] {ds}: absorbed "
                  f"{len(master['cad_emb'])-n0}/{len(gal)} full-mesh ULIP "
                  f"embeddings (coverage {100*cov:.1f}%)")
            if cov < 0.95:
                raise RuntimeError(
                    f"[stage3][fullmesh] {ds}: only {100*cov:.1f}% of the gallery ids "
                    f"have a full-mesh embedding ({len(hit)}/{len(gal)}). "
                    f"Missing e.g. {sorted(gal - hit)[:5]}. "
                    f"_FULLMESH_ID_MODE['{ds}'] does not match the mesh layout.")

    # --- pose-mesh map (native scale) keyed by the GALLERY ids (render-dir
    # names = obj_XXXXXX / gso ids), NOT the fullmesh glob stem (which is
    # "model" for BOP */model.ply layouts). ---
    for oid in ds_gallery_ids:
        nsid = namespaced_id(ds, oid)
        master["pose_mesh"][nsid] = _pose_mesh_path(ds, oid)


def assemble_gallery(target_datasets=(), proxy_ds=PROXY_DATASETS,
                     extra_overrides=None, weights=None, use_partial=True,
                     oscar_cascade=False):
    """Build the union gallery = G_proxy ∪ (exact CADs of each target dataset).

    3a (one big DB): assemble_gallery(TARGET_DATASETS) -> proxies + ALL targets;
        every query dataset retrieves against this single combined gallery.
    3b:              assemble_gallery(())              -> G_proxy only.

    ``oscar_cascade`` = E5 OSCAR baseline (τ-prune + DINO cascade, no shape);
    ``use_partial=False`` = A4 full-mesh shape references.
    """
    target_datasets = tuple(d for d in target_datasets)
    # targets first so the encoders/models are constructed once against a target
    datasets = list(target_datasets) + [d for d in proxy_ds
                                        if d not in target_datasets]

    seed_ds = datasets[0]
    cfg = _base_cfg(seed_ds, extra_overrides, weights=weights,
                    use_partial=use_partial, oscar_cascade=oscar_cascade)
    # Build components ONCE (loads CLIP/DINO/ULIP models + seed dataset caches).
    config, clip_retr, dino_rer, fusion_mod, shape_m = build_pipeline(
        cfg, cad_mesh_items=_mesh_items(seed_ds))

    master = {"dino": {}, "clip_emb": [], "clip_txt": [], "clip_lbl": [],
              "cad_emb": {}, "cad_path": {}, "pose_mesh": {}}
    for ds in datasets:
        _absorb_dataset(ds, clip_retr, dino_rer, shape_m, master,
                        use_partial=use_partial)

    # --- write merged stores back into the (single) component objects ---
    dino_rer._ref_embeddings = master["dino"]
    dino_rer._apply_view_limit()

    if master["clip_emb"]:
        clip_retr._desc_embeddings = torch.stack(master["clip_emb"]).to(
            clip_retr.device)
        clip_retr._desc_texts = master["clip_txt"]
        clip_retr._desc_labels = master["clip_lbl"]

    if shape_m is not None and master["cad_emb"]:
        shape_m._cad_embeddings = master["cad_emb"]
        shape_m._cad_paths = master["cad_path"]
        shape_m._partial_mode = True

    gallery_ids = set(master["dino"].keys())
    return UnionGallery(
        config, clip_retr, dino_rer, fusion_mod, shape_m,
        gallery_ids=gallery_ids,
        id_to_pose_mesh=master["pose_mesh"],
        target_ds=target_datasets,          # tuple of included target datasets
        proxy_ds=tuple(proxy_ds),
        eval_cfg=cfg,
    )


if __name__ == "__main__":
    # Quick self-test: assemble a 3b (proxy-only) gallery and print sizes.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="",
                    help="comma-separated target datasets to include "
                         "(empty = 3b proxies only; 'all' = 3a big DB)")
    ap.add_argument("--proxy", default=",".join(PROXY_DATASETS),
                    help="comma-separated proxy datasets (subset for quick tests)")
    args = ap.parse_args()
    _proxy = tuple(p.strip() for p in args.proxy.split(",") if p.strip())
    if args.targets == "all":
        _targets = TARGET_DATASETS
    else:
        _targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    g = assemble_gallery(_targets, proxy_ds=_proxy)
    from collections import Counter
    by_ds = Counter(split_id(i)[0] for i in g.gallery_ids)
    print(f"[stage3_gallery] union |gallery| = {len(g.gallery_ids)}  by-dataset={dict(by_ds)}")
    print(f"[stage3_gallery] CLIP rows={len(g.clip_retr._desc_labels)}  "
          f"DINO objs={len(g.dino_rer._ref_embeddings)}  "
          f"ULIP objs={len(g.shape_m._cad_embeddings) if g.shape_m else 0}")
    missing = [i for i, (p, _) in g.id_to_pose_mesh.items() if not os.path.isfile(p)]
    print(f"[stage3_gallery] pose-mesh resolved for "
          f"{len(g.id_to_pose_mesh)-len(missing)}/{len(g.id_to_pose_mesh)}; "
          f"missing={len(missing)}")
    if missing[:5]:
        print("  e.g. missing:", missing[:5])
