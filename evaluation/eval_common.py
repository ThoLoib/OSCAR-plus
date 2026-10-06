"""
eval_common.py
==============
Shared evaluation infrastructure for OSCAR+ retrieval experiments.

Dataset-specific scripts provide:
  - An EvalConfig with paths and knobs
  - A to_label_fn(object_id) -> label function
  - A query_factory(k) -> iterator of (roi, gt_label, img_path, category, fname)

This module provides everything else: pipeline init, query processing,
metrics, CSV streaming, and JSON summaries.
"""

import glob as _glob
import json
import os
import sys
from dataclasses import dataclass, field
from math import log2
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OSCAR_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _OSCAR_ROOT not in sys.path:
    sys.path.insert(0, _OSCAR_ROOT)

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.step2_pointcloud import PointCloudResult  # noqa: E402
from pipeline.step3_clip_retrieval import CLIPRetriever  # noqa: E402
from pipeline.step4_dino_reranking import DINOReRanker  # noqa: E402
from pipeline.step5_shape_matching import ShapeMatcher  # noqa: E402
from pipeline.step6_fusion import ScoreFusion  # noqa: E402

RANKING_KEYS = (
    # --- Full-database arms (rank the whole gallery) ------------------------
    "clip_only",             # CLIP image<->text ranking over all objects
    "dino_only_full",        # DINO (topk_softmax) over all objects
    "ulip_only_full",        # ULIP-2 (cross) over all objects
    "clip_dino_ulip_full",   # 3-way weighted fusion (clip+dino+ulip), full DB
    # --- CLIP-shortlist arms (rank only the CLIP candidate set S') ----------
    # S' = {o : sim_text(o) >= clip_tau} with top-clip_fallback_k fallback
    # (OSCAR cascade, Pulli et al.). On the legacy "topk" prune-mode S' is the
    # CLIP top-clip_top_k instead.
    "oscar_maxview",         # DINO best-view (max) over S'  — faithful OSCAR
    "oscar_softmax",         # DINO topk_softmax over S'     — view-agg ablation
    "clip_pruned_dino_ulip", # DINO(softmax)+ULIP fusion over S'
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Pipeline + output settings shared across datasets."""

    ref_dir: str = ""
    desc_file: str = ""
    cad_mesh_glob: str = ""
    result_folder: str = "results"

    topk: List[int] = field(default_factory=lambda: [15])
    TOP_F: int = 20

    clip_top_k: int = 9999
    dino_top_k: int = 9999
    ulip2_top_k: int = 9999
    fusion_top_k: int = 9999
    fusion_method: str = "weighted_sum"

    # CLIP shortlist (S') construction for the *_pruned / oscar_* arms.
    #   clip_prune_mode="topk"      -> S' = CLIP top-clip_top_k (legacy)
    #   clip_prune_mode="threshold" -> S' = {o : sim_text(o) >= clip_tau},
    #        falling back to CLIP top-clip_fallback_k when none clear clip_tau
    #        (OSCAR cascade, Pulli et al. arXiv:2601.07333; tau is empirical
    #        and dataset-specific). The full-DB arms are unaffected.
    clip_prune_mode: str = "topk"
    clip_tau: float = 0.37
    clip_fallback_k: int = 20
    # Defaults = die BASE-Gewichte der Evaluation. Frueher 0/0.5/0.5 — eine
    # dokumentierte Falle (Stage-2-Gewichtsbug): Treiber setzen die Gewichte
    # weiterhin explizit, der Default darf nur nicht mehr davon abweichen.
    weight_clip: float = 0.3
    weight_dino: float = 0.4
    weight_ulip: float = 0.3

    ulip_repo_path: str = "/ulip"
    ulip2_checkpoint: str = "/ulip/checkpoints/ulip2_pointbert_10k.pt"
    ulip2_mode: str = "cross"
    ulip2_use_partial_views: bool = False
    ulip_query_cache_path: str = ""

    # Extra PipelineConfig field overrides applied in build_pipeline()
    # before any component is constructed.  Lets experiment scripts toggle
    # ablation knobs (appearance_encoder, shape_encoder, num_views,
    # ulip2_use_colors, ...) without widening EvalConfig for each one.
    pipeline_overrides: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric helpers (verbatim from retrieval_mi3dor_eval.py)
# ---------------------------------------------------------------------------

def dcg_at_k(rels, k):
    dcg = 0.0
    for i in range(min(len(rels), k)):
        dcg += rels[i] / log2(i + 2)
    return dcg


def ideal_dcg_at_k(n_relevant, k):
    ideal = 0.0
    for i in range(min(n_relevant, k)):
        ideal += 1.0 / log2(i + 2)
    return ideal


def average_precision_from_binary(rels, num_rel_true=None):
    """AP for a binary relevance ranking.

    ``num_rel_true`` (|C|) is the denominator — the total number of relevant
    items in the database. When None (default, back-compat) it falls back to the
    number of relevant items present in ``rels``, which is correct only when
    ``rels`` covers the whole gallery. Pass the true C when scoring a
    pruned/shortlisted ranking so relevant items that were pruned away correctly
    lower the AP.
    """
    rels = np.asarray(rels, dtype=np.int32)
    denom = int(num_rel_true) if num_rel_true is not None else int(rels.sum())
    if denom == 0:
        return 0.0
    precisions_sum = 0.0
    cum = 0
    for i, r in enumerate(rels, start=1):
        if r:
            cum += 1
            precisions_sum += cum / i
    return float(precisions_sum / denom)


def compute_anmrr(ranks, num_rel, K):
    if num_rel == 0:
        return None
    if not ranks:
        avr = K + 1
    else:
        padded = ranks + [K + 1] * max(0, num_rel - len(ranks))
        avr = float(np.mean(padded))
    # Denominator matches Pulli et al.'s original MI3DOR scorer EXACTLY
    # (retrieval_mi3dor_eval.py) so our numbers are directly comparable to the
    # published OSCAR results — this is the operative benchmark definition.
    #
    # NOTE (thesis footnote): textbook MPEG-7 ANMRR uses (K+1) - (num_rel+1)/2
    # here, so that the all-miss case (avr = K+1) maps to NMRR = 1. Pulli's
    # denom omits the +1, which lets NMRR slightly exceed 1 in the worst case
    # (negligible for MI3DOR's |C| = 31..250 → K = 62..500: <2% and ~0.3% at the
    # large-C end). We keep Pulli's form for reproduction; switch to the +1
    # variant if you want the strictly-normalised [0,1] metric.
    denom = K - (num_rel + 1) / 2.0
    if denom <= 0:
        return 0.0
    return (avr - (num_rel + 1) / 2.0) / denom


def mean_ignore_nan(xs):
    xs = np.array(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    return float(xs.mean()) if xs.size > 0 else float("nan")


# ---------------------------------------------------------------------------
# Incremental metric accumulators (constant memory)
# ---------------------------------------------------------------------------

def make_accum():
    return {
        "nn_correct": 0, "ft": 0.0, "st": 0.0, "f1": 0.0,
        "ndcg": 0.0, "ap": [], "anmrr": [], "count": 0,
    }


def update_accum(accum, ids_scores, gt_label, to_label_fn, top_f, num_rel_true):
    """Accumulate one query's metrics against the TRUE relevant-set size.

    ``num_rel_true`` is |C| — the number of gallery objects sharing the query's
    category in the full database. It is passed in, NOT derived from
    ``ids_scores``, so that CLIP-pruned rankings (which contain only a
    shortlist) are still normalised by the true class size rather than by how
    many relevant items happened to survive pruning. Consequences:
      * FT/ST/F1/nDCG/mAP/ANMRR for pruned arms are no longer inflated by a
        collapsed |C|.
      * A query whose shortlist contains no relevant item counts as a genuine
        failure (all-zero contributions) instead of being silently dropped, so
        every arm shares the same denominator (``count``).
    Relevant items that were pruned away simply never appear in ``rels`` and so
    depress recall/precision exactly as they should.
    """
    C = int(num_rel_true)
    if C == 0:
        return  # category genuinely absent from the gallery — not evaluable
    full_labels = [to_label_fn(oid) for oid, _ in ids_scores]
    rels = np.array([1 if lab == gt_label else 0 for lab in full_labels],
                    dtype=int)
    accum["count"] += 1
    if rels.size > 0 and rels[0] == 1:
        accum["nn_correct"] += 1
    # First / Second tier — recall at true C / 2C. numpy slices auto-truncate
    # when the ranking is shorter than the cut (e.g. a 20-item shortlist).
    accum["ft"] += float(rels[:C].sum()) / C
    accum["st"] += float(rels[:2 * C].sum()) / C
    # F1 at TOP_F — recall normalised by true C.
    top_f_eff = min(top_f, rels.size)
    rel_top_f = int(rels[:top_f_eff].sum()) if top_f_eff > 0 else 0
    p = (rel_top_f / top_f_eff) if top_f_eff > 0 else 0.0
    r = rel_top_f / C
    accum["f1"] += (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    # nDCG@2R — dcg over top-2C (dcg_at_k caps at len); idcg over the true C.
    K = 2 * C
    dcg_val = dcg_at_k(rels.tolist(), K)
    idcg_val = ideal_dcg_at_k(C, K)
    accum["ndcg"] += dcg_val / idcg_val if idcg_val > 0 else 0.0
    # AP normalised by true C (pruned-away relevant items count as unretrieved).
    accum["ap"].append(average_precision_from_binary(rels.tolist(), C))
    # ANMRR over window K = 2C; relevant items beyond K (or pruned away) get the
    # miss penalty inside compute_anmrr.
    rels_list = rels.tolist()
    rel_pos = [j + 1 for j, rv in enumerate(rels_list)
               if rv == 1 and (j + 1) <= K]
    accum["anmrr"].append(compute_anmrr(rel_pos, C, K))


def finalize_accum(accum):
    n = accum["count"]
    if n == 0:
        return {k: float("nan") for k in (
            "num_queries", "NN_accuracy", "FT_mean", "ST_mean",
            "F1_mean", "nDCG@2R_mean", "mAP", "ANMRR_mean")}
    return {
        "num_queries": n,
        "NN_accuracy": 100.0 * accum["nn_correct"] / n,
        "FT_mean":      accum["ft"]   / n,
        "ST_mean":      accum["st"]   / n,
        "F1_mean":      accum["f1"]   / n,
        "nDCG@2R_mean": accum["ndcg"] / n,
        "mAP":          float(np.mean(accum["ap"]))    if accum["ap"]    else float("nan"),
        "ANMRR_mean":   float(np.mean(accum["anmrr"])) if accum["anmrr"] else float("nan"),
    }


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def empty_pointcloud_result():
    empty = np.zeros((0, 3), dtype=np.float32)
    return PointCloudResult(
        point_cloud=None,
        points=empty,
        colors=empty,
        num_points=0,
        bbox_min=np.zeros(3, dtype=np.float32),
        bbox_max=np.zeros(3, dtype=np.float32),
        bbox_size=np.zeros(3, dtype=np.float32),
    )


def _filter_dino_result_by_ids(result, id_set, clip_score_map=None):
    """Derive a CLIP-pruned DINO result from a full DINO result.

    Pure id-intersection — scores and order (sorted by dino_score desc)
    are preserved. Optionally backfills ``clip_score`` on the kept
    candidates from a ``{obj_id: clip_score}`` map, so the derived pruned
    result matches what an explicit CLIP-gated DINO run would have
    produced.
    """
    if result is None:
        return None
    import copy as _copy
    kept = []
    for c in result.candidates:
        if c.object_id not in id_set:
            continue
        if clip_score_map is not None and c.object_id in clip_score_map:
            c = _copy.copy(c)
            c.clip_score = float(clip_score_map[c.object_id])
        kept.append(c)
    out = _copy.copy(result)
    out.candidates = kept
    return out


def _build_clip_shortlist(clip_res, cfg):
    """Build the CLIP candidate set S' for the pruned / oscar_* arms.

    Returns ``(shortlist_ids, fell_back)`` where ``shortlist_ids`` is ordered
    by CLIP score (desc).

    * ``clip_prune_mode == "threshold"``: S' = {o : sim_text(o) >= clip_tau}.
      If none clear the threshold, fall back to the CLIP top-``clip_fallback_k``
      (OSCAR cascade behaviour, Pulli et al.). ``fell_back`` flags that case.
    * otherwise (legacy "topk"): S' = CLIP top-``clip_top_k``.

    ``clip_res.candidates`` are assumed sorted by score descending (the CLIP
    retriever guarantees this), so a prefix slice is a valid top-k.
    """
    cands = clip_res.candidates
    mode = getattr(cfg, "clip_prune_mode", "topk")
    if mode == "threshold":
        tau = cfg.clip_tau
        kept = [c.object_id for c in cands if float(c.score) >= tau]
        if kept:
            return kept, False
        # Fallback: top-k text candidates.
        fk = cfg.clip_fallback_k
        return [c.object_id for c in cands[:fk]], True
    # Legacy fixed top-k shortlist.
    k = cfg.clip_top_k
    return [c.object_id for c in cands[:k]], False


def _filter_shape_result_by_ids(result, id_set):
    """Derive a CLIP-pruned ShapeMatchingResult from a full one.

    Pure id-intersection — shape_scores and order are preserved.
    """
    if result is None:
        return None
    import copy as _copy
    kept = [c for c in result.candidates if c.object_id in id_set]
    out = _copy.copy(result)
    out.candidates = kept
    return out


def query_has_depth(query_path):
    d = os.path.dirname(query_path)
    stem = os.path.splitext(os.path.basename(query_path))[0]
    sibling_depth_dir = os.path.join(os.path.dirname(d), "depth")
    if os.path.isdir(sibling_depth_dir):
        for ext in (".png", ".tif", ".tiff", ".npy"):
            if os.path.isfile(os.path.join(sibling_depth_dir, stem + ext)):
                return True
    for suffix in ("_depth.png", "_depth.tif", "_depth.tiff", "_depth.npy"):
        if os.path.isfile(os.path.join(d, stem + suffix)):
            return True
    if os.path.isfile(os.path.join(d, stem + ".npz")):
        return True
    return False


def build_pipeline(cfg, cad_mesh_items=None):
    """Initialise all pipeline components.

    Parameters
    ----------
    cfg : EvalConfig
    cad_mesh_items : list of (obj_id, mesh_path), optional
        When the default basename-based obj_id extraction is wrong (e.g.
        YCBV ``<name>/meshes/model.obj``), pass a pre-built list.

    Returns (pipeline_config, clip_retr, dino_rer, fusion, shape_m).
    """
    config = PipelineConfig(
        description_file=cfg.desc_file,
        reference_images_dir=cfg.ref_dir,
        cad_models_dir="",
        clip_top_k=cfg.clip_top_k,
        dino_top_k=cfg.dino_top_k,
        ulip2_top_k=cfg.ulip2_top_k,
        fusion_top_k=cfg.fusion_top_k,
        fusion_method=cfg.fusion_method,
        weight_clip=cfg.weight_clip,
        weight_dino=cfg.weight_dino,
        weight_ulip=cfg.weight_ulip,
        ulip_repo_path=cfg.ulip_repo_path,
        ulip2_checkpoint=cfg.ulip2_checkpoint,
        ulip2_mode=cfg.ulip2_mode,
        ulip2_use_partial_views=cfg.ulip2_use_partial_views,
    )

    # Apply experiment-level overrides (must happen before components are
    # built, since encoders read the config at construction time).
    for _k, _v in cfg.pipeline_overrides.items():
        if not hasattr(config, _k):
            raise AttributeError(f"pipeline_overrides: unknown "
                                 f"PipelineConfig field '{_k}'")
        setattr(config, _k, _v)
    # Keep EvalConfig's view of partial-view usage in sync so the
    # partial-vs-fullmesh branch below follows the override.
    if "ulip2_use_partial_views" in cfg.pipeline_overrides:
        cfg.ulip2_use_partial_views = config.ulip2_use_partial_views

    print("[init] Loading CLIP descriptions...")
    clip_retr = CLIPRetriever(config)
    clip_retr.load_descriptions()

    print("[init] Loading DINOv2 reference images...")
    dino_rer = DINOReRanker(config)
    dino_rer.load_reference_images()

    fusion_mod = ScoreFusion(config)

    # --- ULIP CAD embeddings ---
    shape_m = None
    if cad_mesh_items is None and cfg.cad_mesh_glob:
        mesh_paths = sorted(_glob.glob(cfg.cad_mesh_glob))
        if mesh_paths:
            cad_mesh_items = [
                (os.path.splitext(os.path.basename(p))[0], p)
                for p in mesh_paths
            ]
        else:
            print(f"[init] WARNING: cad_mesh_glob matched 0 files: "
                  f"{cfg.cad_mesh_glob}")

    if cad_mesh_items:
        shape_m = ShapeMatcher(config)
        # Only ULIP-2 needs an explicit _load_model() call here — it builds
        # the PointBERT architecture (upstream ULIP code always prints
        # "training from scratch for pointbert.", regardless of whether a
        # checkpoint is then loaded onto it). Uni3D is loaded lazily inside
        # encode_pointcloud()/load_cad_models() on first use instead, so
        # calling _load_model() for shape_encoder="uni3d" would build and
        # load a full unused ULIP-2 PointBERT for nothing.
        if getattr(config, "shape_encoder", "ulip2") != "uni3d":
            try:
                shape_m._load_model()
            except Exception as exc:
                print(f"[init] ULIP model load failed: {exc}")
                shape_m = None

        _cache_ok = False
        if shape_m is not None and cfg.ulip2_use_partial_views:
            # Die rohen *_partial.npz sind verzichtbar, sobald ihr Embedding-
            # Cache existiert (er haelt jedes Per-View-Embedding; ein
            # Shape-View-Sweep poolt weiterhin die ersten N je Objekt). Der
            # passende Cache wird ueber die Embedding-Dimension der aktiven
            # Config eindeutig gewaehlt (coloured 1280 / xyz 512 / uni3d 1024).
            # Fehlt beides: harter Fehler — der fruehere stille Full-Mesh-
            # Fallback hat nachweislich Laeufe verfaelscht.
            partial_items = shape_m._collect_partial_items(cfg.ref_dir)
            if not partial_items:
                if not shape_m._load_partial_cache_by_dim(cfg.ref_dir):
                    raise shape_m._partial_cache_error(cfg.ref_dir)
                _mmap = {oid: p for oid, p in cad_mesh_items}
                for oid in shape_m._cad_embeddings:
                    if oid not in shape_m._cad_paths and oid in _mmap:
                        shape_m._cad_paths[oid] = _mmap[oid]
                print(f"[init] ULIP partial-view cache loaded "
                      f"({len(shape_m._cad_embeddings)} models, "
                      f"dim={shape_m._expected_partial_dim()}).")
                _cache_ok = True

        if shape_m is not None and cfg.ulip2_use_partial_views and not _cache_ok:
            mesh_map = {oid: p for oid, p in cad_mesh_items}
            shape_m._partial_view_paths = dict(partial_items)
            cache_path = shape_m._get_partial_cache_path(
                cfg.ref_dir, partial_items)

            if shape_m._try_load_partial_cache(cache_path):
                for oid in shape_m._cad_embeddings:
                    if oid not in shape_m._cad_paths and oid in mesh_map:
                        shape_m._cad_paths[oid] = mesh_map[oid]
                print(f"[init] ULIP partial-view cache loaded "
                      f"({len(shape_m._cad_embeddings)} models).")
            else:
                print(f"[init] Encoding {len(partial_items)} objects "
                      f"(partial views)...")
                ok = 0
                for obj_id, view_files in tqdm(sorted(partial_items.items()),
                                               desc="ULIP partial",
                                               unit="obj"):
                    view_embs = []
                    for view_idx, npz_path in sorted(view_files):
                        try:
                            data = np.load(npz_path)
                            emb = shape_m.encode_pointcloud(
                                data["points"], colors=data.get("colors"))
                            view_embs.append(
                                emb.squeeze(0).detach().cpu())
                        except Exception as e:
                            tqdm.write(
                                f"[warn] {obj_id} view {view_idx}: {e}")
                    if view_embs:
                        shape_m._cad_embeddings[obj_id] = torch.stack(
                            view_embs, dim=0)
                        shape_m._cad_paths[obj_id] = mesh_map.get(
                            obj_id, "")
                        ok += 1
                    elif obj_id in mesh_map:
                        if shape_m._encode_and_cache(
                                obj_id, mesh_map[obj_id]):
                            ok += 1

                print(f"[init] ULIP partial: {ok}/{len(partial_items)} ok.")
                if ok == 0:
                    shape_m = None
                else:
                    shape_m._partial_mode = True
                    shape_m._save_partial_cache(cache_path)

        elif shape_m is not None and not _cache_ok:
            # --- Full-mesh path (legacy) ---
            print(f"[init] Encoding {len(cad_mesh_items)} ULIP CAD meshes...")
            all_paths = [p for _, p in cad_mesh_items]
            cad_dir = os.path.commonpath(all_paths)
            if not os.path.isdir(cad_dir):
                cad_dir = os.path.dirname(cad_dir)
            cache_path = shape_m._get_cache_path(cad_dir, cad_mesh_items)

            if shape_m._try_load_cache(cache_path):
                print(f"[init] ULIP CAD cache loaded "
                      f"({len(shape_m._cad_embeddings)} models).")
            else:
                ok = 0
                for obj_id, mesh_path in tqdm(cad_mesh_items,
                                              desc="ULIP CAD", unit="mesh"):
                    if shape_m._encode_and_cache(obj_id, mesh_path):
                        ok += 1
                print(f"[init] ULIP CAD: {ok}/{len(cad_mesh_items)} ok.")
                if ok == 0:
                    shape_m = None
                else:
                    shape_m._save_cache(cache_path)

    return config, clip_retr, dino_rer, fusion_mod, shape_m


# ---------------------------------------------------------------------------
# ULIP query cache
# ---------------------------------------------------------------------------

def load_ulip_query_cache(cache_path):
    if not cache_path or not os.path.isfile(cache_path):
        return None
    print(f"[ulip-cache] Loading {cache_path}...")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # The keys are the query paths of whichever run wrote the cache, so a
    # different --datasets-root misses every entry and silently re-encodes the
    # queries one by one, which shifts the scores in the fifth decimal. Query
    # file names are unique, so re-key by name whenever that stays injective.
    by_name = {os.path.basename(k): v for k, v in cache.items()}
    if len(by_name) == len(cache):
        cache = by_name
    print(f"[ulip-cache] {len(cache)} embeddings loaded.")
    return cache


def lookup_ulip_query_emb(ulip_cache, img_path):
    """Query embedding for ``img_path``, by full path or by file name."""
    if not ulip_cache:
        return None
    emb = ulip_cache.get(img_path)
    if emb is None:
        emb = ulip_cache.get(os.path.basename(img_path))
    return emb


def pre_encode_ulip_queries(img_paths, shape_m, batch_size=32):
    shape_m._load_image_encoder()
    shape_m.image_encoder.load()
    enc = shape_m.image_encoder
    preprocess = enc.preprocess
    model = enc.model
    device = enc.device

    print(f"[ulip-preenc] Encoding {len(img_paths)} images "
          f"(batch={batch_size})...")
    tensors, valid_paths = [], []
    for p in img_paths:
        try:
            tensors.append(preprocess(Image.open(p).convert("RGB")))
            valid_paths.append(p)
        except Exception as exc:
            tqdm.write(f"[ulip-preenc] skip {p}: {exc}")

    cache = {}
    model.eval()
    for i in tqdm(range(0, len(tensors), batch_size),
                  desc="ULIP pre-enc", unit="batch"):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        with torch.no_grad():
            emb = model(batch)
            emb = F.normalize(emb, p=2, dim=-1)
        for j, p in enumerate(valid_paths[i:i + batch_size]):
            cache[p] = emb[j:j + 1].cpu()

    print(f"[ulip-preenc] Done: {len(cache)} embeddings.")
    return cache


# ---------------------------------------------------------------------------
# Per-query retrieval
# ---------------------------------------------------------------------------

def run_query(pipeline_cfg, clip_retr, dino_rer, fusion_mod, shape_m,
              roi, cfg, ulip_query_emb=None,
              dino_full_top_k=None, ulip_full_top_k=None,
              clip_full_top_k=None):
    """Run one query and produce every ranking arm from a single pass.

    CLIP is scored over the *whole* gallery once; the full-DB arms
    (``clip_only``, ``dino_only_full``, ``ulip_only_full``,
    ``clip_dino_ulip_full``) read those full rankings directly, and the
    CLIP-shortlist arms (``oscar_maxview``, ``oscar_softmax``,
    ``clip_pruned_dino_ulip``) are derived by id-intersecting the full DINO /
    ULIP rankings with the CLIP candidate set ``S'`` — no per-arm re-ranking,
    since DINO/ULIP scores are computed per object independent of the pool.

    ``S'`` is built from the full CLIP scores by :func:`_build_clip_shortlist`
    (threshold-τ with top-k fallback, or legacy top-k). ``dino_score_maxview``
    (best-view) and ``dino_score`` (configured aggregation) are both present on
    every DINO candidate, so the max-view and softmax OSCAR arms share one pass.

    ``*_full_top_k`` control the depth of the full rankings; they should be ≥
    the gallery size so every object survives — the run_evaluation wrapper
    computes them from the loaded reference counts.
    """
    # --- CLIP over the whole gallery (arm: clip_only; source of S') ----------
    clip_k = clip_full_top_k or cfg.clip_top_k
    clip_res = clip_retr.retrieve(roi, top_k=clip_k)
    clip_score_map = {c.object_id: float(c.score)
                      for c in clip_res.candidates}

    # CLIP shortlist S' for the pruned / oscar_* arms.
    shortlist_ids, fell_back = _build_clip_shortlist(clip_res, cfg)
    clip_id_set = set(shortlist_ids)

    # --- Full DINO: single pass over the whole reference set -----------------
    dino_k = dino_full_top_k or cfg.dino_top_k
    dino_res_full = dino_rer.rerank(roi, clip_result=None, top_k=dino_k)
    # NOTE: no clip_score backfill here. The shortlist arm #7 fuses DINO+ULIP
    # *only* (CLIP is a gate, not a summand — per the arm definition), and the
    # full DINO pass leaves clip_score=0 on every candidate, so the fusion's
    # CLIP channel stays zero for fused_du_thresh.
    dino_res_thresh = _filter_dino_result_by_ids(dino_res_full, clip_id_set)

    # --- Full ULIP: single pass over the whole CAD set -----------------------
    ulip_fell_back = False
    shape_res_full = None
    shape_res_thresh = None
    if shape_m is not None:
        ulip_k = ulip_full_top_k or cfg.ulip2_top_k
        prev_mode = pipeline_cfg.ulip2_mode
        pipeline_cfg.ulip2_mode = "cross"
        try:
            if ulip_query_emb is not None:
                cached = ulip_query_emb.to(shape_m.config.device)
                orig_encode = shape_m.encode_image
                shape_m.encode_image = lambda _img: cached
                try:
                    shape_res_full = shape_m.match(
                        empty_pointcloud_result(),
                        top_k=ulip_k,
                        candidate_ids=None,
                        query_image=roi,
                    )
                finally:
                    shape_m.encode_image = orig_encode
            else:
                shape_res_full = shape_m.match(
                    empty_pointcloud_result(),
                    top_k=ulip_k,
                    candidate_ids=None,
                    query_image=roi,
                )
            ulip_fell_back = True
        finally:
            pipeline_cfg.ulip2_mode = prev_mode

        shape_res_thresh = _filter_shape_result_by_ids(
            shape_res_full, clip_id_set,
        )

    # --- Full-DB 3-way fusion (arm: clip_dino_ulip_full) ---------------------
    # CLIP DOES enter the weighted sum here (weight_clip), over the whole gallery.
    full_fusion_k = clip_full_top_k or cfg.fusion_top_k
    fused_full = fusion_mod.fuse(
        clip_res, dino_res_full, shape_res_full, top_k=full_fusion_k,
    )

    # --- CLIP-pruned DINO+ULIP fusion (arm: clip_pruned_dino_ulip) -----------
    # CLIP is used only to select S' (gate), not summed — the thesis cascade.
    fusion_top_k = len(shortlist_ids) if shortlist_ids else cfg.fusion_top_k
    fused_du_thresh = fusion_mod.fuse(
        None, dino_res_thresh, shape_res_thresh, top_k=fusion_top_k,
    )
    # Final decision must be a shortlist candidate: highest fused score in S'.
    if shortlist_ids:
        filtered = [c for c in fused_du_thresh.candidates
                    if c.object_id in clip_id_set]
        filtered.sort(key=lambda c: c.fused_score, reverse=True)
        fused_du_thresh.candidates = filtered
        fused_du_thresh.best_match = filtered[0] if filtered else None

    return {
        "clip_res": clip_res,
        "dino_res_full": dino_res_full,
        "dino_res_thresh": dino_res_thresh,
        "shape_res_full": shape_res_full,
        "shape_res_thresh": shape_res_thresh,
        "fused_full": fused_full,
        "fused_du_thresh": fused_du_thresh,
        "ulip_fell_back": ulip_fell_back,
        "shortlist_size": len(shortlist_ids),
        "shortlist_fallback": fell_back,
    }


# ---------------------------------------------------------------------------
# Ranking extractors  → list of (object_id, score)
# ---------------------------------------------------------------------------

def clip_ranking(clip_res):
    return [(c.object_id, float(c.score)) for c in clip_res.candidates]


def dino_ranking(dino_res):
    return [(c.object_id, float(c.dino_score)) for c in dino_res.candidates]


def dino_ranking_maxview(dino_res):
    """Rank DINO candidates by best-view (max) score — OSCAR aggregation.

    Re-sorts by ``dino_score_maxview`` (the hard per-object best-view
    similarity computed in step4) rather than the configured aggregation,
    so the max-view arm and the softmax arm share one full DINO pass.
    """
    if dino_res is None:
        return []
    ranked = sorted(dino_res.candidates,
                    key=lambda c: getattr(c, "dino_score_maxview", 0.0),
                    reverse=True)
    return [(c.object_id, float(getattr(c, "dino_score_maxview", 0.0)))
            for c in ranked]


def ulip_ranking(shape_res):
    if shape_res is None:
        return []
    return [(c.object_id, float(c.shape_score)) for c in shape_res.candidates]


def fusion_ranking(fusion_res):
    return [(c.object_id, float(c.fused_score)) for c in fusion_res.candidates]


def _cascade_full_ranking(shortlist_ranking, clip_ids):
    """Full ranking for a CLIP-shortlist (cascade) arm.

    A re-ranking cascade does not *shorten* the result list — it reorders the
    CLIP shortlist S' and places it on top, while every object below the
    shortlist keeps its CLIP order. So the output is a FULL ranking of the
    whole gallery: ``[DINO/ULIP-reranked S']  ++  [CLIP tail minus S']``.

    This is what makes the OSCAR/cascade arms comparable to the full-DB arms
    and to OSCAR's published FT: deep recall is inherited from CLIP, and the
    re-ranking only moves the head (where NN / early precision live). Ranking
    order is all that matters to the metrics, so mixing DINO/ULIP scores in the
    head with CLIP scores in the tail is fine.
    """
    placed = {oid for oid, _ in shortlist_ranking}
    tail = [(oid, s) for oid, s in clip_ids if oid not in placed]
    return list(shortlist_ranking) + tail


def _arm_rankings(out):
    """Map each of the 7 RANKING_KEYS to its (object_id, score) ranking.

    Single source of truth shared by the metric accumulation loop and the
    per-query JSON record, so the two never diverge. Order matches
    ``RANKING_KEYS``. The CLIP-shortlist arms are full-gallery cascade rankings
    (reranked S' on top of the CLIP tail) — see :func:`_cascade_full_ranking`.
    """
    clip_ids = clip_ranking(out["clip_res"])
    return {
        # Full-database arms.
        "clip_only":            clip_ids,
        "dino_only_full":       dino_ranking(out["dino_res_full"]),
        "ulip_only_full":       ulip_ranking(out["shape_res_full"]),
        "clip_dino_ulip_full":  fusion_ranking(out["fused_full"]),
        # CLIP-shortlist (S') cascade arms — full rankings (reranked head + CLIP tail).
        "oscar_maxview": _cascade_full_ranking(
            dino_ranking_maxview(out["dino_res_thresh"]), clip_ids),
        "oscar_softmax": _cascade_full_ranking(
            dino_ranking(out["dino_res_thresh"]), clip_ids),
        "clip_pruned_dino_ulip": _cascade_full_ranking(
            fusion_ranking(out["fused_du_thresh"]), clip_ids),
    }


# ---------------------------------------------------------------------------
# Image crop helpers
# ---------------------------------------------------------------------------

def crop_by_bbox(image, bbox):
    """Crop PIL Image by BOP-style [x, y, w, h] bbox."""
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))


def crop_with_mask(image, mask):
    """Crop PIL Image using a binary mask (grey background)."""
    mask_array = np.array(mask) > 0
    if mask_array.sum() == 0:
        return None
    img_array = np.array(image)
    masked = np.full(img_array.shape, (205, 205, 205), dtype=np.uint8)
    masked[mask_array] = img_array[mask_array]
    coords = np.argwhere(mask_array)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return Image.fromarray(masked[y0:y1, x0:x1])


# ---------------------------------------------------------------------------
# JSON output (private helpers)
# ---------------------------------------------------------------------------


def _rel_positions(ids, gt_label, to_label_fn):
    """1-indexed ranks at which relevant (same-category) items appear.

    Relevance is binary category match — the label of the ranked object
    equals the query ground-truth label. This is the complete information a
    category-level metric needs: NN/FT/ST/F1/nDCG/mAP/ANMRR at ANY depth are
    exactly reconstructible from (rel_positions, ranking length, true |C|),
    so future metric changes never require a GPU re-run.
    """
    return [i for i, (oid, _s) in enumerate(ids, start=1)
            if to_label_fn(oid) == gt_label]


def _make_per_query_record(out, gt_label, category, fname, to_label_fn,
                           num_rel_true=None, arm_rankings=None):
    """Build a per-query record for the 7 ranking arms.

    ``pred`` is the top-1 of the primary thesis cascade
    (``clip_pruned_dino_ulip``). Top-5 previews are kept per arm for eyeballing.

    ``eval_trace`` persists, per arm, the true relevant-set size and the
    positions of relevant items down the *full* ranking (not just top-5), plus
    the realized CLIP shortlist size / fallback flag — so every metric can be
    re-scored offline without re-running the pipeline.
    """
    if arm_rankings is None:
        arm_rankings = _arm_rankings(out)

    def _top5(ids, score_key):
        return [{"label": to_label_fn(oid), score_key: s} for oid, s in ids[:5]]

    du_thresh = arm_rankings["clip_pruned_dino_ulip"]
    pred_label = to_label_fn(du_thresh[0][0]) if du_thresh else None

    top5_full_fusion = []
    for c in out["fused_full"].candidates[:5]:
        top5_full_fusion.append({
            "label": to_label_fn(c.object_id),
            "fused_score": float(c.fused_score),
            "dino_score": float(getattr(c, "dino_score", 0.0)),
            "ulip_score": float(getattr(c, "ulip_score", 0.0)),
            "clip_score": float(getattr(c, "clip_score", 0.0)),
        })

    # Compact, full-depth relevance trace — one entry per ranking arm. Enough
    # to recompute any category-level metric offline (see _rel_positions).
    eval_trace = {
        "num_rel_true": (int(num_rel_true)
                         if num_rel_true is not None else None),
        "shortlist_size": out.get("shortlist_size"),
        "shortlist_fallback": out.get("shortlist_fallback"),
        "arms": {name: {"len": len(ids),
                        "rel_positions": _rel_positions(ids, gt_label,
                                                        to_label_fn)}
                 for name, ids in arm_rankings.items()},
    }

    return {
        "category": category,
        "filename": fname,
        "gt": gt_label,
        "pred": pred_label,
        "shortlist_size": out.get("shortlist_size"),
        "shortlist_fallback": out.get("shortlist_fallback"),
        # Full-database arms.
        "clip_candidates": _top5(arm_rankings["clip_only"], "clip_score"),
        "dino_candidates_full": _top5(arm_rankings["dino_only_full"], "dino_score"),
        "ulip_candidates_full": _top5(arm_rankings["ulip_only_full"], "ulip_score"),
        "clip_dino_ulip_full_top5": top5_full_fusion,
        # CLIP-shortlist arms.
        "oscar_maxview_top5": _top5(arm_rankings["oscar_maxview"], "dino_maxview"),
        "oscar_softmax_top5": _top5(arm_rankings["oscar_softmax"], "dino_score"),
        "clip_pruned_dino_ulip_pred": pred_label,
        "clip_pruned_dino_ulip_top5": _top5(
            arm_rankings["clip_pruned_dino_ulip"], "fused_score"),
        "eval_trace": eval_trace,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(cfg, to_label_fn, query_factory, components,
                   ulip_cache=None):
    """Run the full OSCAR+ evaluation.

    Parameters
    ----------
    cfg : EvalConfig
    to_label_fn : callable
        Maps pipeline object_id to the label used for relevance checks.
    query_factory : callable(k) -> iterable of (roi, gt_label, img_path, category, fname)
    components : tuple
        ``(pipeline_config, clip_retr, dino_rer, fusion, shape_m)`` from
        :func:`build_pipeline`.
    ulip_cache : dict, optional
        ``{img_path: tensor(1, embed_dim)}`` pre-computed ULIP query
        embeddings.

    Returns
    -------
    dict mapping *k* to the summary dict for that topk run.
    """
    pipeline_cfg, clip_retr, dino_rer, fusion_mod, shape_m = components
    os.makedirs(cfg.result_folder, exist_ok=True)

    # --- Ground-truth relevant-set sizes (|C| per category) ------------------
    # The relevant set for a query is the set of gallery objects sharing its
    # category — a property of the DATABASE, independent of any per-query
    # shortlisting. Tier metrics (FT/ST/nDCG/mAP/ANMRR) for the CLIP-pruned arms
    # must be normalised by this, not by how many relevant items survived
    # pruning. Built from the union of retrievable object ids (appearance +
    # shape galleries share ids by construction).
    gallery_ids = set(getattr(dino_rer, "_ref_embeddings", {}) or {})
    if shape_m is not None and getattr(shape_m, "_cad_embeddings", None):
        gallery_ids |= set(shape_m._cad_embeddings)
    gallery_label_counts = {}
    for oid in gallery_ids:
        lab = to_label_fn(oid)
        gallery_label_counts[lab] = gallery_label_counts.get(lab, 0) + 1
    if gallery_label_counts:
        _cv = gallery_label_counts.values()
        print(f"[eval] gallery: {len(gallery_ids)} objects across "
              f"{len(gallery_label_counts)} categories "
              f"(|C| {min(_cv)}..{max(_cv)}); tier metrics normalised by true "
              f"|C| for every arm (pruned included).")
    else:
        print("[eval] WARNING: could not determine gallery |C| per category; "
              "tier metrics for pruned arms may be inflated.")

    # --- Auto-expand full-ranking depth so CLIP candidates survive filtering ---
    ref_objects = (len(dino_rer._ref_embeddings)
                   if getattr(dino_rer, "_ref_embeddings", None) else 0)
    cad_objects = (len(shape_m._cad_embeddings)
                   if (shape_m is not None and shape_m._cad_embeddings)
                   else 0)
    dino_full_top_k = max(cfg.dino_top_k, ref_objects) if ref_objects else cfg.dino_top_k
    ulip_full_top_k = max(cfg.ulip2_top_k, cad_objects) if cad_objects else cfg.ulip2_top_k

    auto_dino = dino_full_top_k != cfg.dino_top_k
    auto_ulip = ulip_full_top_k != cfg.ulip2_top_k
    print(
        f"[eval] Full-ranking depth — DINO: {dino_full_top_k} "
        f"({'auto-expanded from ' + str(cfg.dino_top_k) if auto_dino else 'from cfg'}, "
        f"{ref_objects} ref objects); "
        f"ULIP: {ulip_full_top_k} "
        f"({'auto-expanded from ' + str(cfg.ulip2_top_k) if auto_ulip else 'from cfg'}, "
        f"{cad_objects} CAD objects). CLIP-pruned variants are derived "
        f"by id-intersecting these full rankings (no re-ranking)."
    )

    # CLIP scores per DESCRIPTION row and dedups to unique objects afterwards.
    # MI3DOR has ~42 descriptions/object (163k rows), so top_k must span the
    # ROWS, not the objects, or the full CLIP ranking collapses to the handful
    # of objects whose rows dominate the head (e.g. 3848 rows -> ~281 objects).
    # Size it to the description-row count so every object surfaces.
    clip_desc_rows = len(getattr(clip_retr, "_desc_labels", []) or [])
    clip_full_top_k = max(cfg.clip_top_k, clip_desc_rows, ref_objects)
    if clip_full_top_k == cfg.clip_top_k and clip_desc_rows == 0:
        # Retriever did not expose row count; fall back to a large sentinel so
        # retrieve() returns all rows (it caps top_k at len(sims)).
        clip_full_top_k = max(cfg.clip_top_k, 1_000_000)
    print(f"[eval] CLIP full ranking: top_k={clip_full_top_k} over "
          f"{clip_desc_rows} description rows (dedup -> all objects).")

    summaries = {}
    for k in cfg.topk:
        accums = {name: make_accum() for name in RANKING_KEYS}
        ulip_fallback_count = 0
        shortlist_fallback_count = 0
        shortlist_sizes = []
        total_queries = 0
        per_query_records = []

        for roi, gt_label, img_path, category, fname in query_factory(k):
            try:
                ulip_emb = lookup_ulip_query_emb(ulip_cache, img_path)
                out = run_query(
                    pipeline_cfg, clip_retr, dino_rer, fusion_mod,
                    shape_m, roi, cfg, ulip_query_emb=ulip_emb,
                    dino_full_top_k=dino_full_top_k,
                    ulip_full_top_k=ulip_full_top_k,
                    clip_full_top_k=clip_full_top_k,
                )
            except Exception as exc:
                tqdm.write(f"[warn] query failed ({img_path}): {exc}")
                continue

            total_queries += 1
            if out["ulip_fell_back"]:
                ulip_fallback_count += 1
            if out.get("shortlist_fallback"):
                shortlist_fallback_count += 1
            shortlist_sizes.append(out.get("shortlist_size", 0))

            # All 7 arm rankings from this single pass.
            arm_rk = _arm_rankings(out)

            # True relevant-set size for this query's category (same |C| for
            # every arm — the database class count, not the shortlist count).
            num_rel_true = gallery_label_counts.get(gt_label, 0)

            for name in RANKING_KEYS:
                update_accum(accums[name], arm_rk[name], gt_label,
                             to_label_fn, cfg.TOP_F, num_rel_true)

            per_query_records.append(
                _make_per_query_record(out, gt_label, category, fname,
                                      to_label_fn, num_rel_true,
                                      arm_rankings=arm_rk)
            )

        # --- Finalize metrics ---
        variants = {name: finalize_accum(accums[name])
                    for name in RANKING_KEYS}
        primary = variants["clip_pruned_dino_ulip"]

        summary = {
            "primary": "clip_pruned_dino_ulip",
            **primary,
            "variants": variants,
            "config": {
                "ref_dir": cfg.ref_dir,
                "desc_file": cfg.desc_file,
                "cad_mesh_glob": cfg.cad_mesh_glob,
                "result_folder": cfg.result_folder,
                "topk": cfg.topk, "TOP_F": cfg.TOP_F,
                "clip_top_k": cfg.clip_top_k,
                "clip_prune_mode": getattr(cfg, "clip_prune_mode", "topk"),
                "clip_tau": getattr(cfg, "clip_tau", None),
                "clip_fallback_k": getattr(cfg, "clip_fallback_k", None),
                "clip_full_top_k_used": clip_full_top_k,
                "dino_top_k": cfg.dino_top_k,
                "ulip2_top_k": cfg.ulip2_top_k,
                "dino_full_top_k_used": dino_full_top_k,
                "ulip_full_top_k_used": ulip_full_top_k,
                "fusion_top_k": cfg.fusion_top_k,
                "fusion_method": cfg.fusion_method,
                "weight_clip": cfg.weight_clip,
                "weight_dino": cfg.weight_dino,
                "weight_ulip": cfg.weight_ulip,
                "ulip_repo_path": cfg.ulip_repo_path,
                "ulip2_checkpoint": cfg.ulip2_checkpoint,
                "ulip2_mode": cfg.ulip2_mode,
                "ulip2_use_partial_views": cfg.ulip2_use_partial_views,
                "ulip_active": shape_m is not None,
            },
            "ulip_fallback_cross_count": ulip_fallback_count,
            "clip_shortlist_fallback_count": shortlist_fallback_count,
            "clip_shortlist_size_mean": (
                float(np.mean(shortlist_sizes)) if shortlist_sizes else None),
            "clip_shortlist_size_median": (
                float(np.median(shortlist_sizes)) if shortlist_sizes else None),
            "total_queries_seen": total_queries,
        }

        results_path = os.path.join(cfg.result_folder,
                                    f"results_topk_{k}.json")
        summary_path = os.path.join(cfg.result_folder,
                                    f"metrics_summary_topk_{k}.json")
        with open(results_path, "w") as f:
            json.dump(per_query_records, f, indent=2)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # --- Console summary ---
        print(f"\n=== OSCAR+ evaluation (top-k={k}, "
              f"primary=clip_pruned_dino_ulip) ===")
        for key in ("num_queries", "NN_accuracy", "FT_mean", "ST_mean",
                     "F1_mean", "nDCG@2R_mean", "mAP", "ANMRR_mean"):
            print(f"  {key}: {primary.get(key)}")
        print(f"  ulip_fallback_cross_count: {ulip_fallback_count}")
        _pm = getattr(cfg, "clip_prune_mode", "topk")
        if _pm == "threshold":
            _sm = float(np.median(shortlist_sizes)) if shortlist_sizes else 0
            print(f"  clip_shortlist (tau={cfg.clip_tau}): median|S'|={_sm:.1f}, "
                  f"fallback {shortlist_fallback_count}/{total_queries} "
                  f"({100.0*shortlist_fallback_count/max(total_queries,1):.1f}%)")
        print(f"  total_queries_seen: {total_queries}")
        print(f"\n  Per-arm NN / FT / mAP / ANMRR:")
        for name in RANKING_KEYS:
            v = variants[name]
            print(f"    {name:<24s} NN={v.get('NN_accuracy')}  "
                  f"FT={v.get('FT_mean')}  mAP={v.get('mAP')}  "
                  f"ANMRR={v.get('ANMRR_mean')}")
        print(f"\n  Results: {results_path}")
        print(f"  Summary: {summary_path}\n")

        summaries[k] = summary

    return summaries
