# =============================================================================
# pipeline/step4_dino_reranking.py – Thesis Step B1: Appearance Channel S_view
# =============================================================================
#
# Computes the appearance score S_view (thesis Sec. 3.3, Step B1).
#
# Each CAD model is pre-rendered from V viewpoints (template-matching
# paradigm from CNOS, Nguyen et al., 2023). At query time, the ROI image
# and all reference views are encoded; cosine similarity is computed per
# view. Per-object scores are aggregated via softmax-weighted top-k_v
# fusion — a training-free approximation of the learned query-conditioned
# attention in OPEN (Chu et al., 2024, Eq. 2–3).
#
# Default: k_v = 5 (CNOS convention, validated for object identity
# assignment; Nguyen et al., 2023), τ = 0.5.
#
# Encoder alternatives (thesis Sec. 3.5):
#   • DINOv2 (default) — self-supervised ViT; CLS-token descriptor
#     (Oquab et al., 2023). CLS token following CNOS convention.
#   • SigLIP (ablation E4) — sigmoid-loss language-image encoder
#     (Zhai et al., 2023). Used as primary visual stage in ROOMELSA
#     winning entry (Nguyen et al., 2025 — SHREC 2025).
#
# Adapted from: OSCAR – retrieval_combi_clip.py (Pulli et al., 2025)
#
# Outputs:
#   - Refined Top-K candidates with appearance scores
# =============================================================================

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import PipelineConfig
from .step3_clip_retrieval import CLIPRetrievalResult, CLIPCandidate

logger = logging.getLogger(__name__)


def _view_sort_key(fname: str) -> Tuple[int, str]:
    """Numeric sort key for rendered view filenames.

    View files are named with an unpadded trailing index
    (``<obj>_0.png`` … ``<obj>_41.png``), so a plain lexicographic sort
    yields 0, 1, 10, 11, …, 19, 2, 20, … — which breaks the FPS-prefix
    assumption of ablation O4 (config.num_views: the first N views must
    be the N FPS-ordered viewpoints).  Sorting by the parsed integer
    restores rendering order; files without a trailing index fall back
    to lexicographic order after all indexed ones.
    """
    m = re.search(r"(\d+)\.[A-Za-z]+$", os.path.basename(fname))
    return (int(m.group(1)) if m else 10 ** 9, fname)


# ---------------------------------------------------------------------------
# Multi-view aggregation (inspired by OPEN, Chu et al. TCSVT 2024)
# ---------------------------------------------------------------------------

def _aggregate_view_scores(
    scores: torch.Tensor,
    method: str = "topk_softmax",
    top_k: int = 4,
    temperature: float = 0.1,
) -> Tuple[float, int]:
    """Aggregate per-view similarity scores into a single object score.

    Training-free approximation of the learned query-conditioned view
    attention in OPEN (Chu et al., 2024, Eq. 2–3):
      α_k = softmax(sim_k / τ)        — view attention weights
      S_obj = Σ_k α_k · sim_k          — weighted aggregation

    Where OPEN learns the attention weights via a cross-attention module,
    we use the raw cosine similarities as logits with temperature τ to
    produce view weights at inference time (no training required).

    Default k_v = 5 follows the CNOS convention (Nguyen et al., 2023)
    validated for template-based object identity assignment.

    Args:
        scores: (V,) tensor of cosine similarities for V views of one object.
        method: Aggregation strategy.
            "max"           – hard best-view (legacy).
            "mean"          – simple average of all views.
            "softmax"       – softmax-weighted over all views.
            "topk_softmax"  – softmax-weighted over top-k views only.
        top_k: Number of top views to consider (topk_softmax only).
        temperature: Softmax temperature (lower = sharper peaking).

    Returns:
        (aggregated_score, best_view_index)
    """
    best_idx = scores.argmax().item()

    if len(scores) <= 1 or method == "max":
        return scores[best_idx].item(), best_idx

    if method == "mean":
        return scores.mean().item(), best_idx

    if method == "topk_softmax":
        k = min(top_k, len(scores))
        topk_vals, _ = scores.topk(k)
        weights = torch.softmax(topk_vals / temperature, dim=0)
        return (weights * topk_vals).sum().item(), best_idx

    if method == "softmax":
        weights = torch.softmax(scores / temperature, dim=0)
        return (weights * scores).sum().item(), best_idx

    # Unknown method — fall back to max
    logger.warning("Unknown view aggregation method '%s', falling back to max.", method)
    return scores[best_idx].item(), best_idx


# ---------------------------------------------------------------------------
# Datenstruktur fuer DINOv2-Re-Ranking-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class DINOCandidate:
    """A single DINOv2 candidate after re-ranking.

    Attributes:
        object_id: Identifier of the CAD model.
        dino_score: DINOv2 cosine similarity (aggregated view scores).
        dino_score_maxview: Best single-view score (hard max over all
            views). The OSCAR baseline (Pulli et al.) aggregates by max;
            computed independently of ``dino_view_aggregation`` so that
            max-view and softmax arms can be derived from a single pass.
        clip_score: Original CLIP score (for later fusion).
        best_view_path: Path to the most similar rendering.
    """
    object_id: str
    dino_score: float
    clip_score: float
    best_view_path: str = ""
    dino_score_maxview: float = 0.0


@dataclass
class DINOReRankingResult:
    """Result of the DINOv2-based re-ranking (Step 4).

    Attributes:
        candidates: List of the top-K refined candidates.
        query_embedding: DINOv2 embedding of the ROI image.
    """
    candidates: List[DINOCandidate]
    query_embedding: np.ndarray


# ---------------------------------------------------------------------------
# DINOv2 Re-Ranking Modul
# ---------------------------------------------------------------------------

class DINOReRanker:
    """Re-ranks CLIP candidates by visual DINOv2 similarity.

    Compares the ROI image against pre-rendered views of the CAD models
    (8+ views per model, generated via preprocessing/render_views.py).

    This is the core principle of OSCAR:
    1. CLIP filters out semantically unsuitable models.
    2. DINOv2 compares the actual visual appearance.

    Ref: OSCAR – retrieval_combi_clip.py (Stage 2: DINO seg_crop -> ref imgs)

    Usage:
        >>> reranker = DINOReRanker(config)
        >>> reranker.load_reference_images("object_images/ycbv/")
        >>> result = reranker.rerank(roi_image, clip_result, top_k=5)
    """

    CACHE_VERSION = 2  # Bumped: v2 adds SigLIP support
    BATCH_SIZE = 32    # Images per forward pass

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self.processor = None
        self.model = None
        self._encoder_type = getattr(config, "appearance_encoder", "dinov2")

        # Gecachte Referenz-Embeddings: {object_id: [(embedding, path), ...]}
        self._ref_embeddings: Dict[str, List[Tuple[torch.Tensor, str]]] = {}
        # Flache Liste fuer schnellen Batch-Vergleich
        self._all_ref_embs: Optional[torch.Tensor] = None
        self._all_ref_keys: List[Tuple[str, str]] = []  # (object_id, path)

    def _load_model(self):
        """Loads the appearance encoder model (DINOv2 or SigLIP) on first use.

        DINOv2: https://github.com/facebookresearch/dinov2
        SigLIP: https://huggingface.co/google/siglip-base-patch16-224
        """
        if self.model is not None:
            return

        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError(
                "transformers is not installed. Install with:\n"
                "  pip install transformers"
            )

        if self._encoder_type == "siglip":
            model_name = self.config.siglip_model_name
            logger.info("Loading SigLIP model: %s ...", model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).vision_model.to(self.device)
            self.model.eval()
            logger.info("SigLIP vision encoder loaded successfully.")
        else:
            model_name = self.config.dino_model_name
            logger.info("Loading DINOv2 model: %s ...", model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info("DINOv2 loaded successfully.")

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encodes an image into an appearance embedding (DINOv2 or SigLIP).

        Args:
            image: PIL.Image (RGB).

        Returns:
            Normalized tensor (1, D).
        """
        self._load_model()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = self._appearance_features(outputs)
            features = F.normalize(features, p=2, dim=1)
        return features

    def _encode_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Encodes a batch of images into appearance embeddings.

        Args:
            images: List of PIL.Image (RGB).

        Returns:
            Normalized tensor (N, D).
        """
        self._load_model()
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = self._appearance_features(outputs)
            features = F.normalize(features, p=2, dim=1)
        return features

    def _appearance_features(self, outputs) -> torch.Tensor:
        """Global per-image descriptor, per encoder's INTENDED pooling.

        DINOv2 -> its CLS token (or mean patch token), the descriptor it was
        designed around (CNOS convention). SigLIP has NO CLS token — it was
        trained with a multihead-attention-pooling (MAP) head, so its native
        image embedding is ``pooler_output``. The old code ran SigLIP through
        _pool_features too, taking last_hidden_state[:, 0] = an arbitrary FIRST
        PATCH token (a degenerate embedding), which made the DINOv2-vs-SigLIP
        appearance ablation (E4) unfair to SigLIP. Use the MAP head instead."""
        if self._encoder_type == "siglip":
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            # Defensive fallback: mean over patch tokens (NEVER CLS[:,0] for a
            # model without a CLS token).
            return outputs.last_hidden_state.mean(dim=1)
        return self._pool_features(outputs.last_hidden_state)

    def _pool_features(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """Pool patch tokens into a single feature vector.

        Default: CLS token (index 0) — the CNOS convention (Nguyen et al.,
        2023) uses CLS-token cosine similarity for DINOv2 descriptors.
        Mean pooling over patch tokens is retained as legacy option.

        Args:
            last_hidden_state: (B, num_tokens, D) from the ViT.

        Returns:
            (B, D) pooled features.
        """
        pooling = getattr(self.config, "dino_pooling", "cls")
        if pooling == "cls":
            return last_hidden_state[:, 0]
        else:
            return last_hidden_state.mean(dim=1)

    # -------------------------------------------------------------------
    # Cache-Logik
    # -------------------------------------------------------------------

    @staticmethod
    def _dir_fingerprint(ref_dir: str) -> str:
        """Machine-independent fingerprint over the reference views.

        Hashes the sorted (relative-path, size) of every view.  Sizes are
        byte-stable across machines (they survive file copies unchanged),
        whereas the previous mtime-based fingerprint changed after any
        copy/download — which broke reuse of a cache precomputed on
        another PC.  Still changes if views are added/removed/edited.
        """
        entries = []
        for root, _dirs, files in os.walk(ref_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    p = os.path.join(root, f)
                    rel = os.path.relpath(p, ref_dir)
                    try:
                        entries.append(f"{rel}:{os.path.getsize(p)}")
                    except OSError:
                        entries.append(f"{rel}:missing")
        entries.sort()
        raw = (f"v{DINOReRanker.CACHE_VERSION}:{len(entries)}:"
               + "|".join(entries))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self, ref_dir: str) -> str:
        """Path to the cache file inside ref_dir.

        The cache always stores ALL available views.  View filtering
        (config.num_views) is applied after loading so that the same cache
        serves any num_views ablation (O4: V in {8, 16, 42}).
        """
        fp = self._dir_fingerprint(ref_dir)
        if self._encoder_type == "siglip":
            model_tag = self.config.siglip_model_name.replace("/", "_")
            # "_map" marks the MAP-head (pooler_output) embeddings introduced
            # 2026-08-25; a pre-existing patch-0-token cache must NOT be reused.
            return os.path.join(ref_dir, f".siglip_cache_{model_tag}_vall_map_{fp}.pt")
        model_tag = self.config.dino_model_name.replace("/", "_")
        # Pooling mode MUST be part of the cache key: query and gallery must be
        # pooled identically, so a CLS-pooled cache is invalid for a mean run.
        # CLS keeps the historical name (no suffix) so the existing gallery
        # cache is reused; non-cls variants get their own file.
        pooling = getattr(self.config, "dino_pooling", "cls")
        pool_tag = "" if pooling == "cls" else f"_{pooling}"
        return os.path.join(ref_dir, f".dino_cache_{model_tag}_vall{pool_tag}_{fp}.pt")

    def _try_load_cache(self, ref_dir: str) -> bool:
        """Attempts to load cached embeddings.

        Returns:
            True if the cache was loaded, False otherwise.
        """
        cache_file = self._cache_path(ref_dir)
        if not os.path.isfile(cache_file):
            return False
        try:
            data = torch.load(cache_file, map_location=self.device, weights_only=True)
            self._all_ref_embs = data["embeddings"].to(self.device)
            self._all_ref_keys = data["keys"]
            # Rebuild per-object dict
            self._ref_embeddings.clear()
            for i, (obj_id, path) in enumerate(self._all_ref_keys):
                emb = self._all_ref_embs[i]
                self._ref_embeddings.setdefault(obj_id, []).append((emb, path))
            logger.info(
                "DINOv2 embeddings loaded from cache: %d objects, %d views (%s)",
                len(self._ref_embeddings), len(self._all_ref_keys),
                os.path.basename(cache_file),
            )
            return True
        except Exception as e:
            logger.warning("Cache load attempt failed: %s", e)
            return False

    def _save_cache(self, ref_dir: str) -> None:
        """Saves the computed embeddings as a .pt cache."""
        cache_file = self._cache_path(ref_dir)
        try:
            torch.save(
                {
                    "embeddings": self._all_ref_embs.cpu(),
                    "keys": self._all_ref_keys,
                },
                cache_file,
            )
            logger.info("DINOv2 embeddings saved: %s", cache_file)
        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    # -------------------------------------------------------------------
    # Referenzbilder laden (mit Batch + Cache)
    # -------------------------------------------------------------------

    def load_reference_images(self, ref_dir: Optional[str] = None) -> None:
        """Loads and encodes pre-rendered reference images of all CAD models.

        Expects the OSCAR directory layout:
            ref_dir/
                object_label_1/
                    view_001.png
                    view_002.png
                    ...
                object_label_2/
                    ...

        Optimizations over the original:
        - Batched DINOv2 forward passes (BATCH_SIZE images at a time)
        - Disk cache: embeddings are stored as .pt and loaded immediately
          on a subsequent call (fingerprint-based).

        Adapted from: OSCAR – retrieval_combi_clip.py:load_ref_dino_embeddings()

        Args:
            ref_dir: Path to the reference-images directory.
                     If None, config.reference_images_dir is used.
        """
        self._load_model()

        ref_dir = ref_dir or self.config.reference_images_dir
        if not ref_dir:
            raise ValueError("No reference_images_dir configured.")

        # --- Schnellpfad: Cache laden ---
        if self._try_load_cache(ref_dir):
            self._apply_view_limit()
            return

        # --- Kein Cache -> Bilder batchweise encodieren ---
        logger.info(f"Loading reference images from: {ref_dir} (no cache, computing embeddings...)")

        # Schritt 1: Alle Bildpfade sammeln
        # Always encode ALL available views so the cache is reusable across
        # num_views ablations (O4: V in {8, 16, 42}).  View filtering is
        # applied after loading via _apply_view_limit().
        all_paths: List[str] = []
        all_labels: List[str] = []
        for label in sorted(os.listdir(ref_dir)):
            label_dir = os.path.join(ref_dir, label)
            if not os.path.isdir(label_dir):
                continue
            view_files = sorted([
                fname for fname in os.listdir(label_dir)
                if fname.lower().endswith((".png", ".jpg", ".jpeg"))
                and not fname.endswith("_bg.png")
            ], key=_view_sort_key)
            for fname in view_files:
                all_paths.append(os.path.join(label_dir, fname))
                all_labels.append(label)

        if not all_paths:
            logger.warning("No reference images found in %s", ref_dir)
            return

        total = len(all_paths)
        logger.info("  %d reference images found, encoding in batches of %d...",
                     total, self.BATCH_SIZE)

        # Schritt 2: Batch-Encoding
        embs_list: List[torch.Tensor] = []
        keys_list: List[Tuple[str, str]] = []

        batch_imgs: List[Image.Image] = []
        batch_keys: List[Tuple[str, str]] = []

        for i, (img_path, label) in enumerate(zip(all_paths, all_labels)):
            try:
                img = Image.open(img_path).convert("RGB")
            except (OSError, IOError) as e:
                logger.warning(f"Error loading {img_path}: {e}")
                continue

            batch_imgs.append(img)
            batch_keys.append((label, img_path))

            if len(batch_imgs) >= self.BATCH_SIZE:
                batch_emb = self._encode_batch(batch_imgs)  # (B, D)
                embs_list.append(batch_emb.cpu())
                keys_list.extend(batch_keys)
                n_done = len(keys_list)
                if n_done % (self.BATCH_SIZE * 10) == 0 or n_done == total:
                    logger.info("  ... %d / %d encoded (%.0f%%)",
                                n_done, total, 100.0 * n_done / total)
                batch_imgs.clear()
                batch_keys.clear()

        # Letzter partieller Batch
        if batch_imgs:
            batch_emb = self._encode_batch(batch_imgs)
            embs_list.append(batch_emb.cpu())
            keys_list.extend(batch_keys)

        if not embs_list:
            logger.warning("No embeddings computed.")
            return

        # Schritt 3: Zusammenfuegen
        all_embs = torch.cat(embs_list, dim=0)  # (N, D)
        self._all_ref_embs = all_embs.to(self.device)
        self._all_ref_keys = keys_list

        # Per-Object Dict aufbauen
        self._ref_embeddings.clear()
        for i, (obj_id, path) in enumerate(keys_list):
            emb = self._all_ref_embs[i]
            self._ref_embeddings.setdefault(obj_id, []).append((emb, path))

        logger.info(
            "Reference embeddings computed: %d objects, %d views total.",
            len(self._ref_embeddings), len(keys_list),
        )

        # Schritt 4: Cache speichern (all views)
        self._save_cache(ref_dir)

        # Schritt 5: View-Limit anwenden (nach Cache-Speicherung)
        self._apply_view_limit()

    def _apply_view_limit(self) -> None:
        """Filter loaded embeddings to config.num_views per object.

        The cache always stores ALL views.  This method trims to the
        first N views per object so that ablation O4 (V in {8, 16, 42})
        works without rebuilding the cache.
        """
        # Re-sort by numeric view index first: caches built before the
        # natural-sort fix (and any cache, defensively) may hold views in
        # lexicographic order, which would make views[:N] a non-FPS subset.
        for obj_id, views in self._ref_embeddings.items():
            views.sort(key=lambda ep: _view_sort_key(ep[1]))

        max_views = getattr(self.config, "num_views", None)
        if max_views is None:
            return  # Use all views

        trimmed_embeddings: Dict[str, list] = {}
        for obj_id, views in self._ref_embeddings.items():
            trimmed_embeddings[obj_id] = views[:max_views]

        total_before = sum(len(v) for v in self._ref_embeddings.values())
        self._ref_embeddings = trimmed_embeddings
        total_after = sum(len(v) for v in self._ref_embeddings.values())

        if total_after < total_before:
            logger.info(
                "View limit applied: %d → %d views (num_views=%d)",
                total_before, total_after, max_views,
            )

    def rerank(
        self,
        roi_image: Image.Image,
        clip_result: Optional[CLIPRetrievalResult] = None,
        top_k: Optional[int] = None,
    ) -> DINOReRankingResult:
        """Re-ranks candidates by visual DINOv2 similarity.

        If clip_result is passed, only the CLIP candidates are compared
        (faster). Without clip_result, all loaded reference images are
        searched (full search).

        Args:
            roi_image: ROI image of the segmented object (Step 1).
            clip_result: Result of the CLIP search (Step 3), optional.
                         If None, all loaded objects are compared.
            top_k: Number of final candidates (overrides config).

        Returns:
            DINOReRankingResult with refined candidates.
        """
        if not self._ref_embeddings:
            raise RuntimeError(
                "Reference images not loaded. Call load_reference_images() first."
            )

        top_k = top_k or self.config.dino_top_k

        # --- ROI DINOv2 Embedding ---
        query_emb = self.encode_image(roi_image)  # (1, D)

        # --- Kandidatenpool bestimmen ---
        if clip_result is not None:
            # Nur CLIP-Kandidaten vergleichen (schneller)
            clip_score_map = {c.object_id: c.score for c in clip_result.candidates}
            search_ids = [c.object_id for c in clip_result.candidates
                          if c.object_id in self._ref_embeddings]
            mode_label = f"CLIP-filtered ({len(search_ids)} objects)"
        else:
            # Alle geladenen Objekte vergleichen (volle Suche)
            clip_score_map = {}
            search_ids = list(self._ref_embeddings.keys())
            mode_label = f"full search ({len(search_ids)} objects)"

        encoder_label = "SigLIP" if self._encoder_type == "siglip" else "DINOv2"
        logger.info("%s rerank mode: %s", encoder_label, mode_label)

        candidate_embs = []
        candidate_keys = []
        for obj_id in search_ids:
            for emb, path in self._ref_embeddings[obj_id]:
                candidate_embs.append(emb)
                candidate_keys.append((obj_id, path))

        if not candidate_embs:
            logger.warning("No reference images found for the candidates.")
            return DINOReRankingResult(
                candidates=[],
                query_embedding=query_emb.cpu().numpy(),
            )

        # --- Cosine Similarity berechnen ---
        cand_tensor = torch.stack(candidate_embs).to(self.device)  # (K, D)
        sims = (query_emb @ cand_tensor.T).squeeze(0)  # (K,)

        # --- Group view scores by object ---
        obj_view_scores: Dict[str, List[Tuple[float, str]]] = {}
        for idx, (obj_id, path) in enumerate(candidate_keys):
            obj_view_scores.setdefault(obj_id, []).append(
                (sims[idx].item(), path)
            )

        # --- Aggregate per-object using configurable strategy ---
        agg_method = self.config.dino_view_aggregation
        agg_topk = self.config.dino_view_topk
        agg_tau = self.config.dino_view_temperature

        scored_objects: List[Tuple[str, float, float, str]] = []
        for obj_id, view_list in obj_view_scores.items():
            view_scores_t = torch.tensor(
                [s for s, _ in view_list], device=self.device
            )
            agg_score, best_local_idx = _aggregate_view_scores(
                view_scores_t, method=agg_method, top_k=agg_topk,
                temperature=agg_tau,
            )
            # Hard best-view score (OSCAR baseline aggregation), computed
            # alongside the configured aggregation from the same per-view sims.
            maxview_score = float(view_scores_t.max().item())
            best_path = view_list[best_local_idx][1]
            scored_objects.append(
                (obj_id, agg_score, maxview_score, best_path))

        # --- Sort by aggregated score ---
        scored_objects.sort(key=lambda x: x[1], reverse=True)

        candidates = []
        for obj_id, dino_score, maxview_score, best_path in scored_objects[:top_k]:
            candidates.append(DINOCandidate(
                object_id=obj_id,
                dino_score=dino_score,
                clip_score=clip_score_map.get(obj_id, 0.0),
                best_view_path=best_path,
                dino_score_maxview=maxview_score,
            ))

        logger.info(
            "%s Re-Ranking (%s, k=%d, τ=%.2f): %d candidates "
            "(Top: %s, score=%.4f)",
            encoder_label, agg_method, agg_topk, agg_tau, len(candidates),
            candidates[0].object_id, candidates[0].dino_score,
        )

        return DINOReRankingResult(
            candidates=candidates,
            query_embedding=query_emb.cpu().numpy(),
        )