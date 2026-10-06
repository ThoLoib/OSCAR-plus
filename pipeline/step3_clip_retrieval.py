# =============================================================================
# pipeline/step3_clip_retrieval.py – Thesis Step B1: Semantic Channel S_text
# =============================================================================
#
# Computes the text-based semantic score S_text (thesis Sec. 3.3, Step B1).
#
# CLIP (Radford et al., 2021) performs image–text alignment via contrastive
# pretraining. OSCAR (Pulli et al., 2025) establishes CLIP as competitive
# for caption-based CAD retrieval in the training-free setting.
#
# The text channel uses offline-generated natural-language descriptions of
# each CAD model. At query time, the ROI image embedding is compared against
# all description embeddings via cosine similarity.
#
# In the thesis default (full-database scoring), all candidates are scored.
# The OSCAR-style cascade (CLIP top-k → DINOv2/ULIP) is retained as
# ablation O2 (Pulli et al., 2025).
#
# Model:
#   • CLIP ViT-B/32 (Radford et al., 2021)
#     Ref: https://github.com/openai/CLIP
#
# Adapted from: OSCAR – object_retrieval/retrieval_combi_clip.py
#
# Outputs:
#   - List of (object_id, similarity_score) tuples, sorted by score
# =============================================================================

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F

from .config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur für CLIP-Retrieval-Ergebnisse
# ---------------------------------------------------------------------------

@dataclass
class CLIPCandidate:
    """A single CLIP candidate.

    Attributes:
        object_id: Identifier of the CAD model / object.
        score: Cosine similarity between query and description.
        description: The matched text description.
    """
    object_id: str
    score: float
    description: str = ""


@dataclass
class CLIPRetrievalResult:
    """Result of the CLIP-based candidate search (Step 3).

    Attributes:
        candidates: List of the top-K candidates, sorted by score.
        query_embedding: CLIP embedding of the ROI image (for later fusion).
        all_scores: Full score vector against all descriptions.
    """
    candidates: List[CLIPCandidate]
    query_embedding: np.ndarray
    all_scores: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# CLIP Retrieval Modul
# ---------------------------------------------------------------------------

class CLIPRetriever:
    """Semantic object search via CLIP image-text matching.

    Compares the ROI image against pre-generated text descriptions of the
    CAD models in the object database (the OSCAR principle).

    The retrieval process:
    1. ROI image → CLIP image encoder → image embedding
    2. CAD descriptions → CLIP text encoder → text embeddings (precomputed)
    3. Cosine similarity → top-K candidates

    Ref: OSCAR Pipeline – retrieval_combi_clip.py (encode_image_clip, encode_texts_clip)

    Usage:
        >>> retriever = CLIPRetriever(config)
        >>> retriever.load_descriptions("object_database/ycbv/descriptions.json")
        >>> result = retriever.retrieve(roi_image, top_k=20)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self.model = None
        self.preprocess = None

        # Vorab berechnete Embeddings der Beschreibungen
        self._desc_embeddings: Optional[torch.Tensor] = None
        self._desc_texts: List[str] = []
        self._desc_labels: List[str] = []

    def _load_model(self):
        """Loads the CLIP model on first use.

        Ref: https://github.com/openai/CLIP
        """
        if self.model is not None:
            return

        logger.info(f"Loading CLIP model: {self.config.clip_model_name}...")
        try:
            import clip
            self.model, self.preprocess = clip.load(
                self.config.clip_model_name, device=self.device
            )
            logger.info("CLIP loaded successfully.")
        except ImportError:
            raise ImportError(
                "CLIP is not installed. Install with:\n"
                "  pip install git+https://github.com/openai/CLIP.git\n"
                "Ref: https://github.com/openai/CLIP"
            )

    # Foreign-view descriptions: 3-digit zero-padded view names (``_002.png``)
    # are stray renders, not one of the 42 real views (``_0.png``..``_41.png``).
    # The image loader quarantines them (see stage2_mi3dor.
    # _quarantine_foreign_views, SAME regex); the descriptions JSON was built
    # before that and still carries them, giving ~1817 MI3DOR objects a 43rd
    # (foreign) description that biases CLIP scoring. Drop them so CLIP scores
    # exactly the 42 real views per object, consistent with DINO/ULIP.
    _FOREIGN_VIEW_RE = re.compile(r"_[0-9]{3,}\.png$")

    @staticmethod
    def _load_object_descriptions(desc_file: str) -> Tuple[List[str], List[str]]:
        """Loads object descriptions from an OSCAR-compatible JSON file.

        Format: {object_id: {"image_descriptions": {"view_name": "text", ...}}}

        Foreign-view descriptions (3-digit ``_NNN.png`` view keys) are skipped
        so CLIP scores only the 42 real views, matching the image quarantine.

        Args:
            desc_file: Path to the JSON file.

        Returns:
            (texts, labels) – list of all description texts and their label IDs.
        """
        with open(desc_file, "r") as f:
            descriptions = json.load(f)

        texts: List[str] = []
        labels: List[str] = []
        n_foreign = 0
        for obj_id, entry in descriptions.items():
            for view_name, text in entry.get("image_descriptions", {}).items():
                if CLIPRetriever._FOREIGN_VIEW_RE.search(str(view_name)):
                    n_foreign += 1
                    continue
                texts.append(text)
                labels.append(obj_id)
        if n_foreign:
            logger.info(
                "CLIP descriptions: skipped %d foreign-view (_NNN.png) entries; "
                "keeping only the 42 real views per object.", n_foreign)
        return texts, labels

    def load_descriptions(
        self,
        desc_file: Optional[str] = None,
        id_to_label: Optional[Dict[str, str]] = None,
    ) -> None:
        """Loads and encodes all CAD object descriptions.

        The descriptions are passed through the CLIP text encoder once and
        then cached.

        Args:
            desc_file: Path to the descriptions JSON (OSCAR format).
                       If None, config.description_file is used.
            id_to_label: Optional mapping from object IDs to labels.
        """
        self._load_model()

        desc_file = desc_file or self.config.description_file
        if not desc_file:
            raise ValueError("No description_file configured.")

        logger.info(f"Loading descriptions from: {desc_file}")
        self._desc_texts, self._desc_labels = self._load_object_descriptions(desc_file)

        # Optional: IDs zu menschenlesbaren Labels umwandeln
        if id_to_label:
            self._desc_labels = [
                id_to_label.get(lbl, lbl) for lbl in self._desc_labels
            ]

        cache_path = self._cache_path(desc_file)
        if self._try_load_cache(cache_path):
            return

        logger.info(f"Encoding {len(self._desc_texts)} descriptions with CLIP...")
        self._desc_embeddings = self._encode_texts_batch(self._desc_texts)
        logger.info("Description embeddings computed.")
        self._save_cache(cache_path)

    def _cache_path(self, desc_file: str) -> str:
        """Cache path for the text embeddings, next to the descriptions file.

        Fingerprint = CLIP model name + description texts (content, not
        path/mtime) → stable across machines, like the DINO/ULIP caches.
        Labels are deliberately NOT part of the fingerprint: a different
        id_to_label mapping does not change the encoded texts, so the cache
        stays valid.
        """
        model_tag = self.config.clip_model_name.replace("/", "_")
        raw = f"v1:{len(self._desc_texts)}\n" + "\n".join(self._desc_texts)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.dirname(os.path.abspath(desc_file))
        return os.path.join(cache_dir, f".clip_text_cache_{model_tag}_{digest}.pt")

    def _try_load_cache(self, cache_path: str) -> bool:
        """Loads cached text embeddings if present."""
        if not os.path.isfile(cache_path):
            return False
        try:
            data = torch.load(cache_path, map_location=self.device, weights_only=True)
            self._desc_embeddings = data["embeddings"].to(self.device)
            logger.info("CLIP text cache loaded: %s", os.path.basename(cache_path))
            return True
        except Exception as e:
            logger.warning(
                "CLIP text cache could not be loaded (%s), re-encoding.", e
            )
            return False

    def _save_cache(self, cache_path: str) -> None:
        """Saves the text embeddings to disk."""
        try:
            torch.save({"embeddings": self._desc_embeddings.cpu()}, cache_path)
            logger.info("CLIP text embeddings saved: %s", cache_path)
        except OSError as e:
            logger.warning("Could not save CLIP text cache: %s", e)

    def _encode_texts_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> torch.Tensor:
        """Encodes a list of texts into CLIP embeddings.

        Args:
            texts: List of description texts.
            batch_size: Batch size for encoding.

        Returns:
            Normalized tensor (N, D) on the configured device.
        """
        import clip

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_text(tokens)
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb)

        return torch.cat(all_embeddings, dim=0)

    def encode_image(self, image) -> torch.Tensor:
        """Encodes an image into a CLIP embedding.

        Args:
            image: PIL.Image (RGB).

        Returns:
            Normalized tensor (1, D).
        """
        self._load_model()
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
        return F.normalize(emb, p=2, dim=1)

    def retrieve(
        self,
        roi_image,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        text_query: Optional[str] = None,
        text_query_weight: float = 0.0,
    ) -> CLIPRetrievalResult:
        """Finds the top-K semantically most similar CAD models.

        Args:
            roi_image:          PIL.Image of the segmented object (Step 1).
            top_k:              Number of candidates (overrides config).
            threshold:          Minimum similarity (optional, alternative to top_k).
            text_query:         Optional text query (e.g. "yellow mustard bottle").
                                If given, image and text similarity are mixed:
                                ``score = (1-w)·img_sim + w·text_sim``
            text_query_weight:  Weight of the text query (default: 0.0).

        Returns:
            CLIPRetrievalResult with sorted candidates.
        """
        if self._desc_embeddings is None:
            raise RuntimeError(
                "Descriptions not loaded. Call load_descriptions() first."
            )

        top_k = top_k or self.config.clip_top_k

        # --- Image Embedding ---
        query_emb = self.encode_image(roi_image)  # (1, D)

        # --- Cosine Similarity: Bild vs. Beschreibungen ---
        img_sims = (query_emb @ self._desc_embeddings.T).squeeze(0)  # (M,)

        # --- Optional: Text-Query mischen ---
        if text_query:
            import clip
            tokens = clip.tokenize([text_query], truncate=True).to(self.device)
            with torch.no_grad():
                txt_emb = self.model.encode_text(tokens)
            txt_emb = F.normalize(txt_emb, p=2, dim=1)  # (1, D)
            txt_sims = (txt_emb @ self._desc_embeddings.T).squeeze(0)  # (M,)
            sims = (1.0 - text_query_weight) * img_sims + text_query_weight * txt_sims
            logger.debug(
                "CLIP text_query=%r (w=%.2f) mixed in.", text_query, text_query_weight
            )
        else:
            sims = img_sims

        # --- Top-K oder Threshold-basierte Filterung ---
        if threshold is not None:
            keep_mask = sims >= threshold
            if keep_mask.sum() == 0:
                logger.warning(
                    f"No candidate above threshold {threshold}. "
                    f"Falling back to top-{top_k}."
                )
                keep_indices = sims.topk(top_k).indices
            else:
                keep_indices = keep_mask.nonzero(as_tuple=True)[0]
                # Sortiere nach Score
                keep_scores = sims[keep_indices]
                sorted_order = keep_scores.argsort(descending=True)
                keep_indices = keep_indices[sorted_order][:top_k]
        else:
            keep_indices = sims.topk(min(top_k, len(sims))).indices

        # --- Kandidaten aufbauen ---
        # keep_indices is score-descending in both branches, so the first row
        # seen for each object is its best (max) view — object score = max over
        # its description rows. Gather all kept scores to CPU in ONE transfer to
        # avoid a per-row .item() GPU sync (161k syncs/query on MI3DOR ≈ +1s);
        # the result is bit-identical, just ~1s/query faster.
        keep_indices_list = keep_indices.tolist()
        keep_scores_list = sims[keep_indices].cpu().tolist()
        candidates = []
        seen_objects = set()  # Deduplizierung auf Objekt-Ebene
        for pos, idx in enumerate(keep_indices_list):
            obj_id = self._desc_labels[idx]
            # Pro Objekt den besten Score behalten (erste = höchste)
            if obj_id not in seen_objects:
                candidates.append(CLIPCandidate(
                    object_id=obj_id,
                    score=keep_scores_list[pos],
                    description=self._desc_texts[idx],
                ))
                seen_objects.add(obj_id)

        logger.info(
            f"CLIP retrieval: {len(candidates)} candidates "
            f"(top score: {candidates[0].score:.4f} – {candidates[0].object_id})"
        )

        return CLIPRetrievalResult(
            candidates=candidates,
            query_embedding=query_emb.cpu().numpy(),
            all_scores=sims.cpu().numpy(),
        )

    def get_candidate_labels(self, result: CLIPRetrievalResult) -> List[str]:
        """Extracts the object IDs from a CLIP retrieval result."""
        return [c.object_id for c in result.candidates]
