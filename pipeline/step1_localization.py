# =============================================================================
# pipeline/step1_localization.py – Thesis Step A: Scene Grounding
# =============================================================================
#
# Implements the prompt-conditioned detection paradigm from OSCAR
# (Pulli et al., 2025) extended with mask post-processing.
#
# Pipeline:
#   prompt → GroundingDINO (bounding box) → SAM2.1 (segmentation mask)
#         → Mask refinement (largest CC + dilation)
#
# Models:
#   • GroundingDINO – Open-Set Object Detection (Liu et al., 2023)
#     Detection confidence threshold 0.3 follows OSCAR convention.
#     Ref: https://github.com/IDEA-Research/GroundingDINO
#
#   • SAM2.1 – Segment Anything Model 2.1 (Ravi et al., 2024)
#     Ref: https://github.com/facebookresearch/sam2
#
# Mask post-processing (thesis Sec. 3.2, Step A):
#   1. Largest connected component retention — mitigates SAM
#      over-segmentation and fragmentation (Almazroey et al., 2025;
#      Zhang et al., 2024; Tai et al., 2025). Applied before CLIP
#      feature extraction following SAMURAI (Vo et al., 2025).
#   2. Mask dilation — compensates for depth-shadow effect at object
#      boundaries in structured-light sensors (Shen et al., 2013) and
#      flying-pixel artefacts in ToF sensors (Chugunov et al., 2021).
#      Clean masks are critical for downstream ICP-based geometric
#      alignment (Caraffa et al., 2025 — FreeZe).
#
# Outputs:
#   - RGB image (original)
#   - Segmentation mask (binary, H×W)
#   - Bounding box [x_min, y_min, x_max, y_max]
#   - ROI crop (cropped image of the object)
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2

import numpy as np
import torch
from PIL import Image

from .config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenstruktur für Lokalisierungsergebnisse
# ---------------------------------------------------------------------------

@dataclass
class LocalizationResult:
    """Result of object localization (Step 1).

    Attributes:
        rgb_image: Original RGB image (PIL).
        mask: Binary segmentation mask (H, W), bool.
        bbox: Bounding box [x_min, y_min, x_max, y_max] in pixels.
        roi_image: Cropped image of the detected object.
        confidence: Detection confidence.
        prompt: Language prompt used.
    """
    rgb_image: Image.Image
    mask: np.ndarray
    bbox: List[float]
    roi_image: Image.Image
    confidence: float
    prompt: str


# ---------------------------------------------------------------------------
# Lokalisierungsmodul
# ---------------------------------------------------------------------------

class ObjectLocalizer:
    """Localizes objects in RGB images with GroundingDINO + SAM2.1.

    Uses HuggingFace transformers directly (no LangSAM required).
    GroundingDINO provides bounding boxes, SAM2.1 segments precisely.

    Ref:
        GroundingDINO: https://huggingface.co/IDEA-Research/grounding-dino-base
        SAM2.1: https://huggingface.co/facebook/sam2.1-hiera-large

    Usage:
        >>> localizer = ObjectLocalizer(config)
        >>> result = localizer.localize(image, "mayonnaise tube")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self._gdino_model = None
        self._gdino_processor = None
        self._sam_model = None
        self._sam_processor = None

    def _load_model(self):
        """Loads GroundingDINO + SAM2.1 via HuggingFace transformers."""
        if self._gdino_model is not None:
            return

        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
            Sam2Config,
            Sam2Model,
            Sam2Processor,
        )

        gdino_id = self.config.grounding_dino_model
        sam_id = self.config.sam_model

        logger.info("Loading GroundingDINO (%s)...", gdino_id)
        self._gdino_processor = AutoProcessor.from_pretrained(gdino_id)
        self._gdino_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id)
            .to(self.device)
            .eval()
        )

        logger.info("Loading SAM2.1 (%s)...", sam_id)
        self._sam_processor = Sam2Processor.from_pretrained(sam_id)
        # The facebook/sam2.1-hiera-large checkpoint declares model_type
        # "sam2_video" in its config.json, but Sam2Model (image segmentation)
        # expects "sam2".  The architectures are compatible for single-image
        # use; only the metadata differs.  Load config explicitly and fix
        # model_type to suppress the spurious warning.
        sam_config = Sam2Config.from_pretrained(sam_id)
        sam_config.model_type = "sam2"
        self._sam_model = (
            Sam2Model.from_pretrained(sam_id, config=sam_config)
            .to(self.device)
            .eval()
        )

        logger.info("GroundingDINO + SAM2.1 loaded successfully.")

    def _detect(self, rgb_image: Image.Image, prompt: str):
        """Runs GroundingDINO detection.

        Returns:
            Dict with 'scores' (Tensor), 'labels' (list[str]), 'boxes' (Tensor).
        """
        # GroundingDINO erwartet Prompt mit abschließendem Punkt
        text = prompt.strip()
        if not text.endswith("."):
            text += "."

        inputs = self._gdino_processor(
            images=rgb_image, text=text, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._gdino_model(**inputs)

        results = self._gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.detection_confidence,
            text_threshold=self.config.detection_confidence,
            target_sizes=[(rgb_image.height, rgb_image.width)],
        )
        return results[0]  # single image

    def _segment(self, rgb_image: Image.Image, bbox: List[float]) -> np.ndarray:
        """Produces a SAM2.1 mask from a bounding-box prompt.

        Args:
            rgb_image: RGB image.
            bbox: [x1, y1, x2, y2] in pixel coordinates.

        Returns:
            Binary mask (H, W), bool.
        """
        input_boxes = [[[bbox[0], bbox[1], bbox[2], bbox[3]]]]
        inputs = self._sam_processor(
            images=rgb_image, input_boxes=input_boxes, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._sam_model(**inputs)

        masks = self._sam_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )
        # masks[0] shape: (1, 3, H, W) – 3 Masken pro Box, beste wählen
        iou_scores = outputs.iou_scores.cpu()  # (1, 1, 3)
        best_mask_idx = iou_scores[0, 0].argmax().item()
        return masks[0][0, best_mask_idx].numpy().astype(bool)

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """Post-process the segmentation mask (thesis Sec. 3.2, Step A).

        1. Largest connected component — SAM-family models exhibit
           over-segmentation under challenging conditions (Almazroey et al.,
           2025), degraded quality for low-contrast objects (Zhang et al.,
           2024), and fragmentation under occlusion (Tai et al., 2025).
           SAMURAI (Vo et al., 2025) applies this step before feature
           extraction to reduce background noise influence on scores.

        2. Mask dilation — compensates for depth-shadow at object
           boundaries: structured-light sensors leave missing-depth regions
           due to projector/sensor viewpoint disparity (Shen et al., 2013);
           ToF sensors produce flying-pixel artefacts (Chugunov et al., 2021).
           FreeZe (Caraffa et al., 2025) notes that clean masks are critical
           for ICP-based geometric alignment in Step C.

        Args:
            mask: Binary mask (H, W), bool.

        Returns:
            Refined binary mask (H, W), bool.
        """
        mask_uint8 = mask.astype(np.uint8)

        # 1. Largest connected component
        if getattr(self.config, "mask_largest_cc", True):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_uint8, connectivity=8
            )
            if num_labels > 2:  # background + at least 2 components
                # stats[:, cv2.CC_STAT_AREA] — label 0 is background
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_label = 1 + areas.argmax()
                mask_uint8 = (labels == largest_label).astype(np.uint8)
                logger.debug(
                    "Mask: kept largest CC (label %d, %d px) out of %d components",
                    largest_label, stats[largest_label, cv2.CC_STAT_AREA],
                    num_labels - 1,
                )

        # 2. Dilation for depth-shadow compensation
        dilation_iters = getattr(self.config, "mask_dilation_iterations", 1)
        if dilation_iters > 0:
            kernel_size = getattr(self.config, "mask_dilation_kernel", 5)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (kernel_size, kernel_size)
            )
            mask_uint8 = cv2.dilate(
                mask_uint8, kernel, iterations=dilation_iters
            )
            logger.debug(
                "Mask: dilated with %dx%d kernel, %d iteration(s)",
                kernel_size, kernel_size, dilation_iters,
            )

        return mask_uint8.astype(bool)

    def localize(
        self,
        rgb_image: Image.Image,
        prompt: str,
        top_k: int = 1,
    ) -> Optional[LocalizationResult]:
        """Localizes an object in the image from a language prompt.

        Args:
            rgb_image: RGB input image (PIL).
            prompt: Natural-language description of the target object,
                    e.g. "mayonnaise tube".
            top_k: Number of detections requested (default: 1, the best).

        Returns:
            LocalizationResult with mask, bbox and ROI crop,
            or None if no object was found.
        """
        self._load_model()

        det = self._detect(rgb_image, prompt)

        if len(det["scores"]) == 0:
            logger.warning("No object found for prompt '%s'.", prompt)
            return None

        # Beste Detektion auswählen (höchste Konfidenz)
        best_idx = det["scores"].argmax().item()
        bbox = det["boxes"][best_idx].cpu().tolist()  # [x1, y1, x2, y2]
        confidence = det["scores"][best_idx].item()
        label = det["labels"][best_idx]

        logger.info(
            "Object found: '%s' (confidence: %.3f, bbox: %s)",
            label, confidence, bbox,
        )

        # SAM2.1-Segmentierung mit BBox-Prompt
        mask_np = self._segment(rgb_image, bbox)

        # --- Mask post-processing (thesis Step A) ---
        mask_np = self._refine_mask(mask_np)

        # --- ROI-Ausschnitt erzeugen ---
        roi_image = self._extract_roi(rgb_image, mask_np, bbox)

        return LocalizationResult(
            rgb_image=rgb_image,
            mask=mask_np,
            bbox=bbox,
            roi_image=roi_image,
            confidence=confidence,
            prompt=prompt,
        )

    def localize_all(
        self,
        rgb_image: Image.Image,
        prompt: str,
    ) -> List[LocalizationResult]:
        """Returns all detected objects (not just the best one).

        Useful when several instances of the same object type are in the image.

        Args:
            rgb_image: RGB input image.
            prompt: Language prompt.

        Returns:
            List of LocalizationResult objects, sorted by confidence.
        """
        self._load_model()

        det = self._detect(rgb_image, prompt)

        if len(det["scores"]) == 0:
            logger.warning("No objects found for prompt '%s'.", prompt)
            return []

        results = []
        sorted_indices = det["scores"].argsort(descending=True)

        for idx in sorted_indices:
            idx = idx.item()
            bbox = det["boxes"][idx].cpu().tolist()
            confidence = det["scores"][idx].item()

            mask_np = self._segment(rgb_image, bbox)
            roi_image = self._extract_roi(rgb_image, mask_np, bbox)

            results.append(LocalizationResult(
                rgb_image=rgb_image,
                mask=mask_np,
                bbox=bbox,
                roi_image=roi_image,
                confidence=confidence,
                prompt=prompt,
            ))

        return results

    @staticmethod
    def _extract_roi(
        image: Image.Image,
        mask: np.ndarray,
        bbox: List[float],
        background_color: Tuple[int, int, int] = (205, 205, 205),
    ) -> Image.Image:
        """Produces an ROI crop with a neutral background.

        Adapted from OSCAR – object_retrieval/i2i_seg_clip.py:crop_with_mask()

        Args:
            image: Original image (PIL).
            mask: Binary mask (bool, H×W).
            bbox: [x1, y1, x2, y2] bounding box.
            background_color: Background color.

        Returns:
            Cropped PIL image.
        """
        img_array = np.array(image)
        canvas = np.full_like(img_array, background_color, dtype=np.uint8)
        canvas[mask] = img_array[mask]

        # Bounding Box aus Maske für sauberen Zuschnitt
        coords = np.argwhere(mask)
        if len(coords) == 0:
            # Fallback auf BBox
            x1, y1, x2, y2 = [int(c) for c in bbox]
            return Image.fromarray(canvas[y1:y2, x1:x2])

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return Image.fromarray(canvas[y0:y1, x0:x1])
