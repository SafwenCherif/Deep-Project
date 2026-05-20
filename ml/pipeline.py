"""
ml/pipeline.py
==============
The full two-stage cascade pipeline.

Input sizes are auto-detected from each model's input_shape in loader.py
and passed in at call time — so the pipeline can never send the wrong
tensor shape to a model.
"""

import os
import logging
from typing import Any

import numpy as np
from PIL import Image
import tensorflow as tf

from ml.image_utils import (
    preprocess_for_classifier,
    preprocess_for_segmentor,
    build_segmentation_overlay,
)

log = logging.getLogger(__name__)

# ── Tuneable thresholds only (sizes come from the model objects) ───────────
CLASSIFICATION_THRESHOLD: float = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.25"))
SEGMENTATION_THRESHOLD:   float = float(os.getenv("SEGMENTATION_THRESHOLD",   "0.50"))


# ── Stage 1: Classification ────────────────────────────────────────────────

def classify(
    pil_image: Image.Image,
    classifier: tf.keras.Model,
    classifier_size: int,
) -> tuple[float, str]:
    """
    Run the ResNet50 classifier.

    The model uses softmax with 2 outputs:
        index 0 → Healthy probability
        index 1 → Tumour probability   <- we use this

    classifier_size is detected from model.input_shape in loader.py
    (e.g. 224) and is guaranteed to be correct.
    """
    x = preprocess_for_classifier(pil_image, target_size=classifier_size)
    probs = classifier.predict(x, verbose=0)    # shape: (1, 2)

    tumour_prob: float = float(probs[0, 1])
    status = "tumour_detected" if tumour_prob >= CLASSIFICATION_THRESHOLD else "healthy"

    log.info(
        f"  Classification -> p(tumour)={tumour_prob:.4f}  "
        f"threshold={CLASSIFICATION_THRESHOLD}  "
        f"input_size={classifier_size}  status={status}"
    )
    return tumour_prob, status


# ── Stage 2: Segmentation ──────────────────────────────────────────────────

def segment(
    pil_image: Image.Image,
    segmentor: tf.keras.Model,
    segmentor_size: int,
) -> dict:
    """
    Run the UNet segmentor on the confirmed-tumour image.

    segmentor_size is detected from model.input_shape in loader.py
    (e.g. 256) and is guaranteed to be correct.
    """
    x = preprocess_for_segmentor(pil_image, target_size=segmentor_size)
    raw_output = segmentor.predict(x, verbose=0)  # shape: (1, H, W, 1) or (1, H, W)

    # Squeeze batch + channel dims -> 2-D float32 probability map
    prob_mask: np.ndarray = raw_output[0].squeeze()

    log.info(
        f"  Segmentation  -> mask shape={prob_mask.shape}  "
        f"p_min={prob_mask.min():.3f}  p_max={prob_mask.max():.3f}  "
        f"p_mean={prob_mask.mean():.3f}  input_size={segmentor_size}"
    )

    overlay_b64, heatmap_b64, coverage_pct = build_segmentation_overlay(
        pil_image=pil_image,
        prob_mask=prob_mask,
        seg_size=segmentor_size,
        mask_threshold=SEGMENTATION_THRESHOLD,
    )

    log.info(f"  Tumour coverage: {coverage_pct:.2f}% of image pixels")

    return {
        "segmentation_overlay": overlay_b64,
        "confidence_heatmap": heatmap_b64,
        "segmentation_coverage_pct": round(coverage_pct, 4),
    }


# ── Full cascade pipeline ──────────────────────────────────────────────────

def run_pipeline(
    pil_image: Image.Image,
    classifier: tf.keras.Model,
    classifier_size: int,
    segmentor: tf.keras.Model,
    segmentor_size: int,
) -> dict[str, Any]:
    """
    Orchestrates the two-stage pipeline.

    Healthy path  ->  returns classification result only (fast).
    Tumour path   ->  classification then segmentation, returns both.
    """

    # Stage 1
    tumour_prob, status = classify(pil_image, classifier, classifier_size)

    result: dict[str, Any] = {
        "status": status,
        "tumour_probability": round(tumour_prob, 6),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "segmentation_overlay": None,
        "confidence_heatmap": None,
        "segmentation_coverage_pct": None,
    }

    # Stage 2 (only for tumour cases)
    if status == "tumour_detected":
        seg_result = segment(pil_image, segmentor, segmentor_size)
        result.update(seg_result)

    return result