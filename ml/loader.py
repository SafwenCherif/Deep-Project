"""
ml/loader.py
============
Loads both Keras models exactly once at application startup.

Expected model files (configure paths via environment variables or .env):
  CLASSIFIER_PATH  →  brain_tumor_resnet50_pipeline_f.keras
  SEGMENTOR_PATH   →  Brain_Tumor_Segmentation_FTL_Winner.keras
"""

import os
import logging

import tensorflow as tf
import tensorflow.keras.backend as K

log = logging.getLogger(__name__)

# ── File paths ─────────────────────────────────────────────────────────────
# Override with environment variables in production.
CLASSIFIER_PATH = os.getenv(
    "CLASSIFIER_PATH",
    "models/brain_tumor_resnet50_pipeline_f.keras",
)
SEGMENTOR_PATH = os.getenv(
    "SEGMENTOR_PATH",
    "models/Brain_Tumor_Segmentation_FTL_Winner.keras",
)


# ── Custom loss required to load the segmentation model ───────────────────
def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=0.75):
    """
    Focal Tversky Loss — must be registered as a custom object
    so Keras can deserialize the segmentor weights correctly.

    alpha=0.3  →  mild FP penalty
    beta=0.7   →  heavy FN penalty (don't miss real tumour pixels)
    gamma=0.75 →  focal factor (focus on hard boundary pixels)
    """
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))

    TP = K.sum(y_true_f * y_pred_f)
    FP = K.sum((1 - y_true_f) * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))

    tversky_index = (TP + K.epsilon()) / (
        TP + alpha * FP + beta * FN + K.epsilon()
    )
    return K.pow((1 - tversky_index), gamma)


def _extract_input_size(model: tf.keras.Model) -> int:
    """
    Read the spatial input size (H or W, assumed square) straight from
    the model's own input_shape — the single source of truth.
    Handles both (None, H, W, C) and (H, W, C) shapes.
    """
    shape = model.input_shape  # e.g. (None, 224, 224, 3)
    # shape[1] is H when the first dim is the batch dim (None)
    size = shape[1] if shape[0] is None else shape[0]
    if size is None:
        raise ValueError(
            f"Cannot auto-detect input size from model.input_shape={shape}. "
            "Set CLASSIFIER_INPUT_SIZE / SEGMENTOR_INPUT_SIZE manually in .env"
        )
    return int(size)


def _load_classifier(path: str) -> tuple[tf.keras.Model, int]:
    """Load the ResNet50 classifier. Returns (model, input_size)."""
    log.info(f"  Loading classifier from: {path}")
    model = tf.keras.models.load_model(path, compile=False)
    size = _extract_input_size(model)
    log.info(
        f"  Classifier → input_shape={model.input_shape}  "
        f"output_shape={model.output_shape}  detected_size={size}"
    )
    return model, size


def _load_segmentor(path: str) -> tuple[tf.keras.Model, int]:
    """
    Load the UNet segmentor with focal_tversky_loss as a custom object
    (required so Keras can deserialise the .keras file correctly).
    Returns (model, input_size).
    """
    log.info(f"  Loading segmentor  from: {path}")
    model = tf.keras.models.load_model(
        path,
        custom_objects={"focal_tversky_loss": focal_tversky_loss},
        compile=False,
    )
    size = _extract_input_size(model)
    log.info(
        f"  Segmentor  → input_shape={model.input_shape}  "
        f"output_shape={model.output_shape}  detected_size={size}"
    )
    return model, size


def load_models() -> dict:
    """
    Entry point called once at FastAPI startup.

    Returns a dict with keys:
        'classifier'           – tf.keras.Model
        'classifier_size'      – int  (e.g. 224)
        'segmentor'            – tf.keras.Model
        'segmentor_size'       – int  (e.g. 256)

    Sizes are read directly from each model's input_shape, so they are
    always correct regardless of what is (or isn't) set in .env.
    """
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"Classifier model not found at '{CLASSIFIER_PATH}'. "
            "Set the CLASSIFIER_PATH environment variable to the correct path."
        )
    if not os.path.exists(SEGMENTOR_PATH):
        raise FileNotFoundError(
            f"Segmentor model not found at '{SEGMENTOR_PATH}'. "
            "Set the SEGMENTOR_PATH environment variable to the correct path."
        )

    classifier, clf_size = _load_classifier(CLASSIFIER_PATH)
    segmentor,  seg_size = _load_segmentor(SEGMENTOR_PATH)

    log.info(f"✅  Classifier input size : {clf_size}×{clf_size}")
    log.info(f"✅  Segmentor  input size : {seg_size}×{seg_size}")

    return {
        "classifier":      classifier,
        "classifier_size": clf_size,
        "segmentor":       segmentor,
        "segmentor_size":  seg_size,
    }