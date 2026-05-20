"""
ml/image_utils.py
=================
Shared preprocessing helpers for both the classifier and segmentor.
Keeps all NumPy / PIL image manipulation in one place.
"""

import io
import base64
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── Preprocessing ──────────────────────────────────────────────────────────

def preprocess_for_classifier(pil_image: Image.Image, target_size: int = 224) -> np.ndarray:
    """
    Resize → RGB → float32 → divide by 255.
    Returns shape (1, target_size, target_size, 3) — batch of one.

    Matches the classifier's training pipeline:
        normalization: "divide_by_255"
        color_space:   "RGB"
    """
    img = pil_image.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # (1, H, W, 3)


def preprocess_for_segmentor(pil_image: Image.Image, target_size: int = 256) -> np.ndarray:
    """
    Resize → RGB → float32 → divide by 255.
    Returns shape (1, target_size, target_size, 3).

    The segmentor was trained at whatever resolution your friend used.
    Default is 256 — change the SEGMENTOR_INPUT_SIZE env var if needed.
    """
    img = pil_image.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # (1, H, W, 3)


# ── Visualisation helpers ──────────────────────────────────────────────────

def _pil_to_display_array(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a uint8 NumPy array suitable for imshow."""
    arr = np.array(pil_image, dtype=np.float32)
    if arr.max() <= 1.0:
        arr = (arr * 255.0)
    return arr.astype(np.uint8)


def build_segmentation_overlay(
    pil_image: Image.Image,
    prob_mask: np.ndarray,
    seg_size: int,
    mask_threshold: float = 0.5,
    overlay_alpha: float = 0.6,
) -> Tuple[str, str, float]:
    """
    Generate two visualisation images and tumour coverage metric.

    Parameters
    ----------
    pil_image       : original PIL MRI image (any size)
    prob_mask       : raw float32 probability map, shape (seg_size, seg_size)
    seg_size        : segmentor output resolution
    mask_threshold  : pixel probability to call it tumour
    overlay_alpha   : opacity of the red overlay (0–1)

    Returns
    -------
    overlay_b64     : base64 PNG — original MRI + red tumour mask
    heatmap_b64     : base64 PNG — jet-colourmap probability heatmap
    coverage_pct    : % of pixels flagged as tumour
    """
    # Resize original image to match the segmentor's output resolution
    display_img = pil_image.resize((seg_size, seg_size), Image.LANCZOS)
    img_arr = _pil_to_display_array(display_img)

    binary_mask = (prob_mask > mask_threshold).astype(np.float32)
    coverage_pct = float(binary_mask.mean() * 100)

    # ── 1. Red overlay image ───────────────────────────────────────────
    overlay_b64 = _render_overlay(img_arr, binary_mask, overlay_alpha)

    # ── 2. Jet confidence heatmap ──────────────────────────────────────
    heatmap_b64 = _render_heatmap(prob_mask)

    return overlay_b64, heatmap_b64, coverage_pct


def _render_overlay(img_arr: np.ndarray, binary_mask: np.ndarray, alpha: float) -> str:
    """Render the original MRI with a red tumour mask overlay → base64 PNG."""
    masked = np.ma.masked_where(binary_mask == 0, binary_mask)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(img_arr, cmap="gray")
    ax.imshow(masked, cmap="Reds", alpha=alpha, vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _render_heatmap(prob_mask: np.ndarray) -> str:
    """Render the raw probability map with jet colormap → base64 PNG."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    im = ax.imshow(prob_mask, cmap="jet", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("AI Confidence Heatmap", fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")