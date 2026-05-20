"""
Pydantic schemas for request / response validation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    Unified response for both healthy and tumour cases.
    Segmentation fields are None when the scan is healthy.
    """

    status: str = Field(
        ...,
        description="'healthy' or 'tumour_detected'",
        examples=["tumour_detected"],
    )
    tumour_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw softmax probability for the Tumour class (0–1)",
        examples=[0.87],
    )
    classification_threshold: float = Field(
        default=0.25,
        description="Decision threshold used for this prediction",
    )

    # ── Segmentation outputs (only when tumour_detected) ─────────────
    segmentation_overlay: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded PNG of the MRI with a red tumour mask overlay. "
            "Present only when status == 'tumour_detected'."
        ),
    )
    confidence_heatmap: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded PNG of the segmentor's raw probability heatmap "
            "(jet colormap: blue=0%, red=100% confidence). "
            "Present only when status == 'tumour_detected'."
        ),
    )
    segmentation_coverage_pct: Optional[float] = Field(
        default=None,
        description="Percentage of image pixels classified as tumour (0–100).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "tumour_detected",
                "tumour_probability": 0.87,
                "classification_threshold": 0.25,
                "segmentation_overlay": "<base64-encoded PNG string>",
                "confidence_heatmap": "<base64-encoded PNG string>",
                "segmentation_coverage_pct": 4.21,
            }
        }