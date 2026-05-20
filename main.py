"""
Brain Tumor Detection & Segmentation API
=========================================
Pipeline:
  1. Receive MRI image (JPEG, PNG, or TIFF)
  2. Run ResNet50 classifier  ->  Healthy / Tumour Suspected
  3. If tumour -> run UNet segmentor -> return red-overlay + heatmap (base64 PNGs)
"""

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from ml.loader import load_models
from ml.pipeline import run_pipeline
from schemas import PredictionResponse

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Model registry (populated at startup) ─────────────────────────────────
models: dict = {}

# ── Accepted MIME types ─────────────────────────────────────────────────────
# TIF/TIFF is included because both models were trained on .tif images.
ACCEPTED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/tif",
    # Browsers sometimes send this for .tif files
    "application/octet-stream",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models once at startup; release at shutdown."""
    log.info("Loading ML models ...")
    models.update(load_models())
    log.info(
        f"Models ready.  "
        f"Classifier input: {models['classifier_size']}x{models['classifier_size']}  "
        f"Segmentor input: {models['segmentor_size']}x{models['segmentor_size']}"
    )
    yield
    models.clear()
    log.info("Models unloaded.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Detection API",
    description="ResNet50 classifier -> UNet segmentor cascade pipeline",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ───────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "classifier_input_size": models.get("classifier_size"),
        "segmentor_input_size": models.get("segmentor_size"),
    }


# ── Main prediction endpoint ───────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Upload an MRI image (.jpg / .png / .tif / .tiff).

    Returns:
    - **status**: "healthy" | "tumour_detected"
    - **tumour_probability**: float (0-1)
    - **segmentation_overlay**: base64 PNG (only when tumour detected)
    - **confidence_heatmap**: base64 PNG (only when tumour detected)
    - **segmentation_coverage_pct**: % of pixels flagged as tumour
    """

    # ── Read raw bytes first (content_type can be unreliable for .tif) ──
    try:
        raw_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

    # ── Let Pillow decide the real format (more reliable than MIME type) ─
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Cannot decode image '{file.filename}': {exc}. "
                "Send a JPEG, PNG, or TIFF file."
            ),
        )

    log.info(
        f"Received: {file.filename}  "
        f"format={pil_image.format}  size={pil_image.size}  "
        f"content_type={file.content_type}"
    )

    # ── Run cascade pipeline ─────────────────────────────────────────────
    try:
        result = run_pipeline(
            pil_image=pil_image,
            classifier=models["classifier"],
            classifier_size=models["classifier_size"],
            segmentor=models["segmentor"],
            segmentor_size=models["segmentor_size"],
        )
    except Exception as exc:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    log.info(
        f"Result: {result['status']}  "
        f"p(tumour)={result['tumour_probability']:.3f}"
    )
    return JSONResponse(content=result)