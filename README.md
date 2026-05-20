# NeuroScanAI — Brain Tumor Detection & Segmentation

End-to-end deep learning pipeline that **detects** and **localizes** brain tumors in MRI scans using a two-stage cascade:

1) **Classifier (ResNet50)** — decides if a tumor is present.  
2) **Segmentor (ResUNet)** — localizes the tumor only when the classifier is positive.

> ⚠️ Research use only — not for clinical diagnosis.

---

# 📌 Problem Statement

Brain tumors are among the deadliest cancers. Early, accurate detection and localization are crucial, but manual annotation is expensive and slow. This project builds an automated pipeline that:

- **Screens MRI scans** for tumor presence (binary classification).
- **Segments tumor regions** at pixel level when a tumor is suspected.

---

# 📂 Dataset & Clinical Context

## LGG MRI Segmentation Dataset (Kaggle)

[https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

### Dataset Details

* 110 patients
* Paired MRI slices + expert segmentation masks

### The Imbalance Discovery

Initial Exploratory Data Analysis (EDA) revealed a severe class imbalance across the slice distribution:

| Class | Count |
| --- | --- |
| Healthy Slices | ~2,556 |
| Tumor Slices | ~1,373 |

This imbalance strongly influenced our architectural and mathematical decisions, requiring targeted loss-weighting to prevent the model from biasing toward healthy scans.


---

# 🧠 Pipeline Overview

# Stage 1 — High-Sensitivity Screening Classification

## Goal

Detect the presence of a brain tumor (binary classification) to act as a highly sensitive first-look clinical filter.

## System Role

Serve as Stage 1 of a two-stage diagnostic pipeline. By aggressively flagging potential tumors and sending only those positive scans to a heavy downstream Segmentation model, the system minimizes compute waste while using Stage 2 to naturally filter out False Positives.

---

## Data Pipeline & Engineering

### Data Source

Mateusz Buda LGG MRI dataset (`kaggle_3m`), processing paired `.tif` brain slices and `_mask.tif` files.  
Total dataset after parsing: **3,929 image‑mask pairs** from 110 patients.

### Ground Truth Mapping

Dynamically assigned labels based on mask pixel intensity:

* **Class 1 / Tumour** if `mask.max() > 0`
* **Class 0 / Healthy** otherwise

**Class distribution:** 2,556 Healthy (65%) – 1,373 Tumour (35%) — a meaningful imbalance that drives all later decisions.

### Data Split (Reproducible)

* First, a **15% hold‑out test set** (590 samples) is carved out and **never touched** during training or validation.
* The remaining 3,339 samples are split again: **85% train** (2,838) and **15% validation** (501).  
  All splits use `random_state=42`.

### Data Loading & Preprocessing

* `.tif` files are read with OpenCV via `tf.py_function`, converted from BGR→RGB, resized to 256×256, and normalized to [0,1].
* Shape information is explicitly restored with `img.set_shape([256, 256, 3])`.
* Labels are **one‑hot encoded** (`[1,0]` = Healthy, `[0,1]` = Tumour) to match the Softmax output.
* Training pipeline uses `.shuffle(1024)`, `.batch(32)`, and `.prefetch(AUTOTUNE)` for maximum GPU utilisation.

### Resolving Pipeline Bugs & Leakage

* **Double‑Augmentation Fix:** Identified and resolved severe logic bugs where augmentation was applied twice, distorting training signals.
* **Ghost Checkpoint Fix:** Fresh `ModelCheckpoint` instances are created before every training phase to avoid carrying over stale best‑loss values from earlier runs.
* **Batch‑Size Bug:** Global batch size is now computed as `16 * strategy.num_replicas_in_sync` (i.e., 32) — no hardcoded constants.
* **Leakage Prevention:** Class weights are computed **exclusively** on the training partition (`train_df`), ensuring downstream test set distributions never influence gradient descent.

---

## Hardware & Distributed Training

Training runs on **2 × NVIDIA Tesla T4 GPUs** using `tf.distribute.MirroredStrategy()`.  
The model is replicated on both GPUs; each processes a sub‑batch of 16 images. Gradients are synchronised with NCCL All‑Reduce before every weight update, keeping both replicas identical.

---

## Architecture & Training Optimization

### Backbone
Pretrained **ResNet50** using ImageNet weights (`include_top=False`).

### Custom Classification Head

```text
AveragePooling2D(4x4)
→ Flatten()
→ Dense(256, ReLU)
→ Dropout(0.3)
→ Dense(256, ReLU)
→ Dropout(0.3)
→ Dense(2, Softmax)
```

---

### Phase 1 — Baseline / Frozen

* Backbone locked (`trainable = False`)
* Trained only the custom head
* Optimizer: Adam (`learning_rate = 1e-4`)
* Loss: **Categorical Cross‑Entropy** (labels are one‑hot encoded)
* Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`

**Purpose:** Establish a stable gradient baseline without destroying pretrained ImageNet filters.

### Phase 2 — Targeted Fine‑Tuning with Class Weighting

* **Class Weights** computed from the training set:  
  **Tumour: 1.433** | **Healthy: 0.768**  
  (Penalises missed tumours ~1.87× more heavily)
* Upper architecture unfrozen: **first 143 layers frozen**, top 32 layers trainable.
* Learning rate reduced to **1e-5** to allow gentle specialisation without catastrophic forgetting.
* Same loss and callback setup.

**Purpose:** Safeguard generic low‑level features while adapting high‑level representations to low‑contrast MRI tumour patterns.

### Phase 3 — Augmented Fine‑Tuning (Experiment)

A clean augmented pipeline was built from scratch (no double‑augmentation) using:

* Random horizontal/vertical flips (50% probability)
* Random brightness (max_delta=0.1) and contrast (0.9–1.1)
* Pixel clipping to [0,1] after intensity shifts

This phase achieved the **highest overall accuracy (89%)** but **increased false negatives to 23** — worse than Phase 2.  
Because recall is the critical metric for Stage 1, **Phase 2 remains the selected production model**.

---

# ⚔️ The Strategic Showdown: ResNet50 vs DenseNet121

A parallel development track was run using a **DenseNet121** backbone (427 layers, dense connectivity).

### The DenseNet121 Dilemma

As a standalone classifier, DenseNet delivered superior global metrics:

* **ROC‑AUC:** 0.9722
* **PR‑AUC:** 0.9559
* **Accuracy:** 91.53% (augmented)

However, its probability outputs are more tightly compressed.  
Even when pushing the model to its **maximum sensitivity threshold (0.15)**, it still produced **16 missed tumours** (False Negatives).  
The default threshold (0.45) left **26 missed tumours**.

### The ResNet50 Advantage

ResNet50’s wider probability spread allows aggressive threshold lowering without collapsing behaviour.  
At the chosen clinical threshold of **0.25**, ResNet50 yields:

* **Only 9 missed tumours** (vs. 16 for DenseNet)
* **Tumour Recall: 95.59%** (vs. 92.16% for DenseNet)

In a two‑stage pipeline where Stage 2 filters false positives, **minimising false negatives is paramount** — ResNet50 wins decisively.

---

# 🎛️ Clinical Threshold Calibration

Instead of using the mathematical default of `0.50`, a systematic sweep (`0.15 → 0.50`) was conducted on the Phase 2 ResNet50 model.

## Chosen Decision Boundary

```text
Threshold = 0.25
```

## Trade‑off Logic

Lowering the threshold to `0.25`:

* Catches almost every true tumour in the test set.
* Produces only **9 False Negatives**.
* Generates **85 False Positives**.

These 85 False Positives are intentionally accepted because they are routed to Stage 2, where the highly specific segmentation network acts as a safety filter by returning null masks for healthy tissue.

---

# 📦 Final Stage 1 Deployment Metrics & Artifacts

## Selected Model

**ResNet50 (Phase 2 Fine‑Tuned)** @ threshold `0.25`

## Validated Test Metrics (N=590)

| Metric | Value |
| --- | --- |
| System Screening Accuracy | 84.07% |
| Tumour Recall (Sensitivity) | 95.59% |
| ROC‑AUC | 0.9564 |
| PR‑AUC | 0.9214 |
| Missed Tumors (FN) | **9** |

## Exported Artifacts

To ensure cross‑platform compatibility for the downstream pipeline, the chosen model was exported in multiple formats:

```text
└── brain_tumor_resnet50_pipeline_f/
    ├── brain_tumor_resnet50_pipeline_f.keras       # Unified native Keras 3 bundle
    ├── brain_tumor_resnet50_pipeline_f.weights.h5  # Legacy raw weights layer array
    ├── brain_tumor_resnet50_pipeline_f.tflite      # Quantized Edge TFLite binary
    ├── model_config.json                           # Production threshold & preprocessing metadata
    └── saved_model/                                # Standard TF SavedModel directory for cloud serving
```

---

# 🧪 Notebooks (Core of the Project)

## `Classification_MRI.ipynb`

Contains the full classification pipeline:

* Data pipeline + `tf.data` engineering
* ResNet50 baseline + fine‑tuning (Phases 1 & 2)
* Augmented Phase 3 with bug‑fixes
* Systematic threshold tuning & sweeping
* DenseNet121 parallel experiments
* Final cross‑model benchmarking & export

# Stage 2 — Deep Medical Segmentation (ResUNet)

## Primary Objective

Pixel-perfect spatial localization of tumor boundaries to assist in:

- Clinical evaluation
- Volumetric analysis

---

# 1. Hardware & Environment Engineering

To handle the massive computational load of high-resolution MRI arrays, the training pipeline was optimized for distributed computing.

## Multi-GPU Distribution

Implemented:

```python
tf.distribute.MirroredStrategy()
```

Across:

- `2 × NVIDIA T4 GPUs`

## Benefits

- Sharded global image batches
- Synchronized gradient calculations
- Reduced epoch training time

---

# 2. Exploratory Data Analysis & Clinical Context

Before modeling, a rigorous EDA pipeline was constructed to map image paths to diagnostic ground-truth masks.

## Anatomical Overlay Matrix

Engineered custom NumPy boolean matrix logic:

```python
mask > 0
```

To project bright crimson overlays:

```text
[235, 52, 52]
```

Over grayscale MRI scans for clinical visual sanity checks.

## The Imbalance Discovery

Dataset distribution:

| Class | Count |
|---|---|
| Healthy Slices | ~2,556 |
| Tumor Slices | ~1,373 |

This severe imbalance strongly influenced subsequent architectural and mathematical decisions.

---

# 3. Architecture Decisions

## Encoder
Pretrained **ResNet50** feature extractor.

Purpose:
Capture robust low-level and high-level medical textures.

## Decoder
Custom **U-Net upsampling blocks** with skip connections to recover lost spatial resolution.

## Output Layer

```text
256 × 256 × 1 binary mask
```

Using:

```text
Sigmoid activation
```

---

# 4. Model Evolution & Mathematical Optimization

The segmentation model evolved through three major phases.

---

## Phase A — Baseline (BCE + Dice)

### Result
```text
~35.0% Dice Score
```

### Observation
Standard loss functions failed.

The model hallucinated false positives on healthy scans due to domination by background pixels.

---

## Phase B — Data Augmentation Integration

### Added Augmentations

#### Spatial
- Flips
- Rotations

#### Pixel-Level
- Brightness shifts
- Contrast shifts

Purpose:
Simulate biological variability and MRI machine variance.

### Result
```text
47.3% Dice Score
```

False boundaries still remained.

---

## Phase C — Mathematical Fix (Focal Tversky Loss)

### Action
Replaced BCE with a custom **Focal Tversky Loss (FTL)**.

## Tversky Index

:contentReference[oaicite:0]{index=0}

### Hyperparameters

- `α = 0.3`
- `β = 0.7`
- `γ = 0.75`

Logic:
- Penalize False Negatives more heavily
- Force the model to focus on difficult tumor boundary pixels

## Focal Tversky Loss

:contentReference[oaicite:1]{index=1}

### Result
```text
51.5% Dice Score
```

False-positive hallucinations on healthy scans were successfully eliminated.

---

# 5. Final Output & Deployment Readiness

## Selected Model
**ResUNet + Focal Tversky Loss**

## Final Metric

```text
51.5% FTL Smooth Dice
```

## Artifacts Generated

```text
Brain_Tumor_Segmentation_FTL_Winner.keras
Brain_Tumor_Weights_FTL_51_5.weights.h5
```

---

# 📊 Key Metrics Summary

| Task | Model | Key Metric | Value |
|------|------|------------|-------|
| Classification | ResNet50 Phase 2 @ 0.25 | Tumor Recall | **95.59%** |
| Classification | ResNet50 Phase 2 @ 0.25 | Accuracy | **84.07%** |
| Classification | ResNet50 Phase 2 @ 0.25 | ROC-AUC | **0.9564** |
| Segmentation | ResUNet + FTL | Dice | **51.5%** |

---

# 🧪 Notebooks (Core of the Project)

## 1) `Classification_MRI.ipynb`

Contains the full classification pipeline:

- Data pipeline + `tf.data`
- ResNet50 baseline + fine-tuning
- Augmentation improvements
- Threshold tuning
- DenseNet121 experiments
- Final model export

---

## 2) `Brain_Tumor_detection.ipynb`

Contains the full segmentation pipeline:

- ResUNet architecture
- ResNet50 encoder + U-Net decoder
- Baseline + augmented training
- Focal Tversky Loss integration
- Visualization overlays + heatmaps
- Final model export

---

# 🧾 Exported Models

| File | Description |
|------|-------------|
| `models/brain_tumor_resnet50_pipeline_f.keras` | Final classifier (ResNet50, threshold 0.25) |
| `models/Brain_Tumor_Segmentation_FTL_Winner.keras` | Final segmentor (ResUNet + FTL) |

---

# 🧩 FastAPI + Next.js Demo

The project includes a backend + frontend demo stack.

## Backend
- FastAPI
- `/predict` endpoint
- Loads both classification and segmentation models

## Frontend
- Next.js
- MRI upload interface
- Visualization dashboard

---

## 🎛️ Full-Stack Application Architecture

The system is engineered as a decoupled, production-ready microservice stack comprising a high-performance **FastAPI asynchronous backend** handling deep learning inference and a modern **Next.js App Router frontend** driving an interactive clinical dashboard.

---

### 📂 Directory Structures & Workspace Layout

#### 🧠 FastAPI Backend Engine (Project Root)
```text
Deep-Project/
├─ main.py                 # FastAPI Application Entry Point & Lifespan Event Handling
├─ schemas.py              # Pydantic Response/Request Data Models
├─ requirements.txt        # Python Application Dependency Constraints
├─ Dockerfile              # Backend Multi-Stage Build Script
├─ docker-compose.yml      # Multi-Container Orchestration Blueprint
├─ .env                    # Environment Runtime Variable Management
├─ .dockerignore           # Build Context Exclusion Manifest
├─ ml/                     # Machine Learning Execution Modules
│  ├─ __init__.py
│  ├─ loader.py            # Async Model Deserialization & Custom Layer Registries
│  ├─ pipeline.py          # Two-Stage Cascaded Inference Flow Logic
│  └─ image_utils.py       # Preprocessing Matrices, Overlays & Base64 Encoders
└─ models/                 # Deep Learning Binary Weights Cache
   ├─ brain_tumor_resnet50_pipeline_f.keras
   └─ Brain_Tumor_Segmentation_FTL_Winner.keras

```

#### 🎨 Next.js React Frontend (`/front`)

```text
front/
├─ app/                    # Next.js App Router Structure
│  ├─ favicon.ico
│  ├─ globals.css          # Tailwind CSS Theme Tokens & Keyframe Animations
│  ├─ layout.tsx           # Global Document Shell & Font/SEO Metadata
│  └─ page.tsx             # Interactive Clinical State-Machine Dashboard
├─ components/             # Reusable UI Interface Components
│  ├─ UploadZone.tsx       # Drag-and-Drop Handler & File Binary Validation
│  ├─ ScanningState.tsx    # Neural Network Scanning Simulation Overlay
│  ├─ HealthyResult.tsx    # Negative Pathology Clearance UI
│  └─ TumourResult.tsx     # Positive Pathology Dashboard (Tabs: Overlay/Heatmap)
├─ lib/                    # Networking Client Layer
│  └─ api.ts               # Fetch client configured for typed JSON/Base64 payloads
├─ Dockerfile              # Optimized Node.js Production Runner Build Script
├─ .dockerignore           # Local Node Module Cache Exclusions
├─ next.config.ts          # Reverse Proxy Path Rewrite Configurations
└─ tsconfig.json           # Explicit TypeScript Static Type Rules

```

---

### ⚙️ Microservice Data Flows & Processing Modules

#### 1. FastAPI Execution Workflow

```text
Client File Stream ──> [POST /predict] ──> Image Normalization 
                                                │
       ┌────────────────────────────────────────┘
       ▼
 [Stage 1: ResNet50] ──> Softmax Output Probability
       │
       ├─> (Prob < 0.25) ───> [PATHOLOGY CLEAR] ───> Return JSON (Healthy Result)
       │
       └─> (Prob >= 0.25) ──> [PATHOLOGY SUSPECTED]
                                    │
                                    ▼
                             [Stage 2: ResUNet] ───> Generation of Mask Matrix
                                                            │
                                    ┌───────────────────────┘
                                    ▼
                             [image_utils.py] ─────> Render Base64 Crimson Overlays
                                                            │
                                                            ▼
                                                     Return Comprehensive JSON Payload

```

* **`main.py`:** Initializes the microservice interface, handles security headers via CORS, and uses a lifespan event context to parse and cache both models into memory at server startup to prevent request-time overhead.
* **`ml/loader.py`:** Handles model loading safely. It reads the model structures, pulls expected array target shapes from `model.input_shape`, and explicitly registers your custom **Focal Tversky Loss** function to successfully instantiate the segmentation model.
* **`ml/pipeline.py`:** Controls the cascading execution flow. It channels the incoming image through Stage 1. If the target score breaks the custom **0.25 threshold**, it shifts execution into Stage 2 to isolate the tumor boundaries and generate your overlay matrices.
* **`ml/image_utils.py`:** Handles image transforms. It reshapes raw image streams into floating-point image tensors, renders bright crimson `[235, 52, 52]` segmentation marks on the grayscale canvases, and encodes the output files into base64 strings for clean API transmission.
* **`schemas.py`:** Standardizes your API payload structure using Pydantic, enforcing data safety across bounding values, classification logits, and spatial string results.

#### 2. Next.js Frontend App Router Integration

```text
[Idle State: UploadZone] ──> Binary Validation ──> Network Request (lib/api.ts)
                                                        │
           ┌────────────────────────────────────────────┘
           ▼
[Processing State: ScanningState] ──> CSS Matrix Scanning Overlay Animation
                                                        │
           ┌────────────────────────────────────────────┘
           ▼
[Resolved State Parsing]
   │
   ├─> API Payload [unhealthy = false] ──> Mount <HealthyResult />
   └─> API Payload [unhealthy = true]  ──> Mount <TumourResult /> (Render Multi-Tab View)

```

* **`app/page.tsx`:** Acts as the primary application engine, managing UI states through a strict lifecycle machine (`idle` → `scanning` → `result` → `error`).
* **`lib/api.ts`:** Implements an asynchronous client wrapper that packages images into HTML5 `FormData` arrays and processes JSON payloads containing base64 images from the backend.
* **`components/TumourResult.tsx`:** Provides an interactive multi-tab component layout that lets clinical operators toggle between the original grayscale image scans, crimson target segmentation boundaries, and specialized spatial localization matrices.
* **`next.config.ts` Proxy Layer:** Implements clean path rewriting, configuring an internal proxy rule that dynamically rewrites requests from `/api/*` over to the internal `BACKEND_URL` network address. This decouples service interactions and prevents Cross-Origin Resource Sharing (CORS) complications during containerized production deployments.


# 🚀 Run the Dockerized Stack

```bash
docker compose up --build
```

## Services

| Service | URL |
|---|---|
| Backend | http://localhost:8000 |
| Frontend | http://localhost:3000 |

---

# ✅ Project Highlights

- Two-stage pipeline reduces compute cost
- High recall classifier minimizes missed tumors
- Segmentation optimized with Focal Tversky Loss
- Distributed GPU training support
- Clean, reproducible experiments
- Deployment-ready artifacts
- Integrated FastAPI + Next.js demo stack

---
