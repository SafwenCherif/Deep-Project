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

# 📂 Dataset

## LGG MRI Segmentation Dataset (Kaggle)

https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

### Dataset Details
- 110 patients
- Paired MRI slices + expert segmentation masks
- Class imbalance: fewer tumor-positive slices

---

# 🧠 Pipeline Overview

# Stage 1 — High-Sensitivity Screening Classification (ResNet50)

## Goal
Detect the presence of a brain tumor (binary classification) to act as a highly sensitive first-look clinical filter.

## System Role
Serve as Stage 1 of a two-stage diagnostic pipeline. By aggressively flagging potential tumors and sending only those positive scans to a heavy downstream Segmentation model, the system minimizes compute waste while using Stage 2 to naturally filter out False Positives.

---

## Data Pipeline & Engineering

### Data Source
Mateusz Buda LGG MRI dataset (`kaggle_3m`), processing paired `.tif` brain slices and `_mask.tif` files.

### Ground Truth Mapping
Dynamically assigned labels based on mask pixel intensity:

- **Class 1 / Tumour** if `mask.max() > 0`
- **Class 0 / Healthy** otherwise

### Resolving Pipeline Bugs
Identified and resolved severe Double-Augmentation logic bugs that were distorting training signals.

### Class Imbalance Handling
To prevent the model from biasing toward healthy scans, deterministic class weights were calculated directly from the training subset:

- Tumour: `1.43`
- Healthy: `0.77`

These weights were applied to the loss function.

---

## Architecture & Training Optimization

### Backbone
Pretrained **ResNet50** using ImageNet weights.

### Custom Classification Head

```text
AveragePooling2D(4x4)
→ Flatten()
→ Dense(256)
→ Dropout(0.3)
→ Dense(256)
→ Dropout(0.3)
→ Dense(2, Softmax)
```

### Phase 1 — Baseline / Frozen

- Backbone locked
- Trained only the custom head
- Optimizer: Adam (`learning rate = 1e-4`)
- Loss: Categorical Cross-Entropy

Purpose:
Establish a stable gradient baseline.

### Phase 2 — Targeted Fine-Tuning

- Unfroze the upper architecture for specialized medical texture learning
- Kept the first **143 layers frozen** to preserve generic feature extraction
- Reduced Adam learning rate to `1e-5`
- Added:
  - `ModelCheckpoint`
  - `EarlyStopping`
  - `ReduceLROnPlateau`

Purpose:
Safeguard structural weights while specializing for MRI tumor patterns.

---

# The Strategic Showdown: ResNet50 vs DenseNet121

A parallel development track was run using a **DenseNet121** backbone.

## DenseNet121 Dilemma

DenseNet achieved:

- ROC-AUC: `0.9722`
- Accuracy: `91.86%` @ threshold `0.45`

However, its probability distribution was too tightly grouped, resulting in:

- `26` missed tumors (False Negatives)

## ResNet50 Advantage

Although ResNet50 had a slightly lower global ROC-AUC, its architecture allowed aggressive threshold manipulation without collapsing model behavior.

This reduced missed tumors to single digits.

Because this is a two-stage system, minimizing **False Negatives** was prioritized over absolute accuracy.

---

# Clinical Threshold Calibration

Instead of using the mathematical default threshold `0.50`, a systematic threshold sweep (`0.15 → 0.50`) was conducted on the fine-tuned ResNet50 model.

## Chosen Decision Boundary

```text
Threshold = 0.25
```

## Trade-off Logic

Lowering the threshold to `0.25`:

- Successfully caught almost every true tumor
- Produced only `9` False Negatives
- Generated `85` False Positives

These False Positives are intentionally accepted because they are routed to Stage 2, where the segmentation network acts as a safety filter by returning null matrices for healthy tissue.

---

# Final Stage 1 Deployment Metrics & Artifacts

## Selected Model
**ResNet50 (Phase 2 Fine-Tuned)** @ threshold `0.25`

## Metrics

| Metric | Value |
|---|---|
| System Screening Accuracy | 84.07% |
| Tumour Recall | 95.59% |
| ROC-AUC | 0.9564 |
| PR-AUC | 0.9214 |
| Missed Tumors (FN) | 9 |

## Exported Artifacts

```text
brain_tumor_resnet50_pipeline_f.keras
brain_tumor_resnet50_pipeline_f.weights.h5
model_config.json
```

Where:

- `.keras` → unified deployment package
- `.weights.h5` → legacy backend interoperability
- `model_config.json` → runtime threshold configuration (`t = 0.25`)

---

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