# CLAUDE.md — Project Context for AI Assistants

## Project Overview

**SKANN-SSL** is a physics-grounded acoustic representation learning system for:
- Underwater acoustics / sonar / HAVS (Hydroacoustic Vessel Signatures)
- Vessel and machinery acoustic signatures
- Producing interpretable embeddings for clustering and downstream analysis

**Core approach:** Learned filterbank front-ends (SKConv) + self-supervised learning (Barlow Twins) to produce robust acoustic embeddings without labels.

## Architecture Pipeline

```
Stage -1 → 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
```

| Stage | Name | Purpose |
|-------|------|---------|
| -1 | Sample Generation | Synthetic vessel-noise waveforms (physics-inspired) |
| 0 | Preprocessing | Resample, normalize, segment to `[B, 1, T]` |
| 1 | Learned Filterbank | SKConv1D multi-scale front-end → `[B, F, T1]` |
| 2 | 2D Encoder | SKConv2D blocks → embedding `h ∈ ℝᴰ` |
| 3 | SSL Training | Barlow Twins loss (invariance + decorrelation) |
| 4 | Augmentation Engine | Physics-consistent positive pairs for SSL |
| 5 | Embedding & Clustering | HDBSCAN clustering, UMAP visualization |
| 6 | Evaluation | Analytics, robustness tests, confusion matrices |
| 7 | Deployment | ONNX export, quantization |

## Key Principles

1. **Physics before learning** — SPL/PSD quantities retain physical meaning and units
2. **SSL for representation, not classification** — Embeddings support downstream analysis
3. **Synthetic data is first-class** — Stage -1 enables controlled experiments before real data
4. **Interpretability matters** — Embeddings should be analyzable, not black-box

## Repository Structure

- `stages/` — Stage-specific code and READMEs
- `docs/` — Theory, specs, and design documents
- `data/` — Main dataset folder (local-only, not committed)
- `ROADMAP.md` — Full pipeline specification

## GUI Demo (Stage 7)

Interactive tkinter GUI for vessel classification demonstration.

**Location:** `stages/stage7_deployment/demo/`

**Structure:**
```
stages/stage7_deployment/demo/
├── skann_ssl_demo_v2.py       # GUI demo (HybridSKEncoderV2 + radar plot UI)
├── requirements.txt           # torch, numpy, pandas, matplotlib, joblib, sounddevice
├── model/                     # Model artefacts (.joblib) — gitignored
└── data/
    ├── manifest.csv           # Clip metadata (committed)
    └── tensors/               # Audio tensors — gitignored
```

**Setup:** Model and tensor files are gitignored (too large). After cloning, copy from training artefacts or symlink to `data/prototype_dataset/tensors/`. See `stages/stage7_deployment/demo/README.md`.

**Data Policy:** Do not edit, move, or commit files under `data/`. Treat datasets as local-only artefacts.

## Operational Anchors (Current Baseline)

- **Stage 3 training:** `stages/stage3_ssl/README.md`, `stage3_TRAIN_EXPORT_NOTES.md`
- **Stage 6 evaluation:** `stages/stage6_evaluation/README.md`, `CONFUSION_MATRIX_ANALYSIS.md`

## Document Priority (Conflict Resolution)

1. Dated Decision Records (ADRs) if present
2. `ROADMAP.md`
3. `/docs/*` theory/spec documents
4. Stage READMEs under `stages/`

## What This Project Is NOT

- Not a supervised classification pipeline
- Not a dataset-specific one-off model
- Not a monolithic NN where only "accuracy" matters

## Tech Stack

- PyTorch for neural network components
- Audio at 16 kHz sampling rate
- SKConv (Selective Kernel Convolution) for multi-scale processing
- HDBSCAN/DBSCAN for clustering
- ONNX for deployment export
