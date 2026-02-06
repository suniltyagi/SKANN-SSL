# CLAUDE.md — Project Context for AI Assistants

## Project Overview

**SKANN-SSL** is a physics-grounded acoustic representation learning system for:
- Underwater acoustics / sonar / HAVS (Hydroacoustic Vessel Signatures)
- Vessel and machinery acoustic signatures
- Producing interpretable embeddings for clustering and downstream analysis

**Core approach:** Learned filterbank front-ends (SKConv) + self-supervised learning (Barlow Twins) to produce robust acoustic embeddings without labels.

**Current Version:** V3/V5 (February 2026) — 100% classification accuracy on 12,000 clips

## Architecture Pipeline

```
Stage -1 → 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
```

| Stage | Name | Status | Purpose |
|-------|------|--------|---------|
| -1 | Sample Generation | ✅ Complete | Synthetic vessel-noise waveforms (V5 physics) |
| 0 | Preprocessing | ✅ Complete | Resample, normalize, segment to `[B, 1, T]` |
| 1 | Learned Filterbank | ✅ Complete | SKConv1D multi-scale front-end → `[B, F, T1]` |
| 2 | 2D Encoder | ✅ Complete | HybridSKEncoderV3 → embedding `h ∈ ℝ⁵¹²` |
| 3 | SSL Training | ✅ Complete | Barlow Twins loss (V3 production bundle) |
| 4 | Augmentation Engine | ⏳ Planned | Physics-consistent positive pairs for SSL |
| 5 | Training Loop | ✅ Complete | Integrated with Stage 3 |
| 6 | Evaluation | ✅ Complete | Confusion matrix, territory mapping (100% accuracy) |
| 7 | Deployment | ✅ Prototype | Demo GUI, ONNX export planned |

## Key Metrics (V3/V5)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 100.0% (11,995/12,000) |
| Silhouette Score | 0.9697 |
| Classes | 5 (cargo, fishing, no_vessel, small_craft, tanker) |
| Dataset | 12,000 clips × 5 seconds @ 16kHz |
| Model Parameters | 175.9M (training) / ~2M (inference) |
| Embedding Dimension | 512 (backbone h) / 256 (projector z) |

## Key Principles

1. **Physics before learning** — SPL/PSD quantities retain physical meaning and units
2. **SSL for representation, not classification** — Embeddings support downstream analysis
3. **Synthetic data is first-class** — Stage -1 enables controlled experiments before real data
4. **Interpretability matters** — Embeddings should be analyzable, not black-box
5. **Non-overlapping shaft rates** — V5 dataset ensures acoustic distinguishability

## Repository Structure

```
SKANN-SSL/
├── README.md              # Project overview with mermaid diagrams
├── ROADMAP.md             # Pipeline specification
├── CLAUDE.md              # This file
│
├── data/
│   └── v5_dataset/        # 12,000 clips (5 classes)
│       ├── tensors/
│       ├── waveforms/
│       └── master_dataset_manifest.csv
│
├── stages/
│   ├── stage_minus1/      # V5 synthetic data generator
│   ├── stage3_ssl/        # SSL training
│   │   ├── train_script.py           # HybridSKEncoderV3 architecture
│   │   └── artifacts/
│   │       ├── SKANN_SSL_V3_Production_Bundle.joblib
│   │       └── vessel_territories_v3.joblib
│   ├── stage6_evaluation/ # Evaluation scripts
│   │   ├── stage6_confusion_matrix.py
│   │   ├── stage6_acoustic_sonar_classifier.py
│   │   └── artifacts/
│   └── stage7_deployment/ # Demo GUI
│
├── archive/               # V1 and V2 preserved for reproducibility
├── docs/                  # Documentation
└── shared/                # Shared utilities
```

## Key Files

| File | Purpose |
|------|---------|
| `stages/stage3_ssl/train_script.py` | HybridSKEncoderV3 model architecture |
| `stages/stage3_ssl/artifacts/SKANN_SSL_V3_Production_Bundle.joblib` | Trained model + embeddings |
| `stages/stage3_ssl/artifacts/vessel_territories_v3.joblib` | Class centroids (512-dim) |
| `data/v5_dataset/master_dataset_manifest.csv` | Dataset metadata |

## Linked Repositories

| Repository | Purpose |
|------------|---------|
| [SKANN-SSL](https://github.com/suniltyagi/SKANN-SSL) | Main development repo |
| [SKANN-SSL-V5-Demo](https://github.com/suniltyagialtair/SKANN-SSL-V5-Demo) | GUI demo application |
| [Underwater-Acoustic-Synthetic-Dataset](https://github.com/suniltyagialtair/Underwater-Acoustic-Synthetic-Dataset) | V5 dataset (12,000 clips) |

## V5 Physics (Critical)

### Non-Overlapping Shaft Rates
| Vessel Class | Shaft Rate (Hz) | RPM |
|--------------|-----------------|-----|
| tanker | 0.8 – 1.8 | 48–108 |
| cargo_ship | 1.5 – 3.5 | 90–210 |
| fishing_vessel | 4.0 – 8.0 | 240–480 |
| small_craft | 15.0 – 30.0 | 900–1800 |

### SK Kernel Coverage
| Kernel | Frequency | Target Signature |
|--------|-----------|------------------|
| k=1023 | 15+ Hz | Shaft rate |
| k=511 | 31+ Hz | Blade pass |
| k=255 | 62+ Hz | Generator |
| k=127 | 125+ Hz | Equipment |
| k=63 | 250+ Hz | Flow noise |
| k=31 | 500+ Hz | Cavitation |

## Document Priority (Conflict Resolution)

1. `ROADMAP.md`
2. Stage READMEs under `stages/`
3. `/docs/*` theory/spec documents
4. This file (CLAUDE.md)

## What This Project Is NOT

- Not a supervised classification pipeline
- Not a dataset-specific one-off model
- Not a monolithic NN where only "accuracy" matters

## Tech Stack

- PyTorch for neural network components
- Audio at 16 kHz sampling rate
- SKConv (Selective Kernel Convolution) for multi-scale processing
- Barlow Twins for self-supervised learning
- ONNX for deployment export (planned)

---

*Last updated: February 2026*
