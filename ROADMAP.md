# SKANN-SSL PROJECT ROADMAP

## Hybrid Selective Kernel Self-Supervised Acoustic Representation Learning System

A complete, modern, scalable self-supervised acoustic representation learning system for underwater acoustics, suitable for sonar, vessel detection, machinery vibration, and environmental acoustics.

---

## System Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SKANN-SSL Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Raw Waveform [B, 1, 16000]                                                 │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────┐                                │
│  │      Stage 1: Learned Filterbank        │                                │
│  │      SKConv1D (kernels 3,5,7,11,15)     │                                │
│  └─────────────────────────────────────────┘                                │
│         │                                                                    │
│         ▼ [B, 64, T]                                                        │
│  ┌─────────────────────────────────────────┐                                │
│  │      Stage 2: 2D Acoustic Encoder       │                                │
│  │      SKConv2D blocks + Global Pool      │                                │
│  └─────────────────────────────────────────┘                                │
│         │                                                                    │
│         ▼ [B, D] embedding h                                                │
│  ┌─────────────────────────────────────────┐                                │
│  │      Stage 3: SSL (Barlow Twins)        │                                │
│  │                                         │                                │
│  │   x ──┬── Aug1 ──┐                      │                                │
│  │       │          ├── Encoder ── Proj ──┬── Loss                          │
│  │       └── Aug2 ──┘                      │                                │
│  └─────────────────────────────────────────┘                                │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────┐                                │
│  │      Clustering (HDBSCAN)               │                                │
│  │      Unsupervised vessel categorization │                                │
│  └─────────────────────────────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage Overview

| Stage | Name | Status | Description |
|-------|------|--------|-------------|
| -1 | Synthetic Data Generation | ✅ Complete | Physics-based waveform generator |
| 0 | Preprocessing | ✅ Complete | DataLoader, normalization, splits |
| 1 | Learned Filterbank | 🔄 Next | SKConv1D multi-branch frontend |
| 2 | 2D Encoder | ⏳ Planned | SKConv2D hierarchical encoder |
| 3 | SSL Training | ⏳ Planned | Barlow Twins self-supervised |
| 4 | Augmentation | ⏳ Planned | Physics-consistent augmentations |
| 5 | Training Loop | ⏳ Planned | Full training pipeline |
| 6 | Evaluation | ⏳ Planned | Clustering, visualization |
| 7 | Deployment | ⏳ Planned | ONNX export |

---

## Project Structure

```
SKANN_SSL/
├── README.md                     # Project overview
├── ROADMAP.md                    # This file
│
├── data/
│   └── prototype_dataset/
│       ├── master_dataset_manifest.csv   # Authoritative source (26 cols)
│       ├── waveforms/                    # Raw Pa waveforms
│       └── tensors/                      # Preprocessed [1,1,16000]
│
├── stages/
│   ├── stage_minus1/             # Synthetic data generator
│   ├── stage0_preprocessing/     # DataLoader, splits, transforms
│   ├── stage1_skconv1d/          # Learned 1D filterbank
│   ├── stage2_encoder/           # 2D hierarchical encoder
│   ├── stage3_ssl/               # Barlow Twins SSL
│   ├── stage4_augmentation/      # Augmentation engine
│   └── stage5_training/          # Training loop
│
├── shared/
│   ├── config.py                 # Global constants
│   └── utils.py                  # Common utilities
│
├── notebooks/
├── checkpoints/
└── outputs/
```

---

## Stage -1: Synthetic Data Generation ✅ COMPLETE

**Objective:** Generate physics-based synthetic waveforms for initial training and validation.

**What Was Built:**
- Sea noise generator based on digitized Knudsen curves (4 sea states)
- Ship noise generator with tonal, broadband, and cavitation components
- Full-factorial dataset covering all design factor combinations
- Physical units (Pascals) with proper SNR mixing

**Dataset:**
- 1,920 clips (4 sea states × 4 vessels × 3 blades × 2 generators × 4 cavitation × 5 reps)
- 1 second duration, 16 kHz sample rate
- 10 Hz - 8,000 Hz frequency band
- 6 dB SNR (ship above sea noise)

**Output:**
- `data/prototype_dataset/master_dataset_manifest.csv` (26 columns)
- `data/prototype_dataset/waveforms/` (raw Pa)
- `data/prototype_dataset/tensors/` (preprocessed)

**Location:** `stages/stage_minus1/`

---

## Stage 0: Preprocessing & Data Standardization ✅ COMPLETE

**Objective:** Ensure all raw signals are consistent, normalized, and ready for model ingestion.

**What Was Built:**
- `SKANNDataset` class using manifest as single source of truth
- Stratified train/val/test splits (70/15/15) preserving class balance
- Transform classes for augmentation (TimeShift, GaussianNoise, AmplitudeScale)
- Integration testing suite

**Preprocessing Applied:**
- DC offset removal
- RMS normalization
- Reshape to `[B, 1, T]` format

**Usage:**
```python
from stages.stage0_preprocessing import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    manifest_path='data/prototype_dataset/master_dataset_manifest.csv',
    batch_size=32
)
```

**Output:** Tensor `[Batch, 1, 16000]`

**Location:** `stages/stage0_preprocessing/`

---

## Stage 1: Learned Filterbank (SKConv1D) — NEXT

**Objective:** Replace STFT/mel spectrogram preprocessing with a learned multi-scale time-domain filterbank.

**Design:**
- Multi-branch SKConv1D with kernels (3, 5, 7, 11, 15)
- Attention-weighted branch fusion
- Each kernel captures different temporal scales:

| Kernel | Duration | Captures |
|--------|----------|----------|
| 3 | 0.19 ms | Transients, clicks |
| 5 | 0.31 ms | Sharp features |
| 7 | 0.44 ms | Tonals |
| 11 | 0.69 ms | Modulation |
| 15 | 0.94 ms | Broadband patterns |

**Input:** `[B, 1, 16000]` from Stage 0
**Output:** `[B, 64, T]` learned time-feature map

**Location:** `stages/stage1_skconv1d/`

---

## Stage 2: Hierarchical 2D Acoustic Encoder

**Objective:** Convert learned time-feature map into high-level 2D embeddings.

**Workflow:**
1. Reshape `[B, 64, T]` → `[B, 1, 64, T]`
2. Apply SKConv2D blocks with progressive downsampling
3. Global pooling
4. Linear projection to embedding h ∈ ℝᴰ

**Input:** `[B, 64, T]` from Stage 1
**Output:** `[B, D]` embedding vector

**Location:** `stages/stage2_encoder/`

---

## Stage 3: Self-Supervised Representation Learning

**Objective:** Train encoder without labels using Barlow Twins.

**Components:**
- Siamese encoder with shared weights
- Projector head g(h)
- Barlow Twins loss: invariance + decorrelation

**Loss Function:**
```
L = λ × Σᵢ(1 - Cᵢᵢ)² + Σᵢ Σⱼ≠ᵢ Cᵢⱼ²
    ├── Invariance: on-diagonal → 1
    └── Redundancy reduction: off-diagonal → 0
```

**Alternatives:** SimCLR, VICReg

**Location:** `stages/stage3_ssl/`

---

## Stage 4: Data Pipeline & Augmentation Engine

**Objective:** Generate physics-consistent positive pairs for SSL training.

**Augmentations:**
| Augmentation | Description | Physics Motivation |
|--------------|-------------|-------------------|
| Random crop | Extract sub-segment | Temporal invariance |
| Time shift | Circular shift | Phase invariance |
| Gain jitter | Amplitude scaling | Distance variations |
| Gaussian noise | Additive noise | Ambient changes |
| Band-pass filter | Frequency filtering | Propagation effects |
| Time masking | Zero segments | Occlusion robustness |

**Location:** `stages/stage4_augmentation/`

---

## Stage 5: Training, Embedding Extraction & Clustering

**Training Loop:**
```
for epoch:
    x1, x2 = augment(x), augment(x)  # Positive pair
    z1, z2 = projector(encoder(x1)), projector(encoder(x2))
    loss = barlow_twins_loss(z1, z2)
    loss.backward()
    optimizer.step()
```

**Clustering:**
- Primary: HDBSCAN (density-based, no k required)
- Alternative: DBSCAN

**Visualization:**
- UMAP projection
- t-SNE projection
- Cluster centroids / averages

**Location:** `stages/stage5_training/`

---

## Stage 6: Evaluation & Extensions

**Tasks:**
- Embedding variance diagnostics
- Invariance robustness tests
- Cluster purity metrics
- Comparison with supervised baseline

---

## Stage 7: Deployment & Export

**Tasks:**
- ONNX export for embedded inference
- Optional quantization (INT8)
- Deploy to ARM/DSP hardware
- Real-time streaming inference

---

## Dataset Plan

### Prototype Dataset (Current)
Synthetic vessel noise generator producing physics-based waveforms for controlled validation of SKConv1D and SKConv2D architecture.

### Future Real-World Datasets
| Dataset | Description |
|---------|-------------|
| NOAA NCEI | Passive acoustic archives with vessel noise |
| MBARI | Hydrophone recordings with vessel pass-bys |
| JAMSTEC | Long-term underwater observatory recordings |
| DCLDE | Marine mammal + vessel acoustic events |

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 10 Hz minimum frequency | Avoids turbulence model extrapolation |
| Learned filterbank | End-to-end optimization vs fixed STFT |
| Selective Kernels | Adaptive receptive fields for multi-scale features |
| Barlow Twins | No negative pairs needed, stable training |
| 16 kHz sample rate | Captures cavitation (up to 8 kHz Nyquist) |
| Manifest-based loading | Single source of truth for dataset |

---

## Deliverables

**Implementations:**
- SKConv1D multi-branch filterbank
- SKConv2D encoder blocks
- HybridSKEncoder (full backbone)
- Barlow Twins SSL wrapper
- Augmentation engine

**Utilities:**
- Training pipeline with logging
- Clustering utilities (HDBSCAN)
- Embedding visualization
- ONNX export
- Integration tests

---

## References

- Li et al., "Selective Kernel Networks" (CVPR 2019)
- Zbontar & LeCun, "Barlow Twins" (ICML 2021)
- Urick, "Principles of Underwater Sound"
- Ross, "Mechanics of Underwater Noise"
- Knudsen et al., "Underwater Ambient Noise" (1948)

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.1.0 | Dec 2025 | Stage -1 complete (synthetic generator) |
| 0.2.0 | Dec 2025 | Stage 0 complete (DataLoader, manifest) |
| 0.3.0 | Dec 2025 | Project restructure (stages/ layout) |

---

*Last updated: December 17, 2025*
