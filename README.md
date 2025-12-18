# SKANN-SSL

**Selective Kernel Audio Neural Networks with Self-Supervised Learning**

A hybrid self-supervised acoustic representation learning system for underwater acoustics, designed for vessel detection and classification from passive sonar data.

---

## Project Overview

SKANN-SSL learns acoustic representations from unlabelled underwater recordings using a multi-stage pipeline:

1. **Raw waveform ingestion** → learned filterbank (no handcrafted spectrograms)
2. **Multi-scale feature extraction** → Selective Kernel convolutions adapt receptive fields
3. **Self-supervised learning** → Barlow Twins learns invariant representations
4. **Unsupervised clustering** → HDBSCAN discovers vessel categories

### Key Innovation

Traditional approaches use fixed STFT/mel spectrograms. SKANN-SSL learns the filterbank end-to-end, allowing the network to discover optimal time-frequency representations for underwater acoustics.

> **See [ROADMAP.md](ROADMAP.md) for detailed stage descriptions, technical decisions, and current progress.**

---

## Pipeline Stages

| Stage | Name | Description | Status |
|-------|------|-------------|--------|
| -1 | Data Generation | Synthetic waveform generator | ✅ Complete |
| 0 | Preprocessing | DataLoader, normalization, splits | ✅ Complete |
| 1 | Learned Filterbank | SKConv1D multi-branch frontend | 🔄 Next |
| 2 | 2D Encoder | SKConv2D hierarchical encoder | ⏳ Planned |
| 3 | SSL Training | Barlow Twins self-supervised | ⏳ Planned |
| 4 | Augmentation | Physics-consistent augmentations | ⏳ Planned |
| 5 | Training Loop | Full training pipeline | ⏳ Planned |
| 6 | Evaluation | Clustering, visualization | ⏳ Planned |
| 7 | Deployment | ONNX export | ⏳ Planned |

---

## Project Structure

```
SKANN_SSL/
├── README.md                     # This file
├── ROADMAP.md                    # Detailed stage descriptions & status
│
├── data/
│   └── prototype_dataset/
│       ├── master_dataset_manifest.csv  # Authoritative dataset map
│       ├── waveforms/                   # Raw waveforms (Pa, float32)
│       └── tensors/                     # Preprocessed [1,1,16000]
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
├── shared/                       # Cross-stage utilities
│   ├── config.py                 # Global constants
│   └── utils.py                  # Common functions
│
├── notebooks/                    # Colab/Jupyter notebooks
│
├── checkpoints/                  # Saved model weights
│   ├── stage1/
│   ├── stage3/
│   └── stage5/
│
└── outputs/                      # Results and artifacts
    ├── embeddings/
    ├── figures/
    └── evaluation/
```

---

## Quick Start

### 1. Mount Google Drive (Colab)
```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/SKANN_SSL')
```

### 2. Load Data
```python
from stages.stage0_preprocessing.dataloader import get_dataloaders

MANIFEST = '/content/drive/MyDrive/SKANN_SSL/data/prototype_dataset/master_dataset_manifest.csv'

train_loader, val_loader, test_loader = get_dataloaders(
    manifest_path=MANIFEST,
    batch_size=32
)
```

### 3. Verify
```python
x, y = next(iter(train_loader))
print(f"Input shape: {x.shape}")   # [32, 1, 16000]
print(f"Labels: {y}")              # tensor([0, 2, 1, 3, ...])
```

---

## Dataset

### Prototype Dataset (Stage -1)

| Property | Value |
|----------|-------|
| Total clips | 1,920 |
| Duration | 1.0 second |
| Sample rate | 16,000 Hz |
| Frequency band | 10 Hz – 8,000 Hz |
| SNR | 6.0 dB (ship above sea) |
| Format | float32, Pascals |

### Full-Factorial Design
```
4 sea states × 4 vessel classes × 3 blade counts × 
2 generator freqs × 4 cavitation levels × 5 repeats = 1,920 clips
```

### Vessel Classes
| Class | Shaft Rate | BPF Range | Cavitation Peak |
|-------|------------|-----------|-----------------|
| small_craft | 15-30 Hz | 45-90 Hz | 5000 Hz |
| fishing_vessel | 4-8 Hz | 12-32 Hz | 1500 Hz |
| cargo_ship | 1.5-2.5 Hz | 6-12.5 Hz | 600 Hz |
| tanker | 1.0-1.5 Hz | 4-9 Hz | 400 Hz |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SKANN-SSL Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Waveform [B, 1, 16000]                                     │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────┐                    │
│  │         Stage 1: SKConv1D               │                    │
│  │  Multi-branch: kernels (3,5,7,11,15)    │                    │
│  │  Attention-weighted fusion              │                    │
│  └─────────────────────────────────────────┘                    │
│         │                                                        │
│         ▼ [B, 64, 16000]                                        │
│  ┌─────────────────────────────────────────┐                    │
│  │         Stage 2: SKConv2D               │                    │
│  │  Reshape → 2D conv blocks → Pool        │                    │
│  └─────────────────────────────────────────┘                    │
│         │                                                        │
│         ▼ [B, D] embedding                                      │
│  ┌─────────────────────────────────────────┐                    │
│  │         Stage 3: Barlow Twins           │                    │
│  │  Siamese encoder + Projector            │                    │
│  │  Loss: invariance + decorrelation       │                    │
│  └─────────────────────────────────────────┘                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────┐                    │
│  │         Clustering (HDBSCAN)            │                    │
│  │  Unsupervised vessel categorization     │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 10 Hz minimum frequency | Avoids turbulence model extrapolation |
| Learned filterbank | End-to-end optimization vs fixed STFT |
| Selective Kernels | Adaptive receptive fields for multi-scale features |
| Barlow Twins | No negative pairs needed, stable training |
| 16 kHz sample rate | Captures cavitation (up to 8 kHz Nyquist) |

---

## Future Datasets

After prototype validation, integrate real-world data:

- **NOAA NCEI** – Passive acoustic archives
- **MBARI** – Hydrophone recordings with vessel pass-bys
- **JAMSTEC** – Long-term underwater observatory
- **DCLDE** – Marine mammal + vessel acoustic events

---

## Requirements

```
torch >= 2.0
numpy
pandas
scikit-learn
matplotlib
```

---

## References

- Li et al., "Selective Kernel Networks" (CVPR 2019)
- Zbontar & LeCun, "Barlow Twins" (ICML 2021)
- Urick, "Principles of Underwater Sound"
- Ross, "Mechanics of Underwater Noise"

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.1.0 | Dec 2025 | Stage -1 complete (synthetic generator) |
| 0.2.0 | Dec 2025 | Stage 0 complete (DataLoader, manifest) |
| 0.3.0 | Dec 2025 | Project restructure (stages/ layout) |

---

## License

Proprietary – Research Use Only

---

## Contact

Project Lead: Oravont
