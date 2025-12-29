# SKANN-SSL: Selective Kernel Audio Neural Networks with Self-Supervised Learning

An underwater acoustic vessel detection and classification system using self-supervised learning. SKANN-SSL learns to identify vessel signatures directly from raw waveforms without requiring labeled training data.

---

## 🎯 Project Status

| Stage | Name | Status | Description |
|-------|------|--------|-------------|
| -1 | Synthetic Data | ✅ Complete | Physics-based waveform generator |
| 0 | Preprocessing | ✅ Complete | DataLoader, normalization, splits |
| 1 | SKConv1D Filterbank | 🔄 Next | Multi-branch learned filterbank |
| 2 | Encoder | ✅ Complete | HybridSKEncoder (34.4M params) |
| 3 | SSL Training | ✅ Complete | Barlow Twins on Dual T4 GPUs |
| 4 | Augmentation | ⏳ Planned | Physics-consistent augmentations |
| 5 | Training Loop | ✅ Complete | Integrated with Stage 3 |
| 6 | Evaluation | ✅ Complete | Confusion matrix, clustering |
| 7 | Deployment | ✅ Complete | Local inference engine |

---

## 🏆 Key Results

| Metric | Value |
|--------|-------|
| **Model Parameters** | 34.4 Million |
| **Embedding Dimension** | 128 |
| **Silhouette Score** | 0.3997 |
| **Training Hardware** | NVIDIA Dual T4 GPUs (DDP) |
| **Dataset Size** | 1,920 synthetic clips |

---

## 🏗️ Architecture

```
Raw Waveform [B, 1, 16000]
        │
        ▼
┌───────────────────────────────────┐
│  BACKBONE 1D (Temporal)           │
│  Conv1d(1→128, k=31, s=4)         │
│  Conv1d(128→128, k=15, s=2)       │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  BACKBONE 2D (Spectral)           │
│  Conv2d stack → AdaptivePool      │
│  Output: 512-dim                  │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  PROJECTOR (Deep MLP)             │
│  512 → 4096 → 8192 → 128          │
└───────────────────────────────────┘
        │
        ▼
   128-dim Acoustic Fingerprint
```

---

## 📁 Project Structure

```
SKANN-SSL/
├── README.md
├── ROADMAP.md
├── .gitignore
│
├── data/
│   └── prototype_dataset/
│       ├── master_dataset_manifest.csv   # 26-column metadata
│       ├── pairing_manifest.csv          # Hard positive pairs
│       ├── waveforms/                    # Raw Pa waveforms
│       └── tensors/                      # Preprocessed [1,1,16000]
│
├── stages/
│   ├── stage_minus1/          # Synthetic data generator
│   ├── stage0_preprocessing/  # DataLoader, splits
│   ├── stage1_skconv1d/       # [NEXT] Multi-branch filterbank
│   ├── stage2_encoder/        # HybridSKEncoder architecture
│   ├── stage3_ssl/            # Barlow Twins training
│   ├── stage4_augmentation/   # [PLANNED] Augmentation engine
│   ├── stage5_training/       # Training utilities
│   ├── stage6_evaluation/     # Confusion matrix, metrics
│   └── stage7_deployment/     # Local inference engine
│
├── docs/                      # Technical documentation (A-F)
├── outputs/                   # Sample results and plots
└── shared/                    # Common utilities
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/suniltyagi/SKANN-SSL.git
cd SKANN-SSL
```

### 2. Load Data
```python
from stages.stage0_preprocessing.dataloader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    manifest_path='data/prototype_dataset/master_dataset_manifest.csv',
    batch_size=32
)
```

### 3. Run Inference (requires production bundle)
```bash
cd stages/stage7_deployment
python acoustic_sonar_classifier3.py
```

---

## 📊 Dataset

The synthetic dataset covers a full-factorial experimental design:

| Factor | Values |
|--------|--------|
| Sea States | SS0, SS1, SS3, SS6 |
| Vessel Classes | small_craft, fishing_vessel, cargo_ship, tanker |
| Blade Counts | 3, 4, 5 |
| Generator Freq | 50 Hz, 60 Hz |
| Cavitation Levels | 0, 1, 2, 3 |
| Repetitions | 5 per combination |
| **Total Clips** | **1,920** |

**Audio Specifications:**
- Sample Rate: 16,000 Hz
- Duration: 1.0 second
- Frequency Band: 10 Hz – 8,000 Hz
- SNR: 6 dB (ship above sea noise)

---

## 🔬 Training Pipeline

### Hard Positive Mining
The `pairing_manifest.py` creates anchor-partner pairs that are:
- Same vessel class (positive)
- Maximally distant in feature space (hard)

This forces the network to learn class-invariant representations.

### Barlow Twins SSL
```python
# Cross-correlation matrix
C = z1_normalized.T @ z2_normalized / batch_size

# Loss: diagonal → 1, off-diagonal → 0
loss = on_diagonal_loss + λ * off_diagonal_loss
```

### Training Configuration
- Epochs: 50
- Batch Size: 4 per GPU (8 effective)
- Optimizer: AdamW, lr=1e-4
- Lambda (λ): 0.0051
- Mixed Precision: ✅ Enabled

---

## 📈 Evaluation

### Confusion Matrix Generator
```bash
cd stages/stage6_evaluation
python confusion_matrix_generator.py
```

Outputs:
- `confusion_matrix.png` — Visual heatmap
- `confusion_report.txt` — Per-class metrics
- `misclassified_clips.csv` — Error analysis

### Centroid-Based Classification
The inference engine compares new acoustic fingerprints against pre-computed class centroids using Euclidean distance.

---

## 📦 Production Assets (Not in Repo)

These files are too large for GitHub. Generate them using the training pipeline:

| File | Size | Description |
|------|------|-------------|
| `SKANN_SSL_Production_Bundle.joblib` | ~150 MB | Model weights + metadata |
| `vessel_territories.joblib` | ~1 KB | Class centroids (128-dim × 4) |
| `BT_ckpt_epoch_*.pth` | ~140 MB | Training checkpoints |

---

## 📚 Documentation

Comprehensive technical documentation in `/docs/`:

| Document | Content |
|----------|---------|
| A.docx | Underwater Acoustics Fundamentals |
| B.docx | Ambient Noise Models (Knudsen, Wenz, Kießling) |
| C.docx | DSP and Sampling Conventions |
| D.docx | Waveform Synthesis Pipeline |
| E.docx | Encoder Architecture & SSL |
| F.docx | Diagnostics & Deployment |
| Technical_Report.docx | System Summary |

---

## 🔮 Roadmap

### Immediate (Stage 1)
- [ ] Implement multi-branch SKConv1D filterbank
- [ ] Kernels: [3, 5, 7, 11, 15] with attention fusion
- [ ] Retrain and compare confusion metrics

### Short-term
- [ ] Address class confusion (especially large vessels)
- [ ] Implement physics-consistent augmentations
- [ ] Holdout validation on unseen data

### Long-term
- [ ] Test on real hydrophone data (ShipEar, NOAA)
- [ ] ONNX export for embedded deployment
- [ ] Real-time streaming inference

---

## 📖 References

- Li et al., "Selective Kernel Networks" (CVPR 2019)
- Zbontar & LeCun, "Barlow Twins" (ICML 2021)
- Urick, "Principles of Underwater Sound"
- Ross, "Mechanics of Underwater Noise"

---

## 📄 License

[Add your license here]

---

*Last Updated: December 2025*
