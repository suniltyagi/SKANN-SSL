# SKANN-SSL: Selective Kernel Audio Neural Networks with Self-Supervised Learning

An underwater acoustic vessel detection and classification system using self-supervised learning. SKANN-SSL learns to identify vessel signatures directly from raw waveforms and uses downstream evaluation + centroid-based logic for inspection and inference.

---

## 🎯 Project Status

| Stage | Name | Status | Description |
|------:|------|:------:|-------------|
| -1 | Synthetic Data | ✅ Complete | Physics-based waveform generator |
| 0 | Preprocessing | ✅ Complete | DataLoader, normalisation, splits |
| 1 | SKConv1D Filterbank | 🔄 Next | Multi-branch learned filterbank |
| 2 | Encoder | ✅ Complete | HybridSKEncoder (34.4M params) |
| 3 | SSL Training | ✅ Complete | Barlow Twins on Dual T4 GPUs (baseline) |
| 4 | Augmentation | ⏳ Planned | Physics-consistent augmentations |
| 5 | Training Loop | ✅ Complete | Integrated with Stage 3 baseline |
| 6 | Evaluation | ✅ Complete | Confusion matrix + operator inspection + centroid mapping |
| 7 | Deployment | ✅ Prototype | Local inference engine (centroid/territory-assisted) |

---

## 🏆 Key Results (Baseline)

| Metric | Value |
|--------|------:|
| **Model Parameters** | 34.4 Million |
| **Embedding Dimension** | 128 |
| **Silhouette Score (cosine)** | 0.3997 |
| **Training Hardware** | NVIDIA Dual T4 GPUs (DDP) |
| **Dataset Size** | 1,920 synthetic clips |
| **Stage 6 Accuracy (first cut)** | ~78% |

---

## 🏗️ Architecture (Encoder + Projector)

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

```text
SKANN-SSL/
├── README.md
├── ROADMAP.md
├── .gitignore
│
├── data/
│   └── prototype_dataset/
│       ├── master_dataset_manifest.csv   # 26-column metadata
│       ├── pairing_manifest.csv          # Hard positive pairs (Stage 3)
│       ├── waveforms/                    # Raw Pa waveforms
│       └── tensors/                      # Preprocessed [1,1,16000]
│
├── stages/
│   ├── stage_minus1/          # Synthetic data generator
│   ├── stage0_preprocessing/  # DataLoader, splits
│   ├── stage1_skconv1d/       # [NEXT] Multi-branch filterbank
│   ├── stage2_encoder/        # HybridSKEncoder architecture
│   ├── stage3_ssl/            # SSL pairing + training + export notes
│   ├── stage4_augmentation/   # [PLANNED] Augmentation engine
│   ├── stage5_training/       # Training utilities
│   ├── stage6_evaluation/     # Confusion matrix + operator inspection
│   └── stage7_deployment/     # Local inference engine
│
├── docs/                      # Technical documentation
└── shared/                    # Common utilities
```

---

## 🚀 Quick Start

### 1) Clone the repository
```bash
git clone https://github.com/suniltyagi/SKANN-SSL.git
cd SKANN-SSL
```

### 2) Load data (Stage 0)
```python
from stages.stage0_preprocessing.dataloader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    manifest_path="data/prototype_dataset/master_dataset_manifest.csv",
    batch_size=32
)
```

### 3) Stage 6 — batch evaluation (confusion matrix)
```bash
python stages/stage6_evaluation/stage6_confusion_matrix.py
```

### 4) Stage 6 — interactive per-clip inspector (radar plot)
```bash
python stages/stage6_evaluation/stage6_acoustic_sonar_classifier.py
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

**Audio specifications**
- Sample Rate: 16,000 Hz
- Duration: 1.0 second
- Frequency Band: 10 Hz – 8,000 Hz
- Nominal SNR: 6 dB (ship above sea noise)

---

## 🔬 Stage 3 — SSL Training (Baseline)

### Hard positive mining (pairing manifest)
`stages/stage3_ssl/pairing_manifest.py` generates anchor–partner pairs that are:
- **Same vessel class** (positive)
- **Maximally distant within class** (hard positives; avoids trivial pairing)

Output:
- `data/prototype_dataset/pairing_manifest.csv`

### Training + export notes
Baseline training was run on Kaggle (Dual T4 GPUs) and exported into a canonical bundle used by evaluation/deployment.  
See:
- `stages/stage3_ssl/README.md`
- `stages/stage3_ssl/stage3_TRAIN_EXPORT_NOTES.md`

---

## 📈 Stage 6 — Evaluation & Operator Inspection

Stage 6 provides:
- batch evaluation (confusion matrix + reports)
- per-clip probability exports (CSV + Markdown)
- interactive inspector for manual verification (radar plot + audit log)
- vessel “territory/centroid” artefact for downstream inference support

For interpretation of the confusion matrix and what the asymmetries imply, see:
- `stages/stage6_evaluation/CONFUSION_MATRIX_ANALYSIS.md`

---

## 📦 Assets and Storage Policy

Some assets are intentionally treated as **run artefacts** (large/volatile) while others are **project-facing**:

- **Tracked in git (small, stable):**
  - Stage 6 vessel territories (`vessel_territories_stage6_*.joblib`)
  - Confusion matrix + report (`confusion_matrix.png`, `confusion_report.txt`)
  - Diagnostics plots (e.g., UMAP under `artifacts/diagnostics/`)
  - Loss history CSV (posterity)

- **Typically not tracked in git (large):**
  - Training checkpoints (`BT_ckpt_epoch_*.pth`, `SKANN_SSL_GPU_Final.pth`)
  - Large encoder bundles (`*.joblib` bundles ~150MB), unless using Git LFS/Releases


- **Stage 3 trained encoder bundle (large, ~150 MB):**
  - `SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib`
  - Stored in Google Drive for convenient download (GitHub-friendly alternative to committing large binaries):
    - `https://drive.google.com/file/d/1DD7VgyfMfdQcUgxS2nVnZnL6AdpobrmP/view?usp=sharing`


The authoritative policy for Stage 3 and Stage 6 is documented in their stage READMEs.

---

## 📚 Documentation

Authoritative technical documentation is maintained in `/docs/`.

Start here:
- `docs/00_DOCUMENT_INDEX.md`

---

## 🔮 Roadmap (next steps)

Immediate (Stage 1):
- Implement multi-branch SKConv1D filterbank
- Kernels: `[3, 5, 7, 11, 15]` with attention fusion
- Retrain and compare Stage-6 evaluation metrics

Short-term:
- Reduce class confusions (especially large-vessel overlaps)
- Add physics-consistent augmentations
- Holdout validation on unseen conditions

Long-term:
- Test on real hydrophone data (ShipEar, NOAA, etc.)
- ONNX export for embedded deployment
- Real-time streaming inference

---

## 📄 Licence

[Add your licence here]

---

*Last updated: January 2026*
