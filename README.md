# SKANN-SSL: Selective Kernel Audio Neural Networks with Self-Supervised Learning

An underwater acoustic vessel detection and classification system using **physics-aware self-supervised learning**.  
SKANN-SSL learns vessel signatures directly from raw waveforms using selective-kernel filterbanks and produces robust embeddings for downstream evaluation and inference.

---

## 🎯 Project Status (V2.1.0 – Production)

| Stage | Name | Status | Description |
|------:|------|:------:|-------------|
| -1 | Synthetic Data | ✅ Complete | Physics-based waveform generator |
| 0 | Preprocessing | ✅ Complete | DataLoader, normalisation, splits |
| 1 | SKConv1D Filterbank | ✅ Implemented | Multi-branch learned filterbank |
| 2 | Encoder | ✅ Complete | HybridSKEncoder with SK frontend |
| 3 | SSL Training | ✅ Complete | Barlow Twins (V2.1.0 production) |
| 4 | Augmentation | ⏳ Planned | Physics-consistent augmentations |
| 5 | Training Loop | ✅ Complete | Integrated with Stage 3 |
| 6 | Evaluation | ✅ Complete | Confusion matrix + territory mapping |
| 7 | Deployment | ✅ Prototype | Local inference engine |

---

## 🏆 Key Results (V2.1.0 – Physics-Aware)

| Metric | Value |
|--------|------:|
| **Model Parameters** | 34.4M (training) / 1.8M (inference) |
| **Embedding Dimension** | 128 |
| **Silhouette Score (cosine)** | **0.8299** |
| **Training Hardware** | NVIDIA T4 ×2 (Kaggle, DDP) |
| **Dataset Size** | 1,920 synthetic clips |
| **SK Kernels** | (31, 63, 127, 255, 511, 1023) |
| **Frequency Coverage** | 15 Hz – 500+ Hz |

### Improvement over V1 Baseline

| Metric | V1 Baseline | V2.1.0 | Change |
|--------|-------------|--------|--------|
| Silhouette Score | 0.3997 | **0.8299** | **+107.6%** |
| Cargo↔Tanker Confusion | 32.7% | ~0% | Resolved |
| Low-Frequency Coverage | ❌ Limited | ✅ Full | Fixed |

---

## 🏗️ Architecture (V2.1.0)

```
Raw Waveform [B, 1, 16000]
        │
        ▼
┌───────────────────────────────────────┐
│  SKFilterbank (Underwater Kernels)    │
│  Kernels: [31, 63, 127, 255, 511, 1023]│
│  Attention-weighted multi-scale fusion│
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Channel Bridge + 2D Backbone         │
│  Conv layers with SyncBatchNorm       │
│  Output: 512-dim                       │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  PROJECTOR (Deep MLP)                  │
│  512 → 4096 → 8192 → 128              │
└───────────────────────────────────────┘
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
├── CHANGELOG.md
│
├── archive/
│   └── v1_baseline_dec2025/     # Archived V1 baseline
│
├── data/
│   └── prototype_dataset/
│
├── stages/
│   ├── stage_minus1/
│   ├── stage0_preprocessing/
│   ├── stage1_skconv1d/
│   ├── stage2_encoder/
│   ├── stage3_ssl/
│   ├── stage4_augmentation/
│   ├── stage5_training/
│   ├── stage6_evaluation/
│   └── stage7_deployment/
│
├── docs/
└── shared/
```

---

## 🚀 Quick Start

### Stage 6 – Evaluation
```bash
python stages/stage6_evaluation/stage6_confusion_matrix.py
```

```bash
python stages/stage6_evaluation/stage6_acoustic_sonar_classifier.py
```

---

## 📦 Historical Versions

The original V1 baseline (December 2025) is preserved for reproducibility:

- Location: `archive/v1_baseline_dec2025/`
- Silhouette (V1): 0.3997
- Architecture: Fixed Conv1D (no SK frontend)

See the archive README for reproduction details.

---

## 🔮 Roadmap (Excerpt)

### Completed
- V2.1.0 Physics-aware SK kernels
- Kaggle DDP training
- Territory-based evaluation

### Next
- Linear probe benchmarking
- Real hydrophone validation
- Edge deployment (ONNX / Jetson)

---

*Last updated: January 2026*


---

## 🧪 V2.1.0 Experiment Provenance

**Experiment ID**  
`2026-01-07_sk_integrated`

**Execution Platform**
- Kaggle notebook environment
- Dual NVIDIA T4 GPUs
- Distributed Data Parallel (DDP)

**Training Regime**
- Self-supervised learning using **Barlow Twins**
- Physics-aware **Selective Kernel (SK) filterbank**
- Cosine similarity metric for embedding evaluation

**Data Regime**
- Synthetic underwater acoustic dataset
- 1,920 clips, 1 second each
- Vessel classes: small craft, fishing vessel, cargo ship, tanker
- Full-factorial coverage of speed, cavitation, blade count, sea state

**Evaluation**
- Primary metric: **Silhouette score (cosine)**
- Achieved: **0.8299**
- Downstream validation via Stage-6 territory-based classification

**Reproducibility Anchor**
- Full run configuration and environment metadata recorded at:
  ```
  stages/stage3_ssl/runs/2026-01-07_sk_integrated/run_metadata.yaml
  ```
