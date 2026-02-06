# SKANN-SSL Complete Project Handover

**Date:** 2025-01-22  
**Project:** SKANN-SSL (Selective Kernel Audio Neural Networks with Self-Supervised Learning)  
**Owner:** Oravont Systems LLP

---

## 1. Project Identity

**Purpose:** Underwater acoustic vessel detection and classification using self-supervised learning. Commercial-grade system for sonar integration, hydrophone networks, acoustic buoys, shipboard deployment.

**Repository:** https://github.com/suniltyagi/SKANN-SSL (public)

**Local clone:** `C:\Users\Admin\uw_project\SKANN-SSL`

---

## 2. Reference Priority (Must-Follow)

1. `ROADMAP.md` (repo root) — Direction, scope
2. `docs/00_CANONICAL_SKANN_SSL_PROJECT_MEMORY.md` — North Star
3. `docs/00_DOCUMENT_INDEX.md` — Where things live
4. `README.md` (repo root) — Current status

---

## 3. Version History

| Version | Status | Key Change | Silhouette Score |
|---------|--------|------------|------------------|
| V1 | Archived | Baseline, fixed Conv1d | 0.3997 |
| V2.1.0 | **Current Production** | SK filterbank, underwater kernels (31–1023) | **0.8299** |
| V3.0.0 | **Planned** | Add no_vessel class (detection capability) | TBD |

**V1 archive location:** `archive/v1_baseline_dec2025/`

---

## 4. Architecture (V2.1.0)

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
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Projector (Deep MLP)                 │
│  512 → 4096 → 8192 → 128              │
└───────────────────────────────────────┘
        │
        ▼
   128-dim Embedding
```

**Model:** 34.4M params (training) / 1.8M (inference)

---

## 5. Pipeline Stages

| Stage | Name | Status | Location |
|-------|------|--------|----------|
| -1 | Synthetic Data | ✅ Complete | `stages/stage_minus1/` |
| 0 | Preprocessing | ✅ Complete | `stages/stage0_preprocessing/` |
| 1 | SKConv1D Filterbank | ✅ Implemented | `stages/stage1_skconv1d/` |
| 2 | Encoder | ✅ Complete | `stages/stage2_encoder/` |
| 3 | SSL Training | ✅ Complete | `stages/stage3_ssl/` |
| 4 | Augmentation | ⏳ Planned | `stages/stage4_augmentation/` |
| 5 | Training Loop | ✅ Complete | `stages/stage5_training/` |
| 6 | Evaluation | ✅ Complete | `stages/stage6_evaluation/` |
| 7 | Deployment | ✅ Prototype | `stages/stage7_deployment/` |

---

## 6. Current Dataset (V2)

| Property | Value |
|----------|-------|
| Total clips | 1,920 |
| Classes | 4 (small_craft, fishing_vessel, cargo_ship, tanker) |
| Duration | 1.0 second |
| Sample rate | 16,000 Hz |
| Frequency band | 10 Hz – 8,000 Hz |
| SNR | 6 dB (ship above sea) |
| Preprocessing | DC removal + RMS normalization |

**Full-factorial design:** 4 sea states × 4 vessels × 3 blades × 2 gen freqs × 4 cavitation levels × 5 reps

**Location:** `data/prototype_dataset/`
- `master_dataset_manifest.csv` — 26-column metadata (authoritative)
- `pairing_manifest.csv` — K=6 hard positive pairs
- `waveforms/` — Raw Pa waveforms
- `tensors/` — Preprocessed [1,1,16000]

---

## 7. Training Infrastructure

| Property | Value |
|----------|-------|
| Platform | Kaggle |
| GPUs | Dual T4 (DDP) |
| Batch size | 4 |
| Epochs | 50 |
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| Scheduler | CosineAnnealing |
| SSL Loss | Barlow Twins (λ=5e-3) |

**No local GPU available** — all training on Kaggle.

**Trained model bundle (V2):** Google Drive (not in repo due to size)  
https://drive.google.com/file/d/1DD7VgyfMfdQcUgxS2nVnZnL6AdpobrmP/view?usp=drive_link

---

## 8. Evaluation & GUI (Stage 6)

**Batch evaluation:**
- Confusion matrix
- Per-class recall
- Misclassified clips CSV

**Interactive GUI:**
- Radar plot visualization
- Per-clip inspection
- Territory/centroid-based classification

**GUI location:** `stages/stage7_deployment/demo/`

**Key artifacts:** `stages/stage6_evaluation/artifacts/`

---

## 9. Dual-Tool Workflow

| Claude Project (this) | Claude Code (terminal) |
|-----------------------|------------------------|
| Planning, strategy, docs | Hands-on code execution |
| Persistent context | Fresh each session |
| No filesystem access | Full repo access |

**Bridge:** `SESSION_LOG.md` — Claude Code writes summary, upload here to sync.

**Auto-loaded context:** `CLAUDE.md` at repo root loads into Claude Code sessions.

**Git commit/push:** Done from local machine, not Claude Code.

---

## 10. V3.0.0 Plan (Next Task)

**Objective:** Add 5th class `no_vessel` (ambient sea noise only) for detection capability.

**Single 5-class model** (not two-stage). Detection happens naturally in embedding space — periodic (vessel) vs. stochastic (no_vessel) structure.

### Dataset Addition

| Class | Clips | Breakdown |
|-------|-------|-----------|
| no_vessel | 480 | 120 per sea state (SS0, SS1, SS3, SS6) |

**V3 total:** 2,400 clips (5 classes × 480 each)

### V3 Class Distribution

| Class | Clips | Clip IDs |
|-------|-------|----------|
| small_craft | 480 | 0–479 |
| fishing_vessel | 480 | 480–959 |
| cargo_ship | 480 | 960–1439 |
| tanker | 480 | 1440–1919 |
| **no_vessel** | **480** | **1920–2399** |
| **Total** | **2,400** | |

### Pairing Strategy for no_vessel (K=3)

| Anchor SS | P1 | P2 | P3 |
|-----------|-----|-----|-----|
| SS0 | SS6 | SS3 | SS1 |
| SS1 | SS6 | SS0 | SS3 |
| SS3 | SS0 | SS1 | SS6 |
| SS6 | SS0 | SS1 | SS3 |

**Fixed rep-to-rep mapping:** Anchor rep N → Partner rep N

**Rationale:** 
- Cross-sea-state pairing forces model to learn "absence of periodicity" as defining feature
- After RMS normalization, amplitude differences between sea states are removed
- Structural difference (periodic vs. stochastic) survives and is the discriminator
- Consistent with vessel hard-positive strategy (maximize within-class variation)

---

## 11. V3 Pipeline Steps

| Step | Task | Where |
|------|------|-------|
| 1 | Generate 480 no_vessel clips | Local (CPU) |
| 2 | Create no_vessel pairing manifest (K=3) | Local |
| 3 | Merge into master + pairing manifests | Local |
| 4 | Upload to Kaggle/Drive | Manual |
| 5 | Update training notebook for 5 classes | Kaggle |
| 6 | Train on Kaggle T4×2 | Kaggle |
| 7 | Export model bundle | Kaggle |
| 8 | Territory mapping (5 classes) | Kaggle/Local |
| 9 | Confusion matrix + GUI | Local |
| 10 | Archive V2 (if V3 successful) | Repo |

---

## 12. Key Files Reference

### Stage -1 (Synthetic Data)
| File | Purpose |
|------|---------|
| `stages/stage_minus1/config.py` | All parameters |
| `stages/stage_minus1/sea_noise.py` | Knudsen model, sea noise generator |
| `stages/stage_minus1/ship_noise.py` | Vessel noise components |
| `stages/stage_minus1/full_factorial_generator.py` | Dataset generator |
| `stages/stage_minus1/generator.py` | Legacy random generator |
| `stages/stage_minus1/infographic.py` | Visualization |

### Stage 0 (Preprocessing)
| File | Purpose |
|------|---------|
| `stages/stage0_preprocessing/dataloader.py` | DataLoader, splits |

### Stage 1-2 (Encoder)
| File | Purpose |
|------|---------|
| `stages/stage1_skconv1d/skconv1d.py` | SKFilterbank |
| `stages/stage2_encoder/train_script.py` | HybridSKEncoder |

### Stage 3 (SSL Training)
| File | Purpose |
|------|---------|
| `stages/stage3_ssl/pairing_manifest.py` | Generates pairing CSV |
| `stages/stage3_ssl/train_script.py` | SSL training script |

### Stage 6-7 (Evaluation & Deployment)
| File | Purpose |
|------|---------|
| `stages/stage6_evaluation/stage6_confusion_matrix.py` | Batch evaluation |
| `stages/stage6_evaluation/stage6_acoustic_sonar_classifier.py` | Interactive GUI |
| `stages/stage7_deployment/demo/` | GUI demo application |

---

## 13. Data Files

| File | Purpose |
|------|---------|
| `data/prototype_dataset/master_dataset_manifest.csv` | 1,920 vessel clips metadata (26 columns) |
| `data/prototype_dataset/pairing_manifest.csv` | K=6 hard positive pairs |
| `data/prototype_dataset/waveforms/` | Raw Pa waveforms |
| `data/prototype_dataset/tensors/` | Preprocessed [1,1,16000] |

---

## 14. Documentation Suite

Located in `docs/`:

- Underwater Acoustics Foundations
- Ambient Noise Models (Knudsen, Wenz, Kießling)
- Parametric Sea-Noise Model
- Ambient Noise Synthesis
- System Architecture and SSL Pipeline
- Document Index (`00_DOCUMENT_INDEX.md`)
- Canonical Project Memory (`00_CANONICAL_SKANN_SSL_PROJECT_MEMORY.md`)

---

## 15. Archive Structure

**V1 Baseline (December 2025):** `archive/v1_baseline_dec2025/`

Contains complete V1 codebase for reproducibility.

---

## 16. Constraints & Notes

- **No local GPU** — all training on Kaggle T4×2
- **Physics-first methodology** — proper acoustic units (Pascals), Knudsen curves
- **Incremental changes preferred** — maintain working functionality
- Large model files stored on Google Drive (Git LFS for smaller files)

---

*End of handover.*
