# Stage 3 — Self-Supervised Learning (SSL) Training & Diagnostics

Stage 3 trains the **SKANN-SSL encoder** using a **Barlow Twins self-supervised objective** and produces
the **canonical representation artefacts** consumed by downstream stages.

This stage is part of the canonical pipeline:

**Stage -1 → 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7**

Stage 3 covers **representation learning**, **embedding-space diagnostics**, and
**territory construction** (but not supervised classification).

---

## Current Version

**V3.0.0 — SKANN-SSL (HybridSKEncoderV3)**

| Parameter | Value |
|-----------|-------|
| Encoder | HybridSKEncoderV3 |
| Total parameters | 175.9M |
| SK kernel sizes | (31, 63, 127, 255, 511, 1023) |
| Backbone output (h) | 512-dim |
| Projector | 512 → 4096 → 8192 → 16384 → 256 |
| SSL method | Barlow Twins |
| Batch size | 4 (intentionally small — see V3_PAIRING_STRATEGY.md) |
| Dataset | V5 — 12,000 clips, 5 seconds @ 16 kHz, 5 classes |
| Training hardware | Google Colab A100 80 GB |

### Key Metrics

| Metric | V1 Baseline | V2.1.0 | V3 (Current) |
|--------|-------------|--------|---------------|
| Silhouette (h) | 0.3997 | 0.8299 | **0.9697** |
| kNN accuracy | — | — | **1.0000** |

Per-class silhouette (h):
- cargo_ship: 0.8923
- fishing_vessel: 0.9600
- no_vessel: 1.0000
- small_craft: 0.9969
- tanker: 0.9994

---

## Responsibilities

Stage 3 **owns**:
- Self-supervised training (Barlow Twins)
- Encoder + projector optimisation
- Embedding-space quality diagnostics (UMAP, t-SNE, silhouette)
- Territory / centroid construction in embedding space
- Export of reusable encoder and territory artefacts

Stage 3 **does NOT own**:
- Supervised classification
- Confusion matrices or accuracy metrics
- Radar plots or operational decision logic

Those belong to **Stage 6 and beyond**.

---

## Directory Structure

```
stage3_ssl/
├── README.md                              ← This file
├── __init__.py
├── barlow_twins.py                        ← Barlow Twins loss function
├── pairing_manifest.py                    ← Hard-positive pairing logic
├── V3_PAIRING_STRATEGY.md                 ← Pairing strategy documentation
├── SKANN_SSL_V3_Training_Colab.ipynb      ← CANONICAL training notebook
│
├── artifacts/
│   ├── diagnostics/
│   │   ├── t-SNE.png                      ← t-SNE visualisation (V3)
│   │   ├── UMAP.png                       ← UMAP visualisation (V3)
│   │   └── territorymap.png               ← Territory map (V3)
│   ├── vessel_territories_v3.joblib       ← Territory centroids (git-tracked)
│   ├── labels.npy                         ← Class labels (git-tracked)
│   ├── loss_history.csv                   ← Training loss log (git-tracked)
│   ├── final_metrics.txt                  ← Final metrics (git-tracked)
│   ├── SKANN_SSL_V3_Production_Bundle.joblib  ← LOCAL ONLY (683 MB, gitignored)
│   ├── embeddings_h.npy                   ← LOCAL ONLY (24.6 MB, gitignored)
│   └── embeddings_z.npy                   ← LOCAL ONLY (12.3 MB, gitignored)
│
└── archive/                               ← Historical notebooks (reference only)
    ├── skann-ssl-v2-1-0-training.ipynb
    ├── skann-ssl-v3-2-0-training-hybrid.ipynb
    ├── SKANN_SSL_V3_3_H100_Training.ipynb
    ├── SKANN_SSL_V3_3_Training.ipynb
    ├── SKANN_SSL_V3_Kaggle_T4x2.ipynb
    ├── SKANN_SSL_V3_Training.ipynb
    └── territory-map-v02-001.ipynb
```

---

## Canonical Training Notebook

There is **one canonical training notebook**:

```
SKANN_SSL_V3_Training_Colab.ipynb
```

This notebook was executed on Google Colab (A100 80 GB) and contains the complete
training run with embedded outputs for provenance. It covers:

| Cell | Purpose |
|------|---------|
| Cell -1 | Clear GPU memory |
| Cell 0 | Environment setup |
| Cell 0.5 | Load V5 dataset from Google Drive |
| Cell 1 | Configuration & hyperparameters |
| Cell 2 | Validate data |
| Cell 3 | SKANN model (SKConv1D + HybridSKEncoderV3) |
| Cell 4 | Dataset (HierarchicalDataset with hard-positive pairing) |
| Cell 5 | Barlow Twins loss & health metrics |
| Cell 6 | Training loop with health monitoring |
| Cell 7 | Extract embeddings (both h and z) |
| Cell 8 | Final silhouette scores |
| Cell 9 | t-SNE visualisation |
| Cell 10 | UMAP visualisation |
| Cell 11 | Territory mapping |
| Cell 12 | Export production bundle |
| Cell 13 | Summary |

---

## Artefacts Produced

### 1. Production bundle (deployment interface)

`artifacts/SKANN_SSL_V3_Production_Bundle.joblib` (683 MB, local only)

Contains: encoder weights, embeddings (h and z), labels, vessel labels, class map,
metrics, and metadata. Also available on Google Drive.

**Consumers:** Stage 6 evaluation, Stage 7 deployment.

### 2. Territory artefact (embedding-space structure)

`artifacts/vessel_territories_v3.joblib` (0.01 MB, git-tracked)

Contains: class centroids and territory boundaries in latent space.

**Consumers:** Stage 6 evaluation, Stage 7 inference.

### 3. Diagnostics (human inspection)

Stored under `artifacts/diagnostics/`. These evaluate **representation geometry**,
not classifier accuracy.

---

## Training Outputs on Google Drive

Full training outputs are stored at:

`/content/drive/MyDrive/SKANN-SSL/v5_output/`

| File | Size |
|------|------|
| BT_ckpt_epoch_005.pth — 025.pth | 2.1 GB each |
| SKANN_SSL_V3_Production_Bundle.joblib | 683.7 MB |
| best_model.pth | 703.5 MB |
| embeddings_h.npy / h_embeddings.npy | 24.6 MB |
| embeddings_z.npy / z_embeddings.npy | 12.3 MB |
| Visualisations (t-SNE, UMAP, territory map) | ~4 MB total |

The Stage 3 encoder bundle is also available at:
https://drive.google.com/file/d/1DD7VgyfMfdQcUgxS2nVnZnL6AdpobrmP/view

---

## Pairing Strategy (Hard-Positive Sampling)

- Script: `pairing_manifest.py`
- Documentation: `V3_PAIRING_STRATEGY.md`

Anchors are paired with **maximally dissimilar within-class examples** ("hard positives")
to enforce invariance across operating conditions while preserving class identity.

---

## Stage Boundary: Stage 3 → Stage 6

**Inputs:**
- V5 dataset tensors (12,000 clips)
- Dataset manifests (master + pairing)

**Outputs:**
- Production bundle (encoder weights + embeddings)
- Territory artefact (centroids)
- Diagnostics (t-SNE, UMAP, territory map)

**Consumers:**
- Stage 6: evaluation & operator inspection
- Stage 7: inference / deployment

---

## Notes

- Large artefacts (bundles, embeddings, checkpoints) are **not committed to git**.
- Historical notebooks are preserved in `archive/` for reference.
- Stage 3 is strictly representation-level; classification begins at Stage 6.
