# Stage 3 — Self-Supervised Learning (SSL) Encoder Training, Diagnostics & Territory Construction

Stage 3 trains the **SKANN‑SSL encoder** using a **Barlow Twins–style self‑supervised objective** and produces
the **canonical representation artefacts** consumed by downstream stages.

This stage is part of the canonical pipeline:

**Stage −1 → 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7**

Stage 3 is intentionally broad: it covers **representation learning**, **embedding‑space diagnostics**, and
**territory construction** (but not supervised classification).

---

## Responsibilities of Stage 3

Stage 3 **owns**:

- Self‑supervised training (Barlow Twins)
- Encoder + projector optimisation
- Embedding‑space quality diagnostics (UMAP, t‑SNE, silhouette)
- Territory / centroid construction in embedding space
- Export of reusable encoder and territory artefacts

Stage 3 **does NOT own**:

- Supervised classification
- Confusion matrices or accuracy metrics
- Radar plots or operational decision logic

Those belong to **Stage 6 and beyond**.

---

## What Stage 3 Produces (Interface Artefacts)

### 1. Canonical encoder bundle (downstream interface)
- `stages/stage3_ssl/artifacts/SKANN_SSL_Production_Bundle.joblib`  
  *(large file, stored externally; git‑ignored)*

This bundle contains:
- Encoder weights
- Label metadata
- Embedding configuration

**Consumers**
- Stage 6 evaluation
- Stage 7 inference / deployment (planned)

---

### 2. Territory artefact (embedding‑space structure)
- `stages/stage3_ssl/artifacts/territories/vessel_territories_v2_1_0.joblib`  
  *(derived, local; git‑ignored)*

This artefact encodes:
- Class centroids
- Territory boundaries in latent space

It is produced by the territory‑mapping notebook and **consumed by Stage 6**.

---

### 3. Diagnostics (human inspection)
Stored under:

```
stages/stage3_ssl/artifacts/diagnostics/
```

Typical outputs:
- `silhouette_analysis.png`
- `tsne.png`
- `territory_map.png`
- `vessel_signature_umap_2d.png`

These evaluate **representation geometry**, not classifier accuracy.

---

## Directory Structure (Current, Authoritative)

```
stage3_ssl/
├── README.md
├── __init__.py
├── train_script.py                 # Canonical SSL training logic
├── pairing_manifest.py
├── barlow_twins.py                 # Helper / reference (non‑authoritative)
│
├── notebooks/
│   ├── skann-ssl-v2-1-0-training.ipynb
│   └── territory-map-v2-01.ipynb
│
├── artifacts/
│   ├── diagnostics/
│   │   ├── silhouette_analysis.png
│   │   ├── tsne.png
│   │   ├── territory_map.png
│   │   └── vessel_signature_umap_2d.png
│   └── territories/
│       └── vessel_territories_v2_1_0.joblib   # local, git‑ignored
│
└── runs/
    ├── 2026-01-07_kaggle_v2.1.0_underwater/
    │   ├── run_metadata.yaml
    │   └── loss_history.txt
    ├── _local_cpu_smoketest/
    └── README.md
```

---

## Notebooks

The `notebooks/` directory contains **execution notebooks** used for running and
post‑processing Stage‑3 workflows. These notebooks orchestrate training and analysis
but do **not** define canonical logic.

- **`skann-ssl-v2-1-0-training.ipynb`**  
  Kaggle execution notebook for the **V2.1.0** SSL run. Handles environment setup,
  dataset wiring, training execution, diagnostics, and export of the production bundle.

- **`territory-map-v2-01.ipynb`**  
  Constructs **embedding‑space territories and centroids** from the Stage‑3 production
  bundle and produces:
  - `artifacts/diagnostics/territory_map.png`
  - `artifacts/territories/vessel_territories_v2_1_0.joblib`

> **Authoritative logic always lives in `train_script.py`.**
> Notebooks must remain thin execution wrappers.

---

## Canonical Training Entry Point

There is **exactly one authoritative training script**:

```
stages/stage3_ssl/train_script.py
```

All training runs (including Kaggle notebooks) must correspond to this logic.
Notebook‑specific cells may exist for execution convenience but must not diverge
architecturally.

---

## Current Mainline Version

**V2.1.0 — Physics‑Aware SKANN‑SSL**

- Encoder: `HybridSKEncoderV2`
- SK kernel sizes: `(31, 63, 127, 255, 511, 1023)`
- Projector: `512 → 4096 → 8192 → 128`
- SSL method: Barlow Twins
- Best silhouette score (cosine): **0.8299**

This version supersedes all V1 baselines.

---

## UMAP / Embedding Diagnostics (Interpretation)

Each point in a UMAP / t‑SNE plot represents a **128‑D embedding** produced by the encoder.
Neighbourhoods in 2‑D approximate neighbourhoods in latent space.

Healthy SSL indicators:
- No representation collapse
- Structured, class‑coherent regions
- Within‑class spread reflecting operating conditions (speed, SNR, background)

Axes have **no physical meaning**; relative geometry is what matters.

---

## Pairing Manifest (Hard‑Positive Sampling)

- Script: `pairing_manifest.py`
- Output: `data/prototype_dataset/pairing_manifest.csv`

Anchors are paired with **maximally dissimilar within‑class examples** (“hard positives”)
to enforce invariance across operating conditions while preserving class identity.

---

## Stage Boundary: Stage 3 → Stage 6

### Inputs
- Dataset tensors
- Dataset manifests
- Pairing manifest

### Outputs
- Encoder bundle (`SKANN_SSL_Production_Bundle.joblib`)
- Territory artefact (`vessel_territories_v2_1_0.joblib`)
- Optional diagnostics

### Consumers
- **Stage 6** evaluation & operator inspection
- **Stage 7** inference / deployment (planned)

---

## Artefact & Git Hygiene

Large or derived artefacts are **not committed to Git**.

Recommended `.gitignore` entries:

```
*.joblib
*.pth
stages/stage3_ssl/runs/
```

---

## Notes

- `_local_cpu_smoketest` runs are non‑authoritative sanity checks.
- Historical V1 artefacts are archived under `archive/`.
- Stage 3 is strictly representation‑level; classification begins at Stage 6.
