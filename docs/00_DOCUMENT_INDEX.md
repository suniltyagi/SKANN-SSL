# SKANN-SSL — Master Document Index (Authoritative)

This document is the **authoritative index and governance contract** for all documentation
in the SKANN-SSL repository.

It defines:
- Canonical document identities and filenames
- The locked pipeline stage model and interpretation rules
- Authority and conflict-resolution hierarchy
- Explicit qualification of deprecated historical terminology
- The correct ingestion order for Claude Project memory

This file is intended to be a **stable, long-term record**.

---

## Non-Negotiable Rule

**Documents must NOT be mapped one-to-one to pipeline stages.**

Pipeline stages (−1…7) describe **processing order in code**.
Documents are **reference specifications** that may inform multiple stages.

---

## Conflict Resolution Hierarchy

If two documents appear to conflict:

1. **Project Roadmap** governs scope, terminology, and system structure.
2. **Foundational theory and specifications** override architecture and implementation.
3. **Architecture documents** override diagnostics and deployment guidance.
4. **Explicitly recorded amendments** override older text.

Silent reinterpretation is not permitted.

---

## Canonical Pipeline Stage Model (LOCKED)

The SKANN-SSL pipeline stages are fixed as:

Stage −1 → Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6 → Stage 7


| Stage | Meaning |
|------:|--------|
| −1 | Synthetic data generation (engineered, code-backed) |
| 0 | Preprocessing & standardisation |
| 1 | SKConv1D learned filterbank |
| 2 | Encoder backbone |
| 3 | Self-supervised learning (Barlow Twins) |
| 4 | Augmentation engine (supporting, non-linear) |
| 5 | Training utilities |
| 6 | Evaluation & diagnostics |
| 7 | Deployment |

Stages define **execution order only** and do not imply document ownership.

---

## Canonical Overview

### SKANN-SSL — Project Roadmap & System Overview

**Files:**
- `ROADMAP.md` *(primary, repository root)*
- `docs/SKANN_SSL_Project_Roadmap.md`
- `docs/tex/SKANN_SSL_Project_Roadmap.tex`

**Role:**  
Top-level system overview, stage ordering, pipeline vs non-pipeline separation,
system flowchart, dataset plan, and deliverables.

**Authority:**  
Canonical for terminology and structure.
Does not override physics, DSP, synthesis, or architecture specifications.

---

---

## Stage-Level Operational Documentation (Repository-Local)

This index intentionally avoids mapping documents one-to-one to pipeline stages. However, the repository also contains **stage-level READMEs and operational notes** that are essential for day-to-day use, reproducibility, and onboarding.

These files:
- are **implementation-facing** and may change more frequently than `/docs/*`
- must **not override** the Roadmap or theory/spec documents
- should be treated as “how to run and interpret the current baseline” references

### Stage 3 — SSL Training (Operational)
**Files (in `stages/stage3_ssl/`):**
- `stages/stage3_ssl/README.md`
- `stages/stage3_ssl/stage3_TRAIN_EXPORT_NOTES.md`
- `stages/stage3_ssl/stage3_legacy_marking_note.md`

**Purpose:**
- baseline training provenance (Kaggle notebook → `.pth` → exported `.joblib`)
- canonical Stage-3 artefact naming and placement
- pairing manifest generation logic (`pairing_manifest.csv`)
- diagnostic plots (e.g., UMAP projection of 128D embeddings)


#### Stage 3 repo snapshot (key files and run outputs)

**Key files (in `stages/stage3_ssl/`):**
- `barlow_twins.py` — legacy/reference implementation (not required for the baseline)
- `minimalgput4x2.ipynb` — Kaggle baseline training + export notebook
- `pairing_manifest.py` — generates `data/prototype_dataset/pairing_manifest.csv`
- `README.md` — Stage 3 operational baseline and stage boundary
- `stage3_legacy_marking_note.md` — legacy/active file guidance
- `stage3_TRAIN_EXPORT_NOTES.md` — train → save `.pth` → export `.joblib` notes
- `train_script_kaggle.py` — preserved Kaggle training script (paths may be Kaggle-specific)
- `train_script_repo.py` — repo-path training script (for future reproducibility)

**Diagnostics (project-facing):**
- `stages/stage3_ssl/artifacts/diagnostics/vessel_signature_umap_2d.png`

**Run provenance (baseline):**
- `stages/stage3_ssl/runs/2025-12-29_kaggle_baseline/loss_history.csv`
- `stages/stage3_ssl/runs/2025-12-29_kaggle_baseline/SKANN_SSL_GPU_Final.pth`
- `stages/stage3_ssl/runs/2025-12-29_kaggle_baseline/plots/vessel_signature_umap_2d.png`

> Note: Run outputs are provenance artefacts; the canonical stage interface consumed downstream is the exported encoder bundle (`*.joblib`) described in the Stage-3 README.


### Stage 6 — Evaluation & Diagnostics (Operational)
**Files (in `stages/stage6_evaluation/`):**
- `stages/stage6_evaluation/README.md`
- `stages/stage6_evaluation/CONFUSION_MATRIX_ANALYSIS.md`

**Purpose:**
- batch evaluation outputs (confusion matrix + report)
- interactive per-clip operator inspection (radar plot + audit log)
- interpretation guidance for row-normalised confusion matrices
- vessel territory/centroid artefact used to support inference


#### Stage 6 artefacts snapshot (key generated outputs)

**Folder:** `stages/stage6_evaluation/artifacts/`

Batch evaluation outputs:
- `confusion_matrix.png` — row-normalised confusion matrix visualisation
- `confusion_report.txt` — textual evaluation summary (accuracy, per-class stats, top confusions)
- `misclassified_clips.csv` — misclassified clip IDs + metadata (where available)
- `per_clip_class_results_confidences.csv` — per-clip class probabilities (machine-readable)
- `per_clip_class_results_confidences.md` — per-clip class probabilities (human-readable)

Interactive inspection outputs:
- `stage6_per_query_results_log.csv` — append-only log from the interactive inspector
- `final_radar_XXXXXX.png` — radar plot for an inspected clip (one per query; examples include
  `final_radar_001234.png`, `final_radar_001529.png`, `final_radar_001544.png`, `final_radar_001646.png`,
  `final_radar_001834.png`, `final_radar_001919.png`)

> Authoritative generation + interpretation:
> - `stages/stage6_evaluation/README.md`
> - `stages/stage6_evaluation/CONFUSION_MATRIX_ANALYSIS.md`


### Figures Used by Documentation
**Files:**
- `docs/figures/flowchart.md`
- `docs/figures/flowchart.tex`

**Purpose:**
- source-of-truth diagram definition(s) referenced by higher-level docs


## Foundational Theory

### SKANN-SSL — Underwater Acoustics Foundations

**Files:**
- `docs/SKANN_SSL_Underwater_Acoustics_Foundations.md`
- `docs/tex/SKANN_SSL_Underwater_Acoustics_Foundations.tex`

**Role:**  
Physical and mathematical foundations: SPL, PSD, propagation, detectability.

---

## Ambient Noise & Environmental Models

### SKANN-SSL — Ambient Noise Models (Knudsen, Wenz, Kießling)

**Files:**
- `docs/SKANN_SSL_Ambient_Noise_Models_Knudsen_Wenz_Kiessling.md`
- `docs/tex/SKANN_SSL_Ambient_Noise_Models_Knudsen_Wenz_Kiessling.tex`

**Role:**  
Conceptual and empirical ambient-noise modelling framework.

**Qualification:**  
This document intentionally does **not** include the full parametric derivation.

---

### SKANN-SSL — Parametric Sea-Noise Model (Knudsen-Based)

**Files:**
- `docs/SKANN_SSL_Parametric_Sea_Noise_Model_Knudsen.md`
- `docs/tex/SKANN_SSL_Parametric_Sea_Noise_Model_Knudsen.tex`

**Role:**  
Standalone analytic derivation of continuous, piecewise sea-noise spectra.

**Explicit Clarification:**  
Historically known as `ocean_noise2`.  
This document is **not Document B** and **does not define a pipeline stage**.

---

## DSP, Sampling & Preprocessing

### SKANN-SSL — DSP & Sampling Standards (Stage 0)

**Files:**
- `docs/SKANN_SSL_DSP_and_Sampling_Standards_Stage0.md`
- `docs/tex/SKANN_SSL_DSP_and_Sampling_Standards_Stage0.tex`

**Role:**  
DSP conventions, sampling, Welch PSD, normalisation, tensor preparation.

---

## Waveform Synthesis & Simulation

### SKANN-SSL — Ambient Noise Synthesis (Stage −1)

**Files:**
- `docs/SKANN_SSL_Ambient_Noise_Synthesis_StageMinus1.md`
- `docs/tex/SKANN_SSL_Ambient_Noise_Synthesis_StageMinus1.tex`

**Role:**  
Synthetic waveform generation from PSD models (IFFT, random phase, OLA).

---

## Core Architecture & Learning

### SKANN-SSL — System Architecture & Self-Supervised Learning Pipeline

**Files:**
- `docs/SKANN_SSL_System_Architecture_and_SSL_Pipeline.md`
- `docs/tex/SKANN_SSL_System_Architecture_and_SSL_Pipeline.tex`

**Role:**  
Encoder architecture, SKConv1D/2D, SSL (Barlow Twins), embeddings, clustering.

---

### SKANN-SSL — Neural Network Architecture & Self-Supervised Learning (Technical Reference Guide)

**Files:**
- `docs/SKANN_SSL_Neural_Network_Architecture_and_SSL_Technical_Reference_Guide.md`

**Role:**
Developer-facing reference guide bridging deep-learning fundamentals (kernels, stride, batch size, tensor shapes)
with SKANN-SSL architectural decisions (Selective Kernel mechanisms) and SSL training logic (Barlow Twins).
Also provides practical deployment guidance (FPGA vs GPU, ONNX/Jetson strategy).

**Authority:**
Supplementary reference. It supports implementation and explanation but does not override:
Canonical Project Memory → Roadmap → Foundational Documents → Architecture spec (Document E) → Deployment.
---

## Evaluation, Deployment & Deliverables

### SKANN-SSL — Full-Scale System Diagnostics, Deployment & Deliverables

**Files:**
- `docs/SKANN_SSL_Full_Scale_System_Diagnostics_Deployment_Deliverables.md`
- `docs/tex/SKANN_SSL_Full_Scale_System_Diagnostics_Deployment_Deliverables.tex`

**Role:**  
Diagnostics, robustness checks, ONNX export, deployment, system deliverables.

---

## Deprecated Historical Terminology (Quarantined)

Earlier drafts used:
- Conceptual labels **Document A–F**
- Filenames such as `ocean_noise2_corrected.docx`

These identifiers are **deprecated** and must not be used for reasoning,
cross-referencing, or future edits.

They are retained **only for historical continuity**.

---

## Recommended Upload Order for Claude Project Memory

**Rule:** Upload Markdown first (semantic authority), followed by LaTeX (rendering artefact).

1. Project Roadmap
   - `ROADMAP.md`
   - `docs/SKANN_SSL_Project_Roadmap.md`
   - `docs/tex/SKANN_SSL_Project_Roadmap.tex`

2. Document Governance
   - `docs/00_DOCUMENT_INDEX.md`

3. Underwater Acoustics Foundations
   - `docs/SKANN_SSL_Underwater_Acoustics_Foundations.md`
   - `docs/tex/SKANN_SSL_Underwater_Acoustics_Foundations.tex`

4. Ambient Noise Models
   - `docs/SKANN_SSL_Ambient_Noise_Models_Knudsen_Wenz_Kiessling.md`
   - `docs/tex/SKANN_SSL_Ambient_Noise_Models_Knudsen_Wenz_Kiessling.tex`

5. Parametric Sea-Noise Model
   - `docs/SKANN_SSL_Parametric_Sea_Noise_Model_Knudsen.md`
   - `docs/tex/SKANN_SSL_Parametric_Sea_Noise_Model_Knudsen.tex`

6. DSP & Sampling Standards (Stage 0)
   - `docs/SKANN_SSL_DSP_and_Sampling_Standards_Stage0.md`
   - `docs/tex/SKANN_SSL_DSP_and_Sampling_Standards_Stage0.tex`

7. Ambient Noise Synthesis (Stage −1)
   - `docs/SKANN_SSL_Ambient_Noise_Synthesis_StageMinus1.md`
   - `docs/tex/SKANN_SSL_Ambient_Noise_Synthesis_StageMinus1.tex`

8. System Architecture & Self-Supervised Learning Pipeline
   - `docs/SKANN_SSL_System_Architecture_and_SSL_Pipeline.md`
   - `docs/tex/SKANN_SSL_System_Architecture_and_SSL_Pipeline.tex`

9. Neural Network Architecture & SSL Technical Reference Guide
   - `docs/SKANN_SSL_Neural_Network_Architecture_and_SSL_Technical_Reference_Guide.md`

10. Full-Scale System Diagnostics, Deployment & Deliverables
   - `docs/SKANN_SSL_Full_Scale_System_Diagnostics_Deployment_Deliverables.md`
   - `docs/tex/SKANN_SSL_Full_Scale_System_Diagnostics_Deployment_Deliverables.tex`


---

**Status:**  
This index is complete, authoritative, and intended to remain stable.
All SKANN-SSL documentation must conform to it.
