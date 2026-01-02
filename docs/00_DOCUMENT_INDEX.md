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

8. System Architecture & SSL Pipeline
   - `docs/SKANN_SSL_System_Architecture_and_SSL_Pipeline.md`
   - `docs/tex/SKANN_SSL_System_Architecture_and_SSL_Pipeline.tex`

9. Diagnostics, Deployment & Deliverables
   - `docs/SKANN_SSL_Full_Scale_System_Diagnostics_Deployment_Deliverables.md`
   - `docs/tex/SKANN_SSL_Full_Scale_System_Diagnostics_Deployment_Deliverables.tex`

---

**Status:**  
This index is complete, authoritative, and intended to remain stable.
All SKANN-SSL documentation must conform to it.
