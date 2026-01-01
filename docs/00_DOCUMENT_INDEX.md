# SKANN-SSL — Master Document Index (Authoritative)

This index defines the hierarchy, authority, and relationships between all documents in the SKANN-SSL project memory.

## Non‑negotiable rule
- **Do not map documents to development stages.** Stages (−1…7) are implementation constructs; documents are references that may be used at any stage.

## Conflict resolution
1. Canonical overview/roadmap documents override everything else for scope and terminology.
2. Foundational theory/specifications override implementation notes.
3. Explicit amendments override older versions (record amendments in the same section).

---

## Canonical Overview

### SKANN-SSL — Project Roadmap & End-to-End System Overview
**Files:**
- SKANN_SSL_Project_Roadmap.md
- SKANN_SSL_Project_Roadmap.tex
- diagram2.svg (flowchart source used in LaTeX)

**Role:** Top-level narrative, execution roadmap, and system overview for SKANN-SSL.

**Authority:** Canonical for scope, terminology, and high-level system decomposition. Does **not** override physics/DSP/synthesis/architecture/deployment specifications.

---

## Foundational Theory and Specifications

### SKANN-SSL — Underwater Acoustics Foundations
**Files:**
- SKANN_SSL_Underwater_Acoustics_Foundations.tex
- SKANN_SSL_Underwater_Acoustics_Foundations.md

**Role:** Physical and mathematical foundations for interpreting ocean-borne sound and defining SKANN acoustic quantities.

**Authority:** Canonical for pressure/RMS/SPL/PSD definitions, propagation (TL), and detectability conventions.

---

## Ambient Noise and Environmental Models

### SKANN-SSL — Ambient Noise Models (Knudsen, Wenz, Kießling)
**Files:**
- SKANN_SSL_Ambient_Noise_Models_Knudsen_Wenz_Kiessling.tex
- SKANN_SSL_Ambient_Noise_Models_Knudsen_Wenz_Kiessling.md

**Role:** Ambient-noise modelling framework (conceptual + empirical + modern parametrisations) for ocean PSDs.

**Authority:** Canonical for ambient PSD shapes, model-selection constraints, and frequency-region interpretations.

### SKANN-SSL — Parametric Sea-Noise Model from Digitised Knudsen Curves
**Files:**
- SKANN_SSL_Parametric_Sea_Noise_Model_Knudsen.tex
- SKANN_SSL_Parametric_Sea_Noise_Model_Knudsen.md

**Role:** Analytic, continuous piecewise parametric sea-noise PSD derived from digitised Knudsen curves (SKANN simulation model).

**Authority:** Canonical for the specific parametric equations/coefficients used for SKANN sea-noise simulation.

---

## DSP, Sampling and Signal Processing

### SKANN-SSL — DSP & Sampling Standards (Stage 0 Preprocessing)
**Files:**
- SKANN_SSL_DSP_and_Sampling_Standards_Stage0.tex
- SKANN_SSL_DSP_and_Sampling_Standards_Stage0.md

**Role:** DSP and preprocessing standards (sampling, resampling, DC removal, windowing/overlap, PSD estimation, normalisation, tensor prep).

**Authority:** Canonical for signal standardisation conventions used across SKANN workflows.

---

## Waveform Synthesis and Simulation

### SKANN-SSL — Ambient Noise Synthesis (Stage −1)
**Files:**
- SKANN_SSL_Ambient_Noise_Synthesis_StageMinus1.tex
- SKANN_SSL_Ambient_Noise_Synthesis_StageMinus1.md

**Role:** Procedure for generating time-domain ambient-noise waveforms from PSD models (PSD → random phase → IFFT → RMS; OLA for long durations).

**Authority:** Canonical for synthetic ambient-noise waveform generation used in SKANN simulations.

---

## Core System Architecture and Learning

### SKANN-SSL — System Architecture & Self-Supervised Learning Pipeline (Stages 0–7)
**Files:**
- SKANN_SSL_System_Architecture_and_SSL_Pipeline.tex
- SKANN_SSL_System_Architecture_and_SSL_Pipeline.md

**Role:** End-to-end architecture specification (encoder, SSL wrapper, augmentation utilities, embedding extraction and clustering workflows).

**Authority:** Canonical for model/training logic and component definitions.

---

## Operationalisation, Evaluation and Deployment

### SKANN-SSL — Full-Scale System: Diagnostics, Deployment & Deliverables
**Files:**
- SKANN_SSL_Full_Scale_System_Diagnostics_Deployment_Deliverables.tex
- SKANN_SSL_Full_Scale_System_Diagnostics_Deployment_Deliverables.md

**Role:** Diagnostics suite, dataset strategy, export/deployment workflow, and deliverables list for operationalising SKANN-SSL.

**Authority:** Canonical for evaluation and operational readiness criteria; does not override foundational specifications.

---

## Optional canonical project memory (if used)
- 00_CANONICAL_SKANN_SSL_PROJECT_MEMORY.md

---

## Recommended upload order for Claude Project Memory
1. SKANN_SSL_Project_Roadmap.md
2. 00_DOCUMENT_INDEX.md (this file)
3. SKANN_SSL_Underwater_Acoustics_Foundations.md and .tex
4. SKANN_SSL_Ambient_Noise_Models_Knudsen_Wenz_Kiessling.md and .tex
5. SKANN_SSL_Parametric_Sea_Noise_Model_Knudsen.md and .tex
6. SKANN_SSL_DSP_and_Sampling_Standards_Stage0.md and .tex
7. SKANN_SSL_Ambient_Noise_Synthesis_StageMinus1.md and .tex
8. SKANN_SSL_System_Architecture_and_SSL_Pipeline.md and .tex
9. SKANN_SSL_Full_Scale_System_Diagnostics_Deployment_Deliverables.md and .tex
