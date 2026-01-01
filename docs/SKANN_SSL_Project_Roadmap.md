# SKANN-SSL — Project Roadmap & End-to-End System Overview

**Document type:** Canonical overview and execution roadmap (not stage-bound).

## System Flowchart (Authoritative Visual)

## SKANN-SSL End-to-End Pipeline

```mermaid
flowchart TD
    A[Raw Audio (x)] --> B[SKConv1D Filterbank]
    B --> C[Learned Spectrogram]
    C --> D[SKConv2D Encoder]
    D --> E[Embedding h ∈ R^d]

    E --> Aug1[Augmentation 1]
    E --> Aug2[Augmentation 2]

    Aug1 --> F[Siamese Encoder f(x)]
    Aug2 --> F

    F --> G[Projector Head g(h)]
    G --> H[Embeddings z₁, z₂]
    H --> L[Barlow Twins Loss]

    classDef blue fill:#eef5ff,stroke:#3b82f6,stroke-width:2px,color:#111;
    classDef green fill:#eafff1,stroke:#22c55e,stroke-width:2px,color:#111;
    classDef orange fill:#fff1e6,stroke:#f97316,stroke-width:2px,color:#111;

    class A,C,E,G,H blue;
    class B,D,Aug1,Aug2,F green;
    class L orange;

---

## Full Roadmap Text (from Word)

## PROJECT ROADMAP
## Hybrid Selective Kernel Self-Supervised Acoustic Representation Learning System

## Initial Sample Creation (Stage -1)
Objective: Provide raw waveform inputs for the system before Stage 0 begins. 
This stage is external to the main ML pipeline and is part of the Dataset Plan (Stage 11).
Tasks: Generate synthetic vessel-noise waveforms using a Python-based generator.
- Produce physics‑inspired acoustic signatures including blade-rate tonals, multi‑harmonics, broadband propulsion noise, cavitation bursts, and Doppler-induced warping.
- Save the generated signals as .wav files at the standard sampling rate of 16 kHz.
- Organise the synthetic dataset into labeled vessel-type folders.
Output: A set of initial waveform files (e.g., .wav) ready for ingestion by Stage 0 (Preprocessing).
Note: This stage does not replace real-world datasets. It exists to support early debugging, model bring-up, augmentation testing, and baseline evaluation before integrating NOAA, MBARI, JAMSTEC, or DCLDE datasets.
## Stage 0 — Preprocessing & Data Standardisation
Objective: Ensure all raw signals are consistent, normalised, and ready for model ingestion.
## Tasks:
Resample audio to a fixed sampling rate.
Remove DC offset.
Apply amplitude normalisation.
Optional high-pass filter.
Silence trimming / energy-gated cropping.
Segment into fixed-length clips.
## Output: Tensor [Batch, 1, T]

## Stage 1 — Learned Filterbank (Raw Waveform Front-End)
Objective: Replace STFT/mel spectrogram preprocessing with a learned multi-scale time-domain filterbank.
## Design:
Multi-branch SKConv1D with kernels (3,5,7,11,15).
Optional dilated kernels.
Optional initial Conv1D (kernel 512, stride 256).
Output: [B, F, T1] Learned time–feature map.
Phenomena Captured: Transients, tonals, modulation, broadband noise.

## Stage 2 — Hierarchical 2D Acoustic Encoder
Objective: Convert learned time–feature map into high-level 2D embeddings.
## Workflow:
## - Reshape [B, F, T1] → [B,1,F,T1]
- Apply SKConv2D blocks
- Global pooling
- Linear projection to h ∈ ℝᴰ

Stage 3 — Self-Supervised Representation Learning (SSL)
Objective: Train encoder without labels using Barlow Twins.
## Components:
Siamese encoder with shared weights.
Projector head g(h).
Barlow Twins loss: invariance + decorrelation.
Optional: SimCLR, VICReg.

## Stage 4 — Data Pipeline & Augmentation Engine
Objective: Generate physics-consistent positive pairs.
## Augmentations:
Random crop
Time shift
Gain jitter
Gaussian noise
Band-pass / low-pass filters
Time masking
Optional: Frequency masking

## Stage 5 — Training, Embedding Extraction & Clustering
## Training Loop: Augment → Encode → Project → Loss → Update.
Embedding Extraction: Discard projector, optional PCA/whitening.
Clustering: HDBSCAN (preferred), DBSCAN (alternative).
Visualisation: UMAP, t-SNE, cluster averages.

## Stage 6 — Optional / Recommended Extensions
Evaluation & analytics
Embedding variance diagnostics
Invariance robustness tests
Correlation heatmaps

## Stage 7 — Deployment & Export
ONNX export for embedded inference
Optional quantisation
Deploy to ARM/DSP hardware
## Stage 8 — Deliverables
- SKConv1D implementation
- SKConv2D implementation
- HybridSKEncoder
- SSL wrapper
- Augmentation engine
- Training pipeline
- Clustering utilities
- ONNX export
- Full TDD



## Stage 9 — Summary & Conclusion
A complete, modern, scalable self-supervised acoustic representation learning system.
Suitable for sonar, HAVS, machinery vibration, and environmental acoustics.
Industry-grade and implementation-ready.

## Flowchart Diagram

## 11. Dataset Plan
For initial testing, the system will use a Synthetic Vessel Noise Generator implemented in Python. 
This generator produces physics-inspired vessel acoustic signatures (blade-rate tonals, broadband noise, cavitation bursts, Doppler-shifted modulations), enabling controlled validation of the SKConv1D and SKConv2D architecture.

For subsequent evaluation using real-world underwater acoustic signals, the following publicly accessible datasets will be used:

• NOAA NCEI Passive Acoustic Archives – Real hydrophone recordings containing vessel noise.
• MBARI Hydrophone Dataset – Includes vessel pass-bys, maritime traffic, and ambient underwater noise.
• JAMSTEC Underwater Observatory Recordings – Long-term hydrophone deployments with merchant ship noise.
• DCLDE Workshop Datasets – Include mixed marine mammal and vessel acoustic events suitable for SSL validation.

These datasets provide diverse underwater vessel noise signatures required for unsupervised clustering, embedding evaluation, and robustness benchmarking.