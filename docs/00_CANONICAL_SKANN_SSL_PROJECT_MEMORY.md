# SKANN-SSL — Canonical Project Memory (Authoritative)

This document defines the **non-negotiable semantic invariants** of the SKANN-SSL project.

It exists to ensure that reasoning systems (including Claude Project memory)
interpret all SKANN-SSL documentation **consistently, correctly, and without inference drift**.

This document overrides all other documents in matters of **conceptual interpretation**.

---

## 1. What SKANN-SSL Is

SKANN-SSL (Selective Kernel Audio Neural Networks with Self-Supervised Learning) is:

- A **physics-grounded acoustic representation learning system**
- Designed for **underwater acoustics, sonar, HAVS, machinery vibration, and environmental sound**
- Built around **learned filterbanks (SKConv)** and **self-supervised learning (Barlow Twins)**
- Explicitly **not** a black-box end-to-end audio classifier

---

## 2. What SKANN-SSL Is Not

SKANN-SSL is **not**:

- A supervised classification pipeline
- A dataset-specific model
- A fixed architecture tied to a single domain
- A single-stage neural network
- A document-to-stage mapping

---

## 3. Canonical Pipeline Stage Model (LOCKED)

The following stage order is **fixed and must never be altered or reinterpreted**:

Stage −1 → 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7


- Stages describe **processing order in code**
- Stages do **not** imply document ownership
- Documents may reference multiple stages
- No document defines or redefines a stage

---

## 4. Physics Before Learning (Invariant)

All learning in SKANN-SSL is **physically grounded**:

- SPL, PSD, and acoustic quantities follow physical definitions
- Synthetic data is generated from **physically interpretable PSD models**
- Learned representations must remain interpretable in frequency and energy terms

Machine learning **does not override physics**.

---

## 5. Synthetic Data Is a First-Class Citizen

Stage −1 (synthetic data generation):

- Is **intentional**, not a placeholder
- Exists to enforce controlled validation
- Is required for architecture and SSL sanity checks
- Is not optional or auxiliary

Real-world data augments, but does not replace, this stage.

---

## 6. Self-Supervised Learning Role (Invariant)

Self-supervised learning in SKANN-SSL:

- Is used for **representation learning**, not classification
- Uses **Barlow Twins–style redundancy reduction**
- Operates on **learned acoustic embeddings**, not raw labels
- Produces embeddings intended for downstream clustering and analysis

---

## 7. Augmentation Is a Support Mechanism

Augmentation (Stage 4):

- Is **non-linear and non-sequential**
- Supports self-supervised learning objectives
- Is not a standalone pipeline stage
- Must not be treated as an independent data generator

---

## 8. Canonical Authority Rules

- Markdown documents define **semantic truth**
- LaTeX documents are **rendering artefacts**
- The Project Roadmap defines **scope and structure**
- This Canonical Memory defines **meaning and interpretation**

If a conflict arises:
**Canonical Memory → Roadmap → Foundational Documents → Architecture → Deployment**

---

## 9. Deprecated Concepts (Explicitly Ignored)

The following are **deprecated and must not be used for reasoning**:

- Document labels A–F
- Filenames such as `ocean_noise2_corrected.docx`
- Any inferred mapping between documents and pipeline stages

These exist only as historical artefacts.

---

## 10. Stability Contract

This document is expected to remain stable.

Changes are permitted **only** if:
- The fundamental architecture philosophy changes, or
- The canonical stage model is formally revised (exceptional)

All other evolution must occur **within** this framework.

---

**Status:**  
This document is the **canonical semantic memory** for SKANN-SSL.
All reasoning must conform to it.
