# SKANN-SSL — Canonical Intent and Stable Assumptions (North Star)

This document captures **stable intent** and **default interpretations** for the SKANN-SSL project.
It is designed to reduce “inference drift” in humans and assistants, while **not constraining legitimate evolution**.

If a conflict arises, prefer:
1. **A dated Decision Record (ADR)** (if present)
2. Project Roadmap (`ROADMAP.md`)
3. `/docs/*` theory/spec documents
4. Stage READMEs and operational notes under `stages/`

> This doc is a *default lens*, not a constitution.

---

## 1. What SKANN-SSL Is (Intent)

SKANN-SSL is a **physics-grounded acoustic representation learning** system intended for:
- underwater acoustics / sonar / HAVS
- vessel and machinery acoustic signatures
- interpretable embeddings suitable for analysis, clustering, and downstream decision logic

Key idea:
- learned filterbank-style front-ends (e.g., SKConv) + self-supervised learning (e.g., Barlow Twins) to produce robust embeddings.

---

## 2. What SKANN-SSL Is Not (Guardrails)

By default, SKANN-SSL is not framed as:
- “just” a supervised classification pipeline
- a dataset-specific one-off model
- a single-stage monolithic NN whose meaning is only “accuracy”

These may be used tactically, but they should not redefine the project’s intent.

---

## 3. Canonical Stage Indexing Convention (Stable)

Default stage order (numbering convention):

Stage −1 → 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7

Meaning:
- stages are a **code organisation and processing order** convention
- documents may span multiple stages
- stage ownership of artefacts can evolve, but stage numbering is kept consistent for repo stability

**Allowed exception:** the project may add stages (e.g., Stage 8) if scope expands; do so via an ADR.

---

## 4. Physics Before Learning (Stable Principle)

Default discipline:
- SPL/PSD quantities retain physical meaning and units
- synthetic generation and evaluation should preserve interpretability in frequency/energy terms

Machine learning is used to learn representations, but should not invalidate:
- unit consistency
- physically implausible transformations
- evaluation logic tied to acoustic quantities

---

## 5. Synthetic Data as First-Class (Default)

Stage −1 exists to enable:
- controlled experiments
- sanity checks for architecture and SSL
- systematic variation (sea states, cavitation, etc.)

Real data is a target for validation and generalisation, but does not remove the value of Stage −1.

---

## 6. SSL Role (Stable Principle)

Default interpretation:
- SSL is primarily for **representation learning**, not direct classification
- embeddings should support downstream analysis (clustering, territories, decision logic)

A supervised head may be used, but it should be treated as a downstream consumer of embeddings.

---

## 7. Augmentation (Living Policy)

Augmentation supports SSL objectives.
The exact augmentation set, constraints, and where it “lives” in the pipeline are **allowed to evolve**.

---

## 8. Operational Anchors (Where to find “how to run baseline”)

Stage-local operational truth for the current baseline:
- Stage 3: `stages/stage3_ssl/README.md`, `stage3_TRAIN_EXPORT_NOTES.md`
- Stage 6: `stages/stage6_evaluation/README.md`, `CONFUSION_MATRIX_ANALYSIS.md`

---

## 9. Deprecated / Historical Concepts (Default Ignore)

Unless explicitly needed for archaeology:
- document labels A–F
- old filenames and historical intermediate docs
- inferred mapping of documents → stages

---

## 10. Change Control (Lightweight)

This doc changes only when:
- the project’s intent materially changes, or
- a stable assumption becomes false

Process:
1. Create an ADR (short) describing the change and rationale
2. Update this document and add a dated entry below

### Change log
### Change log
- 2026-01-04: Converted to “North Star” doc (defaults + change control) instead of absolutist authority.
- 2026-01-04: Updated operational anchors to reflect current Stage 3 + Stage 6 baseline, artefacts, and provenance.

