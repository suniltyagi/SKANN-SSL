# SKANN-SSL — Ambient Noise Models (Knudsen, Wenz, Kießling)

**Role:** Canonical ambient-noise modelling framework for SKANN synthetic data generation.

## Core Mandates (Locked)
- Knudsen curves alone are **invalid below 1 kHz** due to missing MF flattening.
- **Kießling parametrisation is mandatory** for 200–800 Hz mid-frequency behaviour.
- Wenz curves are used for **source-region interpretation**, not direct synthesis.
- Final PSD **must be continuous, differentiable, and sea-state dependent**.

## Frequency Regions (Authoritative)
- **< 200 Hz:** Shipping-dominated (anthropogenic).
- **200–800 Hz:** Mid-frequency transition (flattened slope).
- **1–50 kHz:** Wind/wave surface noise.
- **> 50 kHz:** Thermal noise floor.

## PSD Construction Rules
- Use **logarithmic frequency grid** for digitisation.
- Apply **logistic blending** at LF/MF/HF boundaries.
- Ensure slope continuity to avoid CNN filter artefacts.

## Model Roles
- **Knudsen:** Conceptual HF slope reference only.
- **Wenz:** Empirical envelope for dominance regions.
- **Kießling:** Primary quantitative model for synthesis.

## Output
- SKANN-ready ambient-noise PSD spanning **0–32 kHz**.
- Used directly by Document D for waveform synthesis.

## Authority
- Canonical within ambient-noise modelling scope.
- Governs all ambient PSDs used in Stage −1 synthesis.