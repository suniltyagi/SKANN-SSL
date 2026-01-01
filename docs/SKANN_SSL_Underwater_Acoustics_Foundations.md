# SKANN-SSL — Underwater Acoustics Foundations

**Role:** Foundational physical and mathematical constraints for SKANN-SSL.

## Core Definitions
- Instantaneous acoustic pressure **p(t)** [Pa] is the universal input.
- RMS and mean-square pressure define acoustic energy.
- Underwater reference pressure: **1 μPa**.

## Spectral Quantities
- Fundamental quantity: **pressure PSD (Pa²/Hz)**.
- Noise in a 1 Hz band is obtained by integrating PSD.

## Detectability Rule
- SKANN uses a **+6 dB signal–noise level gap** as detectability threshold.

## Propagation
- Transmission loss uses **spherical spreading**.
- TL = 20 log10(r / 1 m).

## Waveform Synthesis (Mandatory)
- Always synthesise **pressure waveforms**, never squared pressure.
- PSD → amplitude spectrum → random phase → IFFT → RMS scaling.

## Noise Components
- Tonal, broadband, and transient components explicitly modelled.

## Authority
- Canonical within underwater acoustics scope.
- Governs Documents B–D and all synthetic data generation.