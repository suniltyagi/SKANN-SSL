# SKANN-SSL — DSP & Sampling Standards (Stage 0 Preprocessing)

**Role:** Canonical DSP and preprocessing standards for SKANN Stage 0.

## Core Mandates (Locked)
- Standard sampling frequency: **f_s = 32768 Hz**.
- All signals must be resampled with anti-alias filtering.
- DC offset removal is mandatory.
- RMS-based normalisation defines physical amplitude consistency.

## Spectral Analysis Conventions
- FFT sizes are powers of two.
- Windowing: Hann or Hamming windows.
- 50% frame overlap is standard.
- Welch’s method is used for smooth PSD estimation.

## Segmentation & Tensorisation
- Fixed-length segments prepared for Stage 1.
- Zero or symmetric padding applied as required.
- Output tensor shape: **[B, 1, N]** for waveform models.

## Reconstruction & Symmetry
- Hermitian symmetry enforced for real-valued signals.
- One-sided PSD folded correctly into FFT bins.

## Authority
- Canonical within DSP and preprocessing scope.
- Governs all inputs to Stage 1 learned filterbanks.