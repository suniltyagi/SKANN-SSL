# SKANN-SSL — Ambient Noise Synthesis (Stage −1)

**Role:** Canonical synthesis procedure for generating time-domain ambient-noise waveforms for SKANN.

## Core Synthesis Pipeline (Locked)
- Digitised PSD → FFT frequency grid.
- Compute amplitude spectrum from PSD.
- Assign uniform random phase to each bin.
- Impose Hermitian symmetry.
- Inverse FFT to obtain time-domain waveform.
- Apply RMS scaling **once**, post-IFFT.

## Long-Duration Noise
- Use **Overlap–Add (OLA)** synthesis.
- Hann window with **50% hop size**.
- Prevents discontinuities at frame boundaries.

## Frequency Handling
- PSD interpolated in **log-frequency** space.
- FFT grid consistent with Stage 0 sampling standards.

## Multi-Channel Support
- Independent-channel synthesis supported.
- Spatially correlated noise supported via response weighting.

## Integration
- Output waveforms directly compatible with Stage 0 preprocessing.
- Used for SSL pretraining, augmentation, and SNR benchmarking.

## Authority
- Canonical within ambient-noise synthesis scope.
- Governs all synthetic ambient-noise generation in SKANN.