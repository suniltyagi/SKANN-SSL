# Stage −1: Synthetic Waveform Generation System

## Overview

Stage −1 generates synthetic underwater acoustic waveforms for training the SKANN-SSL (Selective Kernel Audio Neural Networks with Self-Supervised Learning) system. It combines physically-motivated models of:

1. **Sea Noise** — Piecewise parametric Knudsen model (4 sea states)
2. **Ship Noise** — Tonal + broadband + cavitation components (4 vessel classes)
3. **No-Vessel** — Ambient ocean noise only (detection capability)

The generator produces a **full-factorial structured dataset** covering all combinations of design factors for systematic ML training and evaluation.

This stage is part of the canonical SKANN-SSL pipeline:

**Stage −1 → 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7**

---

## Version History

| Version | Clips | Duration | Classes | Key Changes |
|---------|-------|----------|---------|-------------|
| V1 | 1,920 | 1.0 s | 4 vessel | Baseline full-factorial |
| V2 | 1,920 | 1.0 s | 4 vessel | SK kernel fix (silhouette 0.83) |
| V3 | 2,400 | 5.0 s | 5 classes | Added no_vessel, 5s for tanker periodicity |
| **V5** | **12,000** | **5.0 s** | **5 classes** | **Non-overlapping shaft rates, 3 resonances, swell fix** |

---

## V5.0.0 Dataset (Current)

### Signal Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sampling Frequency | 16,000 Hz | Nyquist = 8,000 Hz |
| Clip Duration | 5.0 s | Captures tanker periodicity |
| Samples per Clip | 80,000 | N_SAMPLES = FS × DURATION |
| Frequency Resolution | 0.2 Hz | Δf = 1/T |
| Analysis Band | 10–8,000 Hz | Physical constraint |
| Reference Pressure | 1 µPa | Underwater standard |

### Why 5 Seconds?

**1. Tanker Periodicity Problem**
- Tanker shaft rate: ~1 Hz (60 RPM)
- At 1.0s duration: Only 1 propeller rotation captured
- Neural network cannot detect periodicity from a single cycle
- At 5.0s duration: 5 full rotations → clear periodic pattern

**2. Spectral Resolution**
- Δf = 1/T (fundamental DSP limit)
- At 1.0s: Cannot distinguish 1.1 Hz from 1.9 Hz (both in same bin)
- At 5.0s: 0.2 Hz resolution → sharp, distinct spectral lines

**3. Augmentation Survival**
- Barlow Twins crops audio for augmentation
- 1s clip cropped to 50% = 0.5s = half a tanker rotation (identity lost)
- 5s clip cropped to 50% = 2.5s = still 2–3 full cycles (identity preserved)

### Class Distribution

| Class | Clips | Description |
|-------|-------|-------------|
| small_craft | 2,400 | 15–30 Hz shaft rate, high-freq cavitation |
| fishing_vessel | 2,400 | 4–8 Hz shaft rate, mid-freq cavitation |
| cargo_ship | 2,400 | 1.5–2.5 Hz shaft rate, low-freq signature |
| tanker | 2,400 | 1.0–1.5 Hz shaft rate, lowest cavitation peak |
| no_vessel | 2,400 | Ambient sea noise only |
| **Total** | **12,000** | |

---

## V5 Critical Fixes

### 1. Non-Overlapping Shaft Rate Ranges

Previous versions had overlapping boundaries that made vessel classes acoustically indistinguishable at certain shaft rates. V5 enforces strict separation:

| Vessel Class | Shaft Rate (Hz) | RPM | Gap to Next |
|--------------|-----------------|-----|-------------|
| tanker | 1.00 – 1.50 | 60–90 | — |
| cargo_ship | 1.50 – 2.50 | 90–150 | 0.0006 Hz |
| fishing_vessel | 4.00 – 8.00 | 240–480 | 1.50 Hz |
| small_craft | 15.01 – 30.00 | 900–1800 | 7.01 Hz |

The large gaps between fishing_vessel↔small_craft (7 Hz) and cargo_ship↔fishing_vessel (1.5 Hz) ensure acoustic distinguishability.

### 2. Exactly 3 Resonances Per Clip

V5 enforces exactly 3 structural resonances per vessel clip (previously variable):

| Resonance Band | Frequency Range | Physical Source |
|----------------|-----------------|-----------------|
| Band 1 | 50–150 Hz | Hull modes |
| Band 2 | 100–300 Hz | Foundation/mounting |
| Band 3 | 200–500 Hz | Piping/ductwork |

### 3. Corrected Swell Frequency

V5 uses realistic ocean swell periods for cavitation modulation:
- **V5**: 0.05–0.15 Hz (6.7–20 second periods) ✓
- **Previous**: 0.5 Hz (unrealistic 2-second period) ✗

---

## Full-Factorial Experimental Design

### Design Factors (Vessel Classes)

| Factor | Levels | Count |
|--------|--------|-------|
| Sea State | 0, 1, 3, 6 | 4 |
| Vessel Class | small_craft, fishing_vessel, cargo_ship, tanker | 4 |
| Blade Count | 3, 4, 5 | 3 |
| Generator Frequency | 0 Hz (off), 50 Hz | 2 |
| Cavitation Intensity | 0.0, 0.333, 0.667, 1.0 | 4 |
| Repetitions | 25 | 25 |

**Total Vessel Clips:** 4 × 4 × 3 × 2 × 4 × 25 = **9,600**

### Design Factors (No-Vessel Class)

| Factor | Levels | Count |
|--------|--------|-------|
| Sea State | 0, 1, 3, 6 | 4 |
| Repetitions per state | 600 | 600 |

**Total No-Vessel Clips:** 4 × 600 = **2,400**

**Grand Total:** 9,600 + 2,400 = **12,000 clips**

---

## Architecture

### Data Extraction (One-Time Setup)

```
Knudsen Curve Digitization
──────────────────────────

┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  knudsens_curve.svg │────▶│  WebPlotDigitizer   │────▶│  SS*CSV.txt files   │
│  (Reference image)  │     │  (Manual extraction)│     │  (freq, NL pairs)   │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                                                  │
                                     ┌────────────────────────────┘
                                     ▼
                            ┌─────────────────┐
                            │ SS0CSV.txt (Calm)│
                            │ SS1CSV.txt       │
                            │ SS3CSV.txt       │
                            │ SS6CSV.txt       │
                            └─────────────────┘
```

The Knudsen curves were digitized from published ocean ambient noise spectra using 
[WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/). The extracted (frequency, noise level) 
pairs are stored in `data/SS*CSV.txt` and fitted to a piecewise parametric model by `sea_noise.py`.

### Generation Pipeline

```
Stage -1 Pipeline
─────────────────

┌─────────────────────────────────────────────────────────────────┐
│                    Sea Noise Generation                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Knudsen CSV  │───▶│ Piecewise    │───▶│ Colored      │       │
│  │ (SS0,1,3,6)  │    │ Parametric   │    │ Noise via    │       │
│  │              │    │ Model        │    │ FFT/IFFT     │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
└─────────────────────────────────────────────────┼───────────────┘
                                                  │
                              ┌────────────────────┼────────────────┐
                              │                    │                │
                              ▼                    ▼                ▼
                       ┌──────────────┐     ┌──────────────┐  ┌──────────┐
                       │   SNR Mix    │     │  No Vessel   │  │  Output  │
                       │   (6 dB)     │     │  (sea only)  │  │ Waveform │
                       └──────┬───────┘     └──────────────┘  └──────────┘
                              ▲
┌─────────────────────────────┼───────────────────────────────────┐
│                    Ship Noise Generation                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Vessel       │───▶│ Tonal        │───▶│   Combined   │       │
│  │ Parameters   │    │ (shaft,BPF,  │    │   Ship       │       │
│  │              │    │  generator)  │    │   Signal     │       │
│  └──────────────┘    └──────────────┘    │              │       │
│                                          │              │       │
│  ┌──────────────┐    ┌──────────────┐    │              │       │
│  │ Cavitation   │───▶│ Burst Model  │───▶│              │       │
│  │ Intensity    │    │ (physical)   │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ship Noise Model

### Acoustic Components

The ship noise signal comprises six component types:

#### 1. Shaft Rate Harmonics
Fundamental shaft rotation frequency plus harmonics from mechanical imbalance:
```
f_shaft = n_shaft / 60  [Hz]
```
Amplitude decays at 3 dB per harmonic.

#### 2. Blade Pass Frequency (BPF) Harmonics
The dominant tonal component:
```
f_BPF = n_blades × f_shaft  [Hz]
```
Amplitude decays at 4 dB per harmonic.

#### 3. Generator Harmonics
For vessels with electrical generators (50 Hz or 60 Hz systems):
```
f_gen = {50, 100, 150} Hz  or  {60, 120, 180} Hz
```
Level is 10 dB below shaft harmonics.

#### 4. Equipment Running Frequency
Shipboard machinery (pumps, compressors, fans):
```
f_equip = 25 Hz or 30 Hz (depending on generator)
```
Level is 15 dB below shaft harmonics with 4–7 harmonics.

#### 5. Structural Resonances
Three narrowband resonances per clip from ship structure (see V5 fixes above).

#### 6. Broadband Flow Noise
Hydrodynamic turbulence with power-law spectrum:
```
S_flow(f) ∝ f^(rolloff/20)
```
Rolloff varies by vessel class (−3 to −6 dB/octave).

### Cavitation Burst Model

Cavitation is modeled as discrete bubble collapse events using a physical approach:

#### High-Resolution Generation
Bursts are generated at **200 kHz** to capture the µs-scale collapse physics, then downsampled to 16 kHz with anti-aliasing:
```python
FS_BURST = 200000  # 200 kHz generation
# ... burst synthesis ...
cav = scipy_signal.decimate(burst_sig, FS_BURST // FS, zero_phase=True)
```

#### Blade-Gated Timing
Bursts occur synchronised to blade passage with swell modulation:
```python
swell_freq = rng.uniform(0.05, 0.15)  # V5: Realistic ocean swell
t_burst = t_nominal + 0.1 * sin(2π × swell_freq × t) × period
```

#### Burst Physics
- **Collapse time (τ)**: 50–200 µs (Rayleigh collapse)
- **Envelope**: Exponential decay × sinusoidal carrier
- **Peak frequency**: Vessel-dependent (400–5000 Hz)

| Vessel Class | Cavitation Peak | Physical Reason |
|--------------|-----------------|-----------------|
| small_craft | 5000 Hz | Small bubbles collapse quickly |
| fishing_vessel | 1500 Hz | Medium bubble size |
| cargo_ship | 600 Hz | Large bubbles |
| tanker | 400 Hz | Largest bubbles collapse slowly |

---

## Sea Noise Model (Knudsen)

### Piecewise Parametric Model

```
NL(f) = a · log₁₀(f) + b   [dB re 1 µPa²/Hz]
```

Coefficients (a, b) are fitted from digitised Knudsen curves for each sea state.

### Frequency Bands

| Band | Frequency Range | Physical Mechanism |
|------|-----------------|-------------------|
| Turbulence | 10 Hz → f_t | Hydrodynamic turbulence (flat plateau) |
| LF | f_t → 200 Hz | Wind/mechanical (rising) |
| MF | 200 Hz → 500 Hz | Transition shoulder |
| HF | 500 Hz → 8000 Hz | Wind-driven decay |

where f_t ≈ 30–40 Hz is the turbulence-LF transition frequency.

### Sea State Levels

| Sea State | Description | Sea RMS (Pa) | Sea SPL (dB) |
|-----------|-------------|--------------|--------------|
| 0 | Calm | 0.0076 | 77.6 |
| 1 | Light air | 0.0151 | 83.6 |
| 3 | Gentle breeze | 0.0479 | 93.6 |
| 6 | Strong breeze | 0.1514 | 103.6 |

---

## SNR-Controlled Mixing

Ship and sea signals are combined at a controlled signal-to-noise ratio:

```
SNR = L_ship − L_sea = 6 dB
```

The ship waveform is scaled to achieve exactly 6 dB above the sea noise floor:

```python
scale_factor = sea_rms * 10^(SNR_DB / 20) / ship_rms
combined = sea + scale_factor * ship
```

---

## Output Specification

### Dataset Location

**Google Drive:** [SKANN_SSL_V5_Dataset](https://drive.google.com/drive/u/1/folders/1E6vhPnkY8x8YzZ3a-k6PnL_G9gnq5gBo)

**GitHub (Public):** [Underwater-Acoustic-Synthetic-Dataset](https://github.com/suniltyagialtair/Underwater-Acoustic-Synthetic-Dataset)

### File Structure

```
SKANN_SSL_V5_Dataset/
├── waveforms/
│   ├── clip_000000.npy
│   ├── clip_000001.npy
│   └── ... (12,000 files)
├── tensors/
│   ├── tensor_000000.npy
│   ├── tensor_000001.npy
│   └── ... (12,000 files)
├── master_dataset_manifest.csv
└── pairing_manifest.csv
```

### File Formats

| Type | Format | Shape | Units |
|------|--------|-------|-------|
| Waveforms | `.npy` (Float32) | (80000,) | Pascals |
| Tensors | `.npy` (Float32) | (1, 1, 80000) | Normalised |
| Manifest | `.csv` | 12,000 rows × 27 columns | Metadata |

### Manifest Schema (27 Columns)

| Column | Type | Description |
|--------|------|-------------|
| clip_id | int | Unique identifier (0–11999) |
| repeat_index | int | Repetition within combination |
| sea_state | int | Sea state (0, 1, 3, 6) |
| vessel_class | str | Class label |
| n_blades | int | Propeller blade count |
| generator_freq | float | Generator frequency (0 or 50 Hz) |
| cavitation_intensity | float | Cavitation level (0.0–1.0) |
| shaft_rate | float | Shaft rotation frequency (Hz) |
| blade_pass_freq | float | BPF = shaft_rate × n_blades |
| has_cavitation | bool | Whether cavitation is present |
| cavitation_peak_freq | float | Cavitation spectral peak (Hz) |
| n_cavitation_bursts | int | Number of bursts in clip |
| equipment_base_freq | float | Equipment running frequency |
| resonance_freq_1 | float | First resonance (Hz) |
| resonance_freq_2 | float | Second resonance (Hz) |
| resonance_freq_3 | float | Third resonance (Hz) |
| sea_rms_pa | float | Sea noise RMS (Pascals) |
| ship_rms_pa | float | Ship noise RMS (Pascals) |
| combined_rms_pa | float | Combined RMS (Pascals) |
| scale_factor | float | Ship scaling for SNR |
| sea_spl_db | float | Sea SPL (dB re 1 µPa) |
| ship_spl_db | float | Ship SPL (dB re 1 µPa) |
| combined_spl_db | float | Combined SPL (dB re 1 µPa) |
| snr_db | float | Actual SNR (dB) |
| filename | str | Waveform filename |
| tensor_path | str | Tensor relative path |
| waveform_path | str | Waveform relative path |

---

## Tensor Preprocessing (Stage 0 Interface)

### Normalisation Recipe

```python
def preprocess(waveform):
    # 1. DC Removal
    x = waveform - np.mean(waveform)
    
    # 2. RMS Normalisation
    rms = np.sqrt(np.mean(x ** 2)) + 1e-8
    x = x / rms
    
    # 3. Reshape for CNN
    return x.reshape(1, 1, -1).astype(np.float32)
```

### What Remains After Normalisation?

After RMS normalisation, amplitude differences between sea states are removed. What remains is **structural difference**:
- **Vessel clips**: Periodic patterns (shaft harmonics, BPF)
- **No-vessel clips**: Stochastic patterns (Knudsen-shaped, no periodicity)

This structural difference is what Barlow Twins learns to separate.

---

## Source Code

### Repository Structure

```
stages/stage_minus1/
├── README.md                              # This file
├── config.py                              # Central configuration
├── sea_noise.py                           # Knudsen model implementation
├── ship_noise.py                          # Vessel noise physics (V5)
├── full_factorial_generator_v5.py         # V5 Colab/local generator
├── generate_pairing_manifest_v3_2.py      # SSL pairing manifest
├── generate_tensors.py                    # Tensor preprocessing
├── infographic.py                         # Visualization utilities
├── SKANN_SSL_V5_Dataset_Generator.ipynb   # Generation notebook (Colab)
├── SKANN_SSL_V5_Dataset_Infographic.pdf   # Dataset visualization
├── SKANN_SSL_V5_Dataset_Infographic.png   # Dataset visualization
├── __init__.py                            # Package exports
└── data/
    ├── SS0CSV.txt                         # Knudsen curve data (Sea State 0)
    ├── SS1CSV.txt                         # Knudsen curve data (Sea State 1)
    ├── SS3CSV.txt                         # Knudsen curve data (Sea State 3)
    ├── SS6CSV.txt                         # Knudsen curve data (Sea State 6)
    └── knudsens_curve.svg                 # Source reference (WebPlotDigitizer)
```

### Generation (Colab)

```python
from generator_colab import ColabDatasetGenerator

generator = ColabDatasetGenerator(
    output_dir='/content/drive/MyDrive/SKANN_SSL_V5_Dataset',
    reps=25,           # 9,600 vessel clips
    no_vessel_reps=600 # 2,400 no-vessel clips
)

df = generator.generate(checkpoint_interval=500)
```

### Single Clip Generation (Development)

```python
from sea_noise import SeaNoiseGenerator
from ship_noise import ShipNoiseGenerator
import config

# Sea noise only (no_vessel equivalent)
sea_gen = SeaNoiseGenerator(fs=config.FS)
sea_waveform = sea_gen.generate_frame(sea_state=3)

# Ship noise
ship_gen = ShipNoiseGenerator(fs=config.FS)
ship_waveform, params = ship_gen.generate(vessel_class='tanker')

print(f"Shaft rate: {params.shaft_rate:.2f} Hz")
print(f"BPF: {params.blade_pass_freq:.2f} Hz")
```

---

## Stage Boundary

### Inputs
- Knudsen curve data: `data/SS*CSV.txt`
- Configuration: `config.py`

### Outputs (Stage 0 Interface)
- Waveforms: `waveforms/clip_XXXXXX.npy`
- Tensors: `tensors/tensor_XXXXXX.npy`
- Manifest: `master_dataset_manifest.csv`
- Pairing manifest: `pairing_manifest.csv` (for SSL training)

### Consumers
- **Stage 0**: Preprocessing & DataLoader
- **Stage 3**: SSL training (via pairing manifest)

---

## Known Limitations

1. **Doppler not implemented**: Doppler-induced frequency shifts were deprioritised. Can be added as post-processing if needed.

2. **Single-source model**: Each clip contains one vessel. Multi-source scenarios require extension.

3. **Stationary statistics**: Each clip has fixed parameters. Time-varying (e.g., CPA approach) not modeled.

4. **No propagation effects**: Multipath, surface/bottom reflection, and range-dependent attenuation not included.

5. **Synthetic only**: Real-world datasets (NOAA, MBARI, JAMSTEC, DCLDE) should be integrated for final validation.

---

## References

- Knudsen, V. O., et al. (1948). "Underwater Ambient Noise." Journal of Marine Research.
- Ross, D. (1976). "Mechanics of Underwater Noise." Pergamon Press.
- Urick, R. J. (1983). "Principles of Underwater Sound." McGraw-Hill.

---

## Version

- **Document version**: 5.0.1
- **Dataset version**: V5.0.0
- **Date**: February 2026
