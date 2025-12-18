# SKANN-SSL Stage -1: Synthetic Waveform Generator

## Overview

This module generates synthetic underwater acoustic waveforms for training the SKANN-SSL (Selective Kernel Audio Neural Networks with Self-Supervised Learning) system. It combines physically-motivated models of:

1. **Sea Noise** — based on digitized Knudsen curves (4 sea states)
2. **Ship Noise** — tonal + broadband + cavitation components (4 vessel classes)

The generator produces a **full-factorial structured dataset** covering all combinations of design factors for systematic ML training and evaluation.

---

## Architecture

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
                                                  ▼
                                           ┌──────────────┐
                                           │   SNR Mix    │──────▶ Output
                                           │   (6 dB)     │        Waveform
                                           └──────┬───────┘
                                                  ▲
┌─────────────────────────────────────────────────┼───────────────┐
│                    Ship Noise Generation        │               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────┴───────┐       │
│  │ Vessel       │───▶│ Tonal        │───▶│              │       │
│  │ Parameters   │    │ (shaft,BPF,  │    │   Combined   │       │
│  │              │    │  generator,  │    │   Ship       │       │
│  │              │    │  equipment,  │    │   Signal     │       │
│  │              │    │  resonances) │    │              │       │
│  └──────────────┘    └──────────────┘    │              │       │
│                                          │              │       │
│  ┌──────────────┐    ┌──────────────┐    │              │       │
│  │ Broadband    │───▶│ Flow Noise   │───▶│              │       │
│  │ Parameters   │    │ (shaped)     │    │              │       │
│  └──────────────┘    └──────────────┘    │              │       │
│                                          │              │       │
│  ┌──────────────┐    ┌──────────────┐    │              │       │
│  │ Cavitation   │───▶│ Blade-Gated  │───▶│              │       │
│  │ Bursts       │    │ Physical     │    │              │       │
│  │ (200 kHz)    │    │ Model        │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `config.py` | All configuration parameters (signal, sea, ship, factorial design) |
| `sea_noise.py` | Knudsen model and sea noise generator |
| `ship_noise.py` | Ship noise components (tonal, broadband, cavitation bursts) |
| `generator.py` | Random sampling generator (legacy, 500 clips) |
| `full_factorial_generator.py` | **Full-factorial structured dataset generator (1920 clips)** |
| `infographic.py` | Dataset visualization and summary graphics |
| `__init__.py` | Package exports |
| `data/SS*CSV.txt` | Digitized Knudsen curves |

---

## Full-Factorial Dataset Design

The structured dataset covers **all combinations** of design factors:

| Factor | Levels | Count |
|--------|--------|-------|
| Sea State | {0, 1, 3, 6} | 4 |
| Vessel Class | {small_craft, fishing_vessel, cargo_ship, tanker} | 4 |
| Blade Count | {3, 4, 5} | 3 |
| Generator Frequency | {0, 50} Hz | 2 |
| Cavitation Intensity | {0.0, 0.333, 0.667, 1.0} | 4 |
| Repetitions | 5 per combination | 5 |

**Total: 4 × 4 × 3 × 2 × 4 × 5 = 1920 clips**

---

## Quick Start

### Generate Full-Factorial Dataset (Recommended)

```python
from stages.stage_minus1.full_factorial_generator import FullFactorialDatasetGenerator

# Generate 1920 structured clips
generator = FullFactorialDatasetGenerator()
df = generator.generate(verbose=True)

# Output:
# - data/prototype_dataset/waveforms/clip_000000.npy ... clip_001919.npy
# - data/prototype_dataset/master_dataset_manifest.csv (26 columns)
```

### Generate a Single Clip

```python
from stages.stage_minus1.generator import SyntheticDataGenerator
import numpy as np

gen = SyntheticDataGenerator()
rng = np.random.default_rng(42)

# Generate combined ship + sea noise
waveform, metadata = gen.generate_clip(
    sea_state=3,
    vessel_class='cargo_ship',
    snr_db=6.0,
    rng=rng
)

print(f"Shape: {waveform.shape}")       # (16000,)
print(f"Duration: 1.0 second")
print(f"BPF: {metadata.blade_pass_freq:.1f} Hz")
```

### Use Individual Components

```python
from stages.stage_minus1.sea_noise import SeaNoiseGenerator
from stages.stage_minus1.ship_noise import ShipNoiseGenerator

# Sea noise only
sea_gen = SeaNoiseGenerator()
sea_waveform = sea_gen.generate_frame(sea_state=3)

# Ship noise only
ship_gen = ShipNoiseGenerator()
ship_waveform, params = ship_gen.generate(vessel_class='tanker')

# Access generated parameters
print(f"Shaft rate: {params.shaft_rate:.2f} Hz")
print(f"BPF: {params.blade_pass_freq:.2f} Hz")
print(f"Resonances: {params.resonance_freq_1:.1f}, {params.resonance_freq_2:.1f} Hz")
print(f"Cavitation bursts: {params.n_cavitation_bursts}")
```

---

## Signal Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sampling Frequency | 16,000 Hz | Prototype (32 kHz for full system) |
| Clip Duration | 1.0 second | Fixed |
| Samples per Clip | 16,000 | N_SAMPLES |
| Frequency Band | 10 Hz – 8,000 Hz | No simulation below 10 Hz |
| Reference Pressure | 1 µPa | Standard underwater reference |

---

## Sea Noise (Knudsen Model)

The piecewise parametric model:

```
NL(f) = a · log₁₀(f) + b   [dB re 1 µPa²/Hz]
```

| Band | Frequency Range | Physical Mechanism |
|------|-----------------|-------------------|
| Turbulence | 10 Hz → ~38 Hz | Hydrodynamic turbulence (flat plateau) |
| LF | ~38 Hz → 200 Hz | Wind/mechanical (rising) |
| MF | 200 Hz → 500 Hz | Transition shoulder |
| HF | 500 Hz → 8000 Hz | Wind-driven decay |

**Sea States**: 0 (calm), 1 (light air), 3 (gentle breeze), 6 (strong breeze)

---

## Ship Noise Components

### Vessel Classes

| Vessel Class | Shaft Rate (Hz) | BPF Range (Hz) | Cavitation Peak (Hz) |
|--------------|-----------------|----------------|----------------------|
| small_craft | 15-30 | 45-90 | 5000 |
| fishing_vessel | 4-8 | 12-32 | 1500 |
| cargo_ship | 1.5-2.5 | 6-12.5 | 600 |
| tanker | 1.0-1.5 | 4-9 | 400 |

### Acoustic Components

| Component | Description | Frequency Range |
|-----------|-------------|-----------------|
| Shaft Rate Harmonics | f₀ × h (h=1 to n_harmonics) | 10+ Hz |
| Blade Pass Frequency | BPF = f₀ × n_blades, + harmonics | 10+ Hz |
| Generator Harmonics | 50 Hz (or 60 Hz) + harmonics | 50-350 Hz |
| Equipment Running | 25 Hz (50 Hz supply) or 30 Hz (60 Hz supply) | 25-175 Hz |
| Structural Resonances | Hull (50-150), Foundation (100-300), Piping (200-500) | 50-500 Hz |
| Broadband Flow | Flat to 500 Hz, then -3 dB/octave rolloff | 10-8000 Hz |
| Cavitation Bursts | Blade-gated physical model, 200 kHz generation | Vessel-specific peak |

### Cavitation Burst Model

Cavitation is modeled as **discrete bubble collapse events**:
- Generated at 200 kHz to capture µs-scale physics
- Downsampled with anti-aliasing to 16 kHz
- Blade-gated timing (activity windows during blade passage)
- Three burst types: collapse (60%), cloud (30%), sheet (10%)
- Amplitude scaled by CAVITATION_GAIN = 0.001

---

## Output Format

### Waveform Files
- **Location**: `data/prototype_dataset/waveforms/clip_XXXXXX.npy`
- **Shape**: `(16000,)` — 1 second at 16 kHz
- **Type**: `float32`
- **Units**: Pascals (Pa)

### Tensor Files (Preprocessed)
- **Location**: `data/prototype_dataset/tensors/tensor_XXXXXX.npy`
- **Shape**: `(1, 1, 16000)` — ready for Conv1D `[batch, channels, samples]`
- **Preprocessing**: DC removal + RMS normalization

### Master Dataset Manifest (26 columns)

The `master_dataset_manifest.csv` is the **authoritative source** for the dataset, containing all metadata plus file paths.

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | int | Unique identifier (0-1919) |
| `repeat_index` | int | Repetition index (0-4) |
| **Design Factors** | | |
| `sea_state` | int | 0, 1, 3, or 6 |
| `vessel_class` | str | Vessel type |
| `n_blades` | int | Number of propeller blades (3, 4, or 5) |
| `generator_freq` | float | 0.0 or 50.0 Hz |
| `cavitation_intensity` | float | 0.0, 0.333, 0.667, or 1.0 |
| **Derived Values** | | |
| `shaft_rate` | float | Fundamental frequency (Hz) |
| `blade_pass_freq` | float | BPF = shaft_rate × n_blades |
| `has_cavitation` | bool | Whether cavitation is present |
| `cavitation_peak_freq` | float | Weighted average burst carrier (Hz) |
| `n_cavitation_bursts` | int | Number of bursts in this clip |
| `equipment_base_freq` | float | 25.0 or 30.0 Hz |
| `resonance_freq_1` | float | First structural resonance (Hz) |
| `resonance_freq_2` | float | Second structural resonance (Hz) |
| `resonance_freq_3` | float | Third structural resonance (Hz), 0 if only 2 |
| **Acoustic Measurements** | | |
| `sea_rms_pa` | float | RMS of sea noise (Pa) |
| `ship_rms_pa` | float | RMS of scaled ship noise (Pa) |
| `combined_rms_pa` | float | RMS of combined signal (Pa) |
| `scale_factor` | float | Ship scaling factor for SNR |
| `sea_spl_db` | float | Sea noise SPL (dB re 1 µPa) |
| `ship_spl_db` | float | Ship noise SPL (dB re 1 µPa) |
| `combined_spl_db` | float | Combined SPL (dB re 1 µPa) |
| `snr_db` | float | Actual SNR achieved (target: 6.0 dB) |
| **File Paths** | | |
| `tensor_path` | str | Full path to preprocessed tensor |
| `waveform_path` | str | Full path to raw waveform |

---

## Integration with Stage 0

Stage 0 uses the `master_dataset_manifest.csv` directly:

```python
import sys
sys.path.append('/content/drive/MyDrive/SKANN_SSL')

from shared.config import get_colab_paths
from stages.stage0_preprocessing import get_dataloaders, run_integration_test

# Load data using manifest
paths = get_colab_paths()
train_loader, val_loader, test_loader = get_dataloaders(
    manifest_path=paths['manifest'],
    batch_size=32,
    num_workers=0
)

# Verify
run_integration_test(train_loader, val_loader, test_loader)

# Use in training
for x, y in train_loader:
    # x shape: [32, 1, 16000]
    # y: vessel class labels (0-3)
    z = encoder(x)  # Your SKANN encoder
```

---

## Configuration (config.py)

### Full-Factorial Design Parameters

```python
SEA_STATES_DESIGN = [0, 1, 3, 6]
VESSEL_CLASSES_DESIGN = ['small_craft', 'fishing_vessel', 'cargo_ship', 'tanker']
N_BLADES_OPTIONS = [3, 4, 5]
GEN_FREQ_OPTIONS = [0.0, 50.0]
CAV_INTENSITY_LEVELS = [0.0, 0.3333, 0.6667, 1.0]
FULL_FACTORIAL_REPS = 5
```

### Signal Parameters

```python
FS = 16000              # Sampling frequency (Hz)
DURATION = 1.0          # Clip duration (seconds)
N_SAMPLES = 16000       # Samples per clip
MIN_FREQ = 10.0         # Minimum frequency (Hz)
MAX_FREQ = 8000         # Nyquist frequency (Hz)
SHIP_SNR_DB = 6.0       # Target SNR
P_REF = 1e-6            # Reference pressure (1 µPa)
```

---

## Physical Basis

### Knudsen Curves
Sea noise model based on Knudsen et al. (1948), digitized using WebPlotDigitizer and fitted with piecewise log-linear models. The turbulence band (below ~38 Hz) uses a flat plateau at the LF-turbulence intersection.

### Ship Signatures
- **Tonal components**: Ross (1976), Urick (1983)
- **Blade pass frequency**: BPF = shaft_rate × n_blades
- **Cavitation**: Physical burst model based on Brennen (1995), Rayleigh collapse time

### References
- Knudsen, V.O., et al. (1948). "Underwater ambient noise." J. Marine Research.
- Ross, D. (1976). "Mechanics of Underwater Noise." Pergamon Press.
- Urick, R.J. (1983). "Principles of Underwater Sound." McGraw-Hill.
- Brennen, C.E. (1995). "Cavitation and Bubble Dynamics." Oxford.

---

## Project Structure

```
SKANN_SSL/
├── README.md                              # Project overview
│
├── data/
│   ├── README.md
│   └── prototype_dataset/
│       ├── master_dataset_manifest.csv    # Authoritative source (26 cols)
│       ├── waveforms/
│       │   ├── clip_000000.npy
│       │   └── ... (1920 files)
│       └── tensors/
│           ├── tensor_000000.npy
│           └── ... (1920 files)
│
├── stages/
│   ├── stage_minus1/                      # ← This module
│   │   ├── README.md                      # This file
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── sea_noise.py
│   │   ├── ship_noise.py
│   │   ├── generator.py
│   │   ├── full_factorial_generator.py
│   │   ├── infographic.py
│   │   └── data/
│   │       ├── SS0CSV.txt
│   │       ├── SS1CSV.txt
│   │       ├── SS3CSV.txt
│   │       └── SS6CSV.txt
│   │
│   ├── stage0_preprocessing/              # DataLoader, splits
│   ├── stage1_skconv1d/                   # Learned filterbank
│   ├── stage2_encoder/                    # 2D encoder
│   ├── stage3_ssl/                        # Barlow Twins
│   ├── stage4_augmentation/               # Augmentation engine
│   └── stage5_training/                   # Training loop
│
├── shared/
│   ├── config.py                          # Global constants
│   └── utils.py                           # Common utilities
│
├── notebooks/
│   └── SKANN_SSL_Stage_Minus1.ipynb
│
├── checkpoints/
└── outputs/
```

---

## Version History

- **v0.3.1** (December 17, 2025): Project restructure
  - Moved to `stages/stage_minus1/`
  - Added `tensor_path` and `waveform_path` columns to manifest
  - Renamed `metadata.csv` → `master_dataset_manifest.csv`
  - Integration with Stage 0 DataLoader

- **v0.3.0** (December 16, 2025): Full-factorial structured dataset
  - New `full_factorial_generator.py` for systematic coverage
  - 1920 clips covering all design factor combinations
  - Extended `VesselParams` dataclass with tracking fields
  - Metadata now captures all randomized components

- **v0.2.0** (December 14, 2025): Critical bug fixes and enhancements
  - Fixed IFFT scaling (× N / √2) for correct physical units (Pa)
  - Fixed Zone A plateau: flat at NL(f_t) from LF regression
  - Physical cavitation burst model (200 kHz generation)
  - 10 Hz minimum frequency constraint
  - Verified: Sea 97.3 dB, Ship 103.3 dB, SNR 6.0 dB ✓

- **v0.1.0** (2024): Initial prototype implementation

---

## Status

| Task | Status |
|------|--------|
| Sea noise model (Knudsen) | ✅ Complete |
| Ship noise model (tonal + broadband) | ✅ Complete |
| Cavitation burst model | ✅ Complete |
| Full-factorial generator | ✅ Complete |
| 1920-clip dataset | ✅ Generated |
| Tensor preprocessing | ✅ Complete |
| Master manifest with paths | ✅ Complete |
| Stage 0 integration | ✅ Complete |
| Stage 1 SKConv1D | ⏳ Next |

---

## Dependencies

```
numpy>=1.20
pandas>=1.3
scipy>=1.7
matplotlib>=3.4
```

---

## Next Stage

**→ Stage 0: Preprocessing** — DataLoader, stratified splits, transforms

```python
from stages.stage0_preprocessing import get_dataloaders
```
