
# SKANN-SSL Data Directory

## Prototype Dataset

Synthetic underwater acoustic waveforms generated in Stage -1.

---
## Structure

```text
data/prototype_dataset/
├── master_dataset_manifest.csv   # Authoritative source (26 columns)
├── pairing_manifest.csv          # Prescribed pairs for Stage 3 SSL
├── metadata.csv                  # Extended metadata definitions
├── dataset_infographic.png       # Visual summary of dataset stats
├── waveforms/                    # Raw waveforms
│   ├── clip_000000.npy           # Pa, float32, shape (16000,)
│   └── ... (1920 files)
└── tensors/                      # Preprocessed tensors
    ├── tensor_000000.npy         # Normalized, shape (1, 1, 16000)
    └── ... (1920 files)

```

---

## Master Manifest

The `master_dataset_manifest.csv` is the **single source of truth** for the dataset.

### Columns (26)

| Category | Columns |
| --- | --- |
| **Identifiers** | `clip_id`, `repeat_index` |
| **Design Factors** | `sea_state`, `vessel_class`, `n_blades`, `generator_freq`, `cavitation_intensity` |
| **Propulsion** | `shaft_rate`, `blade_pass_freq` |
| **Cavitation** | `has_cavitation`, `cavitation_peak_freq`, `n_cavitation_bursts` |
| **Equipment** | `equipment_base_freq`, `resonance_freq_1`, `resonance_freq_2`, `resonance_freq_3` |
| **Measurements** | `sea_rms_pa`, `ship_rms_pa`, `combined_rms_pa`, `scale_factor` |
| **SPL (dB)** | `sea_spl_db`, `ship_spl_db`, `combined_spl_db`, `snr_db` |
| **Paths** | `tensor_path`, `waveform_path` |

---

## Pairing Manifest (Stage 3 SSL)

The `pairing_manifest.csv` is used specifically for **Stage 3: Self-Supervised Representation Learning (Barlow Twins)**.

* **Purpose:** It pre-defines valid pairs (or indices) for training to ensure reproducibility and physics-consistent sampling.
* **Usage:** The Stage 3 DataLoader reads this manifest to feed the Siamese encoder, ensuring that the augmentations or pairs used for the invariance objective are tracked and consistent.

---

## Dataset Statistics

| Property | Value |
| --- | --- |
| Total clips | 1,920 |
| Duration | 1.0 second |
| Sample rate | 16,000 Hz |
| Frequency band | 10 Hz – 8,000 Hz |
| SNR | 6.0 dB |
| Format | float32 |

**Rationale:** This prototype dataset is intentionally band-limited to ≤ 8 kHz (fs = 16 kHz) for rapid validation; no distinguishing features are expected above 8 kHz for this prototype.

**Assumption:** Downstream models should expect data at a 16 kHz sampling frequency and should not expect information above 8 kHz for this dataset.

---

## Full-Factorial Design

```text
4 sea states × 4 vessel classes × 3 blade counts × 
2 generator freqs × 4 cavitation levels × 5 repeats = 1,920 clips

```

### Factor Levels

| Factor | Levels |
| --- | --- |
| Sea State | 0, 1, 3, 6 |
| Vessel Class | cargo_ship, fishing_vessel, small_craft, tanker |
| Blade Count | 3, 4, 5 |
| Generator Freq | 0 Hz, 50 Hz |
| Cavitation | 0.0, 0.333, 0.667, 1.0 |

---

## Class Distribution

| Vessel Class | Count |
| --- | --- |
| cargo_ship | 480 |
| fishing_vessel | 480 |
| small_craft | 480 |
| tanker | 480 |

---

## File Formats

### Waveforms (`waveforms/clip_*.npy`)

* Raw pressure values in Pascals (Pa)
* Shape: `(16000,)`
* dtype: `float32`

### Tensors (`tensors/tensor_*.npy`)

* DC removed, RMS normalized
* Shape: `(1, 1, 16000)` → `[B, C, T]`
* dtype: `float32`

---

## Loading

### Via DataLoader (Recommended)

```python
from stages.stage0_preprocessing import get_dataloaders

# Load standard data
train_loader, val_loader, test_loader = get_dataloaders(
    manifest_path='data/prototype_dataset/master_dataset_manifest.csv'
)

```

### Direct Load

```python
import numpy as np
import pandas as pd

manifest = pd.read_csv('data/prototype_dataset/master_dataset_manifest.csv')
tensor = np.load(manifest.loc[0, 'tensor_path'])

```

---

## Future Datasets

After prototype validation:

* NOAA NCEI Passive Acoustic Archives
* MBARI Hydrophone Dataset
* JAMSTEC Underwater Observatory
* DCLDE Workshop Datasets