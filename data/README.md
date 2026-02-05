# SKANN-SSL Data Directory

## V5 Dataset (Current)

Synthetic underwater acoustic waveforms for vessel detection and classification using self-supervised learning.

**Version:** V5.0.0 | **Clips:** 12,000 | **Duration:** 5.0s | **Sample Rate:** 16 kHz

---

## Quick Links

| Resource | Link |
|----------|------|
| **Full Documentation** | [Underwater-Acoustic-Synthetic-Dataset (GitHub)](https://github.com/suniltyagialtair/Underwater-Acoustic-Synthetic-Dataset) |
| **Download (Google Drive)** | [SKANN_SSL_V5_Dataset](https://drive.google.com/drive/folders/1E6vhPnkY8x8YzZ3a-k6PnL_G9gnq5gBo) |
| **Generator Code** | [stages/stage_minus1/](../stages/stage_minus1/) |

> **Note:** For complete dataset documentation including physics background, manifest schema, PyTorch DataLoader examples, and citation info, see the [public GitHub repository](https://github.com/suniltyagialtair/Underwater-Acoustic-Synthetic-Dataset).

---

## Directory Structure

```
data/
├── README.md                              # This file
├── infographic.py                         # Visualization generator script
├── SKANN_SSL_V5_Dataset_Infographic.pdf   # Visual summary (vector)
├── SKANN_SSL_V5_Dataset_Infographic.mermaid # Editable diagram source
└── v5_dataset/                            # Dataset (clone or download)
    ├── master_dataset_manifest.csv        # 27-column metadata
    ├── pairing_manifest.csv               # SSL training pairs
    ├── waveforms/                         # Raw audio (Pa, float32)
    │   └── clip_XXXXXX.npy (12,000 files)
    └── tensors/                           # Normalised (1,1,80000)
        └── tensor_XXXXXX.npy (12,000 files)
```

---

## Generated Infographics

Run `infographic.py` to generate visualization charts:

```bash
cd data/
python infographic.py --all
```

| Output | Description |
|--------|-------------|
| `spectrogram_infographic.png` | 4×2 grid: all vessel classes with/without cavitation |
| `vessel_comparison.png` | V5 shaft rate and BPF range comparison |
| `dataset_distribution.png` | Pie/bar charts of class and factor distributions |

---

## Dataset Summary

| Class | Clips | Shaft Rate (Hz) | RPM |
|-------|-------|-----------------|-----|
| tanker | 2,400 | 1.0 – 1.5 | 60–90 |
| cargo_ship | 2,400 | 1.5 – 2.5 | 90–150 |
| fishing_vessel | 2,400 | 4.0 – 8.0 | 240–480 |
| small_craft | 2,400 | 15.0 – 30.0 | 900–1800 |
| no_vessel | 2,400 | — | — |
| **Total** | **12,000** | | |

**V5 Key Feature:** Non-overlapping shaft rate ranges ensure acoustic distinguishability.

---

## Full-Factorial Design

```
Vessel:    4 sea states × 4 classes × 3 blades × 2 gen × 4 cav × 25 reps = 9,600
No-vessel: 4 sea states × 600 reps = 2,400
Total:     12,000 clips
```

---

## Quick Start

```python
import numpy as np
import pandas as pd

# Load manifest
manifest = pd.read_csv('data/v5_dataset/master_dataset_manifest.csv')

# Load waveform (raw, Pascals)
waveform = np.load('data/v5_dataset/waveforms/clip_000000.npy')
print(f"Shape: {waveform.shape}")  # (80000,)

# Load tensor (normalised, CNN-ready)
tensor = np.load('data/v5_dataset/tensors/tensor_000000.npy')
print(f"Shape: {tensor.shape}")  # (1, 1, 80000)

# Filter by class
tankers = manifest[manifest['vessel_class'] == 'tanker']
print(f"Tanker clips: {len(tankers)}")  # 2400
```

---

## Obtaining the Dataset

### Option 1: Clone from GitHub
```bash
cd data/
git clone https://github.com/suniltyagialtair/Underwater-Acoustic-Synthetic-Dataset.git v5_dataset
```

### Option 2: Download from Google Drive
1. Go to: https://drive.google.com/drive/folders/1E6vhPnkY8x8YzZ3a-k6PnL_G9gnq5gBo
2. Download all files
3. Extract to `data/v5_dataset/`

---

## License

The V5 dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## References

See the [public repository](https://github.com/suniltyagialtair/Underwater-Acoustic-Synthetic-Dataset#references) for full references and citation information.
