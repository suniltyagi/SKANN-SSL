# SKANN-SSL V2.2.0 Demo

**Underwater Acoustic Vessel Classification System**

Self-supervised learning approach for classifying vessel types from hydrophone audio signatures.

Location: `stages/stage7_deployment/demo/`

---

## Features

- **4-Class Vessel Classification**: Cargo Ship, Tanker, Fishing Vessel, Small Craft
- **Real-time Audio Playback**: Listen to acoustic signatures while classifying
- **Radar Plot Visualization**: Interactive probability display
- **Physics-Aware Architecture**: Selective Kernel convolutions tuned for underwater acoustics (15-500+ Hz)

---

## Installation

```bash
# 1. Navigate to demo folder
cd stages/stage7_deployment/demo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python skann_ssl_demo_v2.py
```

### Requirements

- Python 3.10+
- Windows / macOS / Linux
- Model and data files (see Setup section below)

---

## Setup (Required After Clone)

The demo requires model bundles and tensor data that are **not committed to git** (too large). After cloning the repo, you need to populate these files.

### Option 1: Copy from Training Artefacts

```bash
# Copy model files from Stage 3 or Stage 6 artefacts
cp ../../stage3_ssl/artifacts/SKANN_SSL_Production_Bundle.joblib model/
cp ../../stage6_evaluation/artifacts/vessel_territories_stage6_*.joblib model/vessel_territories.joblib

# Copy tensors from main data folder
cp ../../../data/prototype_dataset/tensors/*.npy data/tensors/
```

### Option 2: Symlink to Main Data (Recommended)

On Linux/macOS:
```bash
rm -rf data/tensors
ln -s ../../../data/prototype_dataset/tensors data/tensors
```

On Windows (run as Administrator):
```cmd
rmdir data\tensors
mklink /D data\tensors ..\..\..\data\prototype_dataset\tensors
```

---

## Folder Structure

```
stages/stage7_deployment/demo/
├── skann_ssl_demo_v2.py       # Main demo application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── model/
│   ├── .gitkeep               # Placeholder (models gitignored)
│   ├── SKANN_SSL_Production_Bundle.joblib    # Trained encoder
│   └── vessel_territories.joblib              # Classification centroids
└── data/
    ├── manifest.csv           # Clip metadata (committed)
    └── tensors/               # Audio tensors (gitignored)
        ├── .gitkeep           # Placeholder
        ├── tensor_000000.npy
        └── ... (1920 files)
```

---

## Usage

1. **Launch**: Run `python skann_ssl_demo_v2.py`
2. **Select Clip**: Use dropdown or click "Random"
3. **Classify**: Click "CLASSIFY"
4. **Listen**: Audio plays in loop (click "Stop" to mute)
5. **Review**: Radar plot shows class probabilities

---

## Model Performance

| Metric | Value |
|--------|-------|
| Silhouette Score | 0.8299 |
| Embedding Dimension | 128 |
| Vessel Classes | 4 |
| Sample Rate | 16 kHz |
| Clip Duration | 1 second |

### Frequency Coverage (SK Kernels)

| Kernel Size | Frequency | Acoustic Phenomenon |
|-------------|-----------|---------------------|
| 1023 | 15+ Hz | Shaft rate |
| 511 | 31+ Hz | Generator (25 Hz) |
| 255 | 62+ Hz | Generator (50 Hz) |
| 127 | 125+ Hz | Blade pass |
| 63 | 250+ Hz | Hull resonance |
| 31 | 500+ Hz | Cavitation |

---

## Troubleshooting

**"sounddevice not installed"**
```bash
pip install sounddevice
```

**No audio output**
- Check system audio settings
- Demo works without audio (visual classification still functional)

**CUDA errors on CPU machine**
- The demo automatically handles CPU-only environments

**Missing model/tensor files**
- See Setup section above

---

*SKANN-SSL: Selective Kernel Audio Neural Networks with Self-Supervised Learning*
