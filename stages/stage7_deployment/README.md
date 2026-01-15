# Stage 7: Deployment & Local Inference

## Status: ✅ COMPLETE

## Overview

Stage 7 provides a production-ready inference engine that runs on any machine (CPU or GPU) without requiring the original training infrastructure.

---

## Components

### Production Bundle
`SKANN_SSL_Production_Bundle.joblib` contains:
- Model weights (DDP prefixes stripped)
- Class mappings (to_id, to_label)
- Vessel labels list
- Metadata (version, param count, export date)

### Vessel Territories
`vessel_territories.joblib` contains:
- 4 centroids (one per vessel class)
- Each centroid is a 128-dimensional vector
- Represents the "average signature" of each class

---

## Inference Pipeline

```
Input Audio (.npy tensor)
        │
        ▼
┌───────────────────────────┐
│  Load & Preprocess        │
│  [1, 1, 16000] tensor     │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  HybridSKEncoder          │
│  Generate 128-dim         │
│  fingerprint              │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  Distance Calculation     │
│  Euclidean to centroids   │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  Classification           │
│  Nearest centroid wins    │
│  Confidence = 1/(d+0.8)   │
└───────────────────────────┘
        │
        ▼
    Radar Plot Output
```

---

## Files

| File | Description |
|------|-------------|
| `acoustic_sonar_classifier3.py` | Interactive CLI classifier |
| `demo/skann_ssl_demo_v2.py` | GUI demo with audio playback |

---

## GUI Demo Application

For an interactive GUI-based classifier with radar plot visualization and audio playback:

```bash
cd demo/
python skann_ssl_demo_v2.py
```

Features:
- Tkinter GUI with real-time radar plot
- Audio playback of acoustic signatures
- Dropdown clip selection
- Visual confidence display

See `demo/README.md` for setup instructions (model/tensor files required).

---

## Usage

### Setup
Place these files in the same directory:
```
working_directory/
├── acoustic_sonar_classifier3.py
├── train_script.py
├── SKANN_SSL_Production_Bundle.joblib
├── vessel_territories.joblib
├── master_dataset_manifest.csv  (or pairing_manifest.csv)
└── tensors/
    ├── tensor_000000.npy
    ├── tensor_000001.npy
    └── ...
```

### Run
```bash
python acoustic_sonar_classifier3.py
```

### Interactive Session
```
👉 Enter Clip ID or 'q': 22
   [RESULT] Labeled: SMALL_CRAFT | Detected: SMALL_CRAFT (85.4%)
   [PLOT]   Saved as: final_radar_000022.png

👉 Enter Clip ID or 'q': 1918
   [RESULT] Labeled: TANKER | Detected: CARGO_SHIP (63.4%)
   [PLOT]   Saved as: final_radar_001918.png

👉 Enter Clip ID or 'q': q
🌊 Session closed.
```

---

## Radar Plot Interpretation

The radar plot shows confidence distribution across all vessel classes:

- **Green title**: Correct classification
- **Orange title**: Misclassification
- **Needle direction**: Points toward predicted class
- **Needle length**: Proportional to confidence

Example outputs in `/outputs/sample_radar_plots/`

---

## Confidence Calibration

```python
# Raw inverse distance
scores = 1.0 / (distances + 0.8)

# +0.8 offset prevents:
# - 76% appearing as 100% on the radar
# - Division by zero for exact matches
```

---

## API Usage

```python
from acoustic_sonar_classifier3 import AcousticRadarEngine

engine = AcousticRadarEngine(
    bundle_path="SKANN_SSL_Production_Bundle.joblib",
    territories_path="vessel_territories.joblib"
)

probs, pred, conf, actual = engine.classify(clip_id=22)
print(f"Predicted: {pred} ({conf*100:.1f}%)")
print(f"Actual: {actual}")
```

---

## Future Enhancements

- [ ] ONNX export for embedded deployment
- [ ] Real-time streaming inference
- [ ] Batch classification mode
- [ ] REST API wrapper
- [ ] Confidence thresholding (reject low-confidence predictions)
