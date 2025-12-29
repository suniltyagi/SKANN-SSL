# Stage 6: Evaluation & Analysis

## Status: ✅ COMPLETE

## Overview

Stage 6 provides tools to evaluate the trained model's performance, including confusion matrix generation, silhouette scoring, and centroid extraction for deployment.

---

## Tools

### 1. Confusion Matrix Generator

Runs classification on the entire dataset and produces detailed analysis.

```bash
python confusion_matrix_generator.py
```

**Outputs:**
- `confusion_matrix.png` — Visual heatmap (row-normalized percentages)
- `confusion_report.txt` — Per-class precision, recall, confusion pairs
- `misclassified_clips.csv` — Every error with metadata for analysis

**Sample Report Section:**
```
CARGO_SHIP:
  Recall (Sensitivity):  82.3%
  Precision:             79.1%
  Samples:               480
  Confused with:
    → tanker: 45 (9.4%)
    → fishing_vessel: 12 (2.5%)
```

### 2. Centroid Extraction (Kaggle Notebook)

Extracts the mean 128-dimensional embedding for each vessel class.

```python
centroids = {
    class_id: embeddings[labels == class_id].mean(axis=0)
    for class_id in np.unique(labels)
}
joblib.dump(centroids, "vessel_territories.joblib")
```

### 3. Silhouette Scoring

Measures clustering quality using cosine distance (aligned with Barlow Twins objective).

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(embeddings, labels, metric='cosine')
# Result: 0.3997 (>0.2 indicates significant clustering)
```

---

## Files

| File | Description |
|------|-------------|
| `confusion_matrix_generator.py` | Full dataset evaluation tool |
| `notebook54c27d5357.ipynb` | Kaggle notebook for centroid extraction |

---

## Metrics Explained

### Silhouette Score
- **Range**: -1 to +1
- **Interpretation**:
  - > 0.5: Strong clustering
  - 0.25 - 0.5: Reasonable clustering
  - < 0.25: Weak clustering
- **Our Result**: 0.3997 ✅

### Confusion Matrix
- Rows = Actual class
- Columns = Predicted class
- Diagonal = Correct classifications
- Off-diagonal = Errors

---

## Usage

### Local Analysis
```bash
# Ensure these files are in the same directory:
# - SKANN_SSL_Production_Bundle.joblib
# - vessel_territories.joblib
# - master_dataset_manifest.csv (or pairing_manifest.csv)
# - tensors/ folder

python confusion_matrix_generator.py
```

### Kaggle Centroid Extraction
1. Add training output as Input dataset
2. Run `notebook54c27d5357.ipynb`
3. Download `vessel_territories.joblib`

---

## Known Issues & Limitations

### Class Confusion
Large steel-hulled vessels (Tanker, Cargo) show overlap due to:
- Similar low-frequency engine harmonics
- Overlapping cavitation frequencies (400-600 Hz)
- Similar propeller RPM ranges

### Mitigation (Planned)
- Stage 1: Multi-branch SKConv1D for better multi-scale features
- Refined pairing hierarchy emphasizing blade-rate differences
