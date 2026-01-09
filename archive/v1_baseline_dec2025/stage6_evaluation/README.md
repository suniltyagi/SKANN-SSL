# Stage 6 — Evaluation & Operator Inspection

## Status
✅ Stable (v1)

All evaluation and operator-inspection tools are operational and producing
consistent, reproducible artefacts.

---

## Purpose

Stage 6 is the **evaluation and operator-inspection layer** of the SKANN-SSL
pipeline.

This stage:
- evaluates the trained SSL encoder on the full dataset
- exposes systematic class confusions and asymmetries
- enables **interactive, clip-by-clip inspection** using radar plots
- derives and stores **vessel territory / centroid mappings** for downstream inference

🚫 **Stage 6 does not train models.**

---

## Inputs (Consumes)

- **Stage 3 SSL encoder bundle**  
  `stages/stage3_ssl/artifacts/SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib`

- **Dataset tensors and manifest**  
  `data/prototype_dataset/master_dataset_manifest.csv`

---

## Outputs (Produces)

All outputs are written under `stages/stage6_evaluation/artifacts/`.

### Evaluation Artefacts
- `confusion_matrix.png`
- `confusion_report.txt` (UTF-8)
- `misclassified_clips.csv`
- `per_clip_class_results_confidences.csv`
- `per_clip_class_results_confidences.md`

### Interactive Inspection Artefacts
- `final_radar_XXXXXX.png` (one per inspected clip)
- `stage6_per_query_results_log.csv` (append-only operator log)

### Inference Support Artefact
- `vessel_territories_stage6_YYYY-MM-DD.joblib`

---

## Directory Structure

```text
stage6_evaluation/
├── README.md
├── CONFUSION_MATRIX_ANALYSIS.md
├── stage6_confusion_matrix.py
├── stage6_acoustic_sonar_classifier.py
├── stage6_vessel_territory_mapping_and_centroid_extraction.ipynb
└── artifacts/
    ├── confusion_matrix.png
    ├── confusion_report.txt
    ├── misclassified_clips.csv
    ├── per_clip_class_results_confidences.csv
    ├── per_clip_class_results_confidences.md
    ├── stage6_per_query_results_log.csv
    ├── final_radar_XXXXXX.png
    └── vessel_territories_stage6_YYYY-MM-DD.joblib
```
## Tools

### 1) Batch Evaluation — Confusion Matrix

**Script:** `stage6_confusion_matrix.py`

**What it does**
- Runs full-dataset classification over the manifest.
- Computes a row-normalised confusion matrix and overall accuracy.
- Generates a text report with per-class recall/precision and top confusion pairs.
- Exports misclassified clips (with available metadata).
- Exports per-clip class probabilities (CSV + Markdown).

**Run**
```bash
python stages/stage6_evaluation/stage6_confusion_matrix.py
```

**Outputs (written to `artifacts/`)**
- `confusion_matrix.png`
- `confusion_report.txt`
- `misclassified_clips.csv`
- `per_clip_class_results_confidences.csv`
- `per_clip_class_results_confidences.md`

#### Per-clip confidence tables (CSV vs Markdown)

The batch script produces two equivalent per-clip confidence tables:

- `per_clip_class_results_confidences.csv` — machine-readable table for analysis and scripting.
- `per_clip_class_results_confidences.md` — human-readable Markdown rendering of the same table, useful for:
  - quick manual inspection
  - inclusion in reports
  - peer/stakeholder review

Both files contain the same numerical content; only the presentation format differs.

---

### 2) Interactive Operator Inspector — Radar Plot

**Script:** `stage6_acoustic_sonar_classifier.py`

**What it does**
- Evaluates one clip at a time.
- Displays:
  - clip metadata (ID, true label, predicted label)
  - per-class probability radar plot
  - correctness status (correct / incorrect)
- Saves the radar plot and appends a row to an audit log per query.

**Run**
```bash
python stages/stage6_evaluation/stage6_acoustic_sonar_classifier.py
```

**Input**
- numeric `clip_id` (validated against dataset range)

**Outputs (written to `artifacts/`)**
- `final_radar_XXXXXX.png` (one per inspected clip)
- `stage6_per_query_results_log.csv` (append-only)

---

### 3) Vessel Territory & Centroid Extraction

**Notebook:** `stage6_vessel_territory_mapping_and_centroid_extraction.ipynb`

**What it does**
- Analyses SSL embedding space.
- Computes class-wise centroids.
- Maps embedding territories to vessel classes.

**Output (written to `artifacts/`)**
- `vessel_territories_stage6_YYYY-MM-DD.joblib`

This artefact:
- is deterministic and small
- contains no neural-network weights
- is consumed by Stage 7 inference

---

## Evaluation Interpretation

A detailed explanation of:
- the purpose of the confusion matrix
- how to interpret row-normalised results
- what asymmetric confusions imply physically and operationally

is provided in:

➡ `CONFUSION_MATRIX_ANALYSIS.md`

---

## Naming & Artefact Rules
- No dates in script or notebook filenames.
- Dates are allowed in artefact filenames.
- All generated outputs must be written under `stages/stage6_evaluation/artifacts/`.
- README files are the authoritative documentation for each stage.

---

## Stage Boundary (Stage 6 → Stage 7)

**Inputs**
- Stage 3 SSL encoder bundle
- dataset tensors and manifest

**Outputs**
- evaluation artefacts (diagnostic)
- vessel territory / centroid mapping (inference support)

**Consumers**
- Stage 7 (Inference & Deployment)