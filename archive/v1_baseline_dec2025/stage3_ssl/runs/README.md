# Stage 3 SSL Training Runs

This directory contains metadata for SSL training runs.

## Directory Structure

```
runs/
└── YYYY-MM-DD_<run_name>/
    └── run_metadata.yaml    # Training config, hyperparameters, metrics
```

Artifacts (checkpoints, bundles, plots) are stored in `../artifacts/` and `../artifacts/diagnostics/`.

## Baseline Run

- **2025-12-29_kaggle_baseline**: First successful Barlow Twins training on Dual T4 GPUs
  - Silhouette score: 0.3997
  - Stage 6 accuracy: 78.2%
  - UMAP plot: `../artifacts/diagnostics/vessel_signature_umap_2d.png`

## Notes

- Large artifacts (>50 MB) may be stored off-repo (Google Drive, Releases)
- `run_metadata.yaml` captures all hyperparameters for reproducibility
