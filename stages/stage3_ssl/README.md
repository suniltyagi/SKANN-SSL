# Stage 3: Self-Supervised Learning (Barlow Twins)

## Status: ✅ COMPLETE

## Overview

Stage 3 implements the self-supervised learning pipeline using Barlow Twins. The model learns to produce similar embeddings for different acoustic recordings of the same vessel class, without requiring any labels.

---

## Barlow Twins Algorithm

### Core Idea
Given two augmented views of the same signal, the embeddings should be:
1. **Invariant**: Diagonal of cross-correlation → 1
2. **Decorrelated**: Off-diagonal of cross-correlation → 0

### Loss Function

```python
# Normalize embeddings
z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-7)
z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-7)

# Cross-correlation matrix [D × D]
C = z1_norm.T @ z2_norm / batch_size

# Loss components
on_diag = ((C.diag() - 1) ** 2).sum()      # Push diagonal → 1
off_diag = (C ** 2).sum() - (C.diag() ** 2).sum()  # Push off-diagonal → 0

loss = on_diag + λ * off_diag
```

Where λ = 0.0051 (tuned for 128-dim embeddings).

---

## Hard Positive Mining

### The Challenge
Standard contrastive learning uses random augmentations. But for vessel classification, we need pairs that are:
- Same vessel class (positive label)
- Acoustically different (hard positive)

### Solution: Hierarchical Pairing

The `pairing_manifest.py` script creates anchor-partner pairs:

```python
# For each anchor clip, find K=6 partners that are:
# 1. Same vessel class
# 2. MOST DISTANT in weighted feature space

hierarchy = ["n_blades", "sea_state", "cavitation_peak_freq", ...]
weights = [(12-i)**1.5 for i in range(12)]  # n^1.5 decay

# Select K most distant (not nearest!)
top_k_indices = np.argsort(dist_matrix[i])[::-1][:K]
```

This forces the model to learn: "These sound different, but they're the same vessel class."

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch per GPU | 4 |
| World Size | 2 (Dual T4) |
| Effective Batch | 8 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Lambda (λ) | 0.0051 |
| Mixed Precision | ✅ Enabled |

---

## Files

| File | Description |
|------|-------------|
| `pairing_manifest.py` | Generates hard positive pairs |
| `minimalgput4x2.ipynb` | Kaggle training notebook (Dual T4 DDP) |

---

## Training on Kaggle

### Prerequisites
1. Upload `pairing_manifest.csv` to your dataset
2. Ensure tensors are in `/kaggle/working/SKANN-SSL/data/prototype_dataset/tensors/`

### Launch Training
```python
import torch.multiprocessing as mp
from train_script import train_worker

mp.spawn(
    train_worker,
    args=(world_size, manifest_path, epochs, batch_size),
    nprocs=world_size,
    join=True
)
```

---

## Outputs

| File | Description |
|------|-------------|
| `loss_history.txt` | Per-epoch loss values |
| `BT_ckpt_epoch_*.pth` | Checkpoints every 5 epochs |
| `SKANN_SSL_GPU_Final.pth` | Final model weights |
| `SKANN_SSL_Production_Bundle.joblib` | Portable bundle with metadata |

---

## Results

| Metric | Value |
|--------|-------|
| Final Loss | Converged |
| Silhouette Score | 0.3997 |
| Training Time | ~40 minutes on Dual T4 |

---

## References

- Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. ICML 2021.
