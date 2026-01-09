# Stage 2: HybridSKEncoder Architecture

## Status: ✅ COMPLETE

## Overview

Stage 2 implements the core encoder architecture — a 34.4 million parameter hybrid 1D-2D convolutional neural network that transforms raw audio waveforms into 128-dimensional acoustic fingerprints.

---

## Architecture

```
Input: [B, 1, 16000] raw waveform
           │
           ▼
┌──────────────────────────────────────┐
│  BACKBONE 1D (Temporal Processing)   │
│  Conv1d(1→128, k=31, s=4) → BN → ReLU│
│  Conv1d(128→128, k=15, s=2) → BN → ReLU│
│  Output: [B, 128, T']                │
└──────────────────────────────────────┘
           │ .unsqueeze(1)
           ▼
┌──────────────────────────────────────┐
│  BACKBONE 2D (Spectral Processing)   │
│  Conv2d(1→64, 3×3) → BN → ReLU       │
│  Conv2d(64→128, s=(2,2)) → BN → ReLU │
│  Conv2d(128→256, s=(2,1)) → BN → ReLU│
│  Conv2d(256→512, 3×3) → BN → ReLU    │
│  AdaptiveAvgPool2d(1) → Flatten      │
│  Output: [B, 512]                    │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  PROJECTOR (Deep MLP)                │
│  Linear(512→4096) → LayerNorm → ReLU │
│  Linear(4096→8192) → LayerNorm → ReLU│
│  Linear(8192→128)                    │
│  Output: [B, 128]                    │
└──────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LayerNorm in projector | Avoids DDP inplace errors from BatchNorm running buffers |
| Large projector (4096→8192) | Barlow Twins benefits from high-dimensional projection space |
| 128-dim output | Compact fingerprint balancing expressiveness and efficiency |

---

## Files

| File | Description |
|------|-------------|
| `train_script.py` | Complete HybridSKEncoder class + training worker |

---

## Usage

```python
from train_script import HybridSKEncoder

model = HybridSKEncoder(latent_dim=128)
x = torch.randn(32, 1, 16000)  # [B, C, T]
z = model(x)                    # [B, 128]
```

---

## Parameter Count

```
Backbone 1D:    ~270K
Backbone 2D:    ~1.5M
Projector:      ~32.6M
─────────────────────
Total:          ~34.4M
```

---

## Notes

- The 1D backbone uses fixed kernel sizes (31, 15)
- Stage 1 will add multi-branch SKConv1D with attention-weighted fusion
- This should improve multi-scale feature extraction
