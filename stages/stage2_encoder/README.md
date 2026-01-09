# Stage 2 — HybridSKEncoder (V2.1.0)

## Status: ✅ COMPLETE (Production)

## Role in Pipeline

Stage 2 defines the **HybridSKEncoder**, which transforms **Stage‑1 physics‑aware features**
into compact **128‑dimensional acoustic embeddings**.

**Stage position:**
```
Stage −1 → 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
```

- **Stage 1**: SKConv1D / SKFilterbank (physics‑aware, multi‑scale)
- **Stage 2**: Encoder backbone + projector (this stage)
- **Stage 3**: SSL training (Barlow Twins)

---

## Overview

The HybridSKEncoder is a **1D–2D hybrid convolutional architecture** with a deep projection head,
designed for self‑supervised representation learning on underwater acoustic waveforms.

- **Input**: raw waveform `[B, 1, 16000]`
- **Output**: 128‑D embedding (“acoustic fingerprint”)
- **Total parameters (training graph)**: ~34.4M  
- **Encoder‑only parameters (inference)**: ~1.8M

> **Important note**  
> Although the *total* parameter count is similar to V1, V2.1.0 **reallocates representational capacity**
> using a physics‑aware selective‑kernel frontend, leading to a large improvement in embedding quality
> without increasing model size.

---

## Architecture (V2.1.0)

```
Raw Waveform [B, 1, 16000]
        │
        ▼
┌───────────────────────────────────────┐
│  Stage‑1 SKFilterbank (V2.1.0)        │
│  Physics‑aware, multi‑scale kernels   │
│  Attention‑weighted fusion            │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  2D Convolutional Backbone            │
│  Conv2d stack → SyncBatchNorm → ReLU  │
│  AdaptiveAvgPool → Flatten            │
│  Output: [B, 512]                     │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Projector (Deep MLP)                 │
│  512 → 4096 → 8192 → 128              │
│  LayerNorm + ReLU                     │
└───────────────────────────────────────┘
        │
        ▼
   128‑dim Acoustic Fingerprint
```

---

## Parameter Distribution (V2.1.0)

```
Stage‑1 SKFilterbank (shared):   ~0.35M
Stage‑2 2D Backbone:            ~1.50M
Stage‑2 Projector (SSL only):   ~32.6M
────────────────────────────────────────
Total (training graph):         ~34.4M
Encoder‑only (inference):       ~1.8M
```

> The projector is **used only during SSL training (Stage 3)** and is removed for
> downstream evaluation and deployment.

---

## Parameter Distribution (V1 Baseline – for comparison)

```
Fixed Conv1D backbone:           ~0.30M
2D Backbone:                    ~1.50M
Projector (SSL):                ~32.6M
────────────────────────────────────────
Total (training graph):         ~34.4M
Encoder‑only (inference):       ~1.8M
```

### Key Difference vs V2.1.0
- **V1**: Fixed‑scale Conv1D kernels must explain all temporal phenomena  
- **V2.1.0**: Parameters are *functionally specialised* via physics‑aware,
  attention‑weighted kernel branches in Stage‑1

This change improves **information efficiency**, not raw capacity.

---

## Key Design Decisions

| Decision | Rationale |
|--------|-----------|
| Physics‑aware SK frontend | Align kernel scales with vessel dynamics |
| SK moved to Stage‑1 | Clean separation of feature extraction vs representation |
| Large projector | Improves Barlow Twins decorrelation |
| LayerNorm in projector | Stable DDP training |
| 128‑D embedding | Compact yet discriminative |

---

## Files

| File | Purpose |
|----|--------|
| `train_script.py` | HybridSKEncoder definition + training entry point |
| `__init__.py` | Module export |

---

## Usage

```python
from stages.stage2_encoder.train_script import HybridSKEncoder

model = HybridSKEncoder(latent_dim=128)
x = torch.randn(32, 1, 16000)
z = model(x)  # [32, 128]
```

---

## V2.1.0 Notes

- Fixed‑kernel Conv1D used in V1 is **deprecated**
- All kernel‑scale logic now lives in **Stage‑1**
- Stage‑2 focuses purely on **representation capacity**
- Trained exclusively via **Stage‑3 SSL**

See also:
- `stages/stage1_skconv1d/README.md`
- `stages/stage3_ssl/README.md`
