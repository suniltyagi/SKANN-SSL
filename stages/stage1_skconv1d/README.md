# Stage 1 — Physics-Aware SKConv1D Filterbank (V2.1.0)

## Status: ✅ IMPLEMENTED (Production)

## Objective

Stage 1 provides a **learned, multi-scale, time-domain filterbank** for underwater acoustics.
It replaces fixed STFT/mel spectrogram frontends with **physics-aware temporal kernels** and
**attention-weighted fusion**.

This stage is a **shared dependency** for the V2.1.0 encoder:

- **Stage 1** (this stage): SKFilterbank / SKConv1D  
- **Stage 2**: HybridSKEncoder backbone + projector  
- **Stage 3**: SSL training (Barlow Twins)

---

## Why “Physics-Aware”

Underwater vessel signatures contain strong low-frequency structure (shaft/blade fundamentals and
harmonics) that require **long temporal receptive fields**. V2.1.0 therefore uses longer, odd
kernel sizes to cover these dynamics while still supporting multi-scale analysis.

### Default V2.1.0 kernel set (samples)

```
(31, 63, 127, 255, 511, 1023)
```

At 16 kHz, these correspond to approximately:

| Kernel (samples) | Duration (ms) |
|---:|---:|
| 31 | ~1.94 |
| 63 | ~3.94 |
| 127 | ~7.94 |
| 255 | ~15.94 |
| 511 | ~31.94 |
| 1023 | ~63.94 |

---

## Design

### SKConv1D (Selective Kernel 1D)
- Multiple parallel Conv1D branches with different kernel sizes
- Global descriptor from summed branch features
- Attention MLP produces **softmax weights across branches**
- Weighted fusion yields the output feature map

### SKFilterbank (Stage-1 wrapper)
- Input: raw waveform `[B, 1, T]`
- Output: learned feature map `[B, 64, T]` (default)
- Intended to be consumed by Stage 2 encoder

---

## Files

| File | Description |
|------|-------------|
| `skconv1d.py` | `SKConv1D` + `SKFilterbank` (V2.1.0 physics-aware kernels) |
| `README.md` | This file |

---

## Usage

```python
import torch
from stages.stage1_skconv1d.skconv1d import SKFilterbank

fb = SKFilterbank(out_ch=64, kernel_sizes=(31, 63, 127, 255, 511, 1023))
x = torch.randn(8, 1, 16000)  # 1s @ 16kHz
h = fb(x)                     # [8, 64, 16000]
```

---

## Notes

- Stage 1 was **not used** in the archived V1 baseline training script (fixed Conv1D was embedded
  in the encoder).  
- V2.1.0 training notebooks/scripts originally defined SK modules inline; this stage exists to make
  Stage 1 the **single source of truth** for SKFilterbank going forward.
- See `stages/stage3_ssl/runs/2026-01-07_sk_integrated/run_metadata.yaml` for the authoritative V2.1.0 run record.

---

## Reference

- Li et al., *Selective Kernel Networks* (CVPR 2019)
