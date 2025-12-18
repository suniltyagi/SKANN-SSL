# Stage 1: Learned Filterbank (SKConv1D)

## Status: PLANNED

## Objective

Replace STFT/mel spectrogram preprocessing with a learned multi-scale time-domain filterbank.

---

## Design

### Multi-Branch SKConv1D
- Kernel sizes: (3, 5, 7, 11, 15)
- Each branch learns different temporal scales
- Attention-weighted fusion

### Output
- Input: `[B, 1, 16000]` from Stage 0
- Output: `[B, 64, 16000]` learned time-feature map

---

## Temporal Scales

| Kernel | Samples | Duration | Captures |
|--------|---------|----------|----------|
| 3 | 3 | 0.19 ms | Transients, clicks |
| 5 | 5 | 0.31 ms | Sharp features |
| 7 | 7 | 0.44 ms | Tonals |
| 11 | 11 | 0.69 ms | Modulation |
| 15 | 15 | 0.94 ms | Broadband patterns |

---

## Files (Planned)

| File | Description |
|------|-------------|
| `skconv1d.py` | Multi-branch SKConv1D module |
| `attention.py` | Channel attention mechanism |
| `README.md` | This file |

---

## References

- Li et al., "Selective Kernel Networks" (CVPR 2019)
