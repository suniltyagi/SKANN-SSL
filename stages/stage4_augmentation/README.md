# Stage 4: Data Pipeline & Augmentation Engine

## Status: PLANNED

## Objective

Generate physics-consistent positive pairs for self-supervised learning.

---

## Augmentations

| Augmentation | Description | Physics Motivation |
|--------------|-------------|-------------------|
| Random crop | Extract sub-segment | Temporal invariance |
| Time shift | Circular shift | Phase invariance |
| Gain jitter | Amplitude scaling | Distance variations |
| Gaussian noise | Additive noise | Ambient noise changes |
| Band-pass filter | Frequency filtering | Propagation effects |
| Time masking | Zero segments | Occlusion robustness |

---

## Positive Pair Generation

```python
x₁ = augment(x)  # First view
x₂ = augment(x)  # Second view (different random params)
```

Both views should represent the same underlying acoustic event.

---

## Files (Planned)

| File | Description |
|------|-------------|
| `augmentations.py` | Full augmentation suite |
| `pair_dataset.py` | Positive pair DataLoader |
| `README.md` | This file |

---

## Notes

- Transforms defined in Stage 0 are imported here
- Augmentation strength tuned for underwater acoustics
- Must preserve discriminative information
