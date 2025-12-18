# Stage 3: Self-Supervised Representation Learning

## Status: PLANNED

## Objective

Train encoder without labels using Barlow Twins loss.

---

## Barlow Twins

### Loss Function
```
L = λ × Σᵢ(1 - Cᵢᵢ)² + Σᵢ Σⱼ≠ᵢ Cᵢⱼ²
    ├── Invariance term: on-diagonal close to 1
    └── Redundancy reduction: off-diagonal close to 0
```

Where C is the cross-correlation matrix between embeddings of augmented pairs.

### Architecture
```
x₁ ──┐                    ┌── z₁
     ├── Encoder ── Projector ──┤
x₂ ──┘                    └── z₂
                               │
                               ▼
                        Cross-correlation C
                               │
                               ▼
                          Barlow Loss
```

---

## Files (Planned)

| File | Description |
|------|-------------|
| `barlow_twins.py` | Barlow Twins loss and wrapper |
| `projector.py` | MLP projector head |
| `README.md` | This file |

---

## References

- Zbontar & LeCun, "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (ICML 2021)
