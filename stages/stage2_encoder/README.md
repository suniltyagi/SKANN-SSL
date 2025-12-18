# Stage 2: Hierarchical 2D Acoustic Encoder

## Status: PLANNED

## Objective

Convert learned time-feature map into high-level 2D embeddings using SKConv2D blocks.

---

## Workflow

```
[B, 64, 16000]     # From Stage 1
       │
       ▼ Reshape
[B, 1, 64, 16000]  # Treat as 2D: (freq, time)
       │
       ▼ SKConv2D blocks
[B, C, F', T']     # Hierarchical features
       │
       ▼ Global pooling
[B, C]             # Pooled features
       │
       ▼ Linear projection
[B, D]             # Embedding h ∈ ℝᴰ
```

---

## Files (Planned)

| File | Description |
|------|-------------|
| `skconv2d.py` | 2D Selective Kernel blocks |
| `encoder.py` | Full encoder architecture |
| `README.md` | This file |
