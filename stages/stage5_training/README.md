# Stage 5: Training, Embedding Extraction & Clustering

## Status: PLANNED

## Objective

Full training pipeline, embedding extraction, and unsupervised clustering.

---

## Training Loop

```
for epoch in epochs:
    for x in dataloader:
        x₁, x₂ = augment(x), augment(x)  # Positive pair
        z₁, z₂ = model(x₁), model(x₂)    # Embeddings
        loss = barlow_twins_loss(z₁, z₂) # SSL loss
        loss.backward()
        optimizer.step()
```

---

## Embedding Extraction

After training:
1. Discard projector head
2. Extract embeddings from encoder output
3. Optional: PCA/whitening

---

## Clustering

- **Primary**: HDBSCAN (density-based, no k required)
- **Alternative**: DBSCAN

---

## Visualization

- UMAP projection
- t-SNE projection
- Cluster centroids / averages

---

## Files (Planned)

| File | Description |
|------|-------------|
| `trainer.py` | Training loop with logging |
| `embeddings.py` | Extraction and processing |
| `clustering.py` | HDBSCAN/DBSCAN utilities |
| `visualization.py` | UMAP, t-SNE plots |
| `README.md` | This file |
