# SKANN-SSL — System Architecture & Self-Supervised Learning Pipeline (Stages 0–7)

**Role:** Canonical system architecture and SSL training pipeline for SKANN.

## Pipeline Overview (Locked)
- Stage 0: Preprocessing and data standardisation.
- Stage 1: Multi-branch SKConv1D front-end.
- Stage 2: 1D SKA backbone (temporal hierarchy).
- Stage 3: SKConv2D hierarchical encoder.
- Stage 4: HybridSKEncoder (1D + 2D fusion).
- Stage 5: SSL wrapper (Barlow Twins).
- Stage 6: Augmentation engine.
- Stage 7: Embedding extraction and clustering.

## Encoder Architecture
- Learned multi-scale temporal filters (SKConv1D).
- Hierarchical temporal compression (1D SKA).
- Lift to 2D time-frequency space (SKConv2D).
- Global pooling and fusion into a fixed embedding.

## Self-Supervised Learning
- Label-free training using correlated views.
- Barlow Twins redundancy-reduction objective.
- Projection head used only during training.

## Augmentation Strategy
- Time shift, gain perturbation, masking.
- Optional band dropout and micro-noise injection.
- Augmentations preserve physical plausibility.

## Embeddings & Clustering
- Encoder outputs fixed-dimensional embeddings.
- Clustering via k-means, HDBSCAN, or hierarchical methods.
- Supports unlabeled pattern discovery and anomaly detection.

## Authority
- Canonical for SKANN architecture and training logic.
- Governs all model design, training, and inference workflows.