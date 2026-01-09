"""
SKANN-SSL Stage 3 — Barlow Twins helpers (NON-AUTHORITATIVE)

IMPORTANT:
- The canonical Barlow Twins loss used for training is implemented inline in:
    stages/stage3_ssl/train_script.py
  and SHOULD NOT be duplicated elsewhere.

- This module exists only to provide helper utilities for diagnostics / analysis
  (e.g., computing the cross-correlation matrix and its on/off-diagonal terms)
  in a way that is numerically consistent with train_script.py.

Rationale:
- Avoids “two sources of truth” for the loss.
- Keeps train_script.py unchanged (proven path).
"""

from __future__ import annotations

import torch


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    Return a flattened view of the off-diagonal elements of a square matrix.
    """
    n, m = x.shape
    if n != m:
        raise ValueError(f"off_diagonal expects a square matrix, got {n}x{m}")
    # Standard trick: reshape then drop diagonal
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def normalize_batch(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalise features across the batch dimension:
        (z - mean) / (std + eps)

    Matches the numerical style used in stages/stage3_ssl/train_script.py.
    """
    return (z - z.mean(dim=0)) / (z.std(dim=0) + eps)


def cross_correlation(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute cross-correlation matrix:
        C = (z1_norm^T @ z2_norm) / B
    where B is batch size.
    """
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(f"Expected 2D tensors (B,D). Got z1:{z1.shape}, z2:{z2.shape}")
    if z1.shape != z2.shape:
        raise ValueError(f"Shape mismatch: z1:{z1.shape} vs z2:{z2.shape}")

    b = z1.size(0)
    z1n = normalize_batch(z1, eps=eps)
    z2n = normalize_batch(z2, eps=eps)
    return (z1n.T @ z2n) / b


def barlow_terms(
    z1: torch.Tensor,
    z2: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return:
      - C: cross-correlation matrix
      - on_diag: Σ_i (1 - C_ii)^2
      - off_diag: Σ_{i≠j} C_ij^2

    This mirrors the decomposition implied by the canonical loss in train_script.py.
    """
    c = cross_correlation(z1, z2, eps=eps)
    diag = torch.diagonal(c)
    on_diag = (1.0 - diag).pow(2).sum()

    # off-diagonal squared sum = total squared sum - diagonal squared sum
    c2 = c.pow(2)
    off_diag = c2.sum() - diag.pow(2).sum()

    return c, on_diag, off_diag


def barlow_loss_from_terms(on_diag: torch.Tensor, off_diag: torch.Tensor, lambda_offdiag: float) -> torch.Tensor:
    """
    Convenience helper to reconstruct the scalar loss from already-computed terms.
    Not used by train_script.py (canonical training path remains inline).
    """
    return on_diag + (lambda_offdiag * off_diag)
