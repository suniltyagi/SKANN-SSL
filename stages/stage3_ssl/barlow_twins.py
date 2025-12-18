"""
SKANN-SSL Stage 3: Barlow Twins Self-Supervised Learning
=========================================================
Implementation of Barlow Twins loss for learning invariant representations.

Reference: Zbontar et al., "Barlow Twins: Self-Supervised Learning via 
           Redundancy Reduction" (ICML 2021)
"""

import torch
import torch.nn as nn


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    Extract off-diagonal elements from a square matrix.
    
    Args:
        x: Square matrix [N, N]
    Returns:
        Flattened off-diagonal elements [N*(N-1)]
    """
    n, m = x.shape
    assert n == m, "Matrix must be square"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Projector(nn.Module):
    """
    MLP Projector head for Barlow Twins.
    
    Maps encoder embeddings to a space where the loss is computed.
    Following the original paper: 3 layers with BN on hidden layers.
    
    Args:
        in_dim: Input dimension (encoder output)
        hidden: Hidden layer dimension
        out_dim: Output projection dimension
    """
    
    def __init__(
        self,
        in_dim: int = 256,
        hidden: int = 4096,
        out_dim: int = 256
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Linear(hidden, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            # Layer 3 (output)
            nn.Linear(hidden, out_dim, bias=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder embedding [B, in_dim]
        Returns:
            Projected embedding [B, out_dim]
        """
        return self.net(x)


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins Loss Function.
    
    Encourages:
    1. Invariance: Diagonal of cross-correlation → 1 (same info in both views)
    2. Redundancy reduction: Off-diagonal → 0 (decorrelated features)
    
    Loss = Σ_i (1 - C_ii)² + λ * Σ_i Σ_{j≠i} C_ij²
    
    Args:
        lambd: Weight for off-diagonal (redundancy) term
        eps: Small constant for numerical stability in normalization
    """
    
    def __init__(self, lambd: float = 5e-3, eps: float = 1e-12):
        super().__init__()
        self.lambd = lambd
        self.eps = eps
    
    def forward(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Barlow Twins loss.
        
        Args:
            z1: Projections from view 1 [B, D]
            z2: Projections from view 2 [B, D]
        Returns:
            Scalar loss value
        """
        B, D = z1.shape
        
        # Normalize along batch dimension (zero mean, unit std)
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + self.eps)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + self.eps)
        
        # Cross-correlation matrix [D, D]
        c = (z1_norm.T @ z2_norm) / B
        
        # Invariance loss: diagonal elements should be 1
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        
        # Redundancy reduction: off-diagonal elements should be 0
        off_diag = off_diagonal(c).pow_(2).sum()
        
        # Total loss
        loss = on_diag + self.lambd * off_diag
        
        return loss
