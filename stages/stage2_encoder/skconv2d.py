"""
SKANN-SSL Stage 2: Hierarchical 2D Acoustic Encoder
====================================================
Converts 1D filterbank output to fixed-size embeddings via SKConv2D blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Stage 1 filterbank
from stages.stage1_skconv1d.skconv1d import SKFilterbank


def _norm_2d(channels, kind='gn', groups=8):
    """Create 2D normalization layer."""
    if kind == 'bn':
        return nn.BatchNorm2d(channels)
    if kind == 'ln':
        return nn.GroupNorm(1, channels)
    return nn.GroupNorm(min(groups, channels), channels)


class SKConv2D(nn.Module):
    """
    Selective Kernel 2D Convolution Block.
    
    Multiple parallel branches with different 2D kernel sizes,
    fused via learned channel-wise attention.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels  
        kernel_list: List of (H, W) kernel sizes
        stride: (H, W) stride tuple
        reduction: Channel reduction for attention
        norm: Normalization type
        act: Activation type
        residual: Use residual connection
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_list: tuple = ((3, 3), (3, 5), (5, 5), (7, 7)),
        stride: tuple = (1, 2),
        reduction: int = 16,
        norm: str = 'gn',
        act: str = 'gelu',
        residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Multi-branch 2D convolutions
        self.branches = nn.ModuleList()
        for kh, kw in kernel_list:
            ph, pw = kh // 2, kw // 2
            self.branches.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw), 
                         stride=stride, padding=(ph, pw), bias=False)
            )
        
        self.n_branches = len(kernel_list)
        self.out_ch = out_ch
        
        # Attention MLP
        hidden = max(out_ch // reduction, 8)
        self.fc1 = nn.Linear(out_ch, hidden)
        self.fc2 = nn.Linear(hidden, out_ch * self.n_branches)
        
        # Norm, activation, dropout
        self.norm = _norm_2d(out_ch, norm)
        self.act = nn.GELU() if act == 'gelu' else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Residual matching
        self.residual = residual
        self.match = None
        if residual and (in_ch != out_ch or stride != (1, 1)):
            self.match = nn.Conv2d(in_ch, out_ch, kernel_size=1, 
                                   stride=stride, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C_in, H, W]
        Returns:
            Output [B, C_out, H', W']
        """
        # Branch outputs
        feats = [branch(x) for branch in self.branches]
        
        # Global descriptor
        U = torch.stack(feats, dim=1).sum(dim=1)  # [B, C, H, W]
        s = F.adaptive_avg_pool2d(U, 1).flatten(1)  # [B, C]
        
        # Attention
        z = self.fc2(F.relu(self.fc1(s)))
        a = z.view(x.size(0), self.n_branches, self.out_ch)
        a = F.softmax(a, dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, N, C, 1, 1]
        
        # Weighted fusion
        feats_stacked = torch.stack(feats, dim=1)  # [B, N, C, H, W]
        V = (a * feats_stacked).sum(dim=1)  # [B, C, H, W]
        
        # Norm + act + dropout
        out = self.norm(V)
        out = self.act(out)
        out = self.dropout(out)
        
        # Residual
        if self.residual:
            res = x if self.match is None else self.match(x)
            out = out + res
        
        return out


class HybridSKEncoder(nn.Module):
    """
    Complete SKANN-SSL Backbone: Stage 1 + Stage 2.
    
    Pipeline:
        [B, 1, 16000] → SKFilterbank → [B, 64, 16000]
                      → reshape → [B, 1, 64, 16000]
                      → SKConv2D stages → [B, 256, H, W]
                      → global pool → [B, 256]
                      → projection → [B, D]
    
    Args:
        D: Output embedding dimension
        norm: Normalization type for all layers
    """
    
    def __init__(self, D: int = 256, norm: str = 'gn'):
        super().__init__()
        
        # Stage 1: Learned filterbank
        self.filterbank = SKFilterbank(out_ch=64, norm=norm)
        
        # Stage 2: Hierarchical 2D encoder
        # Input: [B, 1, 64, T] after reshape
        self.stage1 = SKConv2D(1, 64, stride=(1, 4), norm=norm)    # [B, 64, 64, T/4]
        self.stage2 = SKConv2D(64, 128, stride=(2, 4), norm=norm)  # [B, 128, 32, T/16]
        self.stage3 = SKConv2D(128, 256, stride=(2, 4), norm=norm) # [B, 256, 16, T/64]
        
        # Global pooling + projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, D),
            nn.LayerNorm(D),
        )
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from raw waveform.
        
        Args:
            x: Raw waveform [B, 1, T]
        Returns:
            Embedding [B, D]
        """
        # Stage 1: 1D filterbank
        h1 = self.filterbank(x)  # [B, 64, T]
        
        # Reshape for 2D processing: [B, 64, T] → [B, 1, 64, T]
        h2 = h1.unsqueeze(1)
        
        # Stage 2: Hierarchical 2D encoding
        h2 = self.stage1(h2)
        h2 = self.stage2(h2)
        h2 = self.stage3(h2)
        
        # Pool + project
        pooled = self.pool(h2)  # [B, 256, 1, 1]
        embedding = self.proj(pooled)  # [B, D]
        
        return embedding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (alias for forward_features)."""
        return self.forward_features(x)
