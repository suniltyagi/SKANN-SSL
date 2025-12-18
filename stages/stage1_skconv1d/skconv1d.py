"""
SKANN-SSL Stage 1: Selective Kernel 1D Filterbank
==================================================
Multi-scale learned filterbank replacing fixed STFT/mel frontends.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_1d(channels, kind='gn', groups=8):
    """Create 1D normalization layer."""
    if kind == 'bn':
        return nn.BatchNorm1d(channels)
    if kind == 'ln':
        return nn.GroupNorm(1, channels)  # LayerNorm equivalent
    return nn.GroupNorm(min(groups, channels), channels)


class SKConv1D(nn.Module):
    """
    Selective Kernel 1D Convolution Block.
    
    Multiple parallel branches with different kernel sizes,
    fused via learned attention weights.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        kernel_sizes: Tuple of kernel sizes for each branch
        stride: Convolution stride
        reduction: Channel reduction ratio for attention
        norm: Normalization type ('gn', 'bn', 'ln')
        act: Activation type ('gelu', 'relu')
        residual: Whether to use residual connection
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_sizes: tuple = (3, 5, 7, 11, 15),
        stride: int = 1,
        reduction: int = 16,
        norm: str = 'gn',
        act: str = 'gelu',
        residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Multi-branch convolutions
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2  # 'same' padding
            self.branches.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, 
                         padding=pad, bias=False)
            )
        
        self.n_branches = len(kernel_sizes)
        self.out_ch = out_ch
        
        # Attention MLP (squeeze-excitation style)
        hidden = max(out_ch // reduction, 8)
        self.fc1 = nn.Linear(out_ch, hidden)
        self.fc2 = nn.Linear(hidden, out_ch * self.n_branches)
        
        # Normalization and activation
        self.norm = _norm_1d(out_ch, norm)
        self.act = nn.GELU() if act == 'gelu' else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Residual connection
        self.residual = residual
        self.match = None
        if residual and (in_ch != out_ch or stride != 1):
            self.match = nn.Conv1d(in_ch, out_ch, kernel_size=1, 
                                   stride=stride, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C_in, T]
        Returns:
            Output tensor [B, C_out, T'] where T' depends on stride
        """
        # Compute all branch outputs
        feats = [branch(x) for branch in self.branches]  # list of [B, C, T]
        
        # Sum for global descriptor
        U = torch.stack(feats, dim=1).sum(dim=1)  # [B, C, T]
        
        # Global average pooling
        s = F.adaptive_avg_pool1d(U, 1).squeeze(-1)  # [B, C]
        
        # Attention weights via MLP
        z = self.fc2(F.relu(self.fc1(s)))  # [B, C * n_branches]
        a = z.view(z.size(0), self.n_branches, self.out_ch)  # [B, N, C]
        a = F.softmax(a, dim=1)  # softmax across branches
        
        # Weighted fusion
        a = a.unsqueeze(-1)  # [B, N, C, 1]
        feats_stacked = torch.stack(feats, dim=1)  # [B, N, C, T]
        V = (a * feats_stacked).sum(dim=1)  # [B, C, T]
        
        # Norm + activation + dropout
        out = self.norm(V)
        out = self.act(out)
        out = self.dropout(out)
        
        # Residual
        if self.residual:
            res = x if self.match is None else self.match(x)
            out = out + res
        
        return out


class SKFilterbank(nn.Module):
    """
    Stage 1: Learned 1D Filterbank.
    
    Converts raw waveform [B, 1, T] to time-feature map [B, 64, T].
    Replaces STFT/mel spectrogram with learnable multi-scale filters.
    
    Args:
        out_ch: Number of output channels (filter banks)
        kernel_sizes: Kernel sizes for multi-scale analysis
        norm: Normalization type
    """
    
    def __init__(
        self,
        out_ch: int = 64,
        kernel_sizes: tuple = (3, 5, 7, 11, 15),
        norm: str = 'gn'
    ):
        super().__init__()
        
        self.stem = SKConv1D(
            in_ch=1,
            out_ch=out_ch,
            kernel_sizes=kernel_sizes,
            stride=1,
            reduction=16,
            norm=norm,
            act='gelu',
            residual=False  # No residual for 1->64 channel expansion
        )
        self.post_norm = _norm_1d(out_ch, norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw waveform [B, 1, T] (T=16000 for 1s @ 16kHz)
        Returns:
            Time-feature map [B, 64, T]
        """
        h = self.stem(x)
        return self.post_norm(h)
