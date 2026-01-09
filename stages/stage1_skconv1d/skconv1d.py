"""
SKANN-SSL Stage 1 — Physics-Aware Selective Kernel 1D Filterbank (V2.1.0)
========================================================================

This module provides a learned, multi-scale, time-domain filterbank designed for
underwater acoustics. It replaces fixed STFT/mel frontends with **physics-aware**
temporal kernels and attention-weighted fusion.

Primary components
- SKConv1D: multi-branch Conv1D block with branch attention (Selective Kernel)
- SKFilterbank: Stage-1 wrapper: raw waveform [B, 1, T] -> features [B, C, T]

Default V2.1.0 kernel set (samples):
    (31, 63, 127, 255, 511, 1023)

At 16 kHz, these correspond to ~1.9 ms to ~63.9 ms receptive fields, enabling
capture of low-frequency vessel dynamics while retaining multi-scale coverage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_1d(channels: int, kind: str = "gn", groups: int = 8) -> nn.Module:
    """
    Create a 1D normalisation layer.

    Args:
        channels: number of channels
        kind: 'gn' (GroupNorm), 'bn' (BatchNorm1d), 'ln' (LayerNorm-like via GroupNorm(1))
        groups: group count for GroupNorm
    """
    kind = (kind or "gn").lower()
    if kind == "bn":
        return nn.BatchNorm1d(channels)
    if kind == "ln":
        return nn.GroupNorm(1, channels)  # LayerNorm equivalent over channels
    # default: GroupNorm
    return nn.GroupNorm(min(groups, channels), channels)


def _act(act: str = "gelu") -> nn.Module:
    act = (act or "gelu").lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    return nn.GELU()


class SKConv1D(nn.Module):
    """
    Selective Kernel 1D Convolution Block (multi-branch Conv1D + attention fusion).

    Multiple parallel Conv1D branches (different kernel sizes) produce feature maps
    which are fused using attention weights learned from a global descriptor.

    Args:
        in_ch: input channels
        out_ch: output channels
        kernel_sizes: tuple of kernel sizes for branches
        stride: convolution stride for each branch
        reduction: reduction ratio for attention MLP (squeeze)
        norm: normalisation type ('gn', 'bn', 'ln')
        act: activation ('gelu' or 'relu')
        residual: add residual connection if shapes allow
        dropout: dropout probability after activation
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_sizes: tuple[int, ...] = (31, 63, 127, 255, 511, 1023),
        stride: int = 1,
        reduction: int = 16,
        norm: str = "gn",
        act: str = "gelu",
        residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(kernel_sizes, (tuple, list)) or len(kernel_sizes) < 2:
            raise ValueError("kernel_sizes must be a tuple/list with >= 2 elements")

        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.n_branches = len(self.kernel_sizes)
        self.out_ch = int(out_ch)

        # Multi-branch convolutions (same padding)
        branches = []
        for k in self.kernel_sizes:
            if k < 3 or k % 2 == 0:
                raise ValueError(f"Kernel size must be odd and >=3, got {k}")
            pad = k // 2
            branches.append(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=k,
                    stride=stride,
                    padding=pad,
                    bias=False,
                )
            )
        self.branches = nn.ModuleList(branches)

        # Attention MLP (squeeze-excitation style)
        hidden = max(out_ch // reduction, 8)
        self.fc1 = nn.Linear(out_ch, hidden)
        self.fc2 = nn.Linear(hidden, out_ch * self.n_branches)

        self.norm = _norm_1d(out_ch, norm)
        self.act = _act(act)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Residual connection (optional)
        self.residual = bool(residual)
        self.match: nn.Module | None = None
        if self.residual and (in_ch != out_ch or stride != 1):
            self.match = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T]
        Returns:
            out: [B, C_out, T'] where T' depends on stride
        """
        # Branch outputs: list of [B, C, T']
        feats = [branch(x) for branch in self.branches]

        # Global descriptor from summed branches
        U = torch.stack(feats, dim=1).sum(dim=1)  # [B, C, T']
        s = F.adaptive_avg_pool1d(U, 1).squeeze(-1)  # [B, C]

        # Attention over branches
        z = self.fc2(F.relu(self.fc1(s)))  # [B, C * n_branches]
        a = z.view(z.size(0), self.n_branches, self.out_ch)  # [B, N, C]
        a = F.softmax(a, dim=1)  # across branches

        # Weighted fusion
        a = a.unsqueeze(-1)  # [B, N, C, 1]
        feats_stacked = torch.stack(feats, dim=1)  # [B, N, C, T']
        V = (a * feats_stacked).sum(dim=1)  # [B, C, T']

        out = self.norm(V)
        out = self.act(out)
        out = self.dropout(out)

        if self.residual:
            res = x if self.match is None else self.match(x)
            out = out + res

        return out


class SKFilterbank(nn.Module):
    """
    Stage 1: Learned 1D Filterbank (Physics-Aware, V2.1.0).

    Converts raw waveform [B, 1, T] to a time-feature map [B, out_ch, T].

    Args:
        out_ch: number of output channels (filterbank channels)
        kernel_sizes: kernel sizes for multi-scale analysis (odd integers)
        norm: normalisation type ('gn', 'bn', 'ln')
        groups: GroupNorm groups if norm='gn'
    """

    def __init__(
        self,
        out_ch: int = 64,
        kernel_sizes: tuple[int, ...] = (31, 63, 127, 255, 511, 1023),
        norm: str = "gn",
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.out_ch = int(out_ch)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)

        # 1 -> out_ch expansion (no residual)
        self.stem = SKConv1D(
            in_ch=1,
            out_ch=self.out_ch,
            kernel_sizes=self.kernel_sizes,
            stride=1,
            reduction=16,
            norm=norm,
            act="gelu",
            residual=False,
            dropout=0.0,
        )
        self.post_norm = _norm_1d(self.out_ch, norm, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T] raw waveform (T=16000 for 1s @ 16kHz)
        Returns:
            h: [B, out_ch, T]
        """
        h = self.stem(x)
        return self.post_norm(h)


__all__ = ["SKConv1D", "SKFilterbank"]
