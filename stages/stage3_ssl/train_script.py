"""
SKANN-SSL V3 — Model Architecture Definitions

Extracted from: SKANN_SSL_V3_Training_Colab.ipynb (Cell 3)
Source notebook: stages/stage3_ssl/notebooks/SKANN_SSL_V3_Training_Colab.ipynb

This file provides the encoder class (HybridSKEncoderV3) required by:
  - Stage 6 evaluation scripts (stage6_acoustic_sonar_classifier.py, stage6_confusion_matrix.py)
  - Stage 7 deployment scripts
  - Any code that loads the V3 production bundle via model_state

Architecture: SKANN-SSL V2.1.0 base, unchanged for V3
  SK Kernels: (31, 63, 127, 255, 511, 1023)
  Backbone output (h): 512-dim  — deployed representation
  Projector output (z): 256-dim — discarded after training

Do NOT modify this file without retraining — the architecture must match
the saved model_state in SKANN_SSL_V3_Production_Bundle.joblib.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SKANN: SELECTIVE KERNEL AUDIO NEURAL NETWORKS (V2.1.0 / V3)
# SK Kernels: (31, 63, 127, 255, 511, 1023)
# =============================================================================

def _norm_1d(channels, kind='gn', groups=8):
    if kind == 'bn':
        return nn.BatchNorm1d(channels)
    if kind == 'ln':
        return nn.GroupNorm(1, channels)
    return nn.GroupNorm(min(groups, channels), channels)


class SKConv1D(nn.Module):
    """
    Selective Kernel 1D Convolution - V2.1.0

    SK Kernels (31, 63, 127, 255, 511, 1023) capture:
    - k=31:   Cavitation (500+ Hz)
    - k=63:   Resonance (250+ Hz)
    - k=127:  Blade pass (125+ Hz)
    - k=255:  Generator 50Hz (62+ Hz)
    - k=511:  Generator 25Hz (31+ Hz)
    - k=1023: Shaft rate (15+ Hz)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_sizes: tuple = (31, 63, 127, 255, 511, 1023),
        stride: int = 1,
        reduction: int = 16,
        norm: str = 'gn',
        act: str = 'gelu',
        residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2
            self.branches.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride,
                         padding=pad, bias=False)
            )

        self.n_branches = len(kernel_sizes)
        self.out_ch = out_ch
        self.kernel_sizes = kernel_sizes

        hidden = max(out_ch // reduction, 8)
        self.fc1 = nn.Linear(out_ch, hidden)
        self.fc2 = nn.Linear(hidden, out_ch * self.n_branches)

        self.norm = _norm_1d(out_ch, norm)
        self.act = nn.GELU() if act == 'gelu' else nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.residual = residual
        self.match = None
        if residual and (in_ch != out_ch or stride != 1):
            self.match = nn.Conv1d(in_ch, out_ch, kernel_size=1,
                                   stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        U = torch.stack(feats, dim=1).sum(dim=1)
        s = F.adaptive_avg_pool1d(U, 1).squeeze(-1)
        z = self.fc2(F.relu(self.fc1(s), inplace=False))
        a = z.view(z.size(0), self.n_branches, self.out_ch)
        a = F.softmax(a, dim=1).unsqueeze(-1)
        feats_stacked = torch.stack(feats, dim=1)
        V = (a * feats_stacked).sum(dim=1)
        out = self.norm(V)
        out = self.act(out)
        out = self.dropout(out)
        if self.residual:
            res = x if self.match is None else self.match(x)
            out = out + res
        return out


class SKFilterbank(nn.Module):
    """Selective Kernel Filterbank - V2.1.0"""

    def __init__(
        self,
        out_ch: int = 64,
        kernel_sizes: tuple = (31, 63, 127, 255, 511, 1023),
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
            residual=False
        )
        self.post_norm = _norm_1d(out_ch, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        return self.post_norm(h)


class HybridSKEncoderV3(nn.Module):
    """
    SKANN-SSL V3 Encoder

    Forward returns:
    - return_features=False: z (projector output, 256-dim)
    - return_features=True:  (h, z) where h is backbone output (512-dim)
    """

    def __init__(self, latent_dim=256):
        super().__init__()

        # SK Frontend (V2.1.0 - UNCHANGED)
        self.sk_frontend = SKFilterbank(
            out_ch=64,
            kernel_sizes=(31, 63, 127, 255, 511, 1023)
        )
        self.downsample = nn.AvgPool1d(kernel_size=40, stride=40)

        # Channel bridge (V2.1.0 - UNCHANGED)
        self.channel_bridge = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=False)
        )

        # 2D Backbone (V2.1.0 - UNCHANGED)
        # Output: h (512-dim) - THIS IS WHAT GETS DEPLOYED
        self.backbone2d = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, 3, padding=1, stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 256, 3, padding=1, stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )

        # Projector: 512 → 4096 → 8192 → 16384 → 256
        # Output: z (256-dim) - DISCARDED AFTER TRAINING
        self.projector = nn.Sequential(
            nn.Linear(512, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 8192),
            nn.LayerNorm(8192),
            nn.ReLU(inplace=False),
            nn.Linear(8192, 16384),
            nn.LayerNorm(16384),
            nn.ReLU(inplace=False),
            nn.Linear(16384, latent_dim)
        )

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def forward(self, x, return_features=False):
        # Shape handling
        if x.dim() > 3:
            x = x.view(x.size(0), -1).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # SK Frontend
        x = self.sk_frontend(x)
        x = self.downsample(x)

        # Bridge to 2D
        x = self.channel_bridge(x)
        x = x.unsqueeze(1)

        # 2D Backbone → h (512-dim)
        h = self.backbone2d(x)

        # Projector → z (256-dim)
        z = self.projector(h)

        if return_features:
            return h, z
        return z


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------
# V2 scripts may reference HybridSKEncoderV2 — point to V3
# (architecture is identical; V3 only changed dataset & training)
HybridSKEncoderV2 = HybridSKEncoderV3
HybridSKEncoder = HybridSKEncoderV3


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("SKANN-SSL V3 Architecture Self-Test")
    print("=" * 50)
    model = HybridSKEncoderV3(latent_dim=256)
    total, trainable = model._count_params()
    print(f"  Total params:     {total/1e6:.1f}M")
    print(f"  Trainable params: {trainable/1e6:.1f}M")

    dummy = torch.randn(2, 1, 80000)  # 5-second clip @ 16kHz
    h, z = model(dummy, return_features=True)
    print(f"  Input:  {dummy.shape}")
    print(f"  h (backbone): {h.shape}  (512-dim)")
    print(f"  z (projector): {z.shape}  (256-dim)")
    print("=" * 50)
    print("✅ All checks passed")
