"""
SKANN-SSL V2.1.0 Training Script
================================
UNDERWATER-APPROPRIATE KERNEL SIZES

Key Changes from V2.0.x:
1. SK kernels: (3,5,7,11,15) → (31, 63, 127, 255, 511, 1023)
   - Now captures shaft_rate, blade_pass, generator, resonance, cavitation
2. Projector: 512→1024→128 → 512→4096→8192→128 (restored V1 size)
3. Maintains: SyncBatchNorm for DDP compatibility

Frequency Coverage @ 16kHz:
  k=31   → 500+ Hz (cavitation)
  k=63   → 250+ Hz (resonance)
  k=127  → 125+ Hz (blade pass)
  k=255  → 62+ Hz  (generator 50Hz)
  k=511  → 31+ Hz  (generator 25Hz)
  k=1023 → 15+ Hz  (shaft rate)
"""

import os
import sys
import time
import datetime
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def log(msg, rank=None):
    ts = time.strftime('%H:%M:%S')
    if rank is not None:
        print(f"[{ts}][R{rank}] {msg}", flush=True)
    else:
        print(f"[{ts}] {msg}", flush=True)


def log_crash(work_dir, exc, rank=0):
    os.makedirs(work_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(work_dir, f"crash_rank{rank}_{ts}.txt")
    msg = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Rank: {rank}\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write("=" * 50 + "\n")
        f.write(msg)
    print(f"\n🚨 CRASH LOG WRITTEN: {path}")
    print(msg)
    return path


def write_heartbeat(work_dir, epoch, step, loss, rank=0):
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, f"heartbeat_rank{rank}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"epoch={epoch}, step={step}, loss={loss:.6f}\n")
        f.write(f"timestamp={datetime.datetime.now().isoformat()}\n")


class RankLogger:
    def __init__(self, work_dir, rank):
        os.makedirs(work_dir, exist_ok=True)
        self.path = os.path.join(work_dir, f"rank{rank}.log")
        self.rank = rank
        with open(self.path, "w") as f:
            f.write(f"=== Rank {rank} Log Started: {datetime.datetime.now().isoformat()} ===\n")
    
    def log(self, msg):
        ts = time.strftime('%H:%M:%S')
        line = f"[{ts}] {msg}\n"
        print(f"[R{self.rank}] {msg}", flush=True)
        with open(self.path, "a") as f:
            f.write(line)


# ============================================================================
# MODEL COMPONENTS - V2.1.0 WITH UNDERWATER-APPROPRIATE KERNELS
# ============================================================================

def _norm_1d(channels, kind='gn', groups=8):
    """1D normalization layer factory."""
    if kind == 'bn':
        return nn.BatchNorm1d(channels)
    if kind == 'ln':
        return nn.GroupNorm(1, channels)
    return nn.GroupNorm(min(groups, channels), channels)


class SKConv1D(nn.Module):
    """
    Selective Kernel 1D Convolution - V2.1.0
    
    UNDERWATER-APPROPRIATE KERNEL SIZES:
    Default kernels (31, 63, 127, 255, 511, 1023) capture:
    - Cavitation: 400-6300 Hz (k=31)
    - Resonance: 50-500 Hz (k=31-127)
    - Blade pass: 3-150 Hz (k=127-1023)
    - Generator: 25, 50 Hz (k=255-1023)
    - Shaft rate: 15-30 Hz (k=1023)
    """
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_sizes: tuple = (31, 63, 127, 255, 511, 1023),  # V2.1.0: Underwater-appropriate!
        stride: int = 1,
        reduction: int = 16,
        norm: str = 'gn',
        act: str = 'gelu',
        residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Multi-branch convolutions with LARGE kernels for low frequencies
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
        
        # Attention MLP
        hidden = max(out_ch // reduction, 8)
        self.fc1 = nn.Linear(out_ch, hidden)
        self.fc2 = nn.Linear(hidden, out_ch * self.n_branches)
        
        # Normalization and activation (GroupNorm for DDP safety)
        self.norm = _norm_1d(out_ch, norm)
        self.act = nn.GELU() if act == 'gelu' else nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Residual connection
        self.residual = residual
        self.match = None
        if residual and (in_ch != out_ch or stride != 1):
            self.match = nn.Conv1d(in_ch, out_ch, kernel_size=1, 
                                   stride=stride, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute all branch outputs (each with different temporal scale)
        feats = [branch(x) for branch in self.branches]
        
        # Sum for global descriptor
        U = torch.stack(feats, dim=1).sum(dim=1)
        
        # Channel attention: global avg pool → FC → softmax
        s = F.adaptive_avg_pool1d(U, 1).squeeze(-1)
        z = self.fc2(F.relu(self.fc1(s), inplace=False))
        a = z.view(z.size(0), self.n_branches, self.out_ch)
        a = F.softmax(a, dim=1).unsqueeze(-1)
        
        # Weighted fusion of multi-scale features
        feats_stacked = torch.stack(feats, dim=1)
        V = (a * feats_stacked).sum(dim=1)
        
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
    Selective Kernel Filterbank - V2.1.0
    
    Converts raw waveform to time-feature map using
    underwater-appropriate multi-scale kernels.
    """
    
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
            residual=False  # No residual for 1->64 expansion
        )
        self.post_norm = _norm_1d(out_ch, norm)
        
        # Log kernel info
        print(f"    SKFilterbank kernels: {kernel_sizes}")
        print(f"    Frequency coverage @ 16kHz: {16000/kernel_sizes[-1]:.0f}Hz - {16000/kernel_sizes[0]:.0f}Hz")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        return self.post_norm(h)


class HybridSKEncoderV2(nn.Module):
    """
    V2.1.0 Encoder - Underwater-Appropriate Architecture
    
    Key changes from V2.0.x:
    1. SK kernels: (31, 63, 127, 255, 511, 1023) for low-freq capture
    2. Large projector: 512→4096→8192→128 (matches V1)
    3. SyncBatchNorm compatible (standard BatchNorm2d)
    """
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        print("  Building HybridSKEncoderV2 (V2.1.0)...")
        
        # SK Frontend with UNDERWATER-APPROPRIATE kernels
        self.sk_frontend = SKFilterbank(
            out_ch=64,
            kernel_sizes=(31, 63, 127, 255, 511, 1023)  # V2.1.0!
        )
        self.downsample = nn.AvgPool1d(kernel_size=8, stride=8)
        
        # Channel bridge (GroupNorm for safety)
        self.channel_bridge = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=False)
        )
        
        # 2D Backbone with standard BatchNorm2d (will become SyncBN)
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
        
        # LARGE PROJECTOR - restored to V1 size!
        # This is critical for Barlow Twins
        self.projector = nn.Sequential(
            nn.Linear(512, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 8192),
            nn.LayerNorm(8192),
            nn.ReLU(inplace=False),
            nn.Linear(8192, latent_dim)
        )
        
        print(f"    Projector: 512 → 4096 → 8192 → {latent_dim}")
        self._count_params()
    
    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"    Total params: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
    
    def forward(self, x):
        # Robust shape handling
        if x.dim() > 3:
            x = x.view(x.size(0), -1).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        
        # SK Frontend (multi-scale temporal features)
        x = self.sk_frontend(x)
        x = self.downsample(x)
        
        # Bridge to 2D
        x = self.channel_bridge(x)
        x = x.unsqueeze(1)
        
        # 2D Backbone
        x = self.backbone2d(x)
        
        # Project to latent
        return self.projector(x)


# ============================================================================
# DATASET
# ============================================================================

class HierarchicalDataset(Dataset):
    def __init__(self, manifest_path, data_dir=None):
        self.df = pd.read_csv(manifest_path)
        self.data_dir = data_dir or '/kaggle/working/SKANN-SSL/data/prototype_dataset/tensors/'
        self.class_to_id = {c: i for i, c in enumerate(sorted(self.df["vessel_class"].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor_id = int(row['anchor_clip_id'])
        p_ids = str(row['partner_clip_ids']).split('|')
        p_ids = [int(x) for x in p_ids if x.strip()]
        partner_id = np.random.choice(p_ids)
        
        y1 = np.load(os.path.join(self.data_dir, f"tensor_{anchor_id:06d}.npy")).flatten()
        y2 = np.load(os.path.join(self.data_dir, f"tensor_{partner_id:06d}.npy")).flatten()
        
        return (
            torch.from_numpy(y1).float(),
            torch.from_numpy(y2).float(),
            self.class_to_id[row["vessel_class"]]
        )


# ============================================================================
# BARLOW TWINS LOSS
# ============================================================================

def barlow_twins_loss(z1, z2, lambd=5e-3):
    """Barlow Twins loss - no inplace operations."""
    batch_size = z1.size(0)
    
    # Normalize
    z1_mean = z1.mean(dim=0)
    z1_std = z1.std(dim=0) + 1e-6
    z1_norm = (z1 - z1_mean) / z1_std
    
    z2_mean = z2.mean(dim=0)
    z2_std = z2.std(dim=0) + 1e-6
    z2_norm = (z2 - z2_mean) / z2_std
    
    # Cross-correlation matrix
    c = torch.mm(z1_norm.T, z2_norm) / batch_size
    
    # Loss computation (no inplace)
    diag = torch.diagonal(c)
    on_diag_loss = torch.pow(1.0 - diag, 2).sum()
    c_squared = torch.pow(c, 2)
    off_diag_loss = c_squared.sum() - torch.pow(diag, 2).sum()
    
    return on_diag_loss + lambd * off_diag_loss


# ============================================================================
# DDP TRAINING WORKER (with SyncBatchNorm)
# ============================================================================

def train_worker(rank, world_size, manifest_path, data_dir, epochs=50, batch_size=4):
    log_dir = "/kaggle/working/run_logs"
    logger = RankLogger(log_dir, rank)
    
    try:
        logger.log("Initializing process group")
        torch.cuda.set_device(rank)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        logger.log("Process group initialized")
        
        logger.log("Loading dataset")
        dataset = HierarchicalDataset(manifest_path, data_dir)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                           num_workers=2, pin_memory=True, drop_last=True)
        logger.log(f"Dataset loaded: {len(dataset)} samples")
        
        # Build model with SyncBatchNorm
        logger.log("Building model (V2.1.0 - Underwater kernels + Large projector)")
        model = HybridSKEncoderV2(latent_dim=128)
        
        # Convert BatchNorm → SyncBatchNorm for DDP
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.log("Converted BatchNorm → SyncBatchNorm")
        
        model = model.cuda(rank)
        model = DDP(model, device_ids=[rank])
        logger.log("Model wrapped in DDP")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        logger.log("Optimizer ready")
        
        if rank == 0:
            with open('/kaggle/working/loss_history.txt', 'w') as f:
                f.write("epoch,loss\n")
        
        logger.log(f"Starting training: {epochs} epochs")
        
        for epoch in range(1, epochs + 1):
            sampler.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            
            logger.log(f"Epoch {epoch}/{epochs} started")
            
            for step, (y1, y2, _) in enumerate(loader):
                y1 = y1.cuda(rank, non_blocking=True)
                y2 = y2.cuda(rank, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                # Dual forward passes
                emb1 = model(y1)
                emb2 = model(y2)
                loss = barlow_twins_loss(emb1, emb2, lambd=5e-3)
                
                loss.backward()
                optimizer.step()
                
                step_loss = loss.item()
                total_loss += step_loss
                
                if step % 10 == 0:
                    write_heartbeat(log_dir, epoch, step, step_loss, rank)
            
            scheduler.step()
            avg_loss = total_loss / len(loader)
            logger.log(f"Epoch {epoch}/{epochs} complete | Loss: {avg_loss:.4f}")
            
            if rank == 0:
                with open('/kaggle/working/loss_history.txt', 'a') as f:
                    f.write(f"{epoch},{avg_loss:.4f}\n")
                    f.flush()
                    os.fsync(f.fileno())
                
                if epoch % 10 == 0 or epoch == epochs:
                    save_path = f"/kaggle/working/BT_ckpt_epoch_{epoch:03d}.pth"
                    torch.save(
                        {"epoch": epoch, "encoder": model.module.state_dict()},
                        save_path
                    )
                    logger.log(f"Checkpoint saved: {save_path}")
        
        if rank == 0:
            torch.save(model.module.state_dict(), "/kaggle/working/SKANN_SSL_V2_Final.pth")
            logger.log("✅ Training complete! Final model saved.")
        
        dist.destroy_process_group()
        logger.log("Process group destroyed")
        
    except Exception as e:
        log_crash(log_dir, e, rank)
        if dist.is_initialized():
            dist.destroy_process_group()
        raise


# ============================================================================
# SINGLE GPU TRAINING (Fallback)
# ============================================================================

def train_single_gpu(manifest_path, data_dir, epochs=50, batch_size=4):
    log_dir = "/kaggle/working/run_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        log("Single GPU Training Mode (V2.1.0)")
        
        log("Loading dataset")
        dataset = HierarchicalDataset(manifest_path, data_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=2, drop_last=True)
        log(f"Dataset loaded: {len(dataset)} samples")
        
        log("Building model")
        model = HybridSKEncoderV2(latent_dim=128).cuda()
        log("Model built")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        log("Optimizer ready")
        
        with open('/kaggle/working/loss_history.txt', 'w') as f:
            f.write("epoch,loss\n")
        
        log(f"Starting training: {epochs} epochs")
        
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            
            log(f"Epoch {epoch}/{epochs} started")
            
            for step, (y1, y2, _) in enumerate(loader):
                y1, y2 = y1.cuda(), y2.cuda()
                optimizer.zero_grad(set_to_none=True)
                
                z1 = model(y1)
                z2 = model(y2)
                loss = barlow_twins_loss(z1, z2, lambd=5e-3)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if step % 10 == 0:
                    write_heartbeat(log_dir, epoch, step, loss.item(), 0)
            
            scheduler.step()
            avg_loss = total_loss / len(loader)
            log(f"Epoch {epoch}/{epochs} complete | Loss: {avg_loss:.4f}")
            
            with open('/kaggle/working/loss_history.txt', 'a') as f:
                f.write(f"{epoch},{avg_loss:.4f}\n")
            
            if epoch % 10 == 0 or epoch == epochs:
                torch.save({"epoch": epoch, "encoder": model.state_dict()},
                          f"/kaggle/working/BT_ckpt_epoch_{epoch:03d}.pth")
                log(f"Checkpoint saved: epoch {epoch}")
        
        torch.save(model.state_dict(), "/kaggle/working/SKANN_SSL_V2_Final.pth")
        log("✅ Training complete!")
        
    except Exception as e:
        log_crash(log_dir, e, 0)
        raise
