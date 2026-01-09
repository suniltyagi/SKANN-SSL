"""
train_script_repo.py (Stage 3)

Repo-path (local) variant of the Kaggle training script generated in `minimalgput4x2.ipynb`.

Key differences vs Kaggle version:
- Uses repo-relative paths for tensors and outputs by default (works when run from repo root).
- Keeps the model, loss, and training logic unchanged as much as possible.
- Allows overriding paths via environment variables or function arguments.

This file is intended to be committed under:
  stages/stage3_ssl/train_script_repo.py

Keep the original Kaggle script separately (e.g., train_script_kaggle.py) for provenance.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler


# ---------------------------------------------------------
# Paths (repo-relative defaults)
# ---------------------------------------------------------
def _repo_root() -> Path:
    # .../SKANN-SSL/stages/stage3_ssl/train_script_repo.py -> parents[2] = .../SKANN-SSL
    return Path(__file__).resolve().parents[2]


DEFAULT_TENSORS_DIR = _repo_root() / "data" / "prototype_dataset" / "tensors"
DEFAULT_WORK_DIR = _repo_root() / "data" / "prototype_dataset"  # for logs/checkpoints by default


# ---------------------------------------------------------
# HybridSKEncoder: 34.4M Parameters | Stability-First
# ---------------------------------------------------------
class HybridSKEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.backbone1d = nn.Sequential(
            nn.Conv1d(1, 128, 31, stride=4, padding=15),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 15, stride=2, padding=7),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.backbone2d = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=(2, 2)), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=(2, 1)), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1)
        )
        # FIX: Switched to LayerNorm in the projector to eliminate "inplace" version errors
        # caused by BatchNorm running buffers in Distributed mode.
        self.projector = nn.Sequential(
            nn.Linear(512, 4096), nn.LayerNorm(4096), nn.ReLU(),
            nn.Linear(4096, 8192), nn.LayerNorm(8192), nn.ReLU(),
            nn.Linear(8192, latent_dim)
        )

    def forward(self, x):
        if x.dim() > 3:
            x = x.view(x.size(0), -1).unsqueeze(1)
        x = self.backbone1d(x).unsqueeze(1)
        x = self.backbone2d(x)
        return self.projector(x)


# ---------------------------------------------------------
# Hierarchical Dataset
# ---------------------------------------------------------
class HierarchicalDataset(Dataset):
    def __init__(self, manifest_path: str | os.PathLike, tensors_dir: str | os.PathLike | None = None):
        self.df = pd.read_csv(manifest_path)

        # Default: repo-relative tensors directory
        # Override options:
        # - pass tensors_dir explicitly
        # - set env var SKANN_TENSORS_DIR
        env_dir = os.environ.get("SKANN_TENSORS_DIR")
        self.data_dir = Path(tensors_dir or env_dir or DEFAULT_TENSORS_DIR)

        self.class_to_id = {c: i for i, c in enumerate(sorted(self.df["vessel_class"].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p_ids = str(row["partner_clip_ids"]).split("|")

        anchor_id = int(row["anchor_clip_id"])
        partner_id = int(np.random.choice(p_ids))

        y1 = np.load(self.data_dir / f"tensor_{anchor_id:06d}.npy")
        y2 = np.load(self.data_dir / f"tensor_{partner_id:06d}.npy")

        return (
            torch.from_numpy(y1).float().view(1, -1),
            torch.from_numpy(y2).float().view(1, -1),
            self.class_to_id[row["vessel_class"]],
        )


def _choose_backend() -> str:
    # NCCL is standard on Linux + CUDA. On Windows, NCCL typically isn't available.
    if os.name == "nt":
        return "gloo"
    return "nccl" if torch.cuda.is_available() else "gloo"


def train_worker(
    rank: int,
    world_size: int,
    manifest_path: str,
    epochs: int = 50,
    batch_size: int = 4,
    tensors_dir: str | None = None,
    work_dir: str | None = None,
    backend: str | None = None,
):
    """
    DDP worker. Use with torch.multiprocessing.spawn or torchrun.
    """
    backend = backend or _choose_backend()
    work_dir = Path(work_dir) if work_dir else DEFAULT_WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        dist.barrier(device_ids=[rank])
    else:
        dist.barrier()

    try:
        device = rank if torch.cuda.is_available() else "cpu"
        model = DDP(HybridSKEncoder().to(device), device_ids=[rank] if torch.cuda.is_available() else None, find_unused_parameters=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler("cuda") if torch.cuda.is_available() else None
        barlow_lambda = 0.0051

        dataset = HierarchicalDataset(manifest_path, tensors_dir=tensors_dir)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=0,
            pin_memory=False,
        )

        for epoch in range(1, epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)

            total_epoch_loss = 0.0

            for (y1, y2, _) in loader:
                optimizer.zero_grad(set_to_none=True)

                if torch.cuda.is_available():
                    y_combined = torch.cat([y1, y2], dim=0).to(device, non_blocking=True)
                    with autocast("cuda"):
                        z_combined = model(y_combined)
                        z1, z2 = z_combined.chunk(2, dim=0)

                        z1_n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-7)
                        z2_n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-7)

                        c = torch.mm(z1_n.T, z2_n) / z1.shape[0]
                        diag = torch.diagonal(c)
                        on_diag = ((diag - 1) ** 2).sum()
                        off_diag = (c.pow(2).sum() - diag.pow(2).sum())
                        loss = on_diag + barlow_lambda * off_diag

                    assert scaler is not None
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # CPU fallback (slow; intended mainly for smoke testing)
                    y_combined = torch.cat([y1, y2], dim=0).to(device)
                    z_combined = model(y_combined)
                    z1, z2 = z_combined.chunk(2, dim=0)

                    z1_n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-7)
                    z2_n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-7)

                    c = torch.mm(z1_n.T, z2_n) / z1.shape[0]
                    diag = torch.diagonal(c)
                    on_diag = ((diag - 1) ** 2).sum()
                    off_diag = (c.pow(2).sum() - diag.pow(2).sum())
                    loss = on_diag + barlow_lambda * off_diag

                    loss.backward()
                    optimizer.step()

                total_epoch_loss += float(loss.item())

            if rank == 0:
                avg_loss = total_epoch_loss / max(1, len(loader))
                loss_path = work_dir / "loss_history.txt"
                with open(loss_path, "a", encoding="utf-8") as f:
                    f.write(f"{epoch},{avg_loss:.4f}\n")
                    f.flush()
                    os.fsync(f.fileno())

                if epoch % 5 == 0 or epoch == epochs:
                    ckpt_path = work_dir / f"BT_ckpt_epoch_{epoch:03d}.pth"
                    torch.save({"epoch": epoch, "encoder": model.module.state_dict()}, ckpt_path)

                print(f"| Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f} | Process Healthy")

        if rank == 0:
            final_path = work_dir / "SKANN_SSL_GPU_Final.pth"
            torch.save(model.module.state_dict(), final_path)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Optional smoke-test entrypoint (single-process)
    # Example:
    #   python stages/stage3_ssl/train_script_repo.py --manifest data/prototype_dataset/pairing_manifest.csv
    import argparse
    import torch.multiprocessing as mp

    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Path to pairing_manifest.csv (repo-relative or absolute)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--tensors_dir", default=None, help="Override tensors dir (default: repo data/prototype_dataset/tensors)")
    p.add_argument("--work_dir", default=None, help="Where to write loss_history/checkpoints (default: data/prototype_dataset)")
    p.add_argument("--backend", default=None, help="Override distributed backend (default: auto)")
    args = p.parse_args()

    if args.world_size == 1:
        train_worker(
            rank=0,
            world_size=1,
            manifest_path=args.manifest,
            epochs=args.epochs,
            batch_size=args.batch_size,
            tensors_dir=args.tensors_dir,
            work_dir=args.work_dir,
            backend=args.backend,
        )
    else:
        mp.spawn(
            train_worker,
            args=(args.world_size, args.manifest, args.epochs, args.batch_size, args.tensors_dir, args.work_dir, args.backend),
            nprocs=args.world_size,
            join=True,
        )
