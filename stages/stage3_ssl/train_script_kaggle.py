import os, torch, torch.nn as nn, torch.distributed as dist
import pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

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
            nn.Conv2d(64, 128, 3, padding=1, stride=(2,2)), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=(2,1)), nn.BatchNorm2d(256), nn.ReLU(),
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
    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path)
        self.data_dir = '/kaggle/working/SKANN-SSL/data/prototype_dataset/tensors/'
        self.class_to_id = {c: i for i, c in enumerate(sorted(self.df["vessel_class"].unique()))}
            
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p_ids = str(row['partner_clip_ids']).split('|')
        y1 = np.load(os.path.join(self.data_dir, f"tensor_{str(int(row['anchor_clip_id'])).zfill(6)}.npy"))
        y2 = np.load(os.path.join(self.data_dir, f"tensor_{str(np.random.choice(p_ids)).zfill(6)}.npy"))
        return torch.from_numpy(y1).float().view(1, -1), torch.from_numpy(y2).float().view(1, -1), self.class_to_id[row["vessel_class"]]

def train_worker(rank, world_size, manifest_path, epochs=50, batch_size=4):
    # PRE-INIT: Set device and initialize group with explicit IDs
    torch.cuda.set_device(rank)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier(device_ids=[rank]) 
    
    try:
        model = DDP(HybridSKEncoder().to(rank), device_ids=[rank], find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler('cuda')
        barlow_lambda = 0.0051
        
        dataset = HierarchicalDataset(manifest_path)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=DistributedSampler(dataset, world_size, rank),
                            num_workers=0, pin_memory=False)

        for epoch in range(1, epochs + 1):
            loader.sampler.set_epoch(epoch)
            total_epoch_loss = 0
            
            for i, (y1, y2, _) in enumerate(loader):
                # FIX: Set grads to None for better performance and safety
                optimizer.zero_grad(set_to_none=True)
                
                with autocast('cuda'):
                    # FIX: Single forward pass by concatenating views
                    y_combined = torch.cat([y1, y2], dim=0).to(rank)
                    z_combined = model(y_combined)
                    z1, z2 = z_combined.chunk(2, dim=0)
                    
                    # Normalization
                    z1_n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-7)
                    z2_n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-7)
                    
                    # Cross-correlation matrix
                    c = torch.mm(z1_n.T, z2_n) / z1.shape[0]
                    
                    # Out-of-place Loss Math
                    diag = torch.diagonal(c)
                    on_diag = ((diag - 1) ** 2).sum()
                    off_diag = (c.pow(2).sum() - diag.pow(2).sum())
                    loss = on_diag + barlow_lambda * off_diag
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_epoch_loss += loss.item()

            if rank == 0:
                avg_loss = total_epoch_loss / len(loader)
                with open('/kaggle/working/loss_history.txt', 'a') as f:
                    f.write(f"{epoch},{avg_loss:.4f}\n")
                    f.flush(); os.fsync(f.fileno())
                
                if epoch % 5 == 0 or epoch == epochs:
                    torch.save({"epoch": epoch, "encoder": model.module.state_dict()}, f"/kaggle/working/BT_ckpt_epoch_{epoch:03d}.pth")
                print(f"| Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f} | Process Healthy")

        if rank == 0:
            torch.save(model.module.state_dict(), "/kaggle/working/SKANN_SSL_GPU_Final.pth")
            
    finally:
        # GUARANTEED CLEANUP: Prevents resource leaks on crash or exit
        dist.destroy_process_group()