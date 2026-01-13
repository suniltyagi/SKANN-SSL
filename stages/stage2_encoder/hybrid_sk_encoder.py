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