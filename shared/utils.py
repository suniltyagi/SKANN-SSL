"""
SKANN-SSL Shared Utilities
==========================
Common functions used across stages.

Import:
    from shared.utils import set_seed, get_device
"""

import random
import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: torch.nn.Module, input_shape: tuple = (1, 1, 16000)):
    """Print model summary with parameter counts."""
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Input shape: {input_shape}")
    
    # Try forward pass to get output shape
    try:
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, *input_shape[1:])
            y = model(x)
            print(f"Output shape: {tuple(y.shape)}")
    except Exception as e:
        print(f"Could not determine output shape: {e}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    path: Union[str, Path],
    **kwargs
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        **kwargs
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    return checkpoint


def compute_rms(x: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute RMS of signal."""
    if isinstance(x, torch.Tensor):
        return torch.sqrt(torch.mean(x ** 2)).item()
    return float(np.sqrt(np.mean(x ** 2)))


def db_to_amplitude(db: float) -> float:
    """Convert dB to amplitude ratio."""
    return 10 ** (db / 20)


def amplitude_to_db(amp: float) -> float:
    """Convert amplitude ratio to dB."""
    return 20 * np.log10(amp + 1e-10)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = 'metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop
