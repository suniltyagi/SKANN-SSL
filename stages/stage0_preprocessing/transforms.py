"""
SKANN-SSL Stage 0: Transforms
=============================
Data augmentation transforms for audio tensors.

These will be used primarily in Stage 4 (Augmentation Engine),
but are defined here for DataLoader integration.

Usage:
    from stages.stage0_preprocessing.transforms import Compose, TimeShift, GaussianNoise
    
    transform = Compose([
        TimeShift(max_shift=1600),
        GaussianNoise(std=0.01),
        AmplitudeScale(0.8, 1.2)
    ])
    
    dataset = SKANNDataset(manifest_path, transform=transform)
"""

import torch
from typing import List


class TimeShift:
    """
    Circular time shift augmentation.
    
    Shifts the waveform circularly by a random amount,
    preserving periodicity assumptions for underwater acoustics.
    
    Args:
        max_shift: Maximum shift in samples (default 1600 = 100ms at 16kHz)
    """
    def __init__(self, max_shift: int = 1600):
        self.max_shift = max_shift
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=-1)


class GaussianNoise:
    """
    Add Gaussian noise augmentation.
    
    Simulates additional ambient noise variations.
    
    Args:
        std: Standard deviation of noise (default 0.01)
    """
    def __init__(self, std: float = 0.01):
        self.std = std
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class AmplitudeScale:
    """
    Random amplitude scaling augmentation.
    
    Simulates gain variations from different recording conditions.
    
    Args:
        min_scale: Minimum scaling factor (default 0.8)
        max_scale: Maximum scaling factor (default 1.2)
    """
    def __init__(self, min_scale: float = 0.8, max_scale: float = 1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()
        return x * scale


class RandomCrop:
    """
    Random crop from waveform.
    
    Extracts a random segment of specified length.
    
    Args:
        crop_length: Length of crop in samples
    """
    def __init__(self, crop_length: int = 8000):
        self.crop_length = crop_length
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        total_length = x.shape[-1]
        if total_length <= self.crop_length:
            return x
        
        start = torch.randint(0, total_length - self.crop_length, (1,)).item()
        return x[..., start:start + self.crop_length]


class TimeMask:
    """
    Random time masking augmentation.
    
    Zeros out a random contiguous segment, similar to SpecAugment.
    
    Args:
        max_mask_length: Maximum length of mask in samples (default 1600)
    """
    def __init__(self, max_mask_length: int = 1600):
        self.max_mask_length = max_mask_length
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        total_length = x.shape[-1]
        mask_length = torch.randint(1, self.max_mask_length + 1, (1,)).item()
        start = torch.randint(0, total_length - mask_length, (1,)).item()
        
        x_masked = x.clone()
        x_masked[..., start:start + mask_length] = 0
        return x_masked


class Compose:
    """
    Compose multiple transforms sequentially.
    
    Args:
        transforms: List of transform callables
    """
    def __init__(self, transforms: List[callable]):
        self.transforms = transforms
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class RandomApply:
    """
    Apply a transform with given probability.
    
    Args:
        transform: Transform to apply
        p: Probability of applying (default 0.5)
    """
    def __init__(self, transform: callable, p: float = 0.5):
        self.transform = transform
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return self.transform(x)
        return x


# =============================================================================
# Preset Transform Pipelines
# =============================================================================

def get_ssl_transforms(
    time_shift: int = 1600,
    noise_std: float = 0.01,
    scale_range: tuple = (0.8, 1.2)
) -> Compose:
    """
    Get standard SSL augmentation pipeline.
    
    Used for creating positive pairs in Barlow Twins training.
    """
    return Compose([
        TimeShift(max_shift=time_shift),
        GaussianNoise(std=noise_std),
        AmplitudeScale(*scale_range)
    ])


def get_light_transforms() -> Compose:
    """Light augmentation for validation/fine-tuning."""
    return Compose([
        RandomApply(GaussianNoise(std=0.005), p=0.3),
        RandomApply(AmplitudeScale(0.9, 1.1), p=0.3)
    ])


def get_strong_transforms() -> Compose:
    """Strong augmentation for robust SSL training."""
    return Compose([
        TimeShift(max_shift=3200),
        GaussianNoise(std=0.02),
        AmplitudeScale(0.7, 1.3),
        RandomApply(TimeMask(max_mask_length=2400), p=0.5)
    ])
