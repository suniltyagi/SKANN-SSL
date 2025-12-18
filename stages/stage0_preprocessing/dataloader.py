"""
SKANN-SSL Stage 0: DataLoader
=============================
PyTorch Dataset and DataLoader using master_dataset_manifest.csv.

Usage:
    from stages.stage0_preprocessing.dataloader import get_dataloaders
    
    train_loader, val_loader, test_loader = get_dataloaders(
        manifest_path='path/to/master_dataset_manifest.csv',
        batch_size=32
    )
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union

# Import shared config (with fallback for standalone use)
try:
    from shared.config import (
        SAMPLE_RATE, FREQ_MIN, FREQ_MAX, VESSEL_CLASSES,
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
    )
except ImportError:
    # Fallback defaults if shared not available
    SAMPLE_RATE = 16000
    FREQ_MIN = 10
    FREQ_MAX = 8000
    VESSEL_CLASSES = {
        'cargo_ship': 0,
        'fishing_vessel': 1,
        'small_craft': 2,
        'tanker': 3
    }
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_STATE = 42


class SKANNDataset(Dataset):
    """
    PyTorch Dataset for SKANN-SSL underwater acoustic data.
    
    Uses master_dataset_manifest.csv as the authoritative source for:
    - File paths (tensor_path, waveform_path columns)
    - All metadata (26 columns including design factors and measurements)
    """
    
    # Class-level constants
    SAMPLE_RATE = SAMPLE_RATE
    DURATION_SEC = 1.0
    FREQ_MIN = FREQ_MIN
    FREQ_MAX = FREQ_MAX
    VESSEL_CLASSES = VESSEL_CLASSES
    
    REQUIRED_COLUMNS = [
        'clip_id', 'vessel_class', 'sea_state', 'n_blades',
        'cavitation_intensity', 'snr_db', 'tensor_path'
    ]
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        label_column: str = 'vessel_class',
        load_waveforms: bool = False,
        transform: Optional[callable] = None,
        return_metadata: bool = False
    ):
        """
        Initialize dataset from master manifest.
        
        Args:
            manifest_path: Path to master_dataset_manifest.csv
            label_column: Column name for classification labels
            load_waveforms: If True, load raw waveforms instead of tensors
            transform: Optional transform to apply to data
            return_metadata: If True, __getitem__ returns (tensor, label, metadata_dict)
        """
        self.manifest_path = Path(manifest_path)
        self.label_column = label_column
        self.load_waveforms = load_waveforms
        self.transform = transform
        self.return_metadata = return_metadata
        
        self._load_manifest()
        self._validate_manifest()
        
    def _load_manifest(self):
        """Load the master manifest CSV."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        self.manifest = pd.read_csv(self.manifest_path)
        self.manifest = self.manifest.set_index('clip_id')
        
    def _validate_manifest(self):
        """Validate manifest has required columns."""
        missing = set(self.REQUIRED_COLUMNS) - {'clip_id'} - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        
        if self.label_column not in self.manifest.columns:
            raise ValueError(f"Label column '{self.label_column}' not in manifest")
        
        if self.load_waveforms and 'waveform_path' not in self.manifest.columns:
            raise ValueError("waveform_path column required when load_waveforms=True")
        
        self.clip_ids = list(self.manifest.index)
        
        print(f"SKANNDataset initialized from manifest:")
        print(f"  Manifest: {self.manifest_path.name}")
        print(f"  Clips: {len(self.clip_ids)}")
        print(f"  Label column: {self.label_column}")
        print(f"  Mode: {'waveforms' if self.load_waveforms else 'tensors'}")
        
    def __len__(self) -> int:
        return len(self.clip_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load a single sample using path from manifest."""
        clip_id = self.clip_ids[idx]
        row = self.manifest.loc[clip_id]
        
        # Get file path from manifest
        file_path = row['waveform_path'] if self.load_waveforms else row['tensor_path']
        
        # Load data
        data = np.load(file_path)
        
        # Handle tensor shape: [1, 1, 16000] -> [1, 16000]
        if data.ndim == 3:
            data = data.squeeze(0)
        elif data.ndim == 1:
            data = data.reshape(1, -1)
        
        tensor = torch.from_numpy(data).float()
        
        # Apply optional transform
        if self.transform is not None:
            tensor = self.transform(tensor)
        
        # Get label
        label_value = row[self.label_column]
        if isinstance(label_value, str):
            label = self.VESSEL_CLASSES[label_value]
        else:
            label = int(label_value)
        
        if self.return_metadata:
            meta_dict = row.to_dict()
            meta_dict['clip_id'] = clip_id
            return tensor, label, meta_dict
        
        return tensor, label
    
    def get_labels(self) -> np.ndarray:
        """Return all labels as numpy array for stratification."""
        labels = []
        for clip_id in self.clip_ids:
            label_value = self.manifest.loc[clip_id, self.label_column]
            if isinstance(label_value, str):
                labels.append(self.VESSEL_CLASSES.get(label_value, -1))
            else:
                labels.append(int(label_value))
        return np.array(labels)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of samples per class."""
        return self.manifest[self.label_column].value_counts().to_dict()
    
    def get_metadata(self, idx: int) -> pd.Series:
        """Get full metadata for a sample by index."""
        clip_id = self.clip_ids[idx]
        return self.manifest.loc[clip_id]
    
    def filter_by(self, **kwargs) -> List[int]:
        """
        Get indices matching filter criteria.
        
        Example:
            indices = dataset.filter_by(vessel_class='tanker', has_cavitation=True)
        """
        mask = pd.Series([True] * len(self.manifest), index=self.manifest.index)
        for col, val in kwargs.items():
            if col in self.manifest.columns:
                mask &= (self.manifest[col] == val)
        
        matching_ids = self.manifest[mask].index.tolist()
        return [self.clip_ids.index(cid) for cid in matching_ids if cid in self.clip_ids]


def stratified_split(
    dataset: SKANNDataset,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
    stratify_columns: List[str] = None
) -> Tuple[Subset, Subset, Subset]:
    """
    Create stratified train/val/test splits preserving class distribution.
    
    Args:
        dataset: SKANNDataset instance
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for testing (default 0.15)
        random_state: Random seed for reproducibility
        stratify_columns: List of columns for multi-factor stratification
        
    Returns:
        train_subset, val_subset, test_subset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    indices = np.arange(len(dataset))
    
    # Build stratification labels
    if stratify_columns is None:
        stratify_columns = ['vessel_class']
    
    if len(stratify_columns) == 1 and stratify_columns[0] == 'vessel_class':
        labels = dataset.get_labels()
    else:
        labels = []
        for clip_id in dataset.clip_ids:
            row = dataset.manifest.loc[clip_id]
            compound = '_'.join(str(row[col]) for col in stratify_columns)
            labels.append(compound)
        labels = np.array(labels)
    
    # First split: separate test set
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, stratify=labels, random_state=random_state
    )
    
    # Second split: separate train and val
    val_relative = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_relative,
        stratify=labels[train_val_idx], random_state=random_state
    )
    
    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())
    test_subset = Subset(dataset, test_idx.tolist())
    
    print(f"\nStratified split ({'+'.join(stratify_columns)}):")
    print(f"  Train: {len(train_subset)} samples ({100*train_ratio:.0f}%)")
    print(f"  Val:   {len(val_subset)} samples ({100*val_ratio:.0f}%)")
    print(f"  Test:  {len(test_subset)} samples ({100*test_ratio:.0f}%)")
    
    return train_subset, val_subset, test_subset


def get_dataloaders(
    manifest_path: Union[str, Path],
    batch_size: int = 32,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    num_workers: int = 0,
    pin_memory: bool = True,
    random_state: int = RANDOM_STATE,
    label_column: str = 'vessel_class',
    stratify_columns: List[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create train/val/test DataLoaders from manifest.
    
    Args:
        manifest_path: Path to master_dataset_manifest.csv
        batch_size: Batch size for all loaders
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        num_workers: DataLoader workers (use 0 for Colab)
        pin_memory: Pin memory for GPU transfer
        random_state: Random seed
        label_column: Manifest column for labels
        stratify_columns: Columns for stratification
        
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset = SKANNDataset(
        manifest_path=manifest_path,
        label_column=label_column
    )
    
    print(f"\nClass distribution:")
    for cls, count in dataset.get_class_distribution().items():
        print(f"  {cls}: {count}")
    
    train_set, val_set, test_set = stratified_split(
        dataset, train_ratio, val_ratio, test_ratio, random_state, stratify_columns
    )
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# Testing & Validation
# =============================================================================

def validate_batch(batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, any]:
    """Validate a batch meets Stage 0 specifications."""
    tensors, labels = batch
    
    results = {
        'batch_size': tensors.shape[0],
        'tensor_shape': tuple(tensors.shape),
        'dtype': str(tensors.dtype),
        'min_val': tensors.min().item(),
        'max_val': tensors.max().item(),
        'mean': tensors.mean().item(),
        'std': tensors.std().item(),
        'unique_labels': labels.unique().tolist(),
        'has_nan': torch.isnan(tensors).any().item(),
        'has_inf': torch.isinf(tensors).any().item(),
    }
    
    rms_per_sample = torch.sqrt(torch.mean(tensors**2, dim=(1, 2)))
    results['rms_mean'] = rms_per_sample.mean().item()
    results['rms_std'] = rms_per_sample.std().item()
    results['rms_normalized'] = abs(results['rms_mean'] - 1.0) < 0.1
    
    expected_shape = (results['batch_size'], 1, SAMPLE_RATE)
    results['shape_valid'] = tensors.shape == expected_shape
    
    return results


def run_integration_test(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> bool:
    """Run integration tests on DataLoaders."""
    print("\n" + "="*60)
    print("STAGE 0 INTEGRATION TEST")
    print("="*60)
    
    all_passed = True
    
    for name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        print(f"\n--- {name} Loader ---")
        
        batch = next(iter(loader))
        results = validate_batch(batch)
        
        checks = [
            ('Shape valid [B, 1, 16000]', results['shape_valid']),
            ('No NaN values', not results['has_nan']),
            ('No Inf values', not results['has_inf']),
            ('RMS normalized (~1.0)', results['rms_normalized']),
            ('Float32 dtype', results['dtype'] == 'torch.float32'),
        ]
        
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        print(f"  Batch shape: {results['tensor_shape']}")
        print(f"  Value range: [{results['min_val']:.3f}, {results['max_val']:.3f}]")
        print(f"  RMS: {results['rms_mean']:.4f} ± {results['rms_std']:.4f}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓" if all_passed else "SOME TESTS FAILED ✗")
    print("="*60)
    
    return all_passed
