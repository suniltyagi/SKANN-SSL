# Stage 0: Preprocessing & Data Standardization

## Status: ✓ COMPLETE

## Objective

Ensure all raw signals are consistent, normalised, and ready for model ingestion.

---

## What This Stage Does

1. **Loads data** from `master_dataset_manifest.csv` (authoritative source)
2. **Creates PyTorch DataLoader** for training pipeline
3. **Stratified splits** (70/15/15 train/val/test) preserving class balance
4. **Validates** tensor properties (shape, RMS, no NaN/Inf)

---

## Files

| File | Description |
|------|-------------|
| `dataloader.py` | SKANNDataset class and DataLoader factory |
| `transforms.py` | Data augmentation transforms (for Stage 4) |
| `README.md` | This file |

---

## Usage

### Basic Usage
```python
from stages.stage0_preprocessing.dataloader import get_dataloaders

MANIFEST = '/content/drive/MyDrive/SKANN_SSL/data/prototype_dataset/master_dataset_manifest.csv'

train_loader, val_loader, test_loader = get_dataloaders(
    manifest_path=MANIFEST,
    batch_size=32,
    num_workers=0  # Required for Colab
)
```

### With Shared Config
```python
from shared.config import get_colab_paths
from stages.stage0_preprocessing.dataloader import get_dataloaders

paths = get_colab_paths()
train_loader, val_loader, test_loader = get_dataloaders(
    manifest_path=paths['manifest'],
    batch_size=32
)
```

### Access Metadata
```python
# Get underlying dataset
dataset = train_loader.dataset.dataset

# Filter by design factors
tanker_indices = dataset.filter_by(vessel_class='tanker')
cavitating_indices = dataset.filter_by(has_cavitation=True)
ss6_tankers = dataset.filter_by(sea_state=6, vessel_class='tanker')

# Get full metadata for a sample
meta = dataset.get_metadata(idx=0)
print(meta['shaft_rate'], meta['blade_pass_freq'])
```

---

## Output Format

```
Tensor shape: [Batch, 1, T] = [32, 1, 16000]
Labels: Integer tensor [0, 1, 2, 3] mapping to vessel classes
```

---

## Data Splits

| Split | Clips | Batches (BS=32) | Per Class |
|-------|-------|-----------------|-----------|
| Train | 1,344 | 42 | 336 |
| Val | 288 | 9 | 72 |
| Test | 288 | 9 | 72 |

---

## Preprocessing Applied (Stage -1)

```python
x = waveform - mean(waveform)      # DC removal
x = x / rms(x)                      # RMS normalization
x = x.reshape(1, 1, -1)             # [B, C, T] format
```

---

## Integration Test

```python
from stages.stage0_preprocessing.dataloader import run_integration_test

run_integration_test(train_loader, val_loader, test_loader)
```

Expected output:
```
✓ Shape valid [B, 1, 16000]
✓ No NaN values
✓ No Inf values
✓ RMS normalized (~1.0)
✓ Float32 dtype
```

---

## Roadmap Checklist

| Task | Status |
|------|--------|
| Resample to 16 kHz | ✓ (Stage -1) |
| Remove DC offset | ✓ (Stage -1) |
| Amplitude normalization (RMS) | ✓ (Stage -1) |
| Optional high-pass filter | ○ (10 Hz floor in data) |
| Silence trimming | ○ (continuous synthetic) |
| Fixed-length segments | ✓ (1 second) |
| Output Tensor [B, 1, T] | ✓ |
| PyTorch DataLoader | ✓ |
| Train/val/test split | ✓ |

---

## Next Stage

**Stage 1: SKConv1D** — Multi-branch learned filterbank

Input: `[B, 1, 16000]` from this stage  
Output: `[B, 64, 16000]` learned time-feature map
