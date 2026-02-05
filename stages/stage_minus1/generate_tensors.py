"""
SKANN-SSL: Tensor Generator
===========================
Converts v3 raw audio waveforms into model-ready tensors.

- Input:  5-second raw waveforms (from parallel_generator_v7.py)
- Logic:  Applies strict v2 Preprocessing (DC Removal + RMS Normalization)
- Output: Standardized Tensors (Shape: 1, 1, 80000)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = Path("structured_dataset")
WAVE_DIR = BASE_DIR / "waveforms"
TENSOR_DIR = BASE_DIR / "tensors"
MANIFEST_PATH = BASE_DIR / "master_dataset_manifest.csv"

# =========================================================
# THE LOGIC (Strictly from v2 generator.py)
# =========================================================
def apply_v2_normalization(waveform: np.ndarray) -> np.ndarray:
    """
    Applies the exact preprocessing recipe used in v2.
    """
    # 1. DC Removal (Centering the signal)
    x = waveform - np.mean(waveform)
    
    # 2. RMS Normalization (Scaling energy to 1.0)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-8
    x = x / rms
    
    # 3. Reshape for Model (Batch, Channel, Time)
    # Input: (80000,) -> Output: (1, 1, 80000)
    x = x.reshape(1, 1, -1).astype(np.float32)
    
    return x

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    # 1. Setup
    if not MANIFEST_PATH.exists():
        print(f"❌ Error: Manifest not found at {MANIFEST_PATH}")
        return

    print(f"📂 Loading manifest: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH)
    
    print(f"📂 Creating output directory: {TENSOR_DIR}")
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    
    # Lists to store paths for the manifest update
    tensor_paths_col = []
    waveform_paths_col = []
    
    print(f"🚀 Converting {len(df)} v3 waveforms to tensors...")
    
    # 2. Processing Loop
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['filename']
        wave_path = WAVE_DIR / fname
        
        # Define output filename (clip_00000.npy -> tensor_00000.npy)
        tensor_fname = fname.replace("clip_", "tensor_")
        tensor_path = TENSOR_DIR / tensor_fname
        
        try:
            # A. Load Raw Waveform
            if not wave_path.exists():
                raise FileNotFoundError(f"Waveform missing: {wave_path}")
                
            waveform = np.load(wave_path)
            
            # B. Apply Logic
            tensor = apply_v2_normalization(waveform)
            
            # C. Save Tensor
            np.save(tensor_path, tensor)
            
            # D. Store Paths (Relative for portability)
            tensor_paths_col.append(str(tensor_path))
            waveform_paths_col.append(str(wave_path))
            
        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")
            tensor_paths_col.append(None)
            waveform_paths_col.append(None)

    # 3. Update Manifest
    print("💾 Updating manifest with new paths...")
    df['tensor_path'] = tensor_paths_col
    df['waveform_path'] = waveform_paths_col
    
    # Save
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"✅ Success! Tensors created for all v3 clips.")
    print(f"   Manifest updated at: {MANIFEST_PATH}")

if __name__ == "__main__":
    main()