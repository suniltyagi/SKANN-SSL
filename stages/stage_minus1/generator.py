"""
SKANN-SSL Stage -1: Combined Synthetic Waveform Generator
=========================================================
Main entry point for generating synthetic underwater acoustic data.

Combines:
- Sea noise (Knudsen model)
- Ship noise (tonal + broadband + cavitation)

With controlled SNR mixing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import json
import os

from config import (
    FS, N_SAMPLES, DURATION,
    SHIP_SNR_DB, SEA_STATES, VESSEL_CLASSES,
    N_SYNTHETIC_CLIPS, RANDOM_SEED,
    DATA_DIR, WAVEFORM_DIR, TENSOR_DIR, METADATA_FILE
)
from sea_noise import SeaNoiseGenerator
from ship_noise import ShipNoiseGenerator, VesselParams


@dataclass
class ClipMetadata:
    """Metadata for a generated synthetic clip."""
    clip_id: int
    sea_state: int
    vessel_class: str
    shaft_rate: float
    blade_pass_freq: float
    n_blades: int
    has_cavitation: bool
    cavitation_intensity: float
    generator_freqs: List[float]
    snr_db: float
    sea_noise_rms: float
    ship_noise_rms: float
    combined_rms: float


class SyntheticDataGenerator:
    """
    Main generator for SKANN-SSL Stage -1.
    
    Produces synthetic underwater acoustic waveforms by combining
    ship signatures with ambient sea noise at controlled SNR.
    """
    
    def __init__(self, 
                 fs: int = FS, 
                 n_samples: int = N_SAMPLES,
                 snr_db: float = SHIP_SNR_DB):
        """
        Initialize the synthetic data generator.
        
        Args:
            fs: Sampling frequency (Hz)
            n_samples: Samples per clip
            snr_db: Ship-to-sea-noise ratio in dB
        """
        self.fs = fs
        self.n_samples = n_samples
        self.snr_db = snr_db
        self.duration = n_samples / fs
        
        # Initialize sub-generators
        self.sea_gen = SeaNoiseGenerator(fs, n_samples)
        self.ship_gen = ShipNoiseGenerator(fs, n_samples)
    
    def _compute_rms(self, signal: np.ndarray) -> float:
        """Compute RMS of a signal."""
        return np.sqrt(np.mean(signal ** 2))
    
    def _scale_to_snr(self, 
                      ship_signal: np.ndarray, 
                      sea_signal: np.ndarray,
                      target_snr_db: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Scale ship signal to achieve target SNR relative to sea noise.
        
        PHYSICAL MIXING LOGIC:
        ----------------------
        - Sea noise is in physical units (Pa) from Knudsen curves
        - Ship noise is in arbitrary units from spectral synthesis
        - This method converts ship noise to Pa by scaling to target SNR
        
        The scaling preserves the internal structure of ship noise:
        - Tonals and cavitation are scaled by the SAME factor
        - Their relative ratio is preserved exactly
        - No dynamic range compression occurs
        
        Example:
            Sea RMS = 0.005 Pa (Sea State 3)
            Target SNR = 6 dB → factor of 2
            Target Ship RMS = 0.010 Pa
            Raw Ship RMS = 4.0 (arbitrary)
            Scale Factor = 0.010 / 4.0 = 0.0025
            Result: Ship signal converted to Pa, 6 dB above sea
        
        Args:
            ship_signal: Ship noise waveform (arbitrary units)
            sea_signal: Sea noise waveform (Pa)
            target_snr_db: Desired SNR in dB
            
        Returns:
            Tuple of (scaled_ship [Pa], sea_signal [Pa], ship_rms [Pa], sea_rms [Pa])
        """
        sea_rms = self._compute_rms(sea_signal)  # Pa
        
        # Target ship RMS for desired SNR (in Pa)
        # SNR_dB = 20 * log10(ship_rms / sea_rms)
        # ship_rms = sea_rms * 10^(SNR_dB / 20)
        target_ship_rms = sea_rms * (10 ** (target_snr_db / 20))  # Pa
        
        # Scale ship signal from arbitrary units to Pa
        current_ship_rms = self._compute_rms(ship_signal)  # arbitrary
        if current_ship_rms > 0:
            scale_factor = target_ship_rms / current_ship_rms
            scaled_ship = ship_signal * scale_factor  # Now in Pa
        else:
            scaled_ship = ship_signal
        
        return scaled_ship, sea_signal, target_ship_rms, sea_rms
    
    def generate_clip(self,
                      sea_state: Optional[int] = None,
                      vessel_class: Optional[str] = None,
                      snr_db: Optional[float] = None,
                      rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, ClipMetadata]:
        """
        Generate a single synthetic clip.
        
        Args:
            sea_state: Sea state (0, 1, 3, 6). Random if None.
            vessel_class: Vessel class. Random if None.
            snr_db: SNR in dB. Uses default if None.
            rng: Random number generator.
            
        Returns:
            Tuple of (waveform, metadata)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if sea_state is None:
            sea_state = rng.choice(SEA_STATES)
        
        if vessel_class is None:
            vessel_class = rng.choice(list(VESSEL_CLASSES.keys()))
        
        if snr_db is None:
            snr_db = self.snr_db
        
        # Generate sea noise
        sea_signal = self.sea_gen.generate_frame(sea_state=sea_state, rng=rng)
        
        # Generate ship noise
        ship_signal, vessel_params = self.ship_gen.generate(
            vessel_class=vessel_class, rng=rng
        )
        
        # Scale to target SNR
        scaled_ship, sea_signal, ship_rms, sea_rms = self._scale_to_snr(
            ship_signal, sea_signal, snr_db
        )
        
        # Combine
        combined = scaled_ship + sea_signal
        combined_rms = self._compute_rms(combined)
        
        # Create metadata
        metadata = ClipMetadata(
            clip_id=-1,  # Will be set by dataset generator
            sea_state=sea_state,
            vessel_class=vessel_class,
            shaft_rate=vessel_params.shaft_rate,
            blade_pass_freq=vessel_params.blade_pass_freq,
            n_blades=vessel_params.n_blades,
            has_cavitation=vessel_params.has_cavitation,
            cavitation_intensity=vessel_params.cavitation_intensity,
            generator_freqs=[vessel_params.generator_freq] if vessel_params.generator_freq > 0 else [],
            snr_db=snr_db,
            sea_noise_rms=sea_rms,
            ship_noise_rms=ship_rms,
            combined_rms=combined_rms
        )
        
        return combined, metadata
    
    def generate_sea_only(self,
                          sea_state: Optional[int] = None,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Generate sea noise only (for debugging/testing)."""
        if rng is None:
            rng = np.random.default_rng()
        if sea_state is None:
            sea_state = rng.choice(SEA_STATES)
        return self.sea_gen.generate_frame(sea_state=sea_state, rng=rng)
    
    def generate_ship_only(self,
                           vessel_class: Optional[str] = None,
                           rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, VesselParams]:
        """Generate ship noise only (for debugging/testing)."""
        if rng is None:
            rng = np.random.default_rng()
        if vessel_class is None:
            vessel_class = rng.choice(list(VESSEL_CLASSES.keys()))
        return self.ship_gen.generate(vessel_class=vessel_class, rng=rng)


class DatasetGenerator:
    """
    Generates a complete synthetic dataset for SKANN-SSL training.
    """
    
    def __init__(self,
                 n_clips: int = N_SYNTHETIC_CLIPS,
                 output_dir: str = DATA_DIR,
                 seed: int = RANDOM_SEED):
        """
        Initialize dataset generator.
        
        Args:
            n_clips: Number of clips to generate
            output_dir: Output directory
            seed: Random seed for reproducibility
        """
        self.n_clips = n_clips
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        self.waveform_dir = self.output_dir / 'waveforms'
        self.tensor_dir = self.output_dir / 'tensors'
        
        self.generator = SyntheticDataGenerator()
    
    def _ensure_dirs(self):
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.waveform_dir.mkdir(exist_ok=True)
        self.tensor_dir.mkdir(exist_ok=True)
    
    def generate_dataset(self, 
                         save_waveforms: bool = True,
                         save_tensors: bool = True,
                         verbose: bool = True) -> pd.DataFrame:
        """
        Generate the complete dataset.
        
        Args:
            save_waveforms: Save raw waveforms as .npy files
            save_tensors: Save preprocessed tensors
            verbose: Print progress
            
        Returns:
            DataFrame with metadata for all clips
        """
        self._ensure_dirs()
        
        rng = np.random.default_rng(self.seed)
        metadata_list = []
        
        # Balance across sea states and vessel classes
        sea_states = SEA_STATES * (self.n_clips // len(SEA_STATES) + 1)
        vessel_classes = list(VESSEL_CLASSES.keys()) * (self.n_clips // len(VESSEL_CLASSES) + 1)
        
        rng.shuffle(sea_states)
        rng.shuffle(vessel_classes)
        
        for i in range(self.n_clips):
            sea_state = sea_states[i % len(sea_states)]
            vessel_class = vessel_classes[i % len(vessel_classes)]
            
            # Generate clip
            waveform, metadata = self.generator.generate_clip(
                sea_state=sea_state,
                vessel_class=vessel_class,
                rng=rng
            )
            
            # Update clip ID
            metadata.clip_id = i
            
            # Save waveform
            if save_waveforms:
                waveform_path = self.waveform_dir / f'clip_{i:05d}.npy'
                np.save(waveform_path, waveform.astype(np.float32))
            
            # Save tensor (preprocessed: [1, 1, N])
            if save_tensors:
                tensor = self._preprocess(waveform)
                tensor_path = self.tensor_dir / f'tensor_{i:05d}.npy'
                np.save(tensor_path, tensor)
            
            # Store metadata
            metadata_list.append(asdict(metadata))
            
            if verbose and (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{self.n_clips} clips")
        
        # Create DataFrame
        df = pd.DataFrame(metadata_list)
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.csv'
        df.to_csv(metadata_path, index=False)
        
        if verbose:
            print(f"\nDataset generation complete!")
            print(f"  Clips: {self.n_clips}")
            print(f"  Waveforms: {self.waveform_dir}")
            print(f"  Tensors: {self.tensor_dir}")
            print(f"  Metadata: {metadata_path}")
            print(f"\nDataset statistics:")
            print(f"  Sea states: {df['sea_state'].value_counts().to_dict()}")
            print(f"  Vessel classes: {df['vessel_class'].value_counts().to_dict()}")
            print(f"  Cavitation: {df['has_cavitation'].sum()} clips ({100*df['has_cavitation'].mean():.1f}%)")
        
        return df
    
    def _preprocess(self, waveform: np.ndarray) -> np.ndarray:
        """
        Preprocess waveform for SKANN Stage 0.
        
        Operations:
        1. DC offset removal
        2. RMS normalization
        3. Reshape to [1, 1, N]
        
        Args:
            waveform: Raw waveform (N,)
            
        Returns:
            Preprocessed tensor [1, 1, N]
        """
        # DC removal
        x = waveform - np.mean(waveform)
        
        # RMS normalization
        rms = np.sqrt(np.mean(x ** 2)) + 1e-8
        x = x / rms
        
        # Reshape for conv1d: [batch, channels, samples]
        x = x.reshape(1, 1, -1).astype(np.float32)
        
        return x
    
    def load_clip(self, clip_id: int, tensor: bool = True) -> np.ndarray:
        """Load a specific clip by ID."""
        if tensor:
            path = self.tensor_dir / f'tensor_{clip_id:05d}.npy'
        else:
            path = self.waveform_dir / f'clip_{clip_id:05d}.npy'
        return np.load(path)
    
    def load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata."""
        return pd.read_csv(self.output_dir / 'metadata.csv')


def generate_prototype_dataset():
    """Generate the prototype dataset (500 clips)."""
    generator = DatasetGenerator(n_clips=500, seed=42)
    df = generator.generate_dataset(verbose=True)
    return df


def test_generation():
    """Test the synthetic generator."""
    import matplotlib.pyplot as plt
    
    gen = SyntheticDataGenerator()
    rng = np.random.default_rng(42)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Test different combinations
    test_cases = [
        (0, 'small_craft'),
        (1, 'fishing_vessel'),
        (3, 'cargo_ship'),
        (6, 'tanker'),
        (3, 'small_craft'),
        (1, 'cargo_ship'),
    ]
    
    for i, (ss, vc) in enumerate(test_cases):
        ax = axes[i // 2, i % 2]
        
        waveform, metadata = gen.generate_clip(
            sea_state=ss, vessel_class=vc, rng=rng
        )
        
        # Time domain plot (first 100ms)
        t = np.arange(len(waveform)) / gen.fs * 1000
        ax.plot(t[:1600], waveform[:1600], alpha=0.7)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'SS{ss} + {vc}\nSNR={metadata.snr_db}dB, BPF={metadata.blade_pass_freq:.1f}Hz')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('combined_test.png', dpi=150)
    plt.close()
    
    # Spectrum plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, (ss, vc) in enumerate(test_cases[:4]):
        ax = axes[i // 2, i % 2]
        
        waveform, metadata = gen.generate_clip(
            sea_state=ss, vessel_class=vc, rng=rng
        )
        
        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(len(waveform), 1/gen.fs)
        spectrum_db = 20 * np.log10(spectrum + 1e-12)
        
        ax.semilogx(freqs[1:], spectrum_db[1:], alpha=0.7)
        ax.axvline(metadata.blade_pass_freq, color='r', linestyle='--', 
                   alpha=0.5, label=f'BPF: {metadata.blade_pass_freq:.1f}Hz')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(f'SS{ss} + {vc}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, 8000])
    
    plt.tight_layout()
    plt.savefig('combined_spectrum_test.png', dpi=150)
    plt.close()
    
    print("Test complete. Output: combined_test.png, combined_spectrum_test.png")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--generate':
        generate_prototype_dataset()
    else:
        test_generation()
