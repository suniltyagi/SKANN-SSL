"""
SKANN-SSL Stage -1: Sea Noise Generator
=======================================
Generates colored noise based on digitized Knudsen curves.
Implements the methodology from Documents B and D.

Corrected frequency bands:
- Turbulence: 10 Hz → f_t (~35 Hz)
- LF: f_t → 200 Hz  
- MF: 200 Hz → 500 Hz (nearly flat plateau)
- HF: 500 Hz → 100 kHz
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import warnings

from config import (
    FS, N_SAMPLES, FFT_SIZE, NYQUIST, FREQ_RES,
    OLA_OVERLAP, P_REF
)

# Corrected frequency band boundaries
TURB_ANCHOR_1 = (10.0, 108.0)    # (Hz, dB)
TURB_ANCHOR_2 = (40.0, 55.0)     # (Hz, dB)
LF_END = 200.0                   # LF-MF boundary
MF_END = 500.0                   # MF-HF boundary
HF_END = 100000.0                # HF band end


class KnudsenModel:
    """
    Piecewise parametric model of ocean ambient noise based on Knudsen curves.
    
    The model divides the spectrum into four regions:
    1. Turbulence band: steep decay at very low frequencies (10 Hz - f_t)
    2. LF band: transition region (f_t - 200 Hz)
    3. MF band: nearly flat plateau (200 Hz - 500 Hz)
    4. HF band: wind-driven decay (500 Hz - 100 kHz)
    
    Each region follows: NL(f) = a * log10(f) + b
    """
    
    def __init__(self, csv_path: Optional[str] = None, sea_state: int = 3):
        """
        Initialize Knudsen model from digitized CSV data.
        
        Args:
            csv_path: Path to CSV file (index, frequency_Hz, NL_dB)
            sea_state: Sea state identifier (0, 1, 3, or 6)
        """
        self.sea_state = sea_state
        self.coefficients = {}
        self.f_t = None  # Turbulence-LF intersection
        
        if csv_path is not None:
            self._fit_from_csv(csv_path)
        else:
            # Use default SS3 coefficients
            self._set_default_coefficients()
    
    def _set_default_coefficients(self):
        """Set default SS3 coefficients from corrected Document B."""
        # Corrected coefficients for SS3
        self.coefficients = {
            'turb': (-88.01, 108.0),    # Steep turbulence decay
            'lf': (13.30, 36.29),        # Rising LF
            'mf': (-0.033, 66.96),       # Nearly flat MF plateau
            'hf': (-16.22, 110.65),      # HF decay
        }
        self.f_t = 35.1  # Turbulence-LF intersection
    
    def _fit_from_csv(self, csv_path: str):
        """
        Fit piecewise model coefficients from digitized Knudsen curve.
        
        Implements the corrected regression procedure:
        - HF: least-squares regression on 500 Hz - 100 kHz
        - LF: least-squares regression on 100 Hz - 200 Hz
        - MF: algebraic continuity between LF(200) and HF(500)
        - Turbulence: anchor points + intersection with LF
        """
        # Load and sort data
        df = pd.read_csv(csv_path, header=None, names=['idx', 'f', 'NL'])
        df = df.sort_values('f').reset_index(drop=True)
        
        # Store raw data for interpolation
        self.raw_f = df['f'].values
        self.raw_nl = df['NL'].values
        
        # === Turbulence coefficients (from anchor points) ===
        f1, NL1 = TURB_ANCHOR_1  # (10 Hz, 108 dB)
        f2, NL2 = TURB_ANCHOR_2  # (40 Hz, 55 dB)
        a_turb = (NL2 - NL1) / (np.log10(f2) - np.log10(f1))
        b_turb = NL1 - a_turb * np.log10(f1)
        
        # === LF regression (100 Hz - 200 Hz) ===
        lf_mask = (df['f'] >= 100.0) & (df['f'] <= 200.0)
        if lf_mask.sum() >= 2:
            lf_data = df[lf_mask]
            x_lf = np.log10(lf_data['f'].values)
            A_lf = np.vstack([x_lf, np.ones_like(x_lf)]).T
            a_lf, b_lf = np.linalg.lstsq(A_lf, lf_data['NL'].values, rcond=None)[0]
        else:
            # Fallback
            a_lf, b_lf = 13.30, 36.29
        
        # === Intersection frequency f_t ===
        if abs(a_lf - a_turb) > 1e-6:
            log_ft = (b_turb - b_lf) / (a_lf - a_turb)
            self.f_t = 10 ** log_ft
        else:
            self.f_t = 35.0  # Default
        
        # === HF regression (500 Hz - 100 kHz) ===
        hf_mask = (df['f'] >= 500.0) & (df['f'] <= 100000.0)
        if hf_mask.sum() >= 2:
            hf_data = df[hf_mask]
            x_hf = np.log10(hf_data['f'].values)
            A_hf = np.vstack([x_hf, np.ones_like(x_hf)]).T
            a_hf, b_hf_raw = np.linalg.lstsq(A_hf, hf_data['NL'].values, rcond=None)[0]
        else:
            # Fallback
            a_hf, b_hf_raw = -16.22, 110.65
        
        # === MF coefficients (algebraic continuity) ===
        x_200 = np.log10(200.0)
        x_500 = np.log10(500.0)
        
        # NL at 200 Hz from LF line
        NL_200 = a_lf * x_200 + b_lf
        
        # NL at 500 Hz from HF regression
        NL_500 = a_hf * x_500 + b_hf_raw
        
        # MF slope and intercept (connects 200 Hz to 500 Hz)
        a_mf = (NL_500 - NL_200) / (x_500 - x_200)
        b_mf = NL_200 - a_mf * x_200
        
        # Adjust HF intercept to ensure continuity at 500 Hz
        b_hf = NL_500 - a_hf * x_500
        
        # Store coefficients
        self.coefficients = {
            'turb': (a_turb, b_turb),
            'lf': (a_lf, b_lf),
            'mf': (a_mf, b_mf),
            'hf': (a_hf, b_hf),
        }
    
    def _interp_nl(self, df: pd.DataFrame, f_target: float) -> float:
        """Interpolate NL at target frequency in log10(f) space."""
        f = df['f'].values
        nl = df['NL'].values
        x = np.log10(f)
        xt = np.log10(f_target)
        
        idx = np.searchsorted(x, xt)
        if idx == 0:
            return nl[0]
        if idx >= len(x):
            return nl[-1]
        
        x0, x1 = x[idx-1], x[idx]
        y0, y1 = nl[idx-1], nl[idx]
        w = (xt - x0) / (x1 - x0)
        return y0 + w * (y1 - y0)
    
    def nl(self, f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute noise level NL(f) in dB re 1 µPa²/Hz.
        
        Args:
            f: Frequency in Hz (scalar or array)
            
        Returns:
            Noise level in dB
        """
        f = np.atleast_1d(f).astype(float)
        nl = np.zeros_like(f, dtype=float)
        
        # =================================================================
        # CORRECTED 4-ZONE MODEL (Prototype: 10 Hz - 8000 Hz)
        # =================================================================
        # Zone A: Plateau (10 Hz – 35 Hz) - FLAT at realistic value
        # Zone B: LF Transition (35 Hz – 200 Hz) - sea-state dependent
        # Zone C: MF Plateau (200 Hz – 500 Hz) - nearly flat
        # Zone D: HF Decay (500 Hz – 8000 Hz) - wind-driven decay
        # =================================================================
        
        MIN_FREQ = 10.0
        
        # Frequencies below 10 Hz: zero amplitude
        mask_below_min = f < MIN_FREQ
        if mask_below_min.any():
            nl[mask_below_min] = -np.inf
        
        # Get coefficients
        a_lf, b_lf = self.coefficients['lf']
        a_mf, b_mf = self.coefficients['mf']
        a_hf, b_hf = self.coefficients['hf']
        
        # Zone A: Plateau (10 Hz - f_t) 
        # Per ocean_noise_corrected2.docx and user instruction:
        # Keep a flat horizontal line equal to the dB value at f_t
        # NL(f_t) is calculated from the LF regression line at the intersection point
        nl_at_ft = a_lf * np.log10(self.f_t) + b_lf
        nl_plateau = nl_at_ft
        
        mask_plateau = (f >= MIN_FREQ) & (f < self.f_t)
        if mask_plateau.any():
            nl[mask_plateau] = nl_plateau
        
        # Zone B: LF Region (f_t - 200 Hz)
        # Per ocean_noise_corrected2.docx paragraph 7:
        # L = a_LF * log10(f) + b_LF
        mask_lf = (f >= self.f_t) & (f < LF_END)
        if mask_lf.any():
            nl[mask_lf] = a_lf * np.log10(f[mask_lf]) + b_lf
        
        # Zone C: MF Plateau (200 Hz - 500 Hz)
        mask_mf = (f >= LF_END) & (f < MF_END)
        if mask_mf.any():
            nl[mask_mf] = a_mf * np.log10(f[mask_mf]) + b_mf
        
        # Zone D: HF Decay (500 Hz - 8000 Hz)
        mask_hf = f >= MF_END
        if mask_hf.any():
            nl[mask_hf] = a_hf * np.log10(f[mask_hf]) + b_hf
        
        return nl.squeeze()
    
    def psd(self, f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute power spectral density S_pp(f) in Pa²/Hz.
        
        Args:
            f: Frequency in Hz
            
        Returns:
            PSD in Pa²/Hz (linear scale)
        """
        nl = self.nl(f)
        # NL is in dB re 1 µPa²/Hz
        # S_pp = 10^(NL/10) µPa²/Hz = 10^(NL/10) * 1e-12 Pa²/Hz
        return 10 ** (nl / 10) * 1e-12
    
    def __repr__(self):
        return (f"KnudsenModel(SS{self.sea_state}, f_t={self.f_t:.1f}Hz, "
                f"a_hf={self.coefficients['hf'][0]:.2f})")


class SeaNoiseGenerator:
    """
    Generates time-domain sea noise waveforms from Knudsen spectral models.
    
    Implements the synthesis pipeline from Document D:
    1. Construct FFT frequency grid
    2. Interpolate PSD onto grid
    3. Compute amplitude spectrum: |P(f_k)| = sqrt(S_pp(f_k) * Δf)
    4. Assign random phases
    5. Enforce Hermitian symmetry
    6. Inverse FFT
    7. RMS scaling
    """
    
    def __init__(self, fs: int = FS, n_samples: int = N_SAMPLES):
        """
        Initialize sea noise generator.
        
        Args:
            fs: Sampling frequency in Hz
            n_samples: Number of samples per frame
        """
        self.fs = fs
        self.n_samples = n_samples
        self.nyquist = fs // 2
        self.freq_res = fs / n_samples
        
        # FFT frequency grid (positive frequencies only)
        self.n_pos_freqs = n_samples // 2 + 1
        self.freq_grid = np.fft.rfftfreq(n_samples, 1/fs)
        
        # Load Knudsen models for all sea states
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load Knudsen models from CSV files."""
        # Look in multiple locations for CSV files
        base_dir = Path(__file__).parent
        possible_dirs = [
            base_dir / 'data',           # ./data/SS*CSV.txt
            base_dir,                     # ./SS*CSV.txt
            Path.cwd() / 'data',         # cwd/data/SS*CSV.txt
            Path.cwd(),                  # cwd/SS*CSV.txt
        ]
        
        for ss in [0, 1, 3, 6]:
            csv_path = None
            for csv_dir in possible_dirs:
                candidate = csv_dir / f'SS{ss}CSV.txt'
                if candidate.exists():
                    csv_path = candidate
                    break
            
            if csv_path is not None:
                self.models[ss] = KnudsenModel(str(csv_path), sea_state=ss)
            else:
                warnings.warn(f"CSV file not found for SS{ss}, using default SS3 coefficients")
                self.models[ss] = KnudsenModel(sea_state=ss)
    
    def _compute_amplitude_spectrum(self, model: KnudsenModel) -> np.ndarray:
        """
        Compute amplitude spectrum from PSD.
        
        |P(f_k)| = sqrt(S_pp(f_k) * Δf)
        
        Args:
            model: Knudsen model for the desired sea state
            
        Returns:
            Amplitude spectrum (one-sided)
        """
        # Get PSD at each frequency bin
        # Handle DC (f=0) specially
        freqs = self.freq_grid.copy()
        freqs[0] = 1.0  # Avoid log10(0)
        
        psd = model.psd(freqs)
        psd[0] = 0.0  # No DC component
        
        # Limit to Nyquist
        psd[freqs > self.nyquist] = 0.0
        
        # Amplitude spectrum
        amplitude = np.sqrt(psd * self.freq_res)
        
        return amplitude
    
    def generate_frame(self, sea_state: int = 3, 
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generate a single frame of sea noise.
        
        Args:
            sea_state: Sea state (0, 1, 3, or 6)
            rng: Random number generator (for reproducibility)
            
        Returns:
            Time-domain waveform in Pa (N samples)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        model = self.models.get(sea_state, self.models[3])
        
        # Amplitude spectrum (one-sided, in Pa)
        amplitude = self._compute_amplitude_spectrum(model)
        
        # Random phases (uniform 0 to 2π)
        phases = rng.uniform(0, 2 * np.pi, self.n_pos_freqs)
        phases[0] = 0.0  # DC has zero phase
        if self.n_samples % 2 == 0:
            phases[-1] = 0.0  # Nyquist has zero phase for real signal
        
        # Complex spectrum (one-sided)
        spectrum = amplitude * np.exp(1j * phases)
        
        # Inverse FFT (irfft handles Hermitian symmetry automatically)
        # SCALING for physical units (Parseval's theorem):
        # - irfft divides by N internally
        # - For one-sided spectrum, multiply by N to restore amplitude
        # - Divide by sqrt(2) because one-sided PSD represents both +/- frequencies
        # Final factor: N / sqrt(2)
        waveform = np.fft.irfft(spectrum, n=self.n_samples) * self.n_samples / np.sqrt(2)
        
        return waveform
    
    def generate_long(self, duration: float, sea_state: int = 3,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generate long-duration sea noise using overlap-add.
        
        Args:
            duration: Total duration in seconds
            sea_state: Sea state (0, 1, 3, or 6)
            rng: Random number generator
            
        Returns:
            Time-domain waveform
        """
        if rng is None:
            rng = np.random.default_rng()
        
        frame_duration = self.n_samples / self.fs
        hop = int(self.n_samples * (1 - OLA_OVERLAP))
        n_frames = int(np.ceil(duration / (frame_duration * (1 - OLA_OVERLAP))))
        
        # Output buffer
        total_samples = (n_frames - 1) * hop + self.n_samples
        output = np.zeros(total_samples)
        
        # Hann window (COLA-compliant with 50% overlap)
        window = np.hanning(self.n_samples)
        
        for i in range(n_frames):
            frame = self.generate_frame(sea_state, rng)
            frame_windowed = frame * window
            
            start = i * hop
            output[start:start + self.n_samples] += frame_windowed
        
        # Trim to exact duration
        n_out = int(duration * self.fs)
        return output[:n_out]
    
    def get_rms_level(self, sea_state: int = 3) -> float:
        """
        Estimate RMS level for a given sea state.
        
        Integrates PSD over bandwidth to get total power.
        
        Args:
            sea_state: Sea state
            
        Returns:
            Expected RMS pressure in Pa
        """
        model = self.models.get(sea_state, self.models[3])
        
        # Integrate PSD over frequency
        freqs = np.linspace(1, self.nyquist, 1000)
        psd = model.psd(freqs)
        
        # Total power = integral of PSD
        total_power = np.trapz(psd, freqs)
        
        return np.sqrt(total_power)
    
    def get_spl(self, sea_state: int = 3) -> float:
        """
        Get overall SPL for a sea state.
        
        Args:
            sea_state: Sea state
            
        Returns:
            SPL in dB re 1 µPa
        """
        rms = self.get_rms_level(sea_state)
        return 20 * np.log10(rms / P_REF)


def test_sea_noise():
    """Test the sea noise generator."""
    import matplotlib.pyplot as plt
    
    gen = SeaNoiseGenerator()
    
    # Test single frame generation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, ss in enumerate([0, 1, 3, 6]):
        ax = axes[i // 2, i % 2]
        
        # Generate waveform
        waveform = gen.generate_frame(sea_state=ss)
        
        # Compute spectrum for verification
        spectrum = np.abs(np.fft.rfft(waveform))
        freqs = gen.freq_grid
        
        # Convert to dB
        spectrum_db = 20 * np.log10(spectrum + 1e-12)
        
        # Plot
        ax.semilogx(freqs[1:], spectrum_db[1:], alpha=0.7, label='Generated')
        
        # Overlay model
        model = gen.models[ss]
        model_nl = model.nl(freqs[1:])
        ax.semilogx(freqs[1:], model_nl - 60, '--', label='Model (shifted)')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(f'Sea State {ss}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([10, 8000])
    
    plt.tight_layout()
    plt.savefig('sea_noise_test.png', dpi=150)
    plt.close()
    
    print("Sea noise test complete. Output: sea_noise_test.png")
    
    # Print some stats
    for ss in [0, 1, 3, 6]:
        spl = gen.get_spl(ss)
        print(f"SS{ss}: SPL = {spl:.1f} dB re 1 µPa")


if __name__ == '__main__':
    test_sea_noise()
