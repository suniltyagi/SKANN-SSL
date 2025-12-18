"""
SKANN-SSL Stage -1: Ship Noise Generator
=========================================
Generates synthetic ship noise signatures using:
1. Document D spectral approach for tonal + broadband components
2. Physical burst model for cavitation (blade-gated discrete events)

This model moves from "Signal Processing" to "Physical Modeling":
- Cavitation bursts are discrete bubble collapse events
- Bursts are gated by blade passage (activity windows)
- Generated at 200 kHz to capture μs-scale physics, then downsampled

Components:
- Tonal: Shaft rate harmonics, BPF harmonics, generator harmonics (spectral)
- Broadband: Flow noise with frequency-dependent rolloff (spectral)
- Cavitation: Blade-gated burst events with collapse/cloud/sheet types (time-domain)

References:
- Ross (1987) - Mechanics of Underwater Noise
- Brennen (1995) - Cavitation and Bubble Dynamics
- cavitation_with_burst.txt - Physical burst model
"""

import numpy as np
from scipy import signal as scipy_signal
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from config import (
    FS, N_SAMPLES, NYQUIST, FREQ_RES,
    VESSEL_CLASSES, CAVITATION_PROB,
    CAVITATION_INTENSITY_RANGE
)


# =============================================================================
# PHYSICAL CONSTANTS FOR CAVITATION
# =============================================================================
RHO_WATER = 998.0           # Water density [kg/m³]
P_AMBIENT = 101325.0        # Ambient pressure [Pa]
P_VAPOR = 2340.0            # Vapor pressure [Pa]
DELTA_P = P_AMBIENT - P_VAPOR

# High sample rate for burst generation (captures μs-scale physics)
FS_GENERATION = 200000      # 200 kHz - required for 10-100 μs burst resolution


@dataclass
class VesselParams:
    """Parameters defining a vessel's acoustic signature."""
    vessel_class: str
    shaft_rate: float          # Fundamental frequency f0 (Hz)
    n_blades: int              # Number of propeller blades
    blade_pass_freq: float     # BPF = shaft_rate * n_blades (Hz)
    n_shaft_harmonics: int     # Number of shaft harmonics
    n_bpf_harmonics: int       # Number of BPF harmonics
    shaft_harmonic_decay: float    # dB per harmonic
    bpf_harmonic_decay: float      # dB per harmonic
    broadband_level: float     # Relative level (0-1)
    broadband_rolloff: float   # dB per octave (negative)
    has_cavitation: bool
    cavitation_intensity: float    # 0-1
    generator_freq: float      # 50 or 60 Hz (or 0 for none)
    
    # === NEW FIELDS: Populated during generation ===
    # These capture the actual random values used in each clip
    equipment_base_freq: float = 0.0           # 25 or 30 Hz (depends on generator_freq)
    resonance_freq_1: float = 0.0              # First resonance frequency (Hz)
    resonance_freq_2: float = 0.0              # Second resonance frequency (Hz)
    resonance_freq_3: float = 0.0              # Third resonance frequency (Hz), may be 0 if only 2
    cavitation_peak_freq: float = 0.0          # Actual cavitation peak frequency used
    n_cavitation_bursts: int = 0               # Number of cavitation bursts in this clip


class ShipNoiseGenerator:
    """
    Generates ship noise using hybrid approach:
    - Spectral synthesis for tonal + broadband (Document D)
    - Physical burst model for cavitation (blade-gated events)
    
    Cavitation is generated at 200 kHz then downsampled to preserve
    the μs-scale bubble collapse physics while outputting at target fs.
    """
    
    # Cavitation burst scaling factor
    # Raw bursts are ~51 dB above tonals; scale to ~10 dB below
    CAVITATION_GAIN = 0.001
    
    def __init__(self, fs: int = FS, n_samples: int = N_SAMPLES):
        """
        Initialize ship noise generator.
        
        Args:
            fs: Output sampling frequency (Hz) - typically 16000
            n_samples: Number of output samples per frame
        """
        self.fs = fs
        self.n_samples = n_samples
        self.nyquist = fs // 2
        self.freq_res = fs / n_samples
        self.duration = n_samples / fs
        
        # Frequency grid for spectral synthesis
        self.n_bins = n_samples // 2 + 1
        self.freq_grid = np.fft.rfftfreq(n_samples, 1/fs)
        
        # High-rate parameters for cavitation burst generation
        self.fs_gen = FS_GENERATION
        self.n_samples_gen = int(self.duration * self.fs_gen)
    
    def create_vessel_params(self, vessel_class: str,
                             rng: Optional[np.random.Generator] = None) -> VesselParams:
        """
        Create randomized vessel parameters within class-specific ranges.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        vc = VESSEL_CLASSES.get(vessel_class, VESSEL_CLASSES['cargo_ship'])
        
        shaft_rate = rng.uniform(*vc['f0_range'])
        n_blades = rng.integers(vc['n_blades'][0], vc['n_blades'][1] + 1)
        blade_pass_freq = shaft_rate * n_blades
        
        has_cavitation = rng.random() < CAVITATION_PROB
        cavitation_intensity = 0.0
        if has_cavitation:
            cavitation_intensity = rng.uniform(*CAVITATION_INTENSITY_RANGE)
        
        if rng.random() < 0.7:
            generator_freq = rng.choice([50.0, 60.0])
        else:
            generator_freq = 0.0
        
        return VesselParams(
            vessel_class=vessel_class,
            shaft_rate=shaft_rate,
            n_blades=n_blades,
            blade_pass_freq=blade_pass_freq,
            n_shaft_harmonics=vc['n_harmonics'],
            n_bpf_harmonics=vc.get('n_bpf_harmonics', 6),
            shaft_harmonic_decay=vc.get('harmonic_decay', 3.0),
            bpf_harmonic_decay=vc.get('bpf_decay', 4.0),
            broadband_level=vc['broadband_level'],
            broadband_rolloff=vc['broadband_rolloff'],
            has_cavitation=has_cavitation,
            cavitation_intensity=cavitation_intensity,
            generator_freq=generator_freq,
            # New fields - initialized to defaults, populated during generate()
            equipment_base_freq=0.0,
            resonance_freq_1=0.0,
            resonance_freq_2=0.0,
            resonance_freq_3=0.0,
            cavitation_peak_freq=0.0,
            n_cavitation_bursts=0
        )
    
    # =========================================================================
    # SPECTRAL SYNTHESIS (Document D approach) - Tonal + Broadband
    # =========================================================================
    
    def _freq_to_bin(self, freq: float) -> int:
        """Convert frequency (Hz) to nearest bin index."""
        bin_idx = int(round(freq / self.freq_res))
        return np.clip(bin_idx, 0, self.n_bins - 1)
    
    def _add_tonal(self, amplitude_spectrum: np.ndarray,
                   freq: float, amplitude_db: float) -> None:
        """Add a tonal component (spike) to the amplitude spectrum."""
        # CONSTRAINT: No simulation below 10 Hz (prototype limited to 10 Hz - 8000 Hz)
        MIN_FREQ = 10.0
        if freq < MIN_FREQ or freq >= self.nyquist:
            return
        
        bin_idx = self._freq_to_bin(freq)
        amplitude_linear = 10 ** (amplitude_db / 20)
        
        if 0 < bin_idx < self.n_bins:
            amplitude_spectrum[bin_idx] += amplitude_linear
    
    def _add_shaft_harmonics(self, amplitude_spectrum: np.ndarray,
                             params: VesselParams, ref_level_db: float) -> None:
        """Add shaft rate fundamental and harmonics."""
        f0 = params.shaft_rate
        
        for h in range(1, params.n_shaft_harmonics + 1):
            freq = f0 * h
            if freq >= self.nyquist:
                break
            amp_db = ref_level_db - params.shaft_harmonic_decay * (h - 1)
            self._add_tonal(amplitude_spectrum, freq, amp_db)
    
    def _add_bpf_harmonics(self, amplitude_spectrum: np.ndarray,
                           params: VesselParams, ref_level_db: float) -> None:
        """Add blade pass frequency fundamental and harmonics."""
        bpf = params.blade_pass_freq
        bpf_ref = ref_level_db - 3.0
        
        for h in range(1, params.n_bpf_harmonics + 1):
            freq = bpf * h
            if freq >= self.nyquist:
                break
            amp_db = bpf_ref - params.bpf_harmonic_decay * (h - 1)
            self._add_tonal(amplitude_spectrum, freq, amp_db)
    
    def _add_generator_harmonics(self, amplitude_spectrum: np.ndarray,
                                  params: VesselParams, ref_level_db: float) -> None:
        """Add electrical generator harmonics (50/60 Hz)."""
        if params.generator_freq <= 0:
            return
        
        f_gen = params.generator_freq
        gen_ref = ref_level_db - 10.0
        
        for h in range(1, 8):
            freq = f_gen * h
            if freq >= self.nyquist:
                break
            amp_db = gen_ref - 3.0 * (h - 1)
            self._add_tonal(amplitude_spectrum, freq, amp_db)
    
    def _add_equipment_harmonics(self, amplitude_spectrum: np.ndarray,
                                  params: VesselParams, ref_level_db: float,
                                  rng: np.random.Generator) -> float:
        """
        Add machinery/equipment running frequency harmonics.
        
        Most shipboard equipment (pumps, compressors, fans, auxiliary engines)
        runs at ~1500 RPM (25 Hz) or ~1800 RPM (30 Hz) depending on 50/60 Hz supply.
        
        Returns:
            Equipment base frequency used (25.0 or 30.0 Hz)
        """
        # Equipment fundamental: 1500 RPM = 25 Hz (50 Hz supply) or 1800 RPM = 30 Hz (60 Hz supply)
        if params.generator_freq == 60.0:
            f_equip = 30.0  # 1800 RPM for 60 Hz systems
        else:
            f_equip = 25.0  # 1500 RPM for 50 Hz systems
        
        # Equipment is typically 15 dB below main tonal components
        equip_ref = ref_level_db - 15.0
        
        # Add fundamental and harmonics (with some randomness)
        n_harmonics = rng.integers(4, 8)  # 4-7 harmonics
        for h in range(1, n_harmonics + 1):
            freq = f_equip * h
            if freq < 10 or freq >= self.nyquist:
                continue
            # Random amplitude variation ±3 dB
            amp_db = equip_ref - 2.5 * (h - 1) + rng.uniform(-3, 3)
            self._add_tonal(amplitude_spectrum, freq, amp_db)
        
        return f_equip
    
    def _add_structural_resonances(self, amplitude_spectrum: np.ndarray,
                                    params: VesselParams, ref_level_db: float,
                                    rng: np.random.Generator) -> List[float]:
        """
        Add structural/foundation resonance tonals.
        
        Ships have natural resonance frequencies from:
        - Hull modes: 50-150 Hz
        - Foundation/mounting: 100-300 Hz  
        - Piping/ductwork: 200-500 Hz
        
        These create narrow-band tonals that vary per vessel.
        
        Returns:
            List of resonance frequencies actually used (2-3 values)
        """
        # Resonance level: 12-18 dB below main tonals
        res_ref = ref_level_db - 15.0
        
        # Define resonance frequency ranges
        resonance_bands = [
            (50, 150),   # Hull modes
            (100, 300),  # Foundation
            (200, 500),  # Piping
        ]
        
        # Add 2-3 random resonances
        n_resonances = rng.integers(2, 4)
        selected_bands = rng.choice(len(resonance_bands), size=n_resonances, replace=False)
        
        resonance_freqs_used = []
        
        for band_idx in selected_bands:
            f_low, f_high = resonance_bands[band_idx]
            # Random frequency within band
            f_res = rng.uniform(f_low, f_high)
            if f_res < 10 or f_res >= self.nyquist:
                continue
            # Random amplitude ±5 dB
            amp_db = res_ref + rng.uniform(-5, 5)
            self._add_tonal(amplitude_spectrum, f_res, amp_db)
            resonance_freqs_used.append(round(f_res, 2))
        
        return resonance_freqs_used
    
    def _add_broadband(self, amplitude_spectrum: np.ndarray,
                       params: VesselParams, ref_level_db: float) -> None:
        """
        Add broadband flow/turbulence noise with physically realistic spectrum.
        
        Real ship broadband characteristics:
        - Relatively FLAT from 10 Hz to ~500 Hz (machinery/flow noise band)
        - Rolloff ABOVE 500 Hz (high-frequency attenuation)
        - NO low-frequency boost (previous model was wrong)
        
        The previous model used negative rolloff which incorrectly boosted
        low frequencies by 30-40 dB, creating unrealistic divergence from
        sea noise below 150 Hz.
        
        Corrected model:
        - Flat response from MIN_FREQ to f_corner (500 Hz)
        - Rolloff above f_corner at -3 dB/octave
        - Level set relative to tonal reference
        """
        # Broadband level relative to tonals
        broadband_ref_db = ref_level_db - 20.0 + 10 * np.log10(params.broadband_level + 0.1)
        
        # Corner frequency: flat below, rolloff above
        f_corner = 500.0
        
        # Rolloff only applies ABOVE f_corner
        # -3 dB/octave = -10 dB/decade (typical for flow noise at HF)
        rolloff_db_per_octave = -3.0
        rolloff_per_decade = rolloff_db_per_octave * 3.32
        
        for k in range(1, self.n_bins):
            freq = self.freq_grid[k]
            if freq < 10:
                continue
            
            if freq <= f_corner:
                # FLAT response below corner frequency
                psd_db = broadband_ref_db
            else:
                # Rolloff above corner frequency
                psd_db = broadband_ref_db + rolloff_per_decade * np.log10(freq / f_corner)
            
            psd_linear = 10 ** (psd_db / 10)
            amplitude = np.sqrt(psd_linear * self.freq_res)
            amplitude_spectrum[k] += amplitude
    
    def _add_cavitation_spectral(self, amplitude_spectrum: np.ndarray,
                                  params: VesselParams, ref_level_db: float,
                                  rng: np.random.Generator) -> None:
        """
        Add cavitation as a continuous spectral hump (NOT discrete bursts).
        
        This replaces the burst model with a smoother, more continuous
        broadband elevation centered at the cavitation peak frequency.
        The result is horizontal streaks in spectrograms rather than
        vertical barcodes.
        
        Physics:
        - Cavitation creates broadband noise centered around 500-2000 Hz
        - Level depends on cavitation intensity
        - Bandwidth is 1-2 octaves
        """
        if not params.has_cavitation or params.cavitation_intensity < 0.01:
            return
        
        # Get vessel-specific cavitation peak
        vc = VESSEL_CLASSES.get(params.vessel_class, VESSEL_CLASSES['cargo_ship'])
        f_peak = vc.get('cavitation_peak', 1000.0)
        
        # Cavitation bandwidth (1.5 octaves)
        bandwidth_octaves = 1.5
        f_low = f_peak / (2 ** (bandwidth_octaves / 2))
        f_high = f_peak * (2 ** (bandwidth_octaves / 2))
        
        # Cavitation level: scale by intensity, reduced by 0.01x for balance with tonals
        # This puts cavitation about 10-15 dB below tonal peaks
        cav_ref_db = ref_level_db - 10.0 + 10 * np.log10(params.cavitation_intensity + 0.01)
        
        for k in range(1, self.n_bins):
            freq = self.freq_grid[k]
            if freq < f_low or freq > f_high:
                continue
            
            # Gaussian hump shape in log-frequency space
            log_freq = np.log10(freq)
            log_peak = np.log10(f_peak)
            log_width = 0.25  # Width in log-frequency
            
            shape = np.exp(-((log_freq - log_peak) / log_width) ** 2)
            
            # Add some random variation for realism
            variation = rng.uniform(0.8, 1.2)
            
            psd_db = cav_ref_db + 10 * np.log10(shape * variation + 0.001)
            psd_linear = 10 ** (psd_db / 10)
            amplitude = np.sqrt(psd_linear * self.freq_res)
            
            amplitude_spectrum[k] += amplitude
    
    def _generate_tonal_broadband(self, params: VesselParams,
                                   rng: np.random.Generator) -> np.ndarray:
        """
        Generate tonal + broadband components using spectral synthesis.
        Returns time-domain waveform at output sample rate.
        
        Also populates params.equipment_base_freq and params.resonance_freq_* fields.
        """
        amplitude_spectrum = np.zeros(self.n_bins, dtype=np.float64)
        ref_level_db = 0.0
        
        # Tonal components
        self._add_shaft_harmonics(amplitude_spectrum, params, ref_level_db)
        self._add_bpf_harmonics(amplitude_spectrum, params, ref_level_db)
        self._add_generator_harmonics(amplitude_spectrum, params, ref_level_db)
        
        # Equipment harmonics - capture base frequency
        equipment_base_freq = self._add_equipment_harmonics(
            amplitude_spectrum, params, ref_level_db, rng
        )
        params.equipment_base_freq = equipment_base_freq
        
        # Structural resonances - capture frequencies used
        resonance_freqs = self._add_structural_resonances(
            amplitude_spectrum, params, ref_level_db, rng
        )
        # Store in individual fields (pad with 0.0 if fewer than 3)
        params.resonance_freq_1 = resonance_freqs[0] if len(resonance_freqs) > 0 else 0.0
        params.resonance_freq_2 = resonance_freqs[1] if len(resonance_freqs) > 1 else 0.0
        params.resonance_freq_3 = resonance_freqs[2] if len(resonance_freqs) > 2 else 0.0
        
        # Broadband flow noise
        self._add_broadband(amplitude_spectrum, params, ref_level_db)
        
        # NOTE: Cavitation is handled separately via burst model in generate()
        
        # Random phases
        phases = rng.uniform(0, 2 * np.pi, self.n_bins)
        phases[0] = 0.0
        if self.n_samples % 2 == 0:
            phases[-1] = 0.0
        
        complex_spectrum = amplitude_spectrum * np.exp(1j * phases)
        waveform = np.fft.irfft(complex_spectrum, n=self.n_samples)
        
        return waveform
    
    # =========================================================================
    # PHYSICAL BURST MODEL FOR CAVITATION
    # =========================================================================
    
    def _generate_burst_times(self, bpf: float, intensity: float,
                              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate cavitation burst occurrence times with blade passage gating
        and realistic chaos/randomness.
        
        Three Chaos Parameters for realistic horizontal streaking:
        1. Intensity Variation: ±20% per blade (wake non-uniformity)
        2. RPM Jitter: ±2% timing noise (engine governor hunting)
        3. Envelope Wander: 0.5 Hz swell modulation (sea state effect)
        
        This creates natural-looking spectrograms with horizontal bands
        instead of synthetic-looking vertical barcodes.
        """
        blade_period = 1.0 / bpf
        # Add buffer passages to handle jitter at edges
        n_passages = int(self.duration * bpf) + 2
        
        # Physics: Activity window (20-50% of blade period)
        activity_fraction = 0.2 + 0.3 * intensity
        base_bursts = int(3 + 12 * intensity)
        
        burst_times = []
        burst_intensities = []
        
        # CHAOS PARAMETER 3: Envelope Wander (0.5 Hz swell/roll)
        # Random phase so every clip is different
        swell_phase = rng.uniform(0, 2 * np.pi)
        
        for i in range(n_passages):
            # CHAOS PARAMETER 2: RPM Jitter (±2% frequency noise)
            # Applied by jittering the center time of blade pass
            jitter_amount = 0.02 * blade_period
            timing_jitter = rng.normal(0, jitter_amount)
            
            # Nominal center + jitter
            t_center = (i + 0.5) * blade_period + timing_jitter
            
            # CHAOS PARAMETER 1: Intensity Variation per Blade (±20%)
            # Some blades hit harder water than others (wake non-uniformity)
            blade_randomness = max(0, 1.0 + 0.2 * rng.standard_normal())
            
            # Calculate local envelope wander (0.5 Hz drift) for this time
            swell_mod = 1.0 + 0.3 * np.sin(2 * np.pi * 0.5 * t_center + swell_phase)
            
            # Combined intensity for this blade passage
            local_intensity = intensity * blade_randomness * swell_mod
            
            # Define the "Active Window" for this blade pass
            window_start = t_center - activity_fraction * blade_period / 2
            window_end = t_center + activity_fraction * blade_period / 2
            
            # Generate the specific burst events (Poisson distributed)
            n_bursts = rng.poisson(base_bursts)
            
            for _ in range(n_bursts):
                t_burst = rng.uniform(window_start, window_end)
                
                # Only keep bursts within the clip duration
                if 0 < t_burst < self.duration:
                    burst_times.append(t_burst)
                    # Gamma distribution for bubble size, scaled by chaos factors
                    burst_intensities.append(rng.gamma(2, 0.5) * local_intensity)
        
        return np.array(burst_times), np.array(burst_intensities)
    
    def _generate_single_burst(self, t: np.ndarray, t_start: float,
                                burst_duration: float, f_carrier: float,
                                burst_type: str) -> np.ndarray:
        """
        Generate a single cavitation burst waveform.
        
        Burst types:
        - 'collapse': Sharp attack, exponential decay (single bubble) ~10-100 μs
        - 'cloud': Longer, complex envelope (cloud collapse) ~100 μs - 1 ms
        - 'sheet': Oscillatory, quasi-periodic (sheet cavity) ~1-10 ms
        """
        burst = np.zeros_like(t)
        t_rel = t - t_start
        mask = (t_rel >= 0) & (t_rel <= burst_duration * 3)
        
        if not np.any(mask):
            return burst
        
        t_local = t_rel[mask]
        
        if burst_type == 'collapse':
            tau = burst_duration / 3
            envelope = (t_local / tau) * np.exp(-t_local / tau)
            
        elif burst_type == 'cloud':
            tau = burst_duration / 2
            envelope = np.exp(-((t_local - tau) / tau) ** 2)
            envelope += 0.3 * np.exp(-((t_local - 2*tau) / (tau/2)) ** 2)
            
        elif burst_type == 'sheet':
            tau = burst_duration
            envelope = np.sin(np.pi * t_local / burst_duration) ** 2
            envelope *= np.exp(-t_local / (2 * burst_duration))
        
        else:
            envelope = np.exp(-t_local / burst_duration)
        
        # High-frequency carrier with harmonics
        carrier = np.sin(2 * np.pi * f_carrier * t_local)
        carrier += 0.3 * np.sin(2 * np.pi * 2 * f_carrier * t_local)
        carrier += 0.1 * np.sin(2 * np.pi * 3 * f_carrier * t_local)
        
        burst[mask] = envelope * carrier
        
        return burst
    
    def _generate_cavitation_bursts(self, params: VesselParams,
                                     rng: np.random.Generator) -> Tuple[np.ndarray, float, int]:
        """
        Generate cavitation signal with physical burst model.
        
        Generated at 200 kHz to capture μs-scale bubble collapse,
        then downsampled to output sample rate.
        
        Returns:
            Tuple of (signal, cavitation_peak_freq_used, n_bursts)
        """
        if not params.has_cavitation or params.cavitation_intensity < 0.01:
            return np.zeros(self.n_samples), 0.0, 0
        
        # Time array at high sample rate
        t_gen = np.linspace(0, self.duration, self.n_samples_gen)
        signal_gen = np.zeros(self.n_samples_gen)
        
        # Get vessel-specific cavitation parameters
        vc = VESSEL_CLASSES.get(params.vessel_class, VESSEL_CLASSES['cargo_ship'])
        cav_peak = vc.get('cavitation_peak', 1000.0)
        
        # Track the actual peak frequencies used (weighted average)
        peak_freqs_used = []
        peak_weights = []
        
        # Generate burst times
        burst_times, burst_intensities = self._generate_burst_times(
            params.blade_pass_freq, params.cavitation_intensity, rng
        )
        
        n_bursts = len(burst_times)
        
        # Generate each burst
        for t_burst, intensity in zip(burst_times, burst_intensities):
            # Random burst type (weighted)
            r = rng.random()
            if r < 0.6:
                burst_type = 'collapse'
                R_bubble = rng.lognormal(np.log(50e-6), 0.5)
                burst_duration = 0.915 * R_bubble * np.sqrt(RHO_WATER / DELTA_P)
                f_carrier = cav_peak * rng.uniform(0.8, 2.0)
            elif r < 0.9:
                burst_type = 'cloud'
                burst_duration = rng.uniform(0.0005, 0.002)
                f_carrier = cav_peak * rng.uniform(0.5, 1.5)
            else:
                burst_type = 'sheet'
                burst_duration = rng.uniform(0.001, 0.005)
                f_carrier = cav_peak * rng.uniform(0.3, 1.0)
            
            # Track frequencies for metadata
            peak_freqs_used.append(f_carrier)
            peak_weights.append(intensity)
            
            burst = self._generate_single_burst(
                t_gen, t_burst, burst_duration, f_carrier, burst_type
            )
            
            signal_gen += intensity * params.cavitation_intensity * burst
        
        # Calculate weighted average of peak frequencies used
        if len(peak_freqs_used) > 0 and sum(peak_weights) > 0:
            avg_peak_freq = np.average(peak_freqs_used, weights=peak_weights)
        else:
            avg_peak_freq = cav_peak
        
        # Downsample from 200 kHz to output fs
        # Use proper anti-aliasing filter
        downsample_factor = self.fs_gen // self.fs
        
        if downsample_factor > 1:
            # Anti-alias filter before downsampling
            nyq_out = self.fs / 2
            cutoff = 0.9 * nyq_out / (self.fs_gen / 2)
            b, a = scipy_signal.butter(8, cutoff, btype='low')
            signal_filtered = scipy_signal.filtfilt(b, a, signal_gen)
            
            # Decimate
            signal_out = scipy_signal.resample_poly(signal_filtered, 1, downsample_factor)
            
            # Ensure correct length
            if len(signal_out) > self.n_samples:
                signal_out = signal_out[:self.n_samples]
            elif len(signal_out) < self.n_samples:
                signal_out = np.pad(signal_out, (0, self.n_samples - len(signal_out)))
        else:
            signal_out = signal_gen
        
        # SPECTRAL NULLING: Remove energy below MIN_FREQ (10 Hz)
        # The impulsive burst model creates low-frequency spectral leakage
        # that must be removed to respect the 10 Hz - 8000 Hz prototype band
        # Use spectral nulling for a clean brick-wall cutoff
        MIN_FREQ = 10.0
        spectrum = np.fft.rfft(signal_out)
        freqs = np.fft.rfftfreq(len(signal_out), 1/self.fs)
        spectrum[freqs < MIN_FREQ] = 0.0
        signal_out = np.fft.irfft(spectrum, n=len(signal_out))
        
        return signal_out, round(avg_peak_freq, 2), n_bursts
    
    # =========================================================================
    # MAIN GENERATION METHOD
    # =========================================================================
    
    def generate(self, vessel_class: str = 'cargo_ship',
                 params: Optional[VesselParams] = None,
                 rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, VesselParams]:
        """
        Generate ship noise waveform using hybrid approach:
        1. Spectral synthesis for tonal + broadband + equipment + resonances
        2. Physical burst model for cavitation (reduced intensity)
        
        Args:
            vessel_class: Type of vessel
            params: Pre-defined vessel parameters (optional)
            rng: Random number generator
            
        Returns:
            Tuple of (waveform, params)
            
        Note: params object is updated with actual values used:
            - params.equipment_base_freq: 25.0 or 30.0 Hz
            - params.resonance_freq_1/2/3: actual resonance frequencies
            - params.cavitation_peak_freq: weighted average of burst carriers
            - params.n_cavitation_bursts: number of bursts in this clip
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if params is None:
            params = self.create_vessel_params(vessel_class, rng)
        
        # Generate tonal + broadband (spectral synthesis)
        # This also populates params.equipment_base_freq and params.resonance_freq_*
        tonal_broadband = self._generate_tonal_broadband(params, rng)
        
        # Generate cavitation bursts (physical model)
        # This also returns the cavitation peak frequency used and burst count
        cavitation, cav_peak_used, n_bursts = self._generate_cavitation_bursts(params, rng)
        cavitation = cavitation * self.CAVITATION_GAIN
        params.cavitation_peak_freq = cav_peak_used
        params.n_cavitation_bursts = n_bursts
        
        # Combine components - NO NORMALIZATION
        waveform = tonal_broadband + cavitation
        
        return waveform, params


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_ship_noise():
    """Test the ship noise generator with burst cavitation."""
    import matplotlib.pyplot as plt
    
    gen = ShipNoiseGenerator()
    rng = np.random.default_rng(42)
    
    # Test cargo ship with cavitation
    params = gen.create_vessel_params('cargo_ship', rng)
    params.has_cavitation = True
    params.cavitation_intensity = 0.6
    
    waveform, params = gen.generate(params=params, rng=rng)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Time domain
    t_ms = np.arange(len(waveform)) / gen.fs * 1000
    axes[0].plot(t_ms[:3200], waveform[:3200])
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Cargo Ship with Cavitation (BPF={params.blade_pass_freq:.1f} Hz)')
    axes[0].grid(True, alpha=0.3)
    
    # Mark blade passages
    blade_period_ms = 1000 / params.blade_pass_freq
    for i in range(int(200 / blade_period_ms) + 1):
        axes[0].axvline(i * blade_period_ms, color='red', linestyle='--', alpha=0.3)
    
    # Spectrum
    spectrum = np.abs(np.fft.rfft(waveform))
    freqs = gen.freq_grid
    spectrum_db = 20 * np.log10(spectrum + 1e-12)
    
    axes[1].semilogx(freqs[1:], spectrum_db[1:], alpha=0.7)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_title('Spectrum')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([1, 8000])
    
    # Spectrogram
    f, t, Sxx = scipy_signal.spectrogram(waveform, fs=gen.fs, nperseg=256, noverlap=240)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    
    im = axes[2].pcolormesh(t * 1000, f, Sxx_db, shading='gouraud', cmap='inferno')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_title('Spectrogram - Look for vertical streaks (burst events)')
    axes[2].set_ylim([0, 2000])
    plt.colorbar(im, ax=axes[2], label='PSD (dB)')
    
    plt.tight_layout()
    plt.savefig('ship_noise_burst_test.png', dpi=150)
    plt.close()
    
    print("Ship noise burst test complete. Output: ship_noise_burst_test.png")
    print(f"  Shaft rate:           {params.shaft_rate:.2f} Hz")
    print(f"  BPF:                  {params.blade_pass_freq:.2f} Hz")
    print(f"  Cavitation intensity: {params.cavitation_intensity:.2f}")
    print(f"  Cavitation peak freq: {params.cavitation_peak_freq:.2f} Hz")
    print(f"  N cavitation bursts:  {params.n_cavitation_bursts}")
    print(f"  Equipment base freq:  {params.equipment_base_freq:.2f} Hz")
    print(f"  Resonance freqs:      {params.resonance_freq_1:.1f}, {params.resonance_freq_2:.1f}, {params.resonance_freq_3:.1f} Hz")


if __name__ == '__main__':
    test_ship_noise()
