"""
SKANN-SSL Stage -1: Full-Factorial Dataset Generator
====================================================

Generates a structured dataset over:

- Sea state:           SEA_STATES_DESIGN (0,1,3,6)
- Vessel class:        VESSEL_CLASSES_DESIGN
- Blade count:         N_BLADES_OPTIONS (3,4,5)
- Generator frequency: GEN_FREQ_OPTIONS (0,50)
- Cavitation level:    CAV_INTENSITY_LEVELS (0, 0.3333, 0.6667, 1.0)
- Repetitions:         FULL_FACTORIAL_REPS

Total clips: 4 × 4 × 3 × 2 × 4 × 5 = 1920

Uses:
- sea_noise.SeaNoiseGenerator
- ship_noise.ShipNoiseGenerator
- config.py for design and paths

Metadata captures:
- Design factors (controlled inputs)
- Derived values (BPF, actual frequencies used)
- Acoustic measurements (RMS, SPL, SNR)
- Randomized components (resonances, cavitation peak, burst count)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from config import (
    FS, P_REF, SHIP_SNR_DB, DATA_DIR,
    SEA_STATES_DESIGN, VESSEL_CLASSES_DESIGN,
    N_BLADES_OPTIONS, CAV_INTENSITY_LEVELS,
    GEN_FREQ_OPTIONS, FULL_FACTORIAL_REPS,
    STRUCTURED_DATA_DIR
)
from sea_noise import SeaNoiseGenerator
from ship_noise import ShipNoiseGenerator


class FullFactorialDatasetGenerator:
    """
    Full-factorial dataset generator for SKANN-SSL Stage -1.
    
    Generates all combinations of design factors with controlled
    randomization for repetitions. Each clip includes:
    - Sea noise at specified sea state
    - Ship noise with specified vessel class, blade count, generator, cavitation
    - Ship scaled to achieve target SNR (6 dB above sea)
    
    Metadata captures both design factors and actual values used
    (including randomized resonance frequencies and cavitation parameters).
    """

    def __init__(self,
                 output_dir: Optional[str] = None,
                 reps: Optional[int] = None):
        """
        Args:
            output_dir: Output directory for structured dataset. If None, uses STRUCTURED_DATA_DIR.
            reps: Number of repetitions per combination. If None, uses FULL_FACTORIAL_REPS.
        """
        self.output_dir = Path(output_dir) if output_dir is not None else Path(STRUCTURED_DATA_DIR)
        self.waveform_dir = self.output_dir / 'waveforms'
        self.reps = reps if reps is not None else FULL_FACTORIAL_REPS

        self.sea_gen = SeaNoiseGenerator()
        self.ship_gen = ShipNoiseGenerator()

    def _ensure_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.waveform_dir.mkdir(exist_ok=True)

    def generate(self, verbose: bool = True) -> pd.DataFrame:
        """
        Generate the full-factorial dataset.

        Returns:
            pandas.DataFrame with metadata for all generated clips.
        """
        self._ensure_dirs()

        metadata_rows = []
        clip_id = 0

        # Calculate total clips
        total_clips = (len(SEA_STATES_DESIGN) * 
                       len(VESSEL_CLASSES_DESIGN) * 
                       len(N_BLADES_OPTIONS) * 
                       len(GEN_FREQ_OPTIONS) * 
                       len(CAV_INTENSITY_LEVELS) * 
                       self.reps)

        if verbose:
            print("Full-Factorial Dataset Generation")
            print("=" * 50)
            print(f"  Sea states:      {SEA_STATES_DESIGN}")
            print(f"  Vessel classes:  {VESSEL_CLASSES_DESIGN}")
            print(f"  Blades options:  {N_BLADES_OPTIONS}")
            print(f"  Generator freqs: {GEN_FREQ_OPTIONS}")
            print(f"  Cav levels:      {CAV_INTENSITY_LEVELS}")
            print(f"  Repetitions:     {self.reps}")
            print(f"  Total clips:     {total_clips}")
            print()

        for ss in SEA_STATES_DESIGN:
            for vc in VESSEL_CLASSES_DESIGN:
                for n_blades in N_BLADES_OPTIONS:
                    for gen_freq in GEN_FREQ_OPTIONS:
                        for cav_int in CAV_INTENSITY_LEVELS:
                            for rep in range(self.reps):
                                # Deterministic seed for reproducibility
                                seed_base = 10_000 + clip_id
                                rng_main = np.random.default_rng(seed_base)

                                # ---------------------------------------------
                                # Sea Noise
                                # ---------------------------------------------
                                sea_rng = np.random.default_rng(seed_base + 1)
                                sea_wave = self.sea_gen.generate_frame(sea_state=ss, rng=sea_rng)
                                sea_rms = np.sqrt(np.mean(sea_wave**2))

                                # ---------------------------------------------
                                # Ship Params: Start random, override design factors
                                # ---------------------------------------------
                                params = self.ship_gen.create_vessel_params(vc, rng_main)
                                
                                # Override with design factors
                                params.n_blades = n_blades
                                params.blade_pass_freq = params.shaft_rate * params.n_blades
                                params.generator_freq = gen_freq

                                if cav_int <= 0.0:
                                    params.has_cavitation = False
                                    params.cavitation_intensity = 0.0
                                else:
                                    params.has_cavitation = True
                                    params.cavitation_intensity = cav_int

                                # ---------------------------------------------
                                # Generate Ship Noise
                                # ---------------------------------------------
                                ship_rng = np.random.default_rng(seed_base + 2)
                                ship_raw, params = self.ship_gen.generate(params=params, rng=ship_rng)
                                # params now has populated: equipment_base_freq, resonance_freq_*, 
                                # cavitation_peak_freq, n_cavitation_bursts

                                # ---------------------------------------------
                                # Scale Ship to Target SNR
                                # ---------------------------------------------
                                current_ship_rms = np.sqrt(np.mean(ship_raw**2))
                                target_ship_rms = sea_rms * (10 ** (SHIP_SNR_DB / 20))

                                if current_ship_rms > 0:
                                    scale_factor = target_ship_rms / current_ship_rms
                                    ship_wave = ship_raw * scale_factor
                                else:
                                    ship_wave = ship_raw
                                    scale_factor = 1.0

                                ship_rms = np.sqrt(np.mean(ship_wave**2))

                                # ---------------------------------------------
                                # Combine
                                # ---------------------------------------------
                                combined = sea_wave + ship_wave
                                combined_rms = np.sqrt(np.mean(combined**2))

                                # ---------------------------------------------
                                # Compute SPLs and SNR
                                # ---------------------------------------------
                                sea_spl = 20 * np.log10(sea_rms / P_REF + 1e-30)
                                ship_spl = 20 * np.log10(ship_rms / P_REF + 1e-30)
                                comb_spl = 20 * np.log10(combined_rms / P_REF + 1e-30)
                                actual_snr = 20 * np.log10(ship_rms / sea_rms + 1e-30)

                                # ---------------------------------------------
                                # Save Waveform
                                # ---------------------------------------------
                                wf_path = self.waveform_dir / f'clip_{clip_id:06d}.npy'
                                np.save(wf_path, combined.astype(np.float32))

                                # ---------------------------------------------
                                # Metadata Row
                                # ---------------------------------------------
                                row = {
                                    # Identifiers
                                    'clip_id': clip_id,
                                    'repeat_index': rep,
                                    
                                    # Design factors (controlled inputs)
                                    'sea_state': ss,
                                    'vessel_class': vc,
                                    'n_blades': n_blades,
                                    'generator_freq': gen_freq,
                                    'cavitation_intensity': cav_int,
                                    
                                    # Derived propulsion parameters
                                    'shaft_rate': round(params.shaft_rate, 4),
                                    'blade_pass_freq': round(params.blade_pass_freq, 4),
                                    
                                    # Cavitation details
                                    'has_cavitation': params.has_cavitation,
                                    'cavitation_peak_freq': params.cavitation_peak_freq,
                                    'n_cavitation_bursts': params.n_cavitation_bursts,
                                    
                                    # Equipment and resonances (randomized per clip)
                                    'equipment_base_freq': params.equipment_base_freq,
                                    'resonance_freq_1': params.resonance_freq_1,
                                    'resonance_freq_2': params.resonance_freq_2,
                                    'resonance_freq_3': params.resonance_freq_3,
                                    
                                    # Acoustic measurements
                                    'sea_rms_pa': sea_rms,
                                    'ship_rms_pa': ship_rms,
                                    'combined_rms_pa': combined_rms,
                                    'scale_factor': scale_factor,
                                    
                                    'sea_spl_db': round(sea_spl, 2),
                                    'ship_spl_db': round(ship_spl, 2),
                                    'combined_spl_db': round(comb_spl, 2),
                                    'snr_db': round(actual_snr, 4),
                                }
                                metadata_rows.append(row)

                                clip_id += 1

                                if verbose and clip_id % 100 == 0:
                                    print(f"  Generated {clip_id}/{total_clips} clips...")

        # ---------------------------------------------
        # Save Metadata
        # ---------------------------------------------
        df = pd.DataFrame(metadata_rows)
        metadata_path = self.output_dir / 'metadata.csv'
        df.to_csv(metadata_path, index=False)

        if verbose:
            print("\nGeneration complete.")
            print(f"  Total clips: {len(df)}")
            print(f"  Output dir:  {self.output_dir}")
            print(f"  Waveforms:   {self.waveform_dir}")
            print(f"  Metadata:    {metadata_path}")

        return df


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == '__main__':
    generator = FullFactorialDatasetGenerator()
    df = generator.generate(verbose=True)
    
    print("\n" + "=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    print(f"\nBy Sea State:\n{df['sea_state'].value_counts().sort_index()}")
    print(f"\nBy Vessel Class:\n{df['vessel_class'].value_counts()}")
    print(f"\nBy Blade Count:\n{df['n_blades'].value_counts().sort_index()}")
    print(f"\nBy Generator Freq:\n{df['generator_freq'].value_counts().sort_index()}")
    print(f"\nBy Cavitation Intensity:\n{df['cavitation_intensity'].value_counts().sort_index()}")
    print(f"\nSNR Statistics:")
    print(f"  Mean: {df['snr_db'].mean():.4f} dB")
    print(f"  Std:  {df['snr_db'].std():.6f} dB")
    print(f"\nCavitation Bursts (when has_cavitation=True):")
    cav_clips = df[df['has_cavitation']]
    if len(cav_clips) > 0:
        print(f"  Mean bursts: {cav_clips['n_cavitation_bursts'].mean():.1f}")
        print(f"  Min bursts:  {cav_clips['n_cavitation_bursts'].min()}")
        print(f"  Max bursts:  {cav_clips['n_cavitation_bursts'].max()}")
