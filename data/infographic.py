"""
SKANN-SSL V5 Dataset Infographic Generator
===========================================
Generates spectrogram visualizations for the V5 synthetic underwater acoustic dataset.

Usage:
    cd data/
    python infographic.py

Output:
    - spectrogram_infographic.png (all vessel classes, with/without cavitation)
    - vessel_comparison.png (shaft rate and BPF comparison)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from pathlib import Path
import sys

# Add stage_minus1 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'stages' / 'stage_minus1'))

try:
    from ship_noise import ShipNoiseGenerator
    from sea_noise import SeaNoiseGenerator
    from config import VESSEL_CLASSES, FS, N_SAMPLES, DURATION
except ImportError:
    print("Error: Cannot import from stage_minus1. Run from data/ folder.")
    print("Alternatively, set PYTHONPATH to include stages/stage_minus1/")
    sys.exit(1)

# =============================================================================
# V5 CONFIGURATION
# =============================================================================

# V5 Non-overlapping shaft rate ranges
V5_SHAFT_RATES = {
    'tanker':         (1.0, 1.5),    # 60-90 RPM
    'cargo_ship':     (1.5, 2.5),    # 90-150 RPM
    'fishing_vessel': (4.0, 8.0),    # 240-480 RPM
    'small_craft':    (15.0, 30.0),  # 900-1800 RPM
}

# Vessel-specific spectrogram settings
VESSEL_SETTINGS = {
    'tanker':         {'ylim': 500,  'cav_peak': 400},
    'cargo_ship':     {'ylim': 800,  'cav_peak': 600},
    'fishing_vessel': {'ylim': 2000, 'cav_peak': 1500},
    'small_craft':    {'ylim': 5000, 'cav_peak': 5000},
}

VESSEL_ORDER = ['tanker', 'cargo_ship', 'fishing_vessel', 'small_craft']

# V5 Signal parameters
FS_V5 = 16000
DURATION_V5 = 5.0
N_SAMPLES_V5 = int(FS_V5 * DURATION_V5)  # 80,000
SNR_DB = 6.0


# =============================================================================
# SPECTROGRAM INFOGRAPHIC (4x2 Grid)
# =============================================================================

def generate_spectrogram_infographic(output_path='spectrogram_infographic.png'):
    """
    Generate 4x2 spectrogram grid showing all vessel classes with/without cavitation.
    """
    print("Generating spectrogram infographic...")
    
    ship_gen = ShipNoiseGenerator()
    sea_gen = SeaNoiseGenerator()
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    
    for i, vc in enumerate(VESSEL_ORDER):
        settings = VESSEL_SETTINGS[vc]
        shaft_range = V5_SHAFT_RATES[vc]
        
        # Generate params for this vessel class
        rng = np.random.default_rng(42 + i)
        base_params = ship_gen.create_vessel_params(vc, rng)
        
        for j, has_cav in enumerate([False, True]):
            ax = axes[i, j]
            
            # Clone params and set cavitation
            import copy
            params = copy.deepcopy(base_params)
            params.has_cavitation = has_cav
            params.cavitation_intensity = 0.6 if has_cav else 0.0
            
            # Generate ship noise
            ship_rng = np.random.default_rng(42 + i * 10 + j)
            ship_wave, final_params = ship_gen.generate(params=params, rng=ship_rng)
            
            # Generate sea noise (SS3)
            sea_rng = np.random.default_rng(100 + i * 10 + j)
            sea_wave = sea_gen.generate_frame(sea_state=3, rng=sea_rng)
            sea_rms = np.sqrt(np.mean(sea_wave**2))
            
            # Scale ship to 6 dB above sea
            ship_rms = np.sqrt(np.mean(ship_wave**2))
            if ship_rms > 0:
                target_rms = sea_rms * 10**(SNR_DB / 20)
                ship_scaled = ship_wave * (target_rms / ship_rms)
            else:
                ship_scaled = ship_wave
            
            # Combine
            combined = ship_scaled + sea_wave
            
            # Spectrogram (LOFAR-style settings for horizontal streaks)
            nperseg = 2048
            noverlap = int(nperseg * 0.90)
            
            f, t, Sxx = signal.spectrogram(
                combined, fs=FS_V5,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann'
            )
            
            Sxx_db = 10 * np.log10(Sxx + 1e-12)
            
            # Dynamic color scaling
            vmin = np.percentile(Sxx_db, 5)
            vmax = np.percentile(Sxx_db, 99)
            
            im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud',
                               cmap='jet', vmin=vmin, vmax=vmax)
            
            # Cavitation annotation
            if has_cav:
                cav_peak = settings['cav_peak']
                ax.annotate(
                    f'Cav ~{cav_peak:.0f} Hz',
                    xy=(DURATION_V5 - 0.3, cav_peak),
                    fontsize=9, color='white', fontweight='bold',
                    ha='right',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6)
                )
            
            # Title with V5 shaft rate info
            cav_str = "WITH Cavitation" if has_cav else "NO Cavitation"
            ax.set_title(
                f'{vc.upper()} - {cav_str}\n'
                f'Shaft={final_params.shaft_rate:.2f} Hz (V5 range: {shaft_range[0]}-{shaft_range[1]} Hz), '
                f'BPF={final_params.blade_pass_freq:.1f} Hz',
                fontsize=10
            )
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_ylim([0, settings['ylim']])
            
            plt.colorbar(im, ax=ax, label='PSD (dB)', shrink=0.8)
    
    plt.suptitle(
        'SKANN-SSL V5 Dataset: Synthetic Ship Noise Spectrograms\n'
        '(5.0s clips, Sea State 3, SNR = 6 dB, Non-overlapping Shaft Rates)',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


# =============================================================================
# VESSEL COMPARISON CHART
# =============================================================================

def generate_vessel_comparison(output_path='vessel_comparison.png'):
    """
    Generate bar chart comparing shaft rates and BPF ranges across vessel classes.
    """
    print("Generating vessel comparison chart...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Shaft rate ranges
    ax1 = axes[0]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for i, vc in enumerate(VESSEL_ORDER):
        sr_min, sr_max = V5_SHAFT_RATES[vc]
        ax1.barh(i, sr_max - sr_min, left=sr_min, height=0.6, 
                 color=colors[i], edgecolor='black', linewidth=1.5)
        ax1.text(sr_max + 0.5, i, f'{sr_min}-{sr_max} Hz', va='center', fontsize=10)
    
    ax1.set_yticks(range(len(VESSEL_ORDER)))
    ax1.set_yticklabels([vc.replace('_', ' ').title() for vc in VESSEL_ORDER])
    ax1.set_xlabel('Shaft Rate (Hz)')
    ax1.set_title('V5 Non-Overlapping Shaft Rate Ranges', fontsize=12, fontweight='bold')
    ax1.set_xlim([0, 35])
    ax1.grid(axis='x', alpha=0.3)
    
    # BPF ranges (shaft × blades)
    ax2 = axes[1]
    
    for i, vc in enumerate(VESSEL_ORDER):
        sr_min, sr_max = V5_SHAFT_RATES[vc]
        # BPF range with 3-5 blades
        bpf_min = sr_min * 3
        bpf_max = sr_max * 5
        ax2.barh(i, bpf_max - bpf_min, left=bpf_min, height=0.6,
                 color=colors[i], edgecolor='black', linewidth=1.5)
        ax2.text(bpf_max + 2, i, f'{bpf_min:.0f}-{bpf_max:.0f} Hz', va='center', fontsize=10)
    
    ax2.set_yticks(range(len(VESSEL_ORDER)))
    ax2.set_yticklabels([vc.replace('_', ' ').title() for vc in VESSEL_ORDER])
    ax2.set_xlabel('Blade Pass Frequency (Hz)')
    ax2.set_title('BPF Ranges (Shaft × 3-5 Blades)', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 160])
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle('SKANN-SSL V5: Vessel Class Acoustic Signatures', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


# =============================================================================
# DATASET DISTRIBUTION CHART
# =============================================================================

def generate_distribution_chart(manifest_path=None, output_path='dataset_distribution.png'):
    """
    Generate pie/bar charts showing dataset distribution from manifest.
    """
    print("Generating dataset distribution chart...")
    
    if manifest_path is None:
        manifest_path = Path(__file__).parent / 'v5_dataset' / 'master_dataset_manifest.csv'
    
    if not Path(manifest_path).exists():
        print(f"Warning: Manifest not found at {manifest_path}")
        print("Generating chart with expected V5 values...")
        
        # Use expected V5 values
        class_counts = {
            'small_craft': 2400,
            'fishing_vessel': 2400,
            'cargo_ship': 2400,
            'tanker': 2400,
            'no_vessel': 2400
        }
        sea_state_counts = {0: 3000, 1: 3000, 3: 3000, 6: 3000}
        total_clips = 12000
    else:
        df = pd.read_csv(manifest_path)
        class_counts = df['vessel_class'].value_counts().to_dict()
        sea_state_counts = df['sea_state'].value_counts().to_dict()
        total_clips = len(df)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Vessel class distribution
    ax1 = axes[0]
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    ax1.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors[:len(classes)],
            explode=[0.02] * len(classes))
    ax1.set_title(f'Vessel Class Distribution\n(Total: {total_clips:,} clips)', fontweight='bold')
    
    # Sea state distribution
    ax2 = axes[1]
    states = sorted(sea_state_counts.keys())
    state_counts = [sea_state_counts[s] for s in states]
    bars = ax2.bar([f'SS{s}' for s in states], state_counts, color='#3498db', edgecolor='black')
    ax2.set_xlabel('Sea State')
    ax2.set_ylabel('Number of Clips')
    ax2.set_title('Sea State Distribution', fontweight='bold')
    for bar, count in zip(bars, state_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{count:,}', ha='center', fontsize=10)
    
    # Full factorial breakdown
    ax3 = axes[2]
    factors = ['Sea States\n(4)', 'Vessel Classes\n(4)', 'Blade Counts\n(3)',
               'Generator Freq\n(2)', 'Cavitation\n(4)', 'Repetitions\n(25)']
    values = [4, 4, 3, 2, 4, 25]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    bars = ax3.bar(factors, values, color=colors, edgecolor='black')
    ax3.set_ylabel('Number of Levels')
    ax3.set_title('Full Factorial Design Factors\n(4×4×3×2×4×25 = 9,600 vessel clips)', fontweight='bold')
    ax3.set_ylim([0, 30])
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('SKANN-SSL V5 Dataset Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SKANN-SSL V5 Dataset Infographic Generator')
    parser.add_argument('--all', action='store_true', help='Generate all infographics')
    parser.add_argument('--spectrogram', action='store_true', help='Generate spectrogram grid')
    parser.add_argument('--comparison', action='store_true', help='Generate vessel comparison')
    parser.add_argument('--distribution', action='store_true', help='Generate distribution charts')
    parser.add_argument('--manifest', type=str, default=None, help='Path to manifest CSV')
    
    args = parser.parse_args()
    
    # Default to all if no specific option
    if not (args.spectrogram or args.comparison or args.distribution):
        args.all = True
    
    if args.all or args.spectrogram:
        generate_spectrogram_infographic()
    
    if args.all or args.comparison:
        generate_vessel_comparison()
    
    if args.all or args.distribution:
        generate_distribution_chart(manifest_path=args.manifest)
    
    print("\nDone! All infographics generated.")
