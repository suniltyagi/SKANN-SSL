# =============================================================================
# 11. Spectrogram Visualization (All Vessel Classes) - FINAL INFOGRAPHIC
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import copy
import sys

# Ensure your code folder is in the path
# sys.path.insert(0, '/content/drive/MyDrive/SKANN_SSL/code')

from ship_noise import ShipNoiseGenerator
from sea_noise import SeaNoiseGenerator
from config import VESSEL_CLASSES, FS

ship_gen = ShipNoiseGenerator()
sea_gen = SeaNoiseGenerator()
rng = np.random.default_rng(42)

# Vessel-specific settings
vessel_settings = {
    'tanker': {'ylim': 800},
    'cargo_ship': {'ylim': 1000},
    'fishing_vessel': {'ylim': 2500},
    'small_craft': {'ylim': 6000},
}

vessel_classes = ['tanker', 'cargo_ship', 'fishing_vessel', 'small_craft']

fig, axes = plt.subplots(4, 2, figsize=(14, 18))

for i, vc in enumerate(vessel_classes):
    settings = vessel_settings[vc]
    
    # Generate params ONCE for this vessel class (same for both columns)
    base_params = ship_gen.create_vessel_params(vc, rng)
    
    for j, has_cav in enumerate([False, True]):
        ax = axes[i, j]
        
        # Clone params and set cavitation flag
        params = copy.deepcopy(base_params)
        params.has_cavitation = has_cav
        if has_cav:
            params.cavitation_intensity = 0.6
        else:
            params.cavitation_intensity = 0.0
        
        # Generate ship noise (use fresh rng state for each to avoid correlation)
        ship_rng = np.random.default_rng(42 + i * 10 + j)
        ship_wave, _ = ship_gen.generate(params=params, rng=ship_rng)
        
        # Generate sea noise (SS3)
        sea_rng = np.random.default_rng(100 + i * 10 + j)
        sea_wave = sea_gen.generate_frame(sea_state=3, rng=sea_rng)
        sea_rms = np.sqrt(np.mean(sea_wave**2))
        
        # Scale ship to 6 dB above sea
        ship_rms = np.sqrt(np.mean(ship_wave**2))
        target_rms = sea_rms * 10**(6/20)
        
        # Avoid divide by zero if ship is silent
        if ship_rms > 0:
            ship_scaled = ship_wave * (target_rms / ship_rms)
        else:
            ship_scaled = ship_wave
        
        # Combine
        combined = ship_scaled + sea_wave
        
        # Spectrogram with horizontal streak parameters (LOFAR settings)
        nperseg = 2048
        noverlap = int(nperseg * 0.90)
        
        f, t, Sxx = signal.spectrogram(combined, fs=FS, 
                                       nperseg=nperseg, 
                                       noverlap=noverlap, 
                                       window='hann')
        
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        
        # Dynamic color scaling
        vmin = np.percentile(Sxx_db, 5)
        vmax = np.percentile(Sxx_db, 99)
        
        im = ax.pcolormesh(t * 1000, f, Sxx_db, shading='gouraud', 
                           cmap='jet', vmin=vmin, vmax=vmax)
        
        # Text annotation for cavitation (no confusing dashed lines)
        if has_cav:
            cav_peak = VESSEL_CLASSES[vc].get('cavitation_peak', 1000)
            ax.annotate(f'Cav ~{cav_peak:.0f} Hz', 
                        xy=(900, cav_peak), 
                        fontsize=8, color='white', 
                        fontweight='bold',
                        ha='right',
                        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
        
        cav_str = "WITH Cavitation" if has_cav else "NO Cavitation"
        ax.set_title(f'{vc.upper()} - {cav_str}\nShaft={params.shaft_rate:.1f}Hz, BPF={params.blade_pass_freq:.1f}Hz', 
                     fontsize=10)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim([0, settings['ylim']])
        
        plt.colorbar(im, ax=ax, label='PSD (dB)', shrink=0.8)

plt.suptitle('SKANN-SSL Stage -1: Synthetic Ship Noise Spectrograms\n(Sea State 3, SNR = 6 dB)', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('spectrogram_infographic.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: spectrogram_infographic.png")