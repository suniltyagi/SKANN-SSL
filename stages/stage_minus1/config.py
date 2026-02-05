"""
SKANN-SSL Stage -1 Configuration
================================
Centralized parameters for synthetic waveform generation.

PROTOTYPE CONSTRAINT: Frequency band limited to 10 Hz - 8000 Hz
- No simulation below 10 Hz (avoids turbulence extrapolation issues)
- Upper limit is Nyquist (8000 Hz at 16 kHz sample rate)
"""

import numpy as np

# =============================================================================
# SIGNAL PARAMETERS
# =============================================================================
FS = 16000              # Sampling frequency (Hz) - Prototype
FS_FULL = 32000         # Sampling frequency (Hz) - Full system
DURATION = 5.0          # Clip duration (seconds)
N_SAMPLES = int(FS * DURATION)  # Samples per clip (80000 for 5s @ 16kHz)

# FFT parameters
FFT_SIZE = N_SAMPLES    # FFT size matches clip length for Δf = 1 Hz
FREQ_RES = FS / FFT_SIZE  # Frequency resolution (1 Hz)
NYQUIST = FS // 2       # Nyquist frequency (8000 Hz)

# Frequency band constraints (Prototype)
MIN_FREQ = 10.0         # Minimum frequency (Hz) - no simulation below this
MAX_FREQ = NYQUIST      # Maximum frequency (Hz) - 8000 Hz

# =============================================================================
# SEA NOISE PARAMETERS (Knudsen Model)
# =============================================================================
SEA_STATES = [0, 1, 3, 6]  # Available sea states

# Piecewise model band boundaries (Hz) - Corrected per Document B
TURB_START = 10.0       # Turbulence band start
TURB_END = 37.7         # Turbulence-LF intersection (calculated from coefficients)
LF_END = 200.0          # LF-MF boundary
MF_END = 500.0          # MF-HF boundary
HF_END = 100000.0       # HF band end (model limit)

# Turbulence model anchor points (from Document B)
TURB_ANCHOR_1 = (1.0, 108.0)    # (Hz, dB)
TURB_ANCHOR_2 = (40.0, 55.0)    # (Hz, dB)

# Default coefficients for SS3 (will be computed from CSV for all states)
# Format: (a_slope, b_intercept) for NL(f) = a*log10(f) + b
SS3_COEFFS = {
    'turb': (-33.10, 108.0),
    'lf': (9.09, 45.28),
    'mf': (-6.57, 83.15),
    'hf': (-17.11, 113.73),
    'f_t': 30.7
}

# =============================================================================
# SHIP NOISE PARAMETERS
# =============================================================================
# SNR for ship-to-sea-noise ratio
SHIP_SNR_DB = 6.0       # Ship signal 6 dB above sea noise

# Vessel classes with characteristic parameters
# Based on realistic propeller RPM and physics:
# - Large slow ships (tanker/cargo): 60-150 RPM, huge props, low-freq cavitation
# - Medium vessels (fishing): 240-480 RPM, mid-freq cavitation
# - Small fast craft: 900-1800 RPM, small props, high-freq cavitation
VESSEL_CLASSES = {
    'small_craft': {
        'f0_range': (15.0, 30.0),       # Shaft rate (Hz): 900-1800 RPM
        'n_blades': (2, 3),              # Propeller blades
        'n_harmonics': 8,                # Number of shaft harmonics
        'n_bpf_harmonics': 4,            # Number of BPF harmonics
        'harmonic_decay': 3.0,           # dB per harmonic
        'bpf_decay': 4.0,                # dB per BPF harmonic
        'broadband_level': 0.3,          # Relative broadband level
        'broadband_rolloff': -3.0,       # dB/octave rolloff
        'cavitation_peak': 5000.0,       # Cavitation hump center (Hz)
    },
    'fishing_vessel': {
        'f0_range': (4.0, 8.0),          # Shaft rate (Hz): 240-480 RPM
        'n_blades': (3, 4),
        'n_harmonics': 10,
        'n_bpf_harmonics': 5,
        'harmonic_decay': 3.0,
        'bpf_decay': 4.0,
        'broadband_level': 0.4,
        'broadband_rolloff': -4.0,
        'cavitation_peak': 1500.0,       # Cavitation hump center (Hz)
    },
    'cargo_ship': {
        'f0_range': (1.5, 2.5),          # Shaft rate (Hz): 90-150 RPM
        'n_blades': (4, 5),
        'n_harmonics': 12,
        'n_bpf_harmonics': 6,
        'harmonic_decay': 3.0,
        'bpf_decay': 4.0,
        'broadband_level': 0.5,
        'broadband_rolloff': -5.0,
        'cavitation_peak': 600.0,        # Cavitation hump center (Hz): "thumper"
    },
    'tanker': {
        'f0_range': (1.0, 1.5),          # Shaft rate (Hz): 60-90 RPM
        'n_blades': (4, 6),
        'n_harmonics': 15,
        'n_bpf_harmonics': 8,
        'harmonic_decay': 3.0,
        'bpf_decay': 4.0,
        'broadband_level': 0.6,
        'broadband_rolloff': -6.0,
        'cavitation_peak': 400.0,        # Cavitation hump center (Hz): lowest
    }
}

# Tonal component parameters
TONAL_PHASE_JITTER = 0.1    # Phase jitter (radians)
TONAL_FREQ_JITTER = 0.02    # Frequency jitter (fraction)
TONAL_AMP_JITTER = 0.1      # Amplitude jitter (fraction)

# Generator harmonics (50/60 Hz)
GENERATOR_FREQS = [50.0, 100.0, 150.0, 60.0, 120.0, 180.0]
GENERATOR_PROB = 0.7        # Probability of including generator tones

# =============================================================================
# CAVITATION PARAMETERS
# =============================================================================
CAVITATION_PROB = 0.5       # Probability of cavitation
CAVITATION_INTENSITY_RANGE = (0.2, 0.8)
CAVITATION_MOD_FREQ_RANGE = (5.0, 20.0)  # AM modulation frequency (Hz)
CAVITATION_BANDS = [
    {'f_low': 300, 'f_high': 800, 'weight': 0.3},      # Bubble collapse
    {'f_low': 1500, 'f_high': 3000, 'weight': 0.5},    # Mid-range peak
    {'f_low': 4000, 'f_high': 7000, 'weight': 0.2},    # High-frequency component
]

# =============================================================================
# BROADBAND NOISE PARAMETERS
# =============================================================================
FLOW_NOISE_ROLLOFF = -5.0   # dB/octave
FLOW_NOISE_REF_FREQ = 1000.0  # Reference frequency for rolloff

# =============================================================================
# DATASET GENERATION PARAMETERS (Random Sampling)
# =============================================================================
N_SYNTHETIC_CLIPS = 500     # Number of clips for prototype
RANDOM_SEED = 42            # For reproducibility

# Output paths
DATA_DIR = './data'
WAVEFORM_DIR = f'{DATA_DIR}/waveforms'
TENSOR_DIR = f'{DATA_DIR}/tensors'
METADATA_FILE = f'{DATA_DIR}/metadata.csv'

# =============================================================================
# V3.0.0: NO-VESSEL CLASS PARAMETERS
# =============================================================================
NO_VESSEL_CLASS_NAME = 'no_vessel'
NO_VESSEL_REPS_PER_SEA_STATE = 120  # 120 × 4 sea states = 480 clips (matches vessel class count)
NO_VESSEL_START_CLIP_ID = 1920     # Continue numbering after vessel clips

# =============================================================================
# FULL-FACTORIAL DESIGN PARAMETERS (Structured Dataset)
# =============================================================================
SEA_STATES_DESIGN = [0, 1, 3, 6]
VESSEL_CLASSES_DESIGN = ['small_craft', 'fishing_vessel', 'cargo_ship', 'tanker']
N_BLADES_OPTIONS = [3, 4, 5]
GEN_FREQ_OPTIONS = [0.0, 50.0]
CAV_INTENSITY_LEVELS = [0.0, 0.3333, 0.6667, 1.0]
FULL_FACTORIAL_REPS = 5

# Full-factorial output directory
STRUCTURED_DATA_DIR = './structured_dataset'

# Total clips: 4 × 4 × 3 × 2 × 4 × 5 = 1920

# =============================================================================
# OVERLAP-ADD PARAMETERS (for multi-second synthesis)
# =============================================================================
OLA_OVERLAP = 0.5           # 50% overlap
OLA_WINDOW = 'hann'         # Window type

# =============================================================================
# REFERENCE LEVELS
# =============================================================================
P_REF = 1e-6                # Reference pressure (1 µPa)
