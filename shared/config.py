"""
SKANN-SSL Shared Configuration
==============================
Global constants used across all stages.

Import:
    from shared.config import SAMPLE_RATE, VESSEL_CLASSES
"""

# =============================================================================
# Audio Parameters
# =============================================================================

SAMPLE_RATE = 16000          # Hz
DURATION_SEC = 1.0           # seconds per clip
N_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)  # 16000

# Frequency constraints
FREQ_MIN = 10                # Hz (no energy below this)
FREQ_MAX = 8000              # Hz (Nyquist limit)
FREQ_BAND = (FREQ_MIN, FREQ_MAX)

# =============================================================================
# Dataset Parameters
# =============================================================================

N_CLIPS = 1920               # Total clips in prototype dataset
SNR_DB = 6.0                 # Ship above sea noise

# Vessel class mapping (alphabetical)
VESSEL_CLASSES = {
    'cargo_ship': 0,
    'fishing_vessel': 1,
    'small_craft': 2,
    'tanker': 3
}

N_CLASSES = len(VESSEL_CLASSES)

# Inverse mapping
CLASS_NAMES = {v: k for k, v in VESSEL_CLASSES.items()}

# Vessel acoustic characteristics
VESSEL_PARAMS = {
    'small_craft': {
        'shaft_hz': (15, 30),
        'bpf_hz': (45, 90),
        'cavitation_peak': 5000
    },
    'fishing_vessel': {
        'shaft_hz': (4, 8),
        'bpf_hz': (12, 32),
        'cavitation_peak': 1500
    },
    'cargo_ship': {
        'shaft_hz': (1.5, 2.5),
        'bpf_hz': (6, 12.5),
        'cavitation_peak': 600
    },
    'tanker': {
        'shaft_hz': (1.0, 1.5),
        'bpf_hz': (4, 9),
        'cavitation_peak': 400
    }
}

# Design factor levels
SEA_STATES = [0, 1, 3, 6]
BLADE_COUNTS = [3, 4, 5]
GENERATOR_FREQS = [0, 50]  # Hz
CAVITATION_LEVELS = [0.0, 0.333, 0.667, 1.0]
N_REPEATS = 5

# =============================================================================
# Data Splits
# =============================================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# =============================================================================
# Model Architecture (Stage 1+)
# =============================================================================

# SKConv1D kernel sizes
SKCONV1D_KERNELS = [3, 5, 7, 11, 15]

# Default channel dimensions
SKCONV1D_OUT_CHANNELS = 64
SKCONV2D_OUT_CHANNELS = 128
EMBEDDING_DIM = 256

# =============================================================================
# Training Defaults
# =============================================================================

DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 100

# =============================================================================
# Paths (relative to SKANN_SSL root)
# =============================================================================

# These are templates - actual paths set at runtime
MANIFEST_FILENAME = 'master_dataset_manifest.csv'
DATA_DIR = 'data/prototype_dataset'
STAGES_DIR = 'stages'
CHECKPOINTS_DIR = 'checkpoints'
OUTPUTS_DIR = 'outputs'

# =============================================================================
# Colab Paths
# =============================================================================

def get_colab_paths(drive_root: str = '/content/drive/MyDrive/SKANN_SSL'):
    """Get full paths for Colab environment."""
    return {
        'root': drive_root,
        'manifest': f"{drive_root}/data/prototype_dataset/master_dataset_manifest.csv",
        'tensors': f"{drive_root}/data/prototype_dataset/tensors",
        'waveforms': f"{drive_root}/data/prototype_dataset/waveforms",
        'checkpoints': f"{drive_root}/checkpoints",
        'outputs': f"{drive_root}/outputs"
    }


# =============================================================================
# Version
# =============================================================================

__version__ = '0.2.0'
PROJECT_NAME = 'SKANN-SSL'
