"""
SKANN-SSL Shared Module
=======================
Common configuration and utilities.
"""

from .config import (
    SAMPLE_RATE,
    DURATION_SEC,
    N_SAMPLES,
    FREQ_MIN,
    FREQ_MAX,
    VESSEL_CLASSES,
    CLASS_NAMES,
    N_CLASSES,
    VESSEL_PARAMS,
    get_colab_paths,
    __version__
)

from .utils import (
    set_seed,
    get_device,
    count_parameters,
    model_summary,
    ensure_dir,
    save_checkpoint,
    load_checkpoint,
    compute_rms,
    AverageMeter,
    EarlyStopping
)

__all__ = [
    # Config
    'SAMPLE_RATE', 'DURATION_SEC', 'N_SAMPLES',
    'FREQ_MIN', 'FREQ_MAX',
    'VESSEL_CLASSES', 'CLASS_NAMES', 'N_CLASSES', 'VESSEL_PARAMS',
    'get_colab_paths', '__version__',
    # Utils
    'set_seed', 'get_device', 'count_parameters', 'model_summary',
    'ensure_dir', 'save_checkpoint', 'load_checkpoint', 'compute_rms',
    'AverageMeter', 'EarlyStopping'
]
