"""
SKANN-SSL Stage -1: Synthetic Waveform Generator
================================================

This package generates synthetic underwater acoustic waveforms for
training the SKANN-SSL self-supervised learning system.

Components:
- Sea noise: Knudsen-based colored noise generator
- Ship noise: Tonal + broadband + cavitation
- Combined generator: SNR-controlled mixing

Usage:
    from stage_minus1 import SyntheticDataGenerator, DatasetGenerator
    
    # Generate single clip
    gen = SyntheticDataGenerator()
    waveform, metadata = gen.generate_clip(sea_state=3, vessel_class='cargo_ship')
    
    # Generate full dataset
    dataset_gen = DatasetGenerator(n_clips=500)
    df = dataset_gen.generate_dataset()
"""

from .config import (
    FS, N_SAMPLES, DURATION, NYQUIST,
    SEA_STATES, VESSEL_CLASSES, SHIP_SNR_DB,
    N_SYNTHETIC_CLIPS, RANDOM_SEED
)

from .sea_noise import (
    KnudsenModel,
    SeaNoiseGenerator
)

from .ship_noise import (
    VesselParams,
    TonalGenerator,
    BroadbandGenerator,
    CavitationGenerator,
    ShipNoiseGenerator
)

from .generator import (
    ClipMetadata,
    SyntheticDataGenerator,
    DatasetGenerator,
    generate_prototype_dataset
)

__all__ = [
    # Config
    'FS', 'N_SAMPLES', 'DURATION', 'NYQUIST',
    'SEA_STATES', 'VESSEL_CLASSES', 'SHIP_SNR_DB',
    'N_SYNTHETIC_CLIPS', 'RANDOM_SEED',
    # Sea noise
    'KnudsenModel', 'SeaNoiseGenerator',
    # Ship noise
    'VesselParams', 'TonalGenerator', 'BroadbandGenerator',
    'CavitationGenerator', 'ShipNoiseGenerator',
    # Combined
    'ClipMetadata', 'SyntheticDataGenerator', 'DatasetGenerator',
    'generate_prototype_dataset'
]

__version__ = '0.1.0'
