"""
SKANN-SSL V3 Pairing Manifest Generator

This script generates the pairing manifest for Stage 3 SSL training with Barlow Twins.
It supports 5 classes:
  - 4 vessel classes (small_craft, fishing_vessel, cargo_ship, tanker): K=6 cross-parameter pairing
  - 1 no_vessel class: K=3 cross-sea-state pairing

Pairing Strategies:
-------------------
VESSEL CLASSES (K=6):
    Uses weighted Euclidean distance in normalized parameter space to pair each anchor
    with K=6 maximally dissimilar clips within the same class. This creates "hard positives"
    that force the model to learn invariance to operating conditions (sea state, blade count,
    cavitation level, etc.) while preserving class identity.

NO_VESSEL CLASS (K=3):
    Uses cross-sea-state pairing with fixed rep-to-rep mapping. Each anchor at sea state X
    is paired with clips from 3 other sea states, matching repetition indices.
    
    Pairing table:
    | Anchor SS | Partner 1 | Partner 2 | Partner 3 |
    |-----------|-----------|-----------|-----------|
    | SS0       | SS6       | SS3       | SS1       |
    | SS1       | SS6       | SS0       | SS3       |
    | SS3       | SS0       | SS1       | SS6       |
    | SS6       | SS0       | SS1       | SS3       |
    
    Rationale: Cross-sea-state pairing forces model to learn "absence of periodicity" as
    the defining feature. After RMS normalization, amplitude differences are removed;
    structural difference (periodic vessel vs. stochastic ambient) survives as discriminator.

Output:
-------
    data/prototype_dataset/pairing_manifest.csv
    Columns: anchor_clip_id, partner_clip_ids (pipe-delimited), vessel_class

Version: 3.0.0
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Repo-relative paths (so script can be run from anywhere)
REPO_ROOT = Path(__file__).resolve().parents[2] if __file__ else Path.cwd()
DATA_DIR = REPO_ROOT / "data" / "prototype_dataset"

MANIFEST_PATH = DATA_DIR / "master_dataset_manifest.csv"
OUTPUT_PATH = DATA_DIR / "pairing_manifest.csv"

# Pairing config
K_VESSEL = 6  # Partners per anchor for vessel classes
K_NO_VESSEL = 3  # Partners per anchor for no_vessel class

# Hierarchy of features for vessel class pairing (in order of importance)
# Used for weighted distance calculation
VESSEL_HIERARCHY = [
    "n_blades",
    "sea_state", 
    "cavitation_peak_freq",
    "shaft_rate",
    "generator_freq",
    "cavitation_intensity",
    "equipment_base_freq",
    "resonance_freq_1",
    "has_cavitation",
    "resonance_freq_2",
    "n_cavitation_bursts",
    "resonance_freq_3"
]

# Cross-sea-state pairing for no_vessel class
# Each anchor sea state maps to these partner sea states (in priority order)
NO_VESSEL_PAIRING = {
    0: [6, 3, 1],  # SS0 pairs with SS6, SS3, SS1
    1: [6, 0, 3],  # SS1 pairs with SS6, SS0, SS3
    3: [0, 1, 6],  # SS3 pairs with SS0, SS1, SS6
    6: [0, 1, 3],  # SS6 pairs with SS0, SS1, SS3
}


def generate_vessel_pairs(df: pd.DataFrame, vessel_class: str, k: int = K_VESSEL) -> list:
    """
    Generate pairing data for a vessel class using cross-parameter distance.
    
    Pairs each anchor with K maximally dissimilar clips within the same class,
    using weighted Euclidean distance in normalized parameter space.
    
    Args:
        df: DataFrame with all dataset entries
        vessel_class: Name of the vessel class to process
        k: Number of partners per anchor
        
    Returns:
        List of dicts with anchor_clip_id, partner_clip_ids, vessel_class
    """
    # Filter to this class
    class_df = df[df['vessel_class'] == vessel_class].copy()
    
    if len(class_df) == 0:
        print(f"  WARNING: No clips found for vessel class '{vessel_class}'")
        return []
    
    # Compute weights using n^1.5 decay (higher weight for more important features)
    raw_weights = np.array([(len(VESSEL_HIERARCHY) - i) ** 1.5 
                            for i in range(len(VESSEL_HIERARCHY))])
    weights = raw_weights / raw_weights.sum()
    
    # Z-score normalize features
    class_norm = class_df.copy()
    available_features = [f for f in VESSEL_HIERARCHY if f in class_df.columns]
    
    for col in available_features:
        if class_df[col].std() != 0:
            class_norm[col] = (class_df[col] - class_df[col].mean()) / class_df[col].std()
        else:
            class_norm[col] = 0.0
    
    # Build feature matrix
    feats = class_norm[available_features].values
    indices = class_df.index.values
    clip_ids = class_df['clip_id'].values
    
    # Apply weights to features
    feature_weights = weights[:len(available_features)]
    weighted_feats = feats * np.sqrt(feature_weights)
    
    # Compute pairwise distance matrix
    dist_matrix = cdist(weighted_feats, weighted_feats, metric='euclidean')
    
    # Generate pairs: for each anchor, find K most distant clips
    pairing_data = []
    for i in range(len(indices)):
        # Get K most distant (descending order, skip self)
        distances = dist_matrix[i]
        # Set self-distance to -inf so it's not selected
        distances[i] = -np.inf
        top_k_indices = np.argsort(distances)[::-1][:k]
        
        pairing_data.append({
            "anchor_clip_id": int(clip_ids[i]),
            "partner_clip_ids": "|".join(map(str, clip_ids[top_k_indices])),
            "vessel_class": vessel_class
        })
    
    return pairing_data


def generate_no_vessel_pairs(df: pd.DataFrame, k: int = K_NO_VESSEL) -> list:
    """
    Generate pairing data for no_vessel class using cross-sea-state pairing.
    
    Uses fixed rep-to-rep mapping where anchor rep N pairs with partner rep N
    from different sea states. This forces the model to learn "absence of periodicity"
    as the defining characteristic.
    
    Args:
        df: DataFrame with all dataset entries
        k: Number of partners per anchor (should be 3 for cross-sea-state)
        
    Returns:
        List of dicts with anchor_clip_id, partner_clip_ids, vessel_class
    """
    # Filter to no_vessel class
    no_vessel_df = df[df['vessel_class'] == 'no_vessel'].copy()
    
    if len(no_vessel_df) == 0:
        print("  WARNING: No clips found for 'no_vessel' class")
        return []
    
    # Build lookup: sea_state -> rep_index -> clip_id
    lookup = {}
    for ss in NO_VESSEL_PAIRING.keys():
        ss_clips = no_vessel_df[no_vessel_df['sea_state'] == ss]
        lookup[ss] = {row['repeat_index']: row['clip_id'] 
                      for _, row in ss_clips.iterrows()}
    
    pairing_data = []
    
    for _, row in no_vessel_df.iterrows():
        anchor_clip_id = int(row['clip_id'])
        anchor_ss = int(row['sea_state'])
        anchor_rep = int(row['repeat_index'])
        
        # Get partner sea states for this anchor
        partner_sea_states = NO_VESSEL_PAIRING.get(anchor_ss, [])
        
        # Find partner clip_ids (same rep_index from each partner sea state)
        partner_ids = []
        for partner_ss in partner_sea_states[:k]:
            if partner_ss in lookup and anchor_rep in lookup[partner_ss]:
                partner_ids.append(lookup[partner_ss][anchor_rep])
            else:
                # Fallback: find any clip from that sea state
                ss_clips = no_vessel_df[no_vessel_df['sea_state'] == partner_ss]
                if len(ss_clips) > 0:
                    partner_ids.append(int(ss_clips.iloc[0]['clip_id']))
        
        if len(partner_ids) > 0:
            pairing_data.append({
                "anchor_clip_id": anchor_clip_id,
                "partner_clip_ids": "|".join(map(str, partner_ids)),
                "vessel_class": "no_vessel"
            })
        else:
            print(f"  WARNING: No partners found for no_vessel clip {anchor_clip_id}")
    
    return pairing_data


def main():
    """Generate the V3 pairing manifest for 5 classes."""
    
    print("=" * 60)
    print("SKANN-SSL V3 Pairing Manifest Generator")
    print("=" * 60)
    
    # Load master manifest
    print(f"\n📂 Loading manifest from: {MANIFEST_PATH}")
    
    if not MANIFEST_PATH.exists():
        print(f"❌ ERROR: Master manifest not found at {MANIFEST_PATH}")
        print("   Please ensure master_dataset_manifest.csv exists in data/prototype_dataset/")
        return
    
    df = pd.read_csv(MANIFEST_PATH)
    print(f"   Loaded {len(df)} clips")
    
    # Ensure has_cavitation is numeric (for distance calculation)
    if 'has_cavitation' in df.columns:
        df['has_cavitation'] = df['has_cavitation'].astype(float)
    
    # Print class distribution
    print("\n📊 Class Distribution:")
    class_counts = df['vessel_class'].value_counts()
    for cls, count in class_counts.items():
        print(f"   {cls}: {count} clips")
    
    # Generate pairings
    print("\n🔗 Generating Pairings...")
    all_pairs = []
    
    # Vessel classes (K=6)
    vessel_classes = ['small_craft', 'fishing_vessel', 'cargo_ship', 'tanker']
    for vessel_class in vessel_classes:
        if vessel_class in df['vessel_class'].values:
            print(f"   Processing {vessel_class} (K={K_VESSEL})...")
            pairs = generate_vessel_pairs(df, vessel_class, K_VESSEL)
            all_pairs.extend(pairs)
            print(f"      Generated {len(pairs)} anchor-partner sets")
        else:
            print(f"   ⚠️ Skipping {vessel_class} (not in dataset)")
    
    # No_vessel class (K=3)
    if 'no_vessel' in df['vessel_class'].values:
        print(f"   Processing no_vessel (K={K_NO_VESSEL})...")
        pairs = generate_no_vessel_pairs(df, K_NO_VESSEL)
        all_pairs.extend(pairs)
        print(f"      Generated {len(pairs)} anchor-partner sets")
    else:
        print("   ⚠️ Skipping no_vessel (not in dataset)")
    
    # Save to CSV
    print(f"\n💾 Saving pairing manifest to: {OUTPUT_PATH}")
    pairing_df = pd.DataFrame(all_pairs)
    pairing_df.to_csv(OUTPUT_PATH, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ PAIRING MANIFEST GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\n📈 Summary:")
    print(f"   Total anchors: {len(pairing_df)}")
    print(f"   Output file: {OUTPUT_PATH}")
    
    # Breakdown by class
    if 'vessel_class' in pairing_df.columns:
        print("\n   By class:")
        for cls, group in pairing_df.groupby('vessel_class'):
            # Count partners per anchor
            avg_partners = group['partner_clip_ids'].apply(
                lambda x: len(x.split('|'))
            ).mean()
            print(f"      {cls}: {len(group)} anchors, avg {avg_partners:.1f} partners/anchor")
    
    print("\n🎯 Next Steps:")
    print("   1. Verify pairing_manifest.csv in data/prototype_dataset/")
    print("   2. Upload V3 data to Kaggle")
    print("   3. Update training notebook for 5 classes and 80,000 samples input")


if __name__ == "__main__":
    main()
