"""
SKANN-SSL V3.2 Pairing Manifest Generator
==========================================

Generates pairing manifest for SSL training with Barlow Twins.

VESSEL CLASSES (K=6):
    Uses weighted Euclidean distance in normalized parameter space to pair each anchor
    with K=6 maximally dissimilar clips within the same class ("hard positives").
    
    Feature hierarchy (weighted by importance):
    1. n_blades (highest)
    2. sea_state
    3. cavitation_peak_freq
    4. shaft_rate
    5. generator_freq
    6. cavitation_intensity
    7. equipment_base_freq
    8. resonance_freq_1
    9. has_cavitation
    10. resonance_freq_2
    11. n_cavitation_bursts
    12. resonance_freq_3 (lowest)

NO_VESSEL CLASS (K=3):
    Cross-sea-state pairing with fixed rep-to-rep mapping.
    
    | Anchor SS | Partner 1 | Partner 2 | Partner 3 |
    |-----------|-----------|-----------|-----------|
    | SS0       | SS6       | SS3       | SS1       |
    | SS1       | SS6       | SS0       | SS3       |
    | SS3       | SS0       | SS1       | SS6       |
    | SS6       | SS0       | SS1       | SS3       |

Usage:
    python generate_pairing_manifest_v3_2.py

Output:
    pairing_manifest.csv in same folder as master_dataset_manifest.csv
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_ROOT = Path(r"C:\Users\Admin\uw_project\SKANN-SSL\stages\stage_minus1\skann_ssl_v3_dataset")
MANIFEST_PATH = DATASET_ROOT / "master_dataset_manifest.csv"
OUTPUT_PATH = DATASET_ROOT / "pairing_manifest.csv"

# Pairing config
K_VESSEL = 6      # Partners per anchor for vessel classes
K_NO_VESSEL = 3   # Partners per anchor for no_vessel class

# Feature hierarchy for vessel class pairing (in order of importance)
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
NO_VESSEL_PAIRING = {
    0: [6, 3, 1],  # SS0 pairs with SS6, SS3, SS1
    1: [6, 0, 3],  # SS1 pairs with SS6, SS0, SS3
    3: [0, 1, 6],  # SS3 pairs with SS0, SS1, SS6
    6: [0, 1, 3],  # SS6 pairs with SS0, SS1, SS3
}


# =============================================================================
# VESSEL CLASS PAIRING (K=6 Hard Positives)
# =============================================================================

def generate_vessel_pairs(df: pd.DataFrame, vessel_class: str, k: int = K_VESSEL) -> list:
    """
    Generate pairing data for a vessel class using cross-parameter distance.
    
    Pairs each anchor with K maximally dissimilar clips within the same class,
    using weighted Euclidean distance in normalized parameter space.
    
    Args:
        df: Full dataset manifest
        vessel_class: Class to process
        k: Number of partners per anchor
        
    Returns:
        List of dicts with anchor_clip_id, partner_clip_ids, vessel_class
    """
    # Filter to this class
    class_df = df[df['vessel_class'] == vessel_class].copy()
    class_df = class_df.reset_index(drop=True)
    n_clips = len(class_df)
    
    if n_clips < k + 1:
        print(f"   ⚠️ Warning: {vessel_class} has only {n_clips} clips, need at least {k+1}")
        return []
    
    # Get available features (some may be missing)
    available_features = [f for f in VESSEL_HIERARCHY if f in class_df.columns]
    
    if len(available_features) == 0:
        print(f"   ⚠️ Warning: No hierarchy features found for {vessel_class}")
        return []
    
    # Build feature matrix
    feature_matrix = class_df[available_features].copy()
    
    # Handle missing values
    feature_matrix = feature_matrix.fillna(0)
    
    # Convert boolean to numeric
    for col in feature_matrix.columns:
        if feature_matrix[col].dtype == bool:
            feature_matrix[col] = feature_matrix[col].astype(float)
    
    # Normalize each feature to [0, 1]
    for col in feature_matrix.columns:
        col_min = feature_matrix[col].min()
        col_max = feature_matrix[col].max()
        if col_max > col_min:
            feature_matrix[col] = (feature_matrix[col] - col_min) / (col_max - col_min)
        else:
            feature_matrix[col] = 0.0
    
    # Compute weights using power-law decay (n^1.5)
    n_features = len(available_features)
    raw_weights = np.array([(n_features - i) ** 1.5 for i in range(n_features)])
    weights = raw_weights / raw_weights.sum()
    
    # Apply weights to features
    weighted_features = feature_matrix.values * weights
    
    # Compute pairwise distance matrix
    dist_matrix = cdist(weighted_features, weighted_features, metric='euclidean')
    
    # For each anchor, find K most distant (hard positives)
    pairs = []
    clip_ids = class_df['clip_id'].values
    
    for i in range(n_clips):
        anchor_id = int(clip_ids[i])
        
        # Get distances from this anchor, set self-distance to -inf
        distances = dist_matrix[i].copy()
        distances[i] = -np.inf
        
        # Find K most distant (descending order)
        top_k_indices = np.argsort(distances)[::-1][:k]
        partner_ids = [int(clip_ids[j]) for j in top_k_indices]
        
        pairs.append({
            'anchor_clip_id': anchor_id,
            'partner_clip_ids': '|'.join(map(str, partner_ids)),
            'vessel_class': vessel_class
        })
    
    return pairs


# =============================================================================
# NO_VESSEL CLASS PAIRING (K=3 Cross-Sea-State)
# =============================================================================

def generate_no_vessel_pairs(df: pd.DataFrame, k: int = K_NO_VESSEL) -> list:
    """
    Generate pairing data for no_vessel class using cross-sea-state pairing.
    
    Each anchor at sea state X is paired with clips from K other sea states,
    using fixed rep-to-rep mapping (clip N → clip N).
    
    Args:
        df: Full dataset manifest
        k: Number of partners per anchor (should be 3)
        
    Returns:
        List of dicts with anchor_clip_id, partner_clip_ids, vessel_class
    """
    # Filter to no_vessel class
    no_vessel_df = df[df['vessel_class'] == 'no_vessel'].copy()
    
    # Group by sea state
    sea_state_groups = {}
    for ss in NO_VESSEL_PAIRING.keys():
        ss_clips = no_vessel_df[no_vessel_df['sea_state'] == ss].copy()
        ss_clips = ss_clips.sort_values('clip_id').reset_index(drop=True)
        sea_state_groups[ss] = ss_clips
    
    # Verify equal distribution
    group_sizes = {ss: len(grp) for ss, grp in sea_state_groups.items()}
    if len(set(group_sizes.values())) != 1:
        print(f"   ⚠️ Warning: Unequal sea state distribution: {group_sizes}")
    
    clips_per_ss = min(group_sizes.values())
    print(f"      Clips per sea state: {clips_per_ss}")
    
    pairs = []
    
    for anchor_ss, partner_ss_list in NO_VESSEL_PAIRING.items():
        anchor_group = sea_state_groups[anchor_ss]
        
        for rep_idx in range(clips_per_ss):
            anchor_id = int(anchor_group.iloc[rep_idx]['clip_id'])
            
            # Get partner clip IDs (same rep index from each partner sea state)
            partner_ids = []
            for partner_ss in partner_ss_list[:k]:
                partner_group = sea_state_groups[partner_ss]
                if rep_idx < len(partner_group):
                    partner_id = int(partner_group.iloc[rep_idx]['clip_id'])
                    partner_ids.append(partner_id)
            
            if len(partner_ids) == k:
                pairs.append({
                    'anchor_clip_id': anchor_id,
                    'partner_clip_ids': '|'.join(map(str, partner_ids)),
                    'vessel_class': 'no_vessel'
                })
    
    return pairs


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SKANN-SSL V3.2 Pairing Manifest Generator")
    print("=" * 70)
    
    # Load master manifest
    print(f"\n📂 Loading manifest from: {MANIFEST_PATH}")
    
    if not MANIFEST_PATH.exists():
        print(f"❌ ERROR: Master manifest not found at {MANIFEST_PATH}")
        sys.exit(1)
    
    df = pd.read_csv(MANIFEST_PATH)
    print(f"   Loaded {len(df)} clips")
    
    # Ensure has_cavitation is numeric
    if 'has_cavitation' in df.columns:
        df['has_cavitation'] = df['has_cavitation'].astype(float)
    
    # Print class distribution
    print("\n📊 Class Distribution:")
    class_counts = df['vessel_class'].value_counts()
    for cls in sorted(class_counts.index):
        print(f"   {cls}: {class_counts[cls]} clips")
    
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
    print("\n" + "=" * 70)
    print("✅ PAIRING MANIFEST GENERATED SUCCESSFULLY")
    print("=" * 70)
    
    print(f"\n📈 Summary:")
    print(f"   Total anchors: {len(pairing_df)}")
    print(f"   Output file: {OUTPUT_PATH}")
    
    # Breakdown by class
    print("\n   By class:")
    for cls in sorted(pairing_df['vessel_class'].unique()):
        group = pairing_df[pairing_df['vessel_class'] == cls]
        avg_partners = group['partner_clip_ids'].apply(lambda x: len(x.split('|'))).mean()
        print(f"      {cls}: {len(group)} anchors, K={avg_partners:.0f} partners/anchor")
    
    # Verification
    print("\n🔍 Verification:")
    print(f"   Expected vessel anchors: 4 × 2400 = 9600")
    print(f"   Expected no_vessel anchors: 2400")
    print(f"   Expected total: 12000")
    print(f"   Actual total: {len(pairing_df)}")
    
    if len(pairing_df) == 12000:
        print("   ✅ Count matches!")
    else:
        print(f"   ⚠️ Count mismatch! Difference: {12000 - len(pairing_df)}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
