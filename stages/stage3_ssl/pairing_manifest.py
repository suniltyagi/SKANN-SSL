import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# Config
K = 6
hierarchy = ["n_blades", "sea_state", "cavitation_peak_freq", "shaft_rate", "generator_freq", 
             "cavitation_intensity", "equipment_base_freq", "resonance_freq_1", 
             "has_cavitation", "resonance_freq_2", "n_cavitation_bursts", "resonance_freq_3"]

# Load Data
df = pd.read_csv('master_dataset_manifest.csv')
df['has_cavitation'] = df['has_cavitation'].astype(float)

# Math: n^1.5 Weights
raw_weights = np.array([(len(hierarchy) - i)**1.5 for i in range(len(hierarchy))])
weights = raw_weights / raw_weights.sum()

# Normalize Features (Z-score)
df_norm = df.copy()
for col in hierarchy:
    if df[col].std() != 0:
        df_norm[col] = (df[col] - df[col].mean()) / df[col].std()
    else:
        df_norm[col] = 0.0

# Generate Pairs
pairing_data = []
for class_name, group in df_norm.groupby("vessel_class"):
    indices = group.index.values
    feats = group[hierarchy].values
    clip_ids = df.loc[indices, 'clip_id'].values
    weighted_feats = feats * np.sqrt(weights)
    dist_matrix = cdist(weighted_feats, weighted_feats, metric='euclidean')
    
    for i in range(len(indices)):
        top_k_indices = np.argsort(dist_matrix[i])[::-1][:K]
        pairing_data.append({
            "anchor_clip_id": int(clip_ids[i]),
            "partner_clip_ids": "|".join(map(str, clip_ids[top_k_indices])),
            "vessel_class": class_name
        })

pd.DataFrame(pairing_data).to_csv('pairing_manifest.csv', index=False)
print("pairing_manifest.csv generated successfully.")