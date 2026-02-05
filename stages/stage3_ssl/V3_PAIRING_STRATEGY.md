# SKANN-SSL V3 Pairing Strategy

**Document Version:** 3.0.0  
**Date:** 23 January 2026  
**Purpose:** Stage 3 SSL Training (Barlow Twins)

---

## Overview

The V3 pairing manifest supports **5 classes** with differentiated pairing strategies:

| Class Type | Classes | Strategy | K (Partners) |
|------------|---------|----------|--------------|
| Vessel | small_craft, fishing_vessel, cargo_ship, tanker | Cross-parameter distance | 6 |
| Ambient | no_vessel | Cross-sea-state | 3 |

---

## Vessel Classes (K=6)

### Rationale
For vessel classes, we want the model to learn invariance to **operating conditions** while preserving **class identity**. We achieve this by pairing each anchor with **maximally dissimilar clips within the same class** ("hard positives").

### Method: Weighted Euclidean Distance

1. **Feature Hierarchy** (in order of importance):
   ```
   n_blades > sea_state > cavitation_peak_freq > shaft_rate > generator_freq > 
   cavitation_intensity > equipment_base_freq > resonance_freq_1 > has_cavitation > 
   resonance_freq_2 > n_cavitation_bursts > resonance_freq_3
   ```

2. **Weighting**: n^1.5 decay (higher weight for more important features)

3. **Normalization**: Z-score per feature within class

4. **Pairing**: For each anchor, select K=6 most distant clips in weighted feature space

### Effect
Forces the encoder to learn that:
- Same vessel class + different sea states → same embedding
- Same vessel class + different blade counts → same embedding
- Same vessel class + different cavitation levels → same embedding

---

## No_Vessel Class (K=3)

### Rationale
The no_vessel class contains **pure ambient sea noise** with no periodic vessel signatures. The key discriminating feature is the **absence of periodicity** (shaft rate, BPF, generator harmonics).

Cross-sea-state pairing ensures:
1. After RMS normalization, amplitude differences are removed
2. Only **structural difference** (stochastic ambient vs. periodic vessel) survives
3. Model learns "no periodic structure" as the defining characteristic

### Method: Cross-Sea-State Pairing

| Anchor SS | Partner 1 | Partner 2 | Partner 3 |
|-----------|-----------|-----------|-----------|
| SS0 | SS6 | SS3 | SS1 |
| SS1 | SS6 | SS0 | SS3 |
| SS3 | SS0 | SS1 | SS6 |
| SS6 | SS0 | SS1 | SS3 |

**Fixed Rep-to-Rep Mapping**: Anchor repetition N → Partner repetition N

### Effect
Forces the encoder to learn that:
- SS0 ambient (calm) ≈ SS6 ambient (rough) after normalization
- All ambient clips share "absence of periodic structure"
- Ambient forms a coherent cluster distinct from vessel classes

---

## Output Format

**File**: `data/prototype_dataset/pairing_manifest.csv`

| Column | Description |
|--------|-------------|
| `anchor_clip_id` | Integer clip ID of the anchor |
| `partner_clip_ids` | Pipe-delimited list of partner clip IDs |
| `vessel_class` | Class name for verification |

**Example**:
```csv
anchor_clip_id,partner_clip_ids,vessel_class
0,1557|1537|1556|1559|1555|1558,small_craft
1920,2160|2040|1960,no_vessel
```

---

## Integration with Barlow Twins

The pairing manifest feeds into the Stage 3 DataLoader:

```python
# Simplified flow
anchor_tensor = load_tensor(anchor_clip_id)
partner_id = random.choice(partner_clip_ids.split('|'))
partner_tensor = load_tensor(partner_id)

# Augmentation (same for both)
anchor_aug = augment(anchor_tensor)
partner_aug = augment(partner_tensor)

# Barlow Twins learns: anchor_embedding ≈ partner_embedding
```

---

## Expected Outcome

After V3 training, the embedding space should show:

1. **5 distinct clusters** (one per class)
2. **no_vessel cluster** well-separated from vessel clusters
3. **Within-cluster spread** reflecting operating conditions, not random noise
4. **High silhouette score** (target: maintain or exceed V2's 0.8299)

---

## Usage

```bash
# From repo root
cd SKANN-SSL
python stages/stage3_ssl/pairing_manifest.py

# Output: data/prototype_dataset/pairing_manifest.csv
```

---

*End of V3 Pairing Strategy Documentation*
