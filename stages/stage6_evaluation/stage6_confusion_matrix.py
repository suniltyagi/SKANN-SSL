"""
SKANN-SSL Confusion Matrix Generator (V3/V5)
=============================================
Runs classification on entire dataset and generates confusion analysis.

Usage:
    cd SKANN-SSL
    python stages/stage6_evaluation/stage6_confusion_matrix.py

Requires:
    - SKANN_SSL_V3_Production_Bundle.joblib
    - vessel_territories_v3.joblib
    - data/v5_dataset/master_dataset_manifest.csv
    - data/v5_dataset/tensors/ folder with all .npy files
    - stages/stage3_ssl/train_script.py (for HybridSKEncoderV3)
"""

import torch
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Import model from stage3_ssl
sys.path.insert(0, os.path.join(REPO_ROOT, "stages", "stage3_ssl"))
from train_script import HybridSKEncoderV3


def norm_label(s: str) -> str:
    """Normalize label string for consistent matching."""
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


class ConfusionAnalyzer:
    def __init__(self, bundle_path, territories_path):
        print("🔧 Loading SKANN-SSL V3 Assets...")
        self.bundle = joblib.load(bundle_path)
        self.territories = joblib.load(territories_path)
        self.labels = self.bundle["vessel_labels"]
        self.n_classes = len(self.labels)
        
        # Build territory map (label -> centroid)
        self.territory_map = {}
        if "centroids" in self.territories:
            cent_data = self.territories["centroids"]
            if "vessel_labels" in self.territories:
                for lab, vec in zip(self.territories["vessel_labels"], cent_data.values() if isinstance(cent_data, dict) else cent_data):
                    self.territory_map[norm_label(lab)] = np.asarray(vec)
            else:
                for k, v in cent_data.items():
                    self.territory_map[norm_label(k)] = np.asarray(v)
        else:
            # Direct mapping
            for k, v in self.territories.items():
                if not k.startswith('_'):
                    self.territory_map[norm_label(k)] = np.asarray(v)
        
        print(f"   Territories loaded for: {list(self.territory_map.keys())}")
        
        # Find manifest
        self.manifest = None
        manifest_paths = [
            os.path.join(REPO_ROOT, "data", "v5_dataset", "master_dataset_manifest.csv"),
            os.path.join(REPO_ROOT, "data", "v5_dataset", "pairing_manifest.csv"),
        ]
        for path in manifest_paths:
            if os.path.exists(path):
                self.manifest = pd.read_csv(path)
                print(f"📖 Loaded manifest: {path} ({len(self.manifest)} rows)")
                break
        
        if self.manifest is None:
            raise FileNotFoundError("No manifest found!")
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridSKEncoderV3(latent_dim=256).to(self.device)
        self.model.load_state_dict(self.bundle["model_state"])
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
        
        # Initialize confusion matrix
        self.confusion = np.zeros((self.n_classes, self.n_classes), dtype=int)
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.misclassified = []

    def classify_single(self, tensor_path):
        """Classify a single clip tensor file and return (predicted_label, confidence, probs)"""
        tensor_path = str(tensor_path).replace("/", os.sep).replace("\\", os.sep)
        
        if not os.path.isabs(tensor_path):
            # Manifest paths are relative to v5_dataset folder
            tensor_path = os.path.join(REPO_ROOT, "data", "v5_dataset", tensor_path)
        
        if not os.path.exists(tensor_path):
            return None, None, None
        
        # Inference
        audio = np.load(tensor_path)
        tensor = torch.from_numpy(audio).float().view(1, 1, -1).to(self.device)
        
        with torch.no_grad():
            h, z = self.model(tensor, return_features=True)
            fingerprint = h.cpu().numpy().flatten()  # Use h (512-dim), not z
        
        # Distance to centroids
        dists = []
        for lab in self.labels:
            lab_norm = norm_label(lab)
            if lab_norm in self.territory_map:
                dist = np.linalg.norm(fingerprint - self.territory_map[lab_norm])
            else:
                dist = float('inf')
            dists.append(dist)
        
        dists = np.array(dists)
        scores = 1.0 / (dists + 0.8)
        probs = scores / np.sum(scores)
        
        pred_idx = np.argmax(probs)
        return self.labels[pred_idx], float(probs[pred_idx]), probs

    def run_full_analysis(self):
        """Run classification on entire dataset"""
        print("\n🔍 Running full dataset analysis...")
        
        # Determine clip ID column
        id_col = 'clip_id' if 'clip_id' in self.manifest.columns else 'anchor_clip_id'
        
        total = len(self.manifest)
        correct = 0
        processed = 0
        per_clip_rows = []
        
        for idx, row in self.manifest.iterrows():
            clip_id = int(row[id_col])
            actual_label = row['vessel_class']
            
            # Build tensor path
            if 'tensor_path' in row:
                tensor_path = row['tensor_path']
            else:
                tensor_path = f"data/v5_dataset/tensors/tensor_{clip_id:06d}.npy"
            
            pred_label, confidence, probs = self.classify_single(tensor_path)
            
            if pred_label is None:
                continue
            
            processed += 1
            
            row_out = {
                "clip_id": clip_id,
                "actual": actual_label,
                "predicted": pred_label,
                "pred_confidence": float(confidence),
            }
            for k, lab in enumerate(self.labels):
                row_out[f"p_{lab}"] = float(probs[k])
            per_clip_rows.append(row_out)
            
            # Update confusion matrix
            actual_idx = self.label_to_idx[actual_label]
            pred_idx = self.label_to_idx[pred_label]
            self.confusion[actual_idx, pred_idx] += 1
            
            if pred_label == actual_label:
                correct += 1
            else:
                self.misclassified.append({
                    'clip_id': clip_id,
                    'actual': actual_label,
                    'predicted': pred_label,
                    'confidence': confidence,
                    'sea_state': row.get('sea_state', 'N/A'),
                    'n_blades': row.get('n_blades', 'N/A'),
                    'cavitation_intensity': row.get('cavitation_intensity', 'N/A')
                })
            
            # Progress
            if (idx + 1) % 500 == 0:
                print(f"   Processed {idx + 1}/{total} clips...")
        
        # Save per-clip results
        df_pc = pd.DataFrame(per_clip_rows)
        csv_path = os.path.join(ARTIFACTS_DIR, "per_clip_class_results_confidences.csv")
        df_pc.to_csv(csv_path, index=False)
        print(f"🧾 Saved: {csv_path}")
        
        try:
            md_path = os.path.join(ARTIFACTS_DIR, "per_clip_class_results_confidences.md")
            df_pc.to_markdown(md_path, index=False)
            print(f"🧾 Saved: {md_path}")
        except Exception as e:
            print(f"   (Markdown export skipped: {e})")
        
        accuracy = correct / processed * 100 if processed > 0 else 0
        print(f"\n✅ Analysis complete: {correct}/{processed} correct ({accuracy:.1f}%)")
        return accuracy

    def plot_confusion_matrix(self, save_path="confusion_matrix.png"):
        """Generate and save confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Normalize for percentages
        row_sums = self.confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        confusion_pct = self.confusion / row_sums * 100
        
        # Create heatmap
        sns.heatmap(
            confusion_pct,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=self.labels,
            yticklabels=self.labels,
            cbar_kws={'label': 'Percentage (%)'}
        )
        
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('Actual Class', fontsize=12)
        plt.title('SKANN-SSL V3 Confusion Matrix\n(Row-Normalized Percentages)', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {save_path}")

    def generate_report(self, save_path="confusion_report.txt"):
        """Generate detailed text report"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("SKANN-SSL V3 CONFUSION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall accuracy
            total = self.confusion.sum()
            correct = np.trace(self.confusion)
            accuracy = correct / total * 100 if total > 0 else 0
            f.write(f"Overall Accuracy: {correct}/{total} ({accuracy:.1f}%)\n\n")
            
            # Per-class metrics
            f.write("-" * 60 + "\n")
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-" * 60 + "\n")
            
            for i, label in enumerate(self.labels):
                tp = self.confusion[i, i]
                total_actual = self.confusion[i, :].sum()
                total_pred = self.confusion[:, i].sum()
                
                precision = tp / total_pred * 100 if total_pred > 0 else 0
                recall = tp / total_actual * 100 if total_actual > 0 else 0
                
                f.write(f"\n{label.upper()}:\n")
                f.write(f"  Recall (Sensitivity):  {recall:.1f}%\n")
                f.write(f"  Precision:             {precision:.1f}%\n")
                f.write(f"  Samples:               {total_actual}\n")
                
                # Show what this class gets confused with
                if recall < 100:
                    f.write(f"  Confused with:\n")
                    for j, other_label in enumerate(self.labels):
                        if i != j and self.confusion[i, j] > 0:
                            pct = self.confusion[i, j] / total_actual * 100
                            f.write(f"    → {other_label}: {self.confusion[i, j]} ({pct:.1f}%)\n")
            
            # Confusion pairs analysis
            f.write("\n" + "-" * 60 + "\n")
            f.write("TOP CONFUSION PAIRS\n")
            f.write("-" * 60 + "\n")
            
            confusion_pairs = []
            for i in range(self.n_classes):
                for j in range(self.n_classes):
                    if i != j and self.confusion[i, j] > 0:
                        row_sum = self.confusion[i, :].sum()
                        confusion_pairs.append({
                            'actual': self.labels[i],
                            'predicted': self.labels[j],
                            'count': self.confusion[i, j],
                            'pct': self.confusion[i, j] / row_sum * 100 if row_sum > 0 else 0
                        })
            
            confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
            
            for pair in confusion_pairs[:10]:
                f.write(f"\n{pair['actual']} → {pair['predicted']}: "
                       f"{pair['count']} errors ({pair['pct']:.1f}%)\n")
            
            # Metadata correlation (if available)
            if self.misclassified:
                f.write("\n" + "-" * 60 + "\n")
                f.write("MISCLASSIFICATION PATTERNS\n")
                f.write("-" * 60 + "\n")
                
                df_errors = pd.DataFrame(self.misclassified)
                
                if 'sea_state' in df_errors.columns:
                    f.write("\nErrors by Sea State:\n")
                    sea_counts = df_errors['sea_state'].value_counts()
                    for ss, count in sea_counts.items():
                        f.write(f"  SS{ss}: {count} errors\n")
                
                if 'n_blades' in df_errors.columns:
                    f.write("\nErrors by Blade Count:\n")
                    blade_counts = df_errors['n_blades'].value_counts()
                    for nb, count in blade_counts.items():
                        f.write(f"  {nb} blades: {count} errors\n")
        
        print(f"📝 Saved: {save_path}")

    def save_misclassified_csv(self, save_path="misclassified_clips.csv"):
        """Save all misclassified clips to CSV for detailed analysis"""
        if self.misclassified:
            df = pd.DataFrame(self.misclassified)
            df.to_csv(save_path, index=False)
            print(f"📋 Saved: {save_path} ({len(df)} errors)")
        else:
            print("📋 No misclassified clips to save.")


def main():
    print("=" * 60)
    print("SKANN-SSL V3 CONFUSION MATRIX GENERATOR")
    print("=" * 60)
    
    # Initialize with V3 paths
    analyzer = ConfusionAnalyzer(
        bundle_path=os.path.join(REPO_ROOT, "stages", "stage3_ssl", "artifacts", "SKANN_SSL_V3_Production_Bundle.joblib"),
        territories_path=os.path.join(REPO_ROOT, "stages", "stage3_ssl", "artifacts", "vessel_territories_v3.joblib")
    )
    
    # Run analysis
    accuracy = analyzer.run_full_analysis()
    
    analyzer.plot_confusion_matrix(save_path=os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"))
    analyzer.generate_report(save_path=os.path.join(ARTIFACTS_DIR, "confusion_report.txt"))
    analyzer.save_misclassified_csv(save_path=os.path.join(ARTIFACTS_DIR, "misclassified_clips.csv"))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.1f}%")
    print(f"Total Misclassified: {len(analyzer.misclassified)}")
    print("\nOutputs generated:")
    print("  • confusion_matrix.png")
    print("  • confusion_report.txt")
    print("  • misclassified_clips.csv")
    print("  • per_clip_class_results_confidences.csv")


if __name__ == "__main__":
    main()
