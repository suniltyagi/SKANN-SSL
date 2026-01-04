"""
SKANN-SSL Confusion Matrix Generator
=====================================
Runs classification on entire dataset and generates confusion analysis.

Usage:
    python confusion_matrix_generator.py

Requires:
    - SKANN_SSL_Production_Bundle.joblib
    - vessel_territories.joblib
    - master_dataset_manifest.csv (or pairing_manifest.csv)
    - tensors/ folder with all .npy files
    - train_script.py (for HybridSKEncoder)
"""

import torch
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import sys, os

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stage2_encoder")))
from train_script import HybridSKEncoder

# Force script to look in its own directory
#STAGE_DIR = Path(__file__).resolve().parent
#REPO_ROOT = STAGE_DIR.parents[1]   # .../SKANN-SSL


class ConfusionAnalyzer:
    def __init__(self, bundle_path, territories_path):
        print("🔧 Loading SKANN-SSL Assets...")
        self.bundle = joblib.load(bundle_path)
        self.territories = joblib.load(territories_path)
        self.labels = self.bundle["vessel_labels"]
        self.n_classes = len(self.labels)
        
        # Find manifest
        self.manifest = None
        for name in [
            "data/prototype_dataset/master_dataset_manifest.csv",
            "data/prototype_dataset/pairing_manifest.csv",
        ]:
            if os.path.exists(name):
                self.manifest = pd.read_csv(name)
                print(f"📖 Loaded manifest: {name}")
                break

        
        if self.manifest is None:
            raise FileNotFoundError("No manifest found!")
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridSKEncoder().to(self.device)
        self.model.load_state_dict(self.bundle["model_state"])
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
        
        # Initialize confusion matrix
        self.confusion = np.zeros((self.n_classes, self.n_classes), dtype=int)
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.misclassified = []  # Store details of errors

    def classify_single(self, tensor_path):
        """Classify a single clip tensor file and return (predicted_label, confidence)"""

        # tensor_path in manifest is repo-root relative like:
        # data/prototype_dataset/tensors/tensor_000000.npy
        tensor_path = str(tensor_path)

        # normalise separators (handles any / or \ mix)
        tensor_path = tensor_path.replace("/", os.sep).replace("\\", os.sep)

        # if it's relative, make it relative to repo root (you run from repo root)
        if not os.path.isabs(tensor_path):
            # current working dir is repo root in your run command
            tensor_path = os.path.join(os.getcwd(), tensor_path)

        if not os.path.exists(tensor_path):
            return None, None, None

        # Inference
        audio = np.load(tensor_path)
        tensor = torch.from_numpy(audio).float().view(1, 1, -1).to(self.device)

        with torch.no_grad():
            fingerprint = self.model(tensor).cpu().numpy().flatten()

        # Distance to centroids
        dists = [np.linalg.norm(fingerprint - self.territories[i])
                for i in range(self.n_classes)]
        scores = 1.0 / (np.array(dists) + 0.8)
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
        per_clip_rows = []

        
        for idx, row in self.manifest.iterrows():
            clip_id = int(row[id_col])
            actual_label = row['vessel_class']
            
            pred_label, confidence, probs = self.classify_single(row["tensor_path"])
            
            if pred_label is None:
                continue
            
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
            if (idx + 1) % 200 == 0:
                print(f"   Processed {idx + 1}/{total} clips...")
        
        df_pc = pd.DataFrame(per_clip_rows)
        df_pc.to_csv(os.path.join(ARTIFACTS_DIR, "per_clip_class_results_confidences.csv"), index=False)
        print(f"🧾 Saved: {os.path.join(ARTIFACTS_DIR, 'per_clip_class_results_confidences.csv')}")

        df_pc.to_markdown(os.path.join(ARTIFACTS_DIR, "per_clip_class_results_confidences.md"), index=False)
        print(f"🧾 Saved: {os.path.join(ARTIFACTS_DIR, 'per_clip_class_results_confidences.md')}")

        
        accuracy = correct / total * 100
        print(f"\n✅ Analysis complete: {correct}/{total} correct ({accuracy:.1f}%)")
        return accuracy

    def plot_confusion_matrix(self, save_path="confusion_matrix.png"):
        """Generate and save confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Normalize for percentages
        row_sums = self.confusion.sum(axis=1, keepdims=True)
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
        plt.title('SKANN-SSL Confusion Matrix\n(Row-Normalized Percentages)', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {save_path}")

    def generate_report(self, save_path="confusion_report.txt"):
        """Generate detailed text report"""
        with open(save_path, 'w', encoding='utf-8') as f:

            f.write("=" * 60 + "\n")
            f.write("SKANN-SSL CONFUSION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall accuracy
            total = self.confusion.sum()
            correct = np.trace(self.confusion)
            f.write(f"Overall Accuracy: {correct}/{total} ({correct/total*100:.1f}%)\n\n")
            
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
                        confusion_pairs.append({
                            'actual': self.labels[i],
                            'predicted': self.labels[j],
                            'count': self.confusion[i, j],
                            'pct': self.confusion[i, j] / self.confusion[i, :].sum() * 100
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
                
                # By sea state
                if 'sea_state' in df_errors.columns:
                    f.write("\nErrors by Sea State:\n")
                    sea_counts = df_errors['sea_state'].value_counts()
                    for ss, count in sea_counts.items():
                        f.write(f"  SS{ss}: {count} errors\n")
                
                # By blade count
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


def main():
    print("=" * 60)
    print("SKANN-SSL CONFUSION MATRIX GENERATOR")
    print("=" * 60)
    
    # Initialize
    analyzer = ConfusionAnalyzer(
        bundle_path="stages/stage3_ssl/artifacts/SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib",
        territories_path="stages/stage6_evaluation/artifacts/vessel_territories_stage6_2025-12-29.joblib"
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
    




if __name__ == "__main__":
    main()
