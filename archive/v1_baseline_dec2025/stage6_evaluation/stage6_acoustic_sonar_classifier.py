import os
import sys
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import time
from datetime import datetime

# Import encoder from stage2_encoder without packaging changes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stage2_encoder")))
from train_script import HybridSKEncoder

# Stage-6 artifacts directory (canonical)
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


class AcousticRadarEngine:
    def __init__(self, bundle_path, territories_path):
        print("🔍 Synchronizing Acoustic Assets...")
        self.bundle = joblib.load(bundle_path)
        
        # FIX: Corrected variable access
        self.territories = joblib.load(territories_path)
        self.labels = self.bundle["vessel_labels"]
        
        # Manifest (prototype dataset)
        self.manifest = None
        for name in [
            "data/prototype_dataset/master_dataset_manifest.csv",
            "data/prototype_dataset/pairing_manifest.csv",
        ]:
            if os.path.exists(name):
                print(f"📖 Loaded Manifest: {name}")
                self.manifest = pd.read_csv(name)
                break

        if self.manifest is None:
            raise FileNotFoundError("No manifest found under data/prototype_dataset/")

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridSKEncoder().to(self.device)
        self.model.load_state_dict(self.bundle["model_state"])
        self.model.eval()
        print(f"✅ System Ready (Running on CPU).")

    def classify(self, clip_id):
        clip_str = str(int(clip_id)).zfill(6)

        if self.manifest is None:
            return None, None, None, "Missing"

        # Find row by clip_id
        id_col = "clip_id" if "clip_id" in self.manifest.columns else "anchor_clip_id"
        match = self.manifest[self.manifest[id_col].astype(int) == int(clip_id)]
        if match.empty:
            return None, None, None, "Missing"

        actual_label = match["vessel_class"].values[0]
        filename = match["tensor_path"].values[0]

        # Normalise + make absolute relative to repo root
        filename = str(filename).replace("/", os.sep).replace("\\", os.sep)
        if not os.path.isabs(filename):
            filename = os.path.join(os.getcwd(), filename)

        if not os.path.exists(filename):
            return None, None, None, "Missing"

        
        if not os.path.exists(filename): return None, None, None, "Missing"

        # LOOKUP LABEL (Flexible Column Check)
        actual_label = "Unknown"
        if self.manifest is not None:
            # Check for common ID column names
            col = 'clip_id' if 'clip_id' in self.manifest.columns else 'anchor_clip_id'
            match = self.manifest[self.manifest[col].astype(int) == int(clip_id)]
            if not match.empty:
                actual_label = match['vessel_class'].values[0]

        # INFERENCE
        audio = np.load(filename)
        tensor = torch.from_numpy(audio).float().view(1, 1, -1).to(self.device)
        with torch.no_grad():
            fingerprint = self.model(tensor).detach().cpu().numpy().flatten()
        
        # NATURAL SPREAD (76% Scaling Fix)
        # Adding +0.8 to the distance denominator prevents 'Needle Squashing' 
        # so 76% doesn't touch the 100% rim.
        dists = [np.linalg.norm(fingerprint - self.territories[i]) for i in range(len(self.labels))]
        scores = 1.0 / (np.array(dists) + 0.8) 
        probs = scores / np.sum(scores)
        
        return probs, self.labels[np.argmax(probs)], np.max(probs), actual_label

    def plot_radar(self, probabilities, clip_id, pred, conf, actual):
        num_vars = len(self.labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        stats = probabilities.tolist() + [probabilities[0]]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        
        # FIX: Lock the outer rim to 1.0 (100%)
        # This ensures 76% confidence is visually distinct from 100%
        ax.set_ylim(0, 1.0) 
        
        # Draw Data
        ax.fill(angles, stats, color='#3498db', alpha=0.3)
        ax.plot(angles, stats, color='#2980b9', linewidth=4, marker='o')

        ax.set_xticks(angles[:-1])

        # Pretty tick labels (robust mapping) + 2-line labels for long classes
        pretty_labels = []
        for lab in self.labels:
            key = str(lab).strip().lower().replace(" ", "_")   # normalise
            if key == "cargo_ship":
                pretty_labels.append("Cargo\nShip")
            elif key == "small_craft":
                pretty_labels.append("Small\nCraft")
            else:
                pretty_labels.append(str(lab).replace("_", " ").title())

        ax.set_xticklabels(pretty_labels, fontsize=12, fontweight="bold")

        # Move tick labels outward (prevents “jutting into” the circle)
        ax.tick_params(axis="x", pad=18)


        
        # Title colour (fixed, neutral – consistent with radar plot)
        title_color = "#2980b9"   # same blue as radar line

        
        # Status overlay (top-right)
        is_correct = (pred.lower() == actual.lower())
        if actual == "Unknown":
            status_text = "Label\nunknown"
        else:
            status_text = "Correct\nprediction" if is_correct else "Incorrect\nprediction"

        if actual == "Unknown":
            status_color = "#7f8c8d"   # neutral grey
        else:
            status_color = "#27ae60" if is_correct else "#c0392b"

        status = ax.text(
            0.86, 0.98,
            status_text,
            transform=ax.transAxes,
            ha="left", va="top",
            multialignment="left",
            fontsize=12, fontweight="bold",
            color=status_color
        )

        # Header text on the figure canvas (robust; won’t vanish on polar redraws)
        header = (
            f"CLIP ID: {clip_id}\n"
            f"LABELED AS: {actual.upper()}\n"
            f"IDENTIFIED: {pred.upper()} ({conf*100:.1f}%)"
        )
        fig.text(
            0.5, 0.985, header,
            ha="center", va="top",
            fontsize=10, fontweight="bold",
            color="#2980b9"
        )

        # Reserve headroom for the header
        fig.subplots_adjust(top=0.82)
        fig.canvas.draw()
        plt.pause(0.10)

        # Timed flash for 5 seconds, then persist
        end_time = time.time() + 5.0
        visible = True
        while time.time() < end_time:
            status.set_visible(visible)
            plt.pause(0.3)
            visible = not visible

        status.set_visible(True)


       
        
        save_name = os.path.join(ARTIFACTS_DIR, f"final_radar_{clip_id}.png")
        plt.savefig(save_name, dpi=300)
        plt.show(block=False)
        plt.pause(0.001)
        return save_name

if __name__ == "__main__":
    engine = AcousticRadarEngine(
        bundle_path="stages/stage3_ssl/artifacts/SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib", 
        territories_path = "stages/stage6_evaluation/artifacts/vessel_territories_stage6_2025-12-29.joblib"
    )

    # Per-query result log (append-only)
    LOG_PATH = os.path.join(ARTIFACTS_DIR, "stage6_per_query_results_log.csv")
    LOG_FIELDNAMES = (
        ["timestamp", "clip_id", "actual_label", "predicted_label", "pred_confidence", "radar_plot_path"]
        + [f"p_{lab}" for lab in engine.labels]
    )
    LOG_NEEDS_HEADER = not os.path.exists(LOG_PATH)

    while True:
        cid = input("\n👉 Enter Clip ID or 'q': ").strip()

        if cid.lower() == 'q':
            break

        if not cid.isdigit():
            print("   ❌ Please enter a numeric Clip ID (0–1919).")
            continue

        cid_int = int(cid)
        if cid_int < 0 or cid_int > 1919:
            print("   ❌ Clip ID out of range. Valid range is 0–1919.")
            continue

        cid = str(cid_int)

        probs, pred, conf, actual = engine.classify(cid)

        if probs is None:
            print(f"   ❌ Error: Clip {cid} not found.")
            continue

        print(f"   [RESULT] Labeled: {actual.upper()} | Detected: {pred.upper()} ({conf*100:.1f}%)")

        plot_path = engine.plot_radar(probs, cid.zfill(6), pred, conf, actual)
        print(f"   [PLOT]   Saved as: {plot_path}")

        # Append per-query log row
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "clip_id": cid.zfill(6),
            "actual_label": actual,
            "predicted_label": pred,
            "pred_confidence": float(conf),
            "radar_plot_path": plot_path,
        }
        for i, lab in enumerate(engine.labels):
            row[f"p_{lab}"] = float(probs[i])

        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
            if LOG_NEEDS_HEADER:
                w.writeheader()
                LOG_NEEDS_HEADER = False
            w.writerow(row)

        print(f"   [LOG]    Appended to: {LOG_PATH}")

    print("\n🌊 Session closed.")
