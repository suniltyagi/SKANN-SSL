import torch, joblib, numpy as np, os, matplotlib.pyplot as plt
import pandas as pd
from train_script import HybridSKEncoder

# Force the script to look in its own folder for manifest/assets
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class AcousticRadarEngine:
    def __init__(self, bundle_path, territories_path):
        print("🔍 Synchronizing Acoustic Assets...")
        self.bundle = joblib.load(bundle_path)
        
        # FIX: Corrected variable access
        self.territories = joblib.load(territories_path)
        self.labels = self.bundle["vessel_labels"]
        
        # SMART MANIFEST SEARCH
        self.manifest = None
        for name in ["master_dataset_manifest.csv", "pairing_manifest.csv"]:
            if os.path.exists(name):
                print(f"📖 Found Manifest: {name}")
                self.manifest = pd.read_csv(name)
                break
        
        self.device = torch.device("cpu")
        self.model = HybridSKEncoder().to(self.device)
        self.model.load_state_dict(self.bundle["model_state"])
        self.model.eval()
        print(f"✅ System Ready (Running on CPU).")

    def classify(self, clip_id):
        clip_str = str(clip_id).zfill(6)
        # Search for file in root OR tensors folder
        filename = f"tensor_{clip_str}.npy"
        if not os.path.exists(filename):
            filename = os.path.join("tensors", filename)
        
        if not os.path.exists(filename): return None, None, None, "Missing"

        # LOOKUP LABEL (Flexible Column Check)
        actual_label = "Unknown"
        if self.manifest is not None:
            # Check for common ID column names
            col = 'clip_id' if 'clip_id' in self.manifest.columns else 'anchor_clip_id'
            match = self.manifest[self.manifest[col].astype(str).str.contains(str(int(clip_id)))]
            if not match.empty:
                actual_label = match['vessel_class'].values[0]

        # INFERENCE
        audio = np.load(filename)
        tensor = torch.from_numpy(audio).float().view(1, 1, -1)
        with torch.no_grad():
            fingerprint = self.model(tensor).numpy().flatten()
        
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
        ax.set_xticklabels(self.labels, fontsize=12, fontweight='bold')
        
        # Accuracy Color Logic
        title_color = '#27ae60' if pred.lower() == actual.lower() else '#e67e22'
        if actual == "Unknown": title_color = '#2c3e50'

        plt.title(f"CLIP ID: {clip_id}\nLABELED AS: {actual.upper()}\n" + 
                  f"IDENTIFIED: {pred.upper()} ({conf*100:.1f}%)", 
                  size=14, y=1.1, fontweight='bold', color=title_color)
        
        save_name = f"final_radar_{clip_id}.png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close() 
        return save_name

if __name__ == "__main__":
    engine = AcousticRadarEngine(
        bundle_path="SKANN_SSL_Production_Bundle.joblib", 
        territories_path="vessel_territories.joblib"
    )

    while True:
        cid = input("\n👉 Enter Clip ID or 'q': ").strip()
        if cid.lower() == 'q': break
            
        probs, pred, conf, actual = engine.classify(cid)
        if probs is not None:
            print(f"   [RESULT] Labeled: {actual.upper()} | Detected: {pred.upper()} ({conf*100:.1f}%)")
            plot_path = engine.plot_radar(probs, cid.zfill(6), pred, conf, actual)
            print(f"   [PLOT]   Saved as: {plot_path}")
        else:
            print(f"   ❌ Error: Clip {cid} not found.")

    print("\n🌊 Session closed.")