"""
SKANN-SSL V2.1.0 Demo GUI
==========================
Simple GUI for demo video recording.
Select a clip → Click Classify → See radar plot + prediction

Usage:
    python skann_ssl_demo_gui.py

Requirements:
    - SKANN_SSL_Production_Bundle.joblib in artifacts/
    - vessel_territories.joblib in artifacts/
    - Tensor files in data/prototype_dataset/tensors/
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import torch
import torch.nn as nn
from pathlib import Path
import random

# =============================================================================
# CONFIGURATION - Update these paths as needed
# =============================================================================
BUNDLE_PATH = "stages/stage3_ssl/artifacts/SKANN_SSL_Production_Bundle.joblib"
TERRITORIES_PATH = "stages/stage6_evaluation/artifacts/vessel_territories.joblib"
TENSORS_DIR = "data/prototype_dataset/tensors"
MANIFEST_PATH = "data/prototype_dataset/master_dataset_manifest.csv"

# =============================================================================
# MODEL ARCHITECTURE (must match training)
# =============================================================================
class SKConv1D(nn.Module):
    """Selective Kernel 1D Convolution"""
    def __init__(self, in_ch, out_ch, kernel_sizes=(31, 63, 127, 255, 511, 1023), 
                 stride=1, reduction=16):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        d = max(out_ch // reduction, 32)
        self.fc1 = nn.Linear(out_ch, d)
        self.fc2 = nn.Linear(d, out_ch * len(kernel_sizes))
        self.softmax = nn.Softmax(dim=1)
        self.out_ch = out_ch
        self.n_kernels = len(kernel_sizes)
        self.gn = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)
        U = feats.sum(dim=1)
        s = self.pool(U).squeeze(-1)
        z = torch.relu(self.fc1(s))
        attn = self.fc2(z).view(-1, self.n_kernels, self.out_ch)
        attn = self.softmax(attn)
        V = (feats * attn.unsqueeze(-1)).sum(dim=1)
        return self.act(self.gn(V))


class SKFilterbank(nn.Module):
    """Multi-scale filterbank frontend"""
    def __init__(self, out_ch=64, kernel_sizes=(31, 63, 127, 255, 511, 1023)):
        super().__init__()
        self.sk = SKConv1D(1, out_ch, kernel_sizes=kernel_sizes)

    def forward(self, x):
        return self.sk(x)


class HybridSKEncoderV2(nn.Module):
    """V2.1.0 Encoder with underwater-appropriate SK kernels"""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.sk_frontend = SKFilterbank(out_ch=64, 
                                        kernel_sizes=(31, 63, 127, 255, 511, 1023))
        self.pool1d = nn.AvgPool1d(kernel_size=8, stride=8)
        self.channel_bridge = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=False)
        )
        self.backbone2d = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.sk_frontend(x)
        h = self.pool1d(h)
        h = self.channel_bridge(h)
        h = h.unsqueeze(1)
        h = self.backbone2d(h)
        return self.fc(h)


# =============================================================================
# DEMO GUI CLASS
# =============================================================================
class SKANNDemoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SKANN-SSL V2.1.0 Demo")
        self.root.geometry("900x700")
        self.root.configure(bg='#1a1a2e')
        
        # Load model and data
        self.load_model()
        self.load_clips()
        
        # Build GUI
        self.build_gui()
        
    def load_model(self):
        """Load encoder and territories"""
        print("Loading model...")
        
        # Load bundle
        bundle = joblib.load(BUNDLE_PATH)
        
        # Create encoder and load weights
        self.encoder = HybridSKEncoderV2(latent_dim=128)
        self.encoder.load_state_dict(bundle['model_state_dict'])
        self.encoder.eval()
        
        # Load territories
        territories = joblib.load(TERRITORIES_PATH)
        self.centroids = territories['centroids']
        self.class_names = list(self.centroids.keys())
        
        # Store embeddings for reference
        self.ref_embeddings = bundle.get('embeddings', None)
        self.ref_labels = bundle.get('labels', None)
        self.vessel_labels = bundle.get('vessel_labels', self.class_names)
        
        print(f"✅ Loaded model with {len(self.class_names)} classes")
        
    def load_clips(self):
        """Load available clips"""
        import pandas as pd
        
        self.manifest = pd.read_csv(MANIFEST_PATH)
        self.tensor_dir = Path(TENSORS_DIR)
        
        # Get sample clips (mix of classes)
        self.sample_clips = []
        for cls in self.class_names:
            cls_clips = self.manifest[self.manifest['vessel_class'] == cls].head(10)
            for _, row in cls_clips.iterrows():
                self.sample_clips.append({
                    'clip_id': row['clip_id'],
                    'class': row['vessel_class'],
                    'tensor_file': row.get('tensor_filename', f"tensor_{int(row['clip_id']):06d}.npy")
                })
        
        # Shuffle for variety
        random.shuffle(self.sample_clips)
        print(f"✅ Loaded {len(self.sample_clips)} sample clips")
        
    def build_gui(self):
        """Build the GUI layout"""
        # Title
        title = tk.Label(self.root, text="🔊 SKANN-SSL V2.1.0", 
                        font=('Helvetica', 24, 'bold'), 
                        fg='#00d4ff', bg='#1a1a2e')
        title.pack(pady=20)
        
        subtitle = tk.Label(self.root, 
                           text="Underwater Acoustic Vessel Classification",
                           font=('Helvetica', 12), fg='#888', bg='#1a1a2e')
        subtitle.pack()
        
        # Control frame
        control_frame = tk.Frame(self.root, bg='#1a1a2e')
        control_frame.pack(pady=20)
        
        # Clip selector
        tk.Label(control_frame, text="Select Clip:", font=('Helvetica', 11),
                fg='white', bg='#1a1a2e').grid(row=0, column=0, padx=10)
        
        self.clip_var = tk.StringVar()
        clip_options = [f"{c['clip_id']:06d} ({c['class']})" for c in self.sample_clips]
        self.clip_dropdown = ttk.Combobox(control_frame, textvariable=self.clip_var,
                                          values=clip_options, width=30, state='readonly')
        self.clip_dropdown.grid(row=0, column=1, padx=10)
        self.clip_dropdown.current(0)
        
        # Classify button
        self.classify_btn = tk.Button(control_frame, text="🎯 CLASSIFY", 
                                      command=self.classify_clip,
                                      font=('Helvetica', 12, 'bold'),
                                      bg='#00d4ff', fg='black',
                                      padx=20, pady=5)
        self.classify_btn.grid(row=0, column=2, padx=20)
        
        # Random button
        self.random_btn = tk.Button(control_frame, text="🎲 Random", 
                                    command=self.random_clip,
                                    font=('Helvetica', 10),
                                    bg='#333', fg='white',
                                    padx=10, pady=5)
        self.random_btn.grid(row=0, column=3, padx=5)
        
        # Result label
        self.result_var = tk.StringVar(value="Select a clip and click CLASSIFY")
        self.result_label = tk.Label(self.root, textvariable=self.result_var,
                                     font=('Helvetica', 14), fg='#00ff88', bg='#1a1a2e')
        self.result_label.pack(pady=10)
        
        # Confidence bar frame
        self.conf_frame = tk.Frame(self.root, bg='#1a1a2e')
        self.conf_frame.pack(pady=10, fill='x', padx=50)
        
        # Matplotlib figure for radar plot
        self.fig, self.ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
        self.fig.patch.set_facecolor('#1a1a2e')
        self.ax.set_facecolor('#1a1a2e')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(pady=10)
        
        # Ground truth label
        self.truth_var = tk.StringVar(value="")
        self.truth_label = tk.Label(self.root, textvariable=self.truth_var,
                                    font=('Helvetica', 10), fg='#888', bg='#1a1a2e')
        self.truth_label.pack(pady=5)
        
        # Initialize empty radar
        self.draw_radar([0.25, 0.25, 0.25, 0.25], None)
        
    def random_clip(self):
        """Select a random clip"""
        idx = random.randint(0, len(self.sample_clips) - 1)
        self.clip_dropdown.current(idx)
        
    def classify_clip(self):
        """Classify the selected clip"""
        # Get selected clip
        selection = self.clip_dropdown.current()
        clip_info = self.sample_clips[selection]
        
        # Load tensor
        tensor_path = self.tensor_dir / clip_info['tensor_file']
        if not tensor_path.exists():
            # Try alternative naming
            tensor_path = self.tensor_dir / f"tensor_{int(clip_info['clip_id']):06d}.npy"
        
        if not tensor_path.exists():
            self.result_var.set(f"❌ Tensor file not found")
            return
            
        waveform = np.load(tensor_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
            
        # Get embedding
        with torch.no_grad():
            embedding = self.encoder(waveform).numpy().flatten()
        
        # Compute distances to centroids
        distances = {}
        for cls_name, centroid in self.centroids.items():
            # Cosine distance
            cos_sim = np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid))
            distances[cls_name] = 1 - cos_sim  # Convert to distance
        
        # Convert distances to probabilities (softmax-like)
        dist_values = np.array([distances[c] for c in self.class_names])
        # Invert and normalize (smaller distance = higher probability)
        inv_dist = 1 / (dist_values + 0.1)
        probs = inv_dist / inv_dist.sum()
        
        # Get prediction
        pred_idx = np.argmax(probs)
        pred_class = self.class_names[pred_idx]
        confidence = probs[pred_idx] * 100
        
        # Update result
        self.result_var.set(f"🎯 Prediction: {pred_class.upper()}  ({confidence:.1f}%)")
        
        # Update ground truth
        self.truth_var.set(f"Ground Truth: {clip_info['class']}")
        
        # Check if correct
        if pred_class == clip_info['class']:
            self.result_label.config(fg='#00ff88')  # Green
        else:
            self.result_label.config(fg='#ff4444')  # Red
        
        # Draw radar plot
        self.draw_radar(probs, pred_class)
        
        # Update confidence bars
        self.update_conf_bars(probs)
        
    def draw_radar(self, probs, prediction):
        """Draw radar plot with class probabilities"""
        self.ax.clear()
        
        # Setup angles
        angles = np.linspace(0, 2 * np.pi, len(self.class_names), endpoint=False).tolist()
        probs_plot = list(probs) + [probs[0]]  # Close the polygon
        angles += angles[:1]
        
        # Plot
        self.ax.plot(angles, probs_plot, 'o-', linewidth=2, color='#00d4ff')
        self.ax.fill(angles, probs_plot, alpha=0.25, color='#00d4ff')
        
        # Labels
        self.ax.set_xticks(angles[:-1])
        labels = [c.replace('_', '\n').upper() for c in self.class_names]
        self.ax.set_xticklabels(labels, color='white', fontsize=9)
        
        # Styling
        self.ax.set_ylim(0, 1)
        self.ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        self.ax.set_yticklabels(['25%', '50%', '75%', '100%'], color='#666', fontsize=8)
        self.ax.grid(True, color='#333', linestyle='-', linewidth=0.5)
        self.ax.spines['polar'].set_color('#444')
        
        # Title
        if prediction:
            self.ax.set_title(f"Classification: {prediction.replace('_', ' ').title()}", 
                            color='#00d4ff', fontsize=12, pad=20)
        
        self.canvas.draw()
        
    def update_conf_bars(self, probs):
        """Update confidence bars"""
        # Clear existing
        for widget in self.conf_frame.winfo_children():
            widget.destroy()
            
        colors = {'cargo_ship': '#ff6b6b', 'fishing_vessel': '#ffa94d', 
                  'small_craft': '#69db7c', 'tanker': '#748ffc'}
        
        for i, (cls, prob) in enumerate(zip(self.class_names, probs)):
            # Class label
            lbl = tk.Label(self.conf_frame, text=cls.replace('_', ' ').title(),
                          font=('Helvetica', 9), fg='white', bg='#1a1a2e', width=12, anchor='e')
            lbl.grid(row=i, column=0, padx=5, pady=2)
            
            # Progress bar
            bar_frame = tk.Frame(self.conf_frame, bg='#333', height=15, width=300)
            bar_frame.grid(row=i, column=1, padx=5, pady=2)
            bar_frame.pack_propagate(False)
            
            fill = tk.Frame(bar_frame, bg=colors.get(cls, '#00d4ff'), 
                           width=int(300 * prob), height=15)
            fill.pack(side='left')
            
            # Percentage
            pct = tk.Label(self.conf_frame, text=f"{prob*100:.1f}%",
                          font=('Helvetica', 9), fg='#888', bg='#1a1a2e', width=6)
            pct.grid(row=i, column=2, padx=5, pady=2)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SKANNDemoGUI(root)
    root.mainloop()
