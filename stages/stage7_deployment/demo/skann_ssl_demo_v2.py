"""
SKANN-SSL V2.2.0 Demo
=====================
Underwater Acoustic Vessel Classification System

Location: stages/stage7_deployment/demo/

Usage:
    python skann_ssl_demo_v2.py

Requirements:
    pip install -r requirements.txt

NOTE: Requires model/ and data/tensors/ to be populated.
      See README.md for setup instructions.

© 2026 Oravont Systems LLP. All rights reserved.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import threading

# Try to import sounddevice for audio playback
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  sounddevice not installed. Audio playback disabled.")
    print("   Install with: pip install sounddevice")

# =============================================================================
# PATHS (relative to script location)
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BUNDLE_PATH = os.path.join(SCRIPT_DIR, "model", "SKANN_SSL_Production_Bundle.joblib")
TERRITORIES_PATH = os.path.join(SCRIPT_DIR, "model", "vessel_territories.joblib")
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "data", "manifest.csv")
TENSORS_DIR = os.path.join(SCRIPT_DIR, "data", "tensors")

# =============================================================================
# HELPERS
# =============================================================================
def norm_label(s: str) -> str:
    """Normalise label names."""
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def _unwrap_0d(x):
    """Unwrap 0-d numpy object arrays."""
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    return x


def _to_numpy(v):
    """Convert torch tensors / lists into numpy arrays safely."""
    v = _unwrap_0d(v)
    if hasattr(v, "detach") and hasattr(v, "cpu"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def build_territory_map(territories_obj, labels_raw):
    """Build a dict: norm_label(label) -> centroid vector."""
    if not isinstance(territories_obj, dict):
        raise TypeError(f"Unsupported territories type: {type(territories_obj)}")

    structured_keys = {"centroids", "vessel_classes", "class_to_idx", "metadata"}

    if not (structured_keys & set(territories_obj.keys())):
        return {norm_label(k): _to_numpy(v) for k, v in territories_obj.items()}

    if "centroids" not in territories_obj:
        raise KeyError("Territories missing 'centroids' key")

    centroids_obj = _unwrap_0d(territories_obj["centroids"])

    if isinstance(centroids_obj, dict):
        return {norm_label(k): _to_numpy(v) for k, v in centroids_obj.items()}

    centroids_arr = _to_numpy(centroids_obj)
    centroids_arr = _unwrap_0d(centroids_arr)
    centroids_arr = _to_numpy(centroids_arr)

    if "vessel_classes" in territories_obj:
        classes_obj = _unwrap_0d(territories_obj["vessel_classes"])
        classes = [norm_label(c) for c in list(classes_obj)]
        return {classes[i]: _to_numpy(centroids_arr[i]) for i in range(len(classes))}

    if "class_to_idx" in territories_obj:
        c2i = {norm_label(k): int(v) for k, v in territories_obj["class_to_idx"].items()}
        return {lab: _to_numpy(centroids_arr[idx]) for lab, idx in c2i.items()}

    raise KeyError("Territories missing 'vessel_classes' or 'class_to_idx'")


def joblib_load_cpu(path: str):
    """CPU-safe joblib loader for CUDA-saved bundles."""
    _orig = torch.load

    def _torch_load_cpu(*args, **kwargs):
        kwargs["map_location"] = torch.device("cpu")
        return _orig(*args, **kwargs)

    torch.load = _torch_load_cpu
    try:
        return joblib.load(path)
    finally:
        torch.load = _orig


# =============================================================================
# MODEL ARCHITECTURE - V2.1.0 (Exact match to trained weights)
# =============================================================================

def _norm_1d(channels, kind='gn', groups=8):
    """1D normalization layer factory."""
    if kind == 'bn':
        return nn.BatchNorm1d(channels)
    if kind == 'ln':
        return nn.GroupNorm(1, channels)
    return nn.GroupNorm(min(groups, channels), channels)


class SKConv1D(nn.Module):
    """
    Selective Kernel 1D Convolution - V2.1.0
    Underwater-appropriate kernel sizes for vessel acoustics.
    """
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_sizes: tuple = (31, 63, 127, 255, 511, 1023),
        stride: int = 1,
        reduction: int = 16,
        norm: str = 'gn',
        act: str = 'gelu',
        residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Multi-branch convolutions with LARGE kernels for low frequencies
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2
            self.branches.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, 
                         padding=pad, bias=False)
            )
        
        self.n_branches = len(kernel_sizes)
        self.out_ch = out_ch
        self.kernel_sizes = kernel_sizes
        
        # Attention MLP
        hidden = max(out_ch // reduction, 8)
        self.fc1 = nn.Linear(out_ch, hidden)
        self.fc2 = nn.Linear(hidden, out_ch * self.n_branches)
        
        # Normalization and activation
        self.norm = _norm_1d(out_ch, norm)
        self.act = nn.GELU() if act == 'gelu' else nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Residual connection
        self.residual = residual
        self.match = None
        if residual and (in_ch != out_ch or stride != 1):
            self.match = nn.Conv1d(in_ch, out_ch, kernel_size=1, 
                                   stride=stride, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        U = torch.stack(feats, dim=1).sum(dim=1)
        s = F.adaptive_avg_pool1d(U, 1).squeeze(-1)
        z = self.fc2(F.relu(self.fc1(s), inplace=False))
        a = z.view(z.size(0), self.n_branches, self.out_ch)
        a = F.softmax(a, dim=1).unsqueeze(-1)
        feats_stacked = torch.stack(feats, dim=1)
        V = (a * feats_stacked).sum(dim=1)
        out = self.norm(V)
        out = self.act(out)
        out = self.dropout(out)
        if self.residual:
            res = x if self.match is None else self.match(x)
            out = out + res
        return out


class SKFilterbank(nn.Module):
    """Selective Kernel Filterbank - V2.1.0"""
    
    def __init__(
        self,
        out_ch: int = 64,
        kernel_sizes: tuple = (31, 63, 127, 255, 511, 1023),
        norm: str = 'gn'
    ):
        super().__init__()
        
        self.stem = SKConv1D(
            in_ch=1,
            out_ch=out_ch,
            kernel_sizes=kernel_sizes,
            stride=1,
            reduction=16,
            norm=norm,
            act='gelu',
            residual=False
        )
        self.post_norm = _norm_1d(out_ch, norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        return self.post_norm(h)


class HybridSKEncoderV2(nn.Module):
    """V2.1.0 Encoder - Underwater-Appropriate Architecture"""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # SK Frontend with underwater-appropriate kernels
        self.sk_frontend = SKFilterbank(
            out_ch=64,
            kernel_sizes=(31, 63, 127, 255, 511, 1023)
        )
        self.downsample = nn.AvgPool1d(kernel_size=8, stride=8)
        
        # Channel bridge
        self.channel_bridge = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=False)
        )
        
        # 2D Backbone
        self.backbone2d = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, 3, padding=1, stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 256, 3, padding=1, stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        
        # Large projector (critical for Barlow Twins)
        self.projector = nn.Sequential(
            nn.Linear(512, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 8192),
            nn.LayerNorm(8192),
            nn.ReLU(inplace=False),
            nn.Linear(8192, latent_dim)
        )
    
    def forward(self, x):
        if x.dim() > 3:
            x = x.view(x.size(0), -1).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.sk_frontend(x)
        x = self.downsample(x)
        x = self.channel_bridge(x)
        x = x.unsqueeze(1)
        x = self.backbone2d(x)
        return self.projector(x)


# =============================================================================
# DEMO GUI
# =============================================================================
class SKANNDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("SKANN-SSL V2.2.0 — Underwater Vessel Classifier")
        self.root.geometry("900x750")
        self.root.configure(bg='#1a1a2e')
        
        # Audio state
        self.audio_playing = False
        self.current_audio = None
        self.sample_rate = 16000
        
        # Load model and data
        self.load_model()
        self.load_clips()
        
        # Build GUI
        self.build_gui()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        """Clean up audio on window close"""
        self.stop_audio()
        self.root.destroy()
    
    def stop_audio(self):
        """Stop any currently playing audio"""
        if AUDIO_AVAILABLE and self.audio_playing:
            sd.stop()
            self.audio_playing = False
        try:
            self.audio_status_var.set("")
        except:
            pass
    
    def play_audio_loop(self, audio_data):
        """Play audio in a loop with crossfade for smooth looping"""
        if not AUDIO_AVAILABLE:
            return
        
        self.stop_audio()
        audio_looped = self._create_crossfade_loop(audio_data)
        self.current_audio = audio_looped
        self.audio_playing = True
        
        def _loop_play():
            while self.audio_playing:
                try:
                    audio_norm = audio_looped / (np.abs(audio_looped).max() + 1e-8)
                    sd.play(audio_norm, self.sample_rate)
                    sd.wait()
                except Exception as e:
                    print(f"Audio error: {e}")
                    break
        
        self.audio_thread = threading.Thread(target=_loop_play, daemon=True)
        self.audio_thread.start()
    
    def _create_crossfade_loop(self, audio, fade_ms=50):
        """Create seamless loop with crossfade."""
        fade_samples = int(self.sample_rate * fade_ms / 1000)
        fade_samples = min(fade_samples, len(audio) // 4)
        
        if fade_samples < 10:
            return audio
        
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        
        looped = audio.copy()
        looped[:fade_samples] = (audio[:fade_samples] * fade_in + 
                                  audio[-fade_samples:] * fade_out)
        looped = looped[:-fade_samples]
        
        return looped
        
    def load_model(self):
        """Load encoder and territories"""
        print("Loading SKANN-SSL model...")
        
        # CPU-safe bundle load
        self.bundle = joblib_load_cpu(BUNDLE_PATH)
        
        # Get labels
        if "vessel_labels" not in self.bundle:
            raise KeyError("Bundle missing 'vessel_labels'")
        self.labels_raw = list(self.bundle["vessel_labels"])
        self.labels_norm = [norm_label(lab) for lab in self.labels_raw]
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if "model" in self.bundle:
            self.model = self.bundle["model"]
        elif "model_state" in self.bundle:
            self.model = HybridSKEncoderV2()
            self.model.load_state_dict(self.bundle["model_state"])
        else:
            raise KeyError("Bundle missing 'model' or 'model_state'")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load territories
        territories = joblib_load_cpu(TERRITORIES_PATH)
        self.territory_map = build_territory_map(territories, self.labels_raw)
        
        print(f"✅ Model loaded ({len(self.labels_raw)} classes) on {self.device}")
        
    def load_clips(self):
        """Load available clips from manifest"""
        print("Loading audio clips...")
        
        self.manifest = pd.read_csv(MANIFEST_PATH)
        
        self.sample_clips = []
        for cls in self.labels_raw:
            cls_clips = self.manifest[self.manifest['vessel_class'] == cls].head(15)
            for _, row in cls_clips.iterrows():
                clip_id = int(row['clip_id'])
                self.sample_clips.append({
                    'clip_id': clip_id,
                    'class': row['vessel_class'],
                    'tensor_file': f"tensor_{clip_id:06d}.npy"
                })
        
        random.shuffle(self.sample_clips)
        print(f"✅ Loaded {len(self.sample_clips)} sample clips")
        
    def build_gui(self):
        """Build the GUI layout"""
        # Title
        title = tk.Label(self.root, text="🔊 SKANN-SSL V2.2.0", 
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
        
        # Stop audio button
        self.stop_btn = tk.Button(control_frame, text="🔇 Stop", 
                                  command=self.stop_audio,
                                  font=('Helvetica', 10),
                                  bg='#8B0000', fg='white',
                                  padx=10, pady=5)
        self.stop_btn.grid(row=0, column=4, padx=5)
        
        # Audio status label
        self.audio_status_var = tk.StringVar(value="")
        self.audio_status = tk.Label(self.root, textvariable=self.audio_status_var,
                                     font=('Helvetica', 10), fg='#ffa500', bg='#1a1a2e')
        self.audio_status.pack(pady=2)
        
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
                                    font=('Helvetica', 11), fg='#888', bg='#1a1a2e')
        self.truth_label.pack(pady=5)
        
        # Copyright
        copyright_label = tk.Label(self.root, 
                                   text="© 2026 Oravont Systems LLP",
                                   font=('Helvetica', 9), fg='#444', bg='#1a1a2e')
        copyright_label.pack(side='bottom', pady=10)
        
        # Initialize empty radar
        self.draw_radar([0.25] * len(self.labels_raw), None)
        
    def random_clip(self):
        """Select a random clip"""
        self.stop_audio()
        idx = random.randint(0, len(self.sample_clips) - 1)
        self.clip_dropdown.current(idx)
        
    def classify_clip(self):
        """Classify the selected clip"""
        selection = self.clip_dropdown.current()
        clip_info = self.sample_clips[selection]
        clip_id = clip_info['clip_id']
        
        tensor_path = os.path.join(TENSORS_DIR, f"tensor_{clip_id:06d}.npy")
        
        if not os.path.exists(tensor_path):
            self.result_var.set(f"❌ File not found")
            return
            
        # Load audio
        audio = np.load(tensor_path)
        
        # Start audio playback
        if AUDIO_AVAILABLE:
            self.play_audio_loop(audio.flatten())
            self.audio_status_var.set("🔊 Playing audio (looped)")
        
        # Prepare tensor
        tensor = torch.from_numpy(audio).float().view(1, 1, -1).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            fingerprint = self.model(tensor).detach().cpu().numpy().flatten()
        
        # Compute distances
        dists = []
        for lab in self.labels_raw:
            centroid = self.territory_map[norm_label(lab)]
            dist = np.linalg.norm(fingerprint - centroid)
            dists.append(dist)
        
        # Convert to probabilities
        scores = 1.0 / (np.asarray(dists) + 0.8)
        probs = scores / np.sum(scores)
        
        # Get prediction
        pred_idx = int(np.argmax(probs))
        pred_class = self.labels_raw[pred_idx]
        confidence = float(probs[pred_idx])
        actual_class = clip_info['class']
        
        # Update display
        self.result_var.set(f"🎯 Prediction: {pred_class.upper()}  ({confidence*100:.1f}%)")
        self.truth_var.set(f"Ground Truth: {actual_class}")
        
        is_correct = (norm_label(pred_class) == norm_label(actual_class))
        self.result_label.config(fg='#00ff88' if is_correct else '#ff4444')
        
        self.draw_radar(probs, pred_class)
        self.update_conf_bars(probs)
        
    def draw_radar(self, probs, prediction):
        """Draw radar plot with class probabilities"""
        self.ax.clear()
        
        num_vars = len(self.labels_raw)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        probs_plot = list(probs) + [probs[0]]
        angles += angles[:1]
        
        self.ax.fill(angles, probs_plot, alpha=0.25, color='#00d4ff')
        self.ax.plot(angles, probs_plot, 'o-', linewidth=2, color='#00d4ff')
        
        # Pretty labels
        pretty_labels = []
        for lab in self.labels_raw:
            key = norm_label(lab)
            if key == "cargo_ship":
                pretty_labels.append("Cargo\nShip")
            elif key == "small_craft":
                pretty_labels.append("Small\nCraft")
            elif key == "fishing_vessel":
                pretty_labels.append("Fishing\nVessel")
            else:
                pretty_labels.append(str(lab).replace("_", "\n").title())
        
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(pretty_labels, color='white', fontsize=10)
        
        # Highlight predicted class
        if prediction:
            pred_norm = norm_label(prediction)
            for i, lab in enumerate(self.labels_raw):
                if norm_label(lab) == pred_norm:
                    labels = self.ax.get_xticklabels()
                    if i < len(labels):
                        labels[i].set_color('#ff6b6b')
                        labels[i].set_fontweight('bold')
        
        self.ax.set_ylim(0, 1)
        self.ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        self.ax.set_yticklabels(['25%', '50%', '75%', '100%'], color='#666', fontsize=8)
        self.ax.grid(True, color='#333', linestyle='-', linewidth=0.5)
        self.ax.spines['polar'].set_color('#444')
        
        if prediction:
            self.ax.set_title(f"Classification: {prediction.replace('_', ' ').title()}", 
                            color='#00d4ff', fontsize=12, pad=20)
        
        self.canvas.draw()
        
    def update_conf_bars(self, probs):
        """Update confidence bars"""
        for widget in self.conf_frame.winfo_children():
            widget.destroy()
            
        colors = {
            'cargo_ship': '#ff6b6b', 
            'fishing_vessel': '#ffa94d', 
            'small_craft': '#69db7c', 
            'tanker': '#748ffc'
        }
        
        for i, (lab, prob) in enumerate(zip(self.labels_raw, probs)):
            lab_norm = norm_label(lab)
            
            display_name = lab.replace('_', ' ').title()
            lbl = tk.Label(self.conf_frame, text=display_name,
                          font=('Helvetica', 9), fg='white', bg='#1a1a2e', width=14, anchor='e')
            lbl.grid(row=i, column=0, padx=5, pady=2)
            
            bar_frame = tk.Frame(self.conf_frame, bg='#333', height=18, width=300)
            bar_frame.grid(row=i, column=1, padx=5, pady=2)
            bar_frame.pack_propagate(False)
            
            color = colors.get(lab_norm, '#00d4ff')
            fill_width = int(300 * prob)
            fill = tk.Frame(bar_frame, bg=color, width=fill_width, height=18)
            fill.pack(side='left')
            
            pct = tk.Label(self.conf_frame, text=f"{prob*100:.1f}%",
                          font=('Helvetica', 9), fg='#888', bg='#1a1a2e', width=6)
            pct.grid(row=i, column=2, padx=5, pady=2)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SKANN-SSL V2.2.0 — Underwater Vessel Classifier")
    print("© 2026 Oravont Systems LLP")
    print("=" * 60)
    
    # Verify paths
    missing = []
    if not os.path.exists(BUNDLE_PATH):
        missing.append(f"Model bundle: model/SKANN_SSL_Production_Bundle.joblib")
    if not os.path.exists(TERRITORIES_PATH):
        missing.append(f"Territories: model/vessel_territories.joblib")
    if not os.path.exists(MANIFEST_PATH):
        missing.append(f"Manifest: data/manifest.csv")
    if not os.path.isdir(TENSORS_DIR):
        missing.append(f"Tensors: data/tensors/")
    
    if missing:
        print("\n❌ Missing required files:")
        for m in missing:
            print(f"   - {m}")
        print("\nExpected folder structure:")
        print("   SKANN-SSL-Demo/")
        print("   ├── skann_ssl_demo.py")
        print("   ├── requirements.txt")
        print("   ├── model/")
        print("   │   ├── SKANN_SSL_Production_Bundle.joblib")
        print("   │   └── vessel_territories.joblib")
        print("   └── data/")
        print("       ├── manifest.csv")
        print("       └── tensors/")
        print("           ├── tensor_000000.npy")
        print("           └── ...")
        sys.exit(1)
    
    print("✅ All files found")
    print()
    
    root = tk.Tk()
    app = SKANNDemo(root)
    root.mainloop()
