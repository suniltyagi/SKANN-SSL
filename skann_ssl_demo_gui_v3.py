"""
SKANN-SSL V2.1.0 Demo GUI (Fixed)
==================================
Simple GUI for demo video recording.
Select a clip → Click Classify → See radar plot + prediction

Fixes applied from working stage6_acoustic_sonar_classifier.py:
  - CPU-safe bundle loading (joblib_load_cpu)
  - Robust territory map parsing
  - Model import from Stage 3 train_script.py
  - Correct tensor path resolution

Usage:
    python skann_ssl_demo_gui_v2.py

Requirements:
    - SKANN_SSL_Production_Bundle.joblib
    - vessel_territories_v2_1_0.joblib
    - Tensor files in data/prototype_dataset/tensors/
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
# CONFIGURATION - Update these paths to match your setup
# =============================================================================
REPO_ROOT = r"C:\Users\Admin\uw_project\SKANN-SSL"

BUNDLE_PATH = os.path.join(REPO_ROOT, "stages", "stage3_ssl", "artifacts", "SKANN_SSL_Production_Bundle.joblib")
TERRITORIES_PATH = os.path.join(REPO_ROOT, "stages", "stage3_ssl", "artifacts", "territories", "vessel_territories_v2_1_0.joblib")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "prototype_dataset", "master_dataset_manifest.csv")
TENSORS_DIR = os.path.join(REPO_ROOT, "data", "prototype_dataset", "tensors")

# Add Stage 3 to path for model import
sys.path.insert(0, os.path.join(REPO_ROOT, "stages", "stage3_ssl"))

# =============================================================================
# HELPERS (from working Stage 6 code)
# =============================================================================
def norm_label(s: str) -> str:
    """Normalise label names so 'Cargo Ship', 'cargo-ship', 'cargo_ship' all match."""
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def _unwrap_0d(x):
    """Unwrap 0-d numpy object arrays (common in joblib dumps)."""
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
    """
    Build a dict: norm_label(label) -> centroid vector (np.ndarray)
    Supports legacy dict and structured territories packages.
    """
    if not isinstance(territories_obj, dict):
        raise TypeError(f"Unsupported territories type: {type(territories_obj)} (expected dict)")

    structured_keys = {"centroids", "vessel_classes", "class_to_idx", "metadata"}

    # Case A: already label->centroid (and not a structured package)
    if not (structured_keys & set(territories_obj.keys())):
        return {norm_label(k): _to_numpy(v) for k, v in territories_obj.items()}

    # Case B: structured package must have 'centroids'
    if "centroids" not in territories_obj:
        raise KeyError("Structured territories package missing 'centroids' key")

    centroids_obj = _unwrap_0d(territories_obj["centroids"])

    # B1: centroids is itself a dict[label->vector]
    if isinstance(centroids_obj, dict):
        return {norm_label(k): _to_numpy(v) for k, v in centroids_obj.items()}

    # B2: centroids is array-like
    centroids_arr = _to_numpy(centroids_obj)
    centroids_arr = _unwrap_0d(centroids_arr)
    centroids_arr = _to_numpy(centroids_arr)

    # Use 'vessel_classes' to align with rows of centroids
    if "vessel_classes" in territories_obj:
        classes_obj = _unwrap_0d(territories_obj["vessel_classes"])
        classes = [norm_label(c) for c in list(classes_obj)]
        return {classes[i]: _to_numpy(centroids_arr[i]) for i in range(len(classes))}

    # Fallback: 'class_to_idx'
    if "class_to_idx" in territories_obj:
        c2i = {norm_label(k): int(v) for k, v in territories_obj["class_to_idx"].items()}
        return {lab: _to_numpy(centroids_arr[idx]) for lab, idx in c2i.items()}

    raise KeyError("Structured territories has 'centroids' but missing both 'vessel_classes' and 'class_to_idx'.")


def joblib_load_cpu(path: str):
    """
    CPU-safe joblib loader for bundles that contain CUDA-saved torch storages.
    Works even when torch is CPU-only.
    """
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
# DEMO GUI CLASS
# =============================================================================
class SKANNDemoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SKANN-SSL V2.1.0 Demo")
        self.root.geometry("900x750")
        self.root.configure(bg='#1a1a2e')
        
        # Audio playback state
        self.audio_playing = False
        self.current_audio = None
        self.sample_rate = 16000  # Standard for this dataset
        
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
            pass  # GUI might not be ready yet
    
    def play_audio_loop(self, audio_data):
        """Play audio in a loop until stopped, with crossfade for smooth looping"""
        if not AUDIO_AVAILABLE:
            return
        
        self.stop_audio()
        
        # Create crossfaded loop for seamless playback
        audio_looped = self._create_crossfade_loop(audio_data)
        
        self.current_audio = audio_looped
        self.audio_playing = True
        
        def _loop_play():
            while self.audio_playing:
                try:
                    # Normalize audio for playback
                    audio_norm = audio_looped / (np.abs(audio_looped).max() + 1e-8)
                    sd.play(audio_norm, self.sample_rate)
                    sd.wait()  # Wait for playback to finish
                except Exception as e:
                    print(f"Audio playback error: {e}")
                    break
        
        # Run in background thread
        self.audio_thread = threading.Thread(target=_loop_play, daemon=True)
        self.audio_thread.start()
    
    def _create_crossfade_loop(self, audio, fade_ms=50):
        """
        Create a seamless loop by crossfading the end into the beginning.
        fade_ms: crossfade duration in milliseconds
        """
        fade_samples = int(self.sample_rate * fade_ms / 1000)
        fade_samples = min(fade_samples, len(audio) // 4)  # Don't exceed 25% of audio
        
        if fade_samples < 10:
            return audio  # Too short to crossfade
        
        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        
        # Copy audio to avoid modifying original
        looped = audio.copy()
        
        # Crossfade: blend end of audio with beginning
        # The last fade_samples of the audio blend with the first fade_samples
        looped[:fade_samples] = (audio[:fade_samples] * fade_in + 
                                  audio[-fade_samples:] * fade_out)
        
        # Trim the end (since it's now blended into the start)
        looped = looped[:-fade_samples]
        
        return looped
        
    def load_model(self):
        """Load encoder and territories using CPU-safe loader"""
        print("Loading model...")
        
        # Import model class from Stage 3 (authoritative source)
        try:
            from train_script import HybridSKEncoderV2
        except ImportError as e:
            print(f"⚠️  Could not import from train_script.py: {e}")
            print("    Falling back to inline model definition...")
            HybridSKEncoderV2 = self._get_fallback_model_class()
        
        # CPU-safe bundle load
        self.bundle = joblib_load_cpu(BUNDLE_PATH)
        
        if not isinstance(self.bundle, dict):
            raise TypeError(f"Bundle must be a dict. Got: {type(self.bundle)}")
        
        # Get labels
        if "vessel_labels" not in self.bundle:
            raise KeyError("Bundle missing key: 'vessel_labels'")
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
            raise KeyError("Bundle must contain either 'model' or 'model_state'.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load territories (CPU-safe)
        territories = joblib_load_cpu(TERRITORIES_PATH)
        self.territory_map = build_territory_map(territories, self.labels_raw)
        
        # Verify all labels have territories
        missing = [lab for lab in self.labels_norm if lab not in self.territory_map]
        if missing:
            print(f"⚠️  Missing territory centroids for: {missing}")
            print(f"   Available territories: {list(self.territory_map.keys())}")
        
        print(f"✅ Loaded model with {len(self.labels_raw)} classes on {self.device}")
        print(f"   Classes: {self.labels_raw}")
        
    def _get_fallback_model_class(self):
        """Fallback model definition if import fails"""
        import torch.nn as nn
        
        class SKConv1D(nn.Module):
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
            def __init__(self, out_ch=64, kernel_sizes=(31, 63, 127, 255, 511, 1023)):
                super().__init__()
                self.sk = SKConv1D(1, out_ch, kernel_sizes=kernel_sizes)

            def forward(self, x):
                return self.sk(x)

        class HybridSKEncoderV2(nn.Module):
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
        
        return HybridSKEncoderV2
        
    def load_clips(self):
        """Load available clips from manifest"""
        print("Loading clips...")
        
        self.manifest = pd.read_csv(MANIFEST_PATH)
        
        # Get sample clips (mix of classes)
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
        
        # Shuffle for variety in demo
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
        
        # Initialize empty radar
        self.draw_radar([0.25] * len(self.labels_raw), None)
        
    def random_clip(self):
        """Select a random clip"""
        self.stop_audio()  # Stop current audio
        idx = random.randint(0, len(self.sample_clips) - 1)
        self.clip_dropdown.current(idx)
        
    def classify_clip(self):
        """Classify the selected clip"""
        # Get selected clip
        selection = self.clip_dropdown.current()
        clip_info = self.sample_clips[selection]
        clip_id = clip_info['clip_id']
        
        # Build tensor path
        tensor_path = os.path.join(TENSORS_DIR, f"tensor_{clip_id:06d}.npy")
        
        if not os.path.exists(tensor_path):
            self.result_var.set(f"❌ Tensor file not found: {tensor_path}")
            return
            
        # Load audio data
        audio = np.load(tensor_path)
        
        # Start audio playback in loop
        if AUDIO_AVAILABLE:
            self.play_audio_loop(audio.flatten())
            self.audio_status_var.set("🔊 Playing audio (looped)")
        
        # Prepare tensor for model
        tensor = torch.from_numpy(audio).float().view(1, 1, -1).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            fingerprint = self.model(tensor).detach().cpu().numpy().flatten()
        
        # Compute distances to territory centroids (using normalized labels)
        dists = []
        for lab in self.labels_raw:
            lab_norm = norm_label(lab)
            centroid = self.territory_map[lab_norm]
            dist = np.linalg.norm(fingerprint - centroid)
            dists.append(dist)
        
        # Convert distances to probabilities (smaller distance = higher probability)
        scores = 1.0 / (np.asarray(dists) + 0.8)
        probs = scores / np.sum(scores)
        
        # Get prediction
        pred_idx = int(np.argmax(probs))
        pred_class = self.labels_raw[pred_idx]
        confidence = float(probs[pred_idx])
        actual_class = clip_info['class']
        
        # Update result
        self.result_var.set(f"🎯 Prediction: {pred_class.upper()}  ({confidence*100:.1f}%)")
        
        # Update ground truth
        self.truth_var.set(f"Ground Truth: {actual_class}")
        
        # Check if correct
        is_correct = (norm_label(pred_class) == norm_label(actual_class))
        if is_correct:
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
        num_vars = len(self.labels_raw)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        probs_plot = list(probs) + [probs[0]]  # Close the polygon
        angles += angles[:1]
        
        # Plot
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
        
        # Highlight predicted class label
        if prediction:
            pred_norm = norm_label(prediction)
            for i, lab in enumerate(self.labels_raw):
                if norm_label(lab) == pred_norm:
                    labels = self.ax.get_xticklabels()
                    if i < len(labels):
                        labels[i].set_color('#ff6b6b')
                        labels[i].set_fontweight('bold')
        
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
            
        colors = {
            'cargo_ship': '#ff6b6b', 
            'fishing_vessel': '#ffa94d', 
            'small_craft': '#69db7c', 
            'tanker': '#748ffc'
        }
        
        for i, (lab, prob) in enumerate(zip(self.labels_raw, probs)):
            lab_norm = norm_label(lab)
            
            # Class label
            display_name = lab.replace('_', ' ').title()
            lbl = tk.Label(self.conf_frame, text=display_name,
                          font=('Helvetica', 9), fg='white', bg='#1a1a2e', width=14, anchor='e')
            lbl.grid(row=i, column=0, padx=5, pady=2)
            
            # Progress bar
            bar_frame = tk.Frame(self.conf_frame, bg='#333', height=18, width=300)
            bar_frame.grid(row=i, column=1, padx=5, pady=2)
            bar_frame.pack_propagate(False)
            
            color = colors.get(lab_norm, '#00d4ff')
            fill_width = int(300 * prob)
            fill = tk.Frame(bar_frame, bg=color, width=fill_width, height=18)
            fill.pack(side='left')
            
            # Percentage
            pct = tk.Label(self.conf_frame, text=f"{prob*100:.1f}%",
                          font=('Helvetica', 9), fg='#888', bg='#1a1a2e', width=6)
            pct.grid(row=i, column=2, padx=5, pady=2)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Verify paths exist
    print("=" * 60)
    print("SKANN-SSL V2.1.0 Demo GUI")
    print("=" * 60)
    
    missing = []
    if not os.path.exists(BUNDLE_PATH):
        missing.append(f"Bundle: {BUNDLE_PATH}")
    if not os.path.exists(TERRITORIES_PATH):
        missing.append(f"Territories: {TERRITORIES_PATH}")
    if not os.path.exists(MANIFEST_PATH):
        missing.append(f"Manifest: {MANIFEST_PATH}")
    if not os.path.isdir(TENSORS_DIR):
        missing.append(f"Tensors dir: {TENSORS_DIR}")
    
    if missing:
        print("❌ Missing required files:")
        for m in missing:
            print(f"   - {m}")
        print("\nPlease update the paths at the top of this script.")
        sys.exit(1)
    
    print(f"✅ All paths verified")
    print(f"   Bundle: {BUNDLE_PATH}")
    print(f"   Territories: {TERRITORIES_PATH}")
    print()
    
    root = tk.Tk()
    app = SKANNDemoGUI(root)
    root.mainloop()
