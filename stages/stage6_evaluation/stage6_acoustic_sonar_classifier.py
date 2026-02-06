# stages/stage6_evaluation/stage6_acoustic_sonar_classifier.py
# V3/V5 Stage-6 Radar Classifier
# Fixes:
# - CPU-safe bundle load (CUDA-saved tensors on CPU-only torch)
# - Robust structured territories parsing (centroids/vessel_classes/class_to_idx)
# - Label normalisation for stable lookup
# - Manifest-relative tensor path resolution

import os
import sys
import csv
import time
from datetime import datetime

import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 0) Helpers
# -----------------------------
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

    Supports:
      A) legacy dict[label -> centroid]
      B) structured territories package with keys like:
         'centroids', 'vessel_classes', 'class_to_idx', 'metadata', ...
         where centroids may be:
           - (K,D) numpy array
           - list[K] of vectors
           - torch tensor
           - 0-d numpy object containing any of the above
           - dict[label -> vector] stored under 'centroids'
    """
    if not isinstance(territories_obj, dict):
        raise TypeError(f"Unsupported territories type: {type(territories_obj)} (expected dict)")

    structured_keys = {"centroids", "vessel_classes", "class_to_idx", "metadata", "vessel_classes", "class_to_idx"}

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

    # B2: centroids is array-like (possibly boxed)
    centroids_arr = _to_numpy(centroids_obj)

    # If still boxed after conversion (rare), unwrap again
    centroids_arr = _unwrap_0d(centroids_arr)
    centroids_arr = _to_numpy(centroids_arr)

    # Now centroids_arr should be sized (K, D) or list-like K
    # Preferred: 'vessel_classes' aligns with rows of centroids
    if "vessel_classes" in territories_obj:
        classes_obj = _unwrap_0d(territories_obj["vessel_classes"])
        classes = [norm_label(c) for c in list(classes_obj)]
        # Ensure centroids_arr has length
        try:
            k_cent = len(centroids_arr)
        except TypeError as e:
            raise TypeError(
                "centroids could not be sized (still unsized after unwrapping). "
                "Inspect territories['centroids'] type/contents."
            ) from e

        if len(classes) != k_cent:
            raise ValueError(f"vessel_classes length {len(classes)} != centroids rows {k_cent}")

        return {classes[i]: _to_numpy(centroids_arr[i]) for i in range(len(classes))}

    # Fallback: 'class_to_idx' tells which centroid row to use
    if "class_to_idx" in territories_obj:
        c2i = {norm_label(k): int(v) for k, v in territories_obj["class_to_idx"].items()}
        return {lab: _to_numpy(centroids_arr[idx]) for lab, idx in c2i.items()}

    raise KeyError("Structured territories has 'centroids' but missing both 'vessel_classes' and 'class_to_idx'.")


def joblib_load_cpu(path: str):
    """
    CPU-safe joblib loader for bundles that contain CUDA-saved torch storages.
    Works even when torch is CPU-only (torch==...+cpu).
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


def repo_root_from_here() -> str:
    # stage6_evaluation is at stages/stage6_evaluation -> repo root is ../../
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# -----------------------------
# 1) Imports from Stage-3 (optional — V3 bundle stores full model object)
# -----------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stage3_ssl")))
_HAS_ENCODER_CLASS = False
try:
    from barlow_twins import HybridSKEncoderV3  # V3 architecture
    _HAS_ENCODER_CLASS = True
except ImportError:
    try:
        from train_script import HybridSKEncoderV2  # V2 fallback
        HybridSKEncoderV3 = HybridSKEncoderV2
        _HAS_ENCODER_CLASS = True
    except ImportError:
        pass  # Bundle must contain 'model' key (full model object)


# -----------------------------
# 2) Stage-6 artifacts directory
# -----------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


class AcousticRadarEngine:
    def __init__(self, bundle_path: str, territories_path: str, manifest_path: str | None = None):
        print("🔍 Synchronizing Acoustic Assets...")

        # ---- Bundle (CPU-safe load) ----
        self.bundle = joblib_load_cpu(bundle_path)
        if not isinstance(self.bundle, dict):
            raise TypeError(f"Bundle must be a dict. Got: {type(self.bundle)}")

        # ---- Labels ----
        if "vessel_labels" not in self.bundle:
            raise KeyError("Bundle missing key: 'vessel_labels'")
        self.labels_raw = list(self.bundle["vessel_labels"])
        self.labels_norm = [norm_label(lab) for lab in self.labels_raw]

        # ---- Territories ----
        self.territories = joblib.load(territories_path)
        self.territory_map = build_territory_map(self.territories, self.labels_raw)

        missing = [lab for lab in self.labels_norm if lab not in self.territory_map]
        if missing:
            print("❌ Territory/Label mismatch detected.")
            print("   Missing labels in territories (normalised):", missing)
            print("   First 30 territory keys (normalised):", list(self.territory_map.keys())[:30])
            # Also show whether we loaded a structured package
            if isinstance(self.territories, dict):
                print("   Territories top-level keys:", list(self.territories.keys()))
            raise KeyError("Territories do not contain centroids for all bundle labels.")

        # ---- Manifest ----
        if manifest_path is None:
            manifest_path = os.path.join(repo_root_from_here(), "data", "v5_dataset", "master_dataset_manifest.csv")

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        print(f"📖 Loaded Manifest: {os.path.relpath(manifest_path, repo_root_from_here())}")
        self.manifest = pd.read_csv(manifest_path)
        self.manifest_base_dir = os.path.dirname(os.path.abspath(manifest_path))

        # ---- Device + Model ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "model" in self.bundle:
            self.model = self.bundle["model"]
        elif "model_state" in self.bundle:
            if not _HAS_ENCODER_CLASS:
                raise ImportError(
                    "Bundle contains 'model_state' but no encoder class could be imported. "
                    "Ensure barlow_twins.py or train_script.py is in stages/stage3_ssl/."
                )
            self.model = HybridSKEncoderV3()
            self.model.load_state_dict(self.bundle["model_state"])
        else:
            raise KeyError("Bundle must contain either 'model' or 'model_state'.")

        self.model.to(self.device)
        self.model.eval()
        print(f"✅ System Ready (Running on {self.device}).")

    def _resolve_tensor_path(self, filename: str) -> str:
        filename = str(filename).replace("/", os.sep).replace("\\", os.sep)

        # If already absolute, just normalise
        if os.path.isabs(filename):
            return os.path.normpath(filename)

        # Resolve relative to manifest directory
        candidate = os.path.normpath(os.path.join(self.manifest_base_dir, filename))

        # De-duplicate if manifest paths already include "data/v5_dataset/..."
        # Example bad: <base>\data\v5_dataset\tensors\...
        # where base already ends with ...\data\v5_dataset
        base_norm = os.path.normpath(self.manifest_base_dir)
        dup_prefix = os.path.normpath(os.path.join(base_norm, "data", "v5_dataset"))

        if candidate.startswith(dup_prefix + os.sep):
            candidate = os.path.normpath(os.path.join(base_norm, candidate[len(dup_prefix) + 1:]))

        return candidate


    def classify(self, clip_id: str):
        if self.manifest is None:
            return None, None, None, "Missing"

        if not str(clip_id).isdigit():
            return None, None, None, "Missing"

        clip_int = int(clip_id)
        clip_str = str(clip_int).zfill(6)

        # Identify ID column
        id_col = "clip_id" if "clip_id" in self.manifest.columns else "anchor_clip_id"
        match = self.manifest[self.manifest[id_col].astype(int) == clip_int]
        if match.empty:
            return None, None, None, "Missing"

        actual_label_raw = match["vessel_class"].values[0] if "vessel_class" in match.columns else "Unknown"

        # Tensor path from manifest
        if "tensor_path" in match.columns:
            filename = self._resolve_tensor_path(match["tensor_path"].values[0])
        else:
            filename = os.path.join(repo_root_from_here(), "data", "v5_dataset", "tensors", f"tensor_{clip_str}.npy")

        if not os.path.exists(filename):
            print(f"   ❌ Tensor file missing: {filename}")
            return None, None, None, "Missing"

        # INFERENCE
        audio = np.load(filename)
        tensor = torch.from_numpy(audio).float().view(1, 1, -1).to(self.device)

        with torch.no_grad():
            h, z = self.model(tensor, return_features=True)
            fingerprint = h.detach().cpu().numpy().flatten()

        # DISTANCE TO TERRITORIES (aligned by bundle label order)
        dists = [
            np.linalg.norm(fingerprint - self.territory_map[norm_label(lab)])
            for lab in self.labels_raw
        ]
        scores = 1.0 / (np.asarray(dists) + 0.8)
        probs = scores / np.sum(scores)

        pred_idx = int(np.argmax(probs))
        pred_label_raw = self.labels_raw[pred_idx]
        conf = float(np.max(probs))

        return probs, pred_label_raw, conf, str(actual_label_raw)


    def plot_radar(self, probabilities, clip_id, pred, conf, actual):
        display_labels = self.labels_raw

        # --- FIX: compute pred_idx inside plot_radar (robust to formatting differences) ---
        pred_norm = norm_label(pred)
        labels_norm = [norm_label(l) for l in display_labels]
        try:
            pred_idx = labels_norm.index(pred_norm)
        except ValueError:
            # Fallback: if pred isn't in labels for any reason, highlight argmax
            pred_idx = int(np.argmax(probabilities))

        num_vars = len(display_labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        stats = probabilities.tolist() + [probabilities[0]]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.set_ylim(0, 1.0)

        # Base radar (muted)
        ax.fill(angles, stats, alpha=0.15, color="tab:blue")
        ax.plot(angles, stats, linewidth=2, color="tab:blue", alpha=0.6)

        # # Highlight predicted class (the winning axis segment)
        # i0 = pred_idx
        # i1 = pred_idx + 1  # safe because angles/stats are "closed" with one extra point
        # highlight_angles = [angles[i0], angles[i1]]
        # highlight_stats  = [stats[i0], stats[i1]]

        # ax.plot(
            # highlight_angles,
            # highlight_stats,
            # linewidth=6,
            # color="crimson",
            # marker="o",
            # markersize=10,
            # zorder=10
        # )

        ax.set_xticks(angles[:-1])
    
        pretty_labels = []
        for lab in display_labels:
            key = norm_label(lab)
            if key == "cargo_ship":
                pretty_labels.append("Cargo\nShip")
            elif key == "small_craft":
                pretty_labels.append("Small\nCraft")
            else:
                pretty_labels.append(str(lab).replace("_", " ").title())

        ax.set_xticklabels(pretty_labels, fontsize=12)

        # Highlight predicted axis label
        for i, label in enumerate(ax.get_xticklabels()):
            if i == pred_idx:
                label.set_color("crimson")
                label.set_fontweight("bold")
                label.set_fontsize(14)

        ax.tick_params(axis="x", pad=18)

        # Correct/incorrect status (normalised compare)
        is_correct = (norm_label(pred) == norm_label(actual))
        if actual == "Unknown":
            status_text = "Label\nunknown"
        else:
            status_text = "Correct\nprediction" if is_correct else "Incorrect\nprediction"

        status = ax.text(
            0.86, 0.98,
            status_text,
            transform=ax.transAxes,
            ha="left", va="top",
            multialignment="left",
            fontsize=12, fontweight="bold",
        )

        # --- Top header with highlighted IDENTIFIED line ---
        fig.text(
            0.5, 0.985,
            f"CLIP ID: {clip_id}",
            ha="center", va="top",
            fontsize=10, fontweight="bold",
        )
        fig.text(
            0.5, 0.955,
            f"LABELED AS: {str(actual).upper()}",
            ha="center", va="top",
            fontsize=10, fontweight="bold",
        )
        fig.text(
            0.5, 0.925,
            f"IDENTIFIED: {str(pred).upper()} ({conf*100:.1f}%)",
            ha="center", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(facecolor="crimson", alpha=0.18, edgecolor="crimson", boxstyle="round,pad=0.35"),
            color="black",
        )

        fig.subplots_adjust(top=0.82)
        fig.canvas.draw()
        plt.pause(0.10)

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
    _root = repo_root_from_here()
    engine = AcousticRadarEngine(
        bundle_path=os.path.join(_root, "stages", "stage3_ssl", "artifacts", "SKANN_SSL_V3_Production_Bundle.joblib"),
        territories_path=os.path.join(_root, "stages", "stage3_ssl", "artifacts", "vessel_territories_v3.joblib"),
        manifest_path=os.path.join(_root, "data", "v5_dataset", "master_dataset_manifest.csv"),
    )

    LOG_PATH = os.path.join(ARTIFACTS_DIR, "stage6_per_query_results_log.csv")
    LOG_FIELDNAMES = (
        ["timestamp", "clip_id", "actual_label", "predicted_label", "pred_confidence", "radar_plot_path"]
        + [f"p_{lab}" for lab in engine.labels_raw]
    )
    LOG_NEEDS_HEADER = not os.path.exists(LOG_PATH)

    while True:
        cid = input("\n👉 Enter Clip ID or 'q': ").strip()
        if cid.lower() == "q":
            break

        if not cid.isdigit():
            print("   ❌ Please enter a numeric Clip ID (0–11999).")
            continue

        cid_int = int(cid)
        if cid_int < 0 or cid_int > 11999:
            print("   ❌ Clip ID out of range. Valid range is 0–11999.")
            continue

        probs, pred, conf, actual = engine.classify(cid)

        if probs is None:
            print(f"   ❌ Error: Clip {cid} not found / missing tensor.")
            continue

        print(f"   [RESULT] Labeled: {str(actual).upper()} | Detected: {str(pred).upper()} ({conf*100:.1f}%)")

        plot_path = engine.plot_radar(probs, str(cid_int).zfill(6), pred, conf, actual)
        print(f"   [PLOT]   Saved as: {plot_path}")

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "clip_id": str(cid_int).zfill(6),
            "actual_label": actual,
            "predicted_label": pred,
            "pred_confidence": float(conf),
            "radar_plot_path": plot_path,
        }
        for i, lab in enumerate(engine.labels_raw):
            row[f"p_{lab}"] = float(probs[i])


        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
            if LOG_NEEDS_HEADER:
                w.writeheader()
                LOG_NEEDS_HEADER = False
            w.writerow(row)

        print(f"   [LOG]    Appended to: {LOG_PATH}")

    print("\n🌊 Session closed.")
