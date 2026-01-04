# Stage 3 — Training and Bundle Export Notes (Kaggle → Repo)

This note documents the **current baseline** workflow used to produce the Stage-3
encoder bundle artefact.

> This is intended to be referenced from `stages/stage3_ssl/README.md` and kept
> alongside it in the same folder.

---

## Baseline workflow summary

1. **Train on Kaggle (notebook-driven)**
   - Notebook: `stages/stage3_ssl/minimalgput4x2.ipynb`
   - The notebook writes a temporary training script using:
     - `%%writefile train_script.py`

2. **Training outputs (weights as `.pth`)**
   During training / end of training, the script writes PyTorch checkpoints to the Kaggle working directory:
   - `BT_ckpt_epoch_XXX.pth` (periodic)
   - `SKANN_SSL_GPU_Final.pth` (final weights)

3. **Export a portable bundle (`.joblib`)**
   The notebook then packages model weights + label metadata into a Joblib bundle (example filename in Kaggle):
   - `SKANN_SSL_Production_Bundle.joblib`

   The export step (function such as `export_production_bundle()`) typically stores at least:
   - `model_state` (state_dict)
   - `vessel_labels` / label list
   - `class_map` (to_id / to_label)

4. **Copy into the repo under the canonical Stage-3 artefact name**
   Download the exported `.joblib` from Kaggle and place it under:

   - `stages/stage3_ssl/artifacts/SKANN_SSL_Stage3_SSL_Encoder_Bundle.joblib`

   This filename is the **canonical interface** consumed by Stage 6 and Stage 7.

---

## Why this note exists

- The training script (`train_script.py`) is generated inside Kaggle and is not automatically present in the repo.
- The repo should still document *exactly* how the `.joblib` bundle was produced.
- Downstream stages rely on the canonical artefact name and location.

---

## Recommended repo convention (optional)

To preserve provenance without changing the baseline workflow:

- Store the Kaggle-generated training script in the repo as:
  - `stages/stage3_ssl/train_script_kaggle.py`

- Optionally add a repo-path variant for local reproducibility:
  - `stages/stage3_ssl/train_script_repo.py`

These do **not** replace the notebook; they make the baseline easier to reproduce and audit.
