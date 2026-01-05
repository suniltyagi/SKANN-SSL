# SKANN-SSL Project Anomalies Register

**Generated:** 2026-01-05  
**Source:** Repo audit of https://github.com/suniltyagi/SKANN-SSL  
**Status:** Active — to be resolved sequentially

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 3 |
| MEDIUM | 6 (1 resolved) |
| LOW | 4 |
| DOC-ONLY | 2 |

**Resolved:** A7

---

## HIGH Severity

### A1. SKConv1D not integrated into baseline

**Type:** Architecture  
**Status:** OPEN  
**Evidence:** VERIFIED

- **File:** `stages/stage2_encoder/train_script.py`
- **Quote:** `nn.Conv1d(1, 128, 31, stride=4, padding=15)`
- **Expected:** Multi-branch SKConv1D with kernels `(3, 5, 7, 11, 15)` and attention fusion
- **Impact:** The "SK" in SKANN is not active; baseline uses fixed single-kernel convolution
- **Resolution:** Integrate `stages/stage1_skconv1d/skconv1d.py:SKFilterbank` into encoder

---

### A2. Duplicate HybridSKEncoder definitions

**Type:** Code duplication  
**Status:** OPEN  
**Evidence:** VERIFIED

- **File 1:** `stages/stage2_encoder/train_script.py`
  - Quote: `class HybridSKEncoder(nn.Module): ... self.backbone1d = nn.Sequential(...)`
  - Uses fixed Conv1d kernels
  
- **File 2:** `stages/stage2_encoder/skconv2d.py`
  - Quote: `class HybridSKEncoder(nn.Module): ... self.filterbank = SKFilterbank(out_ch=64, norm=norm)`
  - Uses SKFilterbank (Stage 1)

- **Impact:** Confusion about which is authoritative; risk of divergent implementations
- **Resolution:** Keep both; rename `skconv2d.py` version or mark as "v2 / integrated path"

---

### A3. Hardcoded Kaggle paths in training script

**Type:** Portability  
**Status:** OPEN  
**Evidence:** VERIFIED

- **File:** `stages/stage2_encoder/train_script.py`
- **Quote:** `self.data_dir = '/kaggle/working/SKANN-SSL/data/prototype_dataset/tensors/'`
- **Impact:** Script cannot run outside Kaggle without modification
- **Resolution:** Parameterize via config, CLI argument, or environment variable

---

## MEDIUM Severity

### A4. Stage 1 README status is stale

**Type:** Documentation  
**Status:** OPEN  
**Evidence:** VERIFIED

- **File:** `stages/stage1_skconv1d/README.md`
- **Quote:** `"## Status: PLANNED"`
- **Reality:** `skconv1d.py` contains complete `SKConv1D` and `SKFilterbank` implementations
- **Resolution:** Update to `"## Status: IMPLEMENTED (not integrated into baseline)"`

---

### A5. Stage 5 is empty placeholder

**Type:** Missing implementation  
**Status:** OPEN  
**Evidence:** VERIFIED

- **File:** `stages/stage5_training/README.md`
- **Quote:** `"## Status: PLANNED"`
- **Reality:** Only README exists; no `trainer.py`, `embeddings.py`, `clustering.py`, `visualization.py`
- **Impact:** Training loop lives in Stage 3 notebook; Stage 5 is orphaned
- **Resolution:** Either populate with reusable utilities or deprecate/merge into Stage 3

---

### A6. Root README says Stage 1 is "Next" but code exists

**Type:** Documentation  
**Status:** OPEN  
**Evidence:** VERIFIED

- **File:** `README.md` (root)
- **Quote:** `"| 1 | SKConv1D Filterbank | 🔄 Next |"`
- **Reality:** Implementation exists in `stages/stage1_skconv1d/skconv1d.py`
- **Resolution:** Update to `"🔄 Implemented, awaiting integration"`

---

### A7. UMAP hyperparameters unrecorded

**Type:** Reproducibility  
**Status:** ✅ RESOLVED (2026-01-05)  
**Evidence:** VERIFIED

- **File:** `stages/stage3_ssl/README.md`
- **Quote:** `"If these were not recorded for the baseline image, treat this plot as a qualitative snapshot"`
- **Impact:** Cannot reproduce exact UMAP visualization
- **Resolution:** Created `stages/stage3_ssl/runs/2025-12-29_kaggle_baseline/run_metadata.yaml` with:
  - `n_neighbors: 15`
  - `min_dist: 0.1`
  - `metric: cosine`
  - `random_state: 42`
- **Commit:** "Add baseline run metadata with UMAP hyperparameters (resolves A7)"

---

### A8. barlow_twins.py is unused

**Type:** Dead code  
**Status:** OPEN  
**Evidence:** VERIFIED

- **File:** `stages/stage3_ssl/barlow_twins.py`
- **Quote (from README):** `"barlow_twins.py contains a generic Barlow Twins projector/loss implementation and is currently not used by the baseline Kaggle training path (which computes the loss inline)"`
- **Impact:** Standalone module exists but baseline ignores it
- **Resolution:** Either wire into training or mark as reference/legacy

---

### A9. Stage 2 lacks __init__.py

**Type:** Package structure  
**Status:** OPEN  
**Evidence:** VERIFIED (not found in search)

- **Path:** `stages/stage2_encoder/__init__.py` — does not exist
- **Impact:** Cannot import as Python package; requires `sys.path` hacks
- **Resolution:** Add `__init__.py` exporting `HybridSKEncoder`

---

### A10. ONNX export not implemented

**Type:** Missing feature  
**Status:** OPEN  
**Evidence:** VERIFIED

- **File:** `stages/stage7_deployment/README.md`
- **Quote:** `"- [ ] ONNX export for embedded deployment"`
- **Impact:** Cannot deploy to ARM/DSP/edge devices as planned in ROADMAP
- **Resolution:** Implement ONNX export script in Stage 7

---

## LOW Severity

### A11. Fragile sys.path imports in Stage 6

**Type:** Code quality  
**Status:** OPEN  
**Evidence:** VERIFIED

- **Files:** 
  - `stages/stage6_evaluation/stage6_confusion_matrix.py`
  - `stages/stage6_evaluation/stage6_acoustic_sonar_classifier.py`
- **Quote:** `sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stage2_encoder")))`
- **Resolution:** Fix via proper `__init__.py` in Stage 2 (see A9)

---

### A12. Transforms ownership unclear (Stage 0 vs Stage 4)

**Type:** Architecture  
**Status:** OPEN (INFERENCE)  
**Evidence:** Transforms defined in `stages/stage0_preprocessing/transforms.py`; Stage 4 folder exists but content unknown

- **Impact:** Low; transforms work regardless of folder location
- **Resolution:** Clarify ownership in documentation; Stage 0 = basic, Stage 4 = SSL-specific policy

---

### A13. shared/config.py fallback in dataloader

**Type:** Code quality  
**Status:** OPEN  
**Evidence:** VERIFIED

- **File:** `stages/stage0_preprocessing/dataloader.py`
- **Quote:** `except ImportError: SAMPLE_RATE = 16000...` (fallback defaults)
- **Impact:** Potential config drift if `shared/config.py` is missing
- **Resolution:** Remove fallback; require `shared/config.py` to exist

---

### A14. Stage 7 duplicates Stage 6 classifier logic

**Type:** Code duplication  
**Status:** OPEN (INFERENCE)  
**Evidence:** Both `stage6_acoustic_sonar_classifier.py` and `stage7_deployment/acoustic_sonar_classifier3.py` implement similar radar plot + centroid inference

- **Impact:** Maintenance burden; changes need to be applied twice
- **Resolution:** Extract shared logic into common module

---

## DOC-ONLY (Not Repo Bugs)

### D1. Stage numbering differs between spec doc and repo

**Type:** Documentation mismatch  
**Status:** ACKNOWLEDGED  
**Evidence:** VERIFIED

- **Spec doc (`docs/SKANN_SSL_System_Architecture_and_SSL_Pipeline.md`):**
  - Stage 2 = 1D SKA backbone
  - Stage 3 = SKConv2D hierarchical encoder
  - Stage 5 = SSL wrapper
  - Stage 6 = Augmentation engine

- **Repo folders + `docs/00_DOCUMENT_INDEX.md`:**
  - `stage2_encoder/` = HybridSKEncoder
  - `stage3_ssl/` = Barlow Twins SSL
  - `stage4_augmentation/` = Augmentation
  - `stage6_evaluation/` = Evaluation

- **Assessment:** Spec doc describes theoretical decomposition; repo is operational truth
- **Resolution:** Add clarifying note to spec doc; repo folder names are authoritative

---

### D2. Duplicate ROADMAP files

**Type:** Documentation redundancy  
**Status:** ACKNOWLEDGED  
**Evidence:** VERIFIED

- `ROADMAP.md` (root) — canonical
- `docs/SKANN_SSL_Project_Roadmap.md` — duplicate
- `docs/tex/SKANN_SSL_Project_Roadmap.tex` — LaTeX rendering

- **Resolution:** Mark `/docs/` versions as "rendered artefacts from root ROADMAP.md"

---

## Resolution Order (Recommended)

1. **A1** — SKConv1D integration (architectural priority)
2. **A2** — Resolve encoder duplication (depends on A1)
3. **A4, A6** — Update stale READMEs (quick wins)
4. **A9, A11** — Fix package structure (enables cleaner imports)
5. **A3** — Parameterize paths (portability)
6. **A7** — Record UMAP params (reproducibility)
7. **A5** — Decide Stage 5 fate (cleanup)
8. **A8** — Wire or deprecate barlow_twins.py
9. **A10** — ONNX export (deployment milestone)
10. **A12, A13, A14** — Low-priority cleanup

---

*Document will be updated as anomalies are resolved.*
