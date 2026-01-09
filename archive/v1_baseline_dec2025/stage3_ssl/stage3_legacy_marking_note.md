# Legacy / Reference Modules (Stage 3)

## Marking `barlow_twins.py` as legacy

`barlow_twins.py` is **not used** by the current baseline training path (the Kaggle notebook writes and runs `train_script.py`).
You can keep it as a reference implementation, but mark it clearly to avoid confusion.

### Options

1) **README-only (no file rename)**
- In `stages/stage3_ssl/README.md`, add a short section:

  - `barlow_twins.py` — *legacy/reference implementation of projector + Barlow Twins loss (not used in baseline training)*

2) **Header banner in the file (no rename)**
- Add at the very top of `barlow_twins.py`:

  ```python
  # NOTE (LEGACY): This module is currently not used by the baseline training pipeline.
  # Baseline training is executed from `minimalgput4x2.ipynb` which writes and runs `train_script.py`.
  ```

3) **Move into a `legacy/` folder (cleanest)**
- Create `stages/stage3_ssl/legacy/`
- Move `barlow_twins.py` there
- Update README accordingly

Recommendation for now: **Option 1 + Option 2** (no renames, no breakage; still clear).
