# Session Log — 2026-01-16

## Session Summary

This session focused on organizing the SKANN-SSL repository by creating project documentation and relocating the GUI demo to its proper location.

---

## Tasks Completed

### 1. Created CLAUDE.md
- Read `ROADMAP.md` and `docs/00_CANONICAL_SKANN_SSL_PROJECT_MEMORY.md`
- Created `CLAUDE.md` at repo root summarizing essential project context for AI assistants
- Includes: architecture pipeline, key principles, repository structure, operational anchors

### 2. Reviewed SKANN-SSL-Demo Folder
- Verified demo functionality (runs correctly)
- Identified issue: demo had its own `.git` folder (nested repo) and duplicate tensors (~123MB)
- Confirmed 1920 tensor files duplicated from `data/prototype_dataset/tensors/`

### 3. Relocated Demo to Stage 7
Moved demo from `SKANN-SSL-Demo/` (root) to `stages/stage7_deployment/demo/`

**New structure:**
```
stages/stage7_deployment/demo/
├── skann_ssl_demo_v2.py       # Renamed, updated to V2.2.0
├── requirements.txt
├── README.md                  # Updated with setup instructions
├── model/
│   ├── .gitkeep
│   └── *.joblib               # Gitignored
└── data/
    ├── manifest.csv           # Committed (~542KB)
    └── tensors/
        ├── .gitkeep
        └── *.npy              # Gitignored (1920 files)
```

### 4. Updated .gitignore
Added rules to exclude large demo files while keeping placeholders:
```gitignore
stages/stage7_deployment/demo/model/*.joblib
stages/stage7_deployment/demo/data/tensors/*.npy
!stages/stage7_deployment/demo/model/.gitkeep
!stages/stage7_deployment/demo/data/tensors/.gitkeep
!stages/stage7_deployment/demo/data/manifest.csv
```

### 5. Updated Documentation
- `stages/stage7_deployment/README.md` — Added GUI demo section
- `stages/stage7_deployment/demo/README.md` — Updated paths, added setup instructions
- `CLAUDE.md` — Updated demo location reference

### 6. Cleanup
- Removed nested `.git` folder from original demo
- Original `SKANN-SSL-Demo/` folder emptied (manual delete required due to file lock)

---

## Commits

| Hash | Message |
|------|---------|
| `e387513` | Stage7: relocate GUI demo to stages/stage7_deployment/demo/ |

**Pushed to:** `origin/main`

---

## Files Changed

**New files:**
- `CLAUDE.md`
- `stages/stage7_deployment/demo/skann_ssl_demo_v2.py`
- `stages/stage7_deployment/demo/README.md`
- `stages/stage7_deployment/demo/requirements.txt`
- `stages/stage7_deployment/demo/data/manifest.csv`
- `stages/stage7_deployment/demo/model/.gitkeep`
- `stages/stage7_deployment/demo/data/tensors/.gitkeep`

**Modified files:**
- `.gitignore`
- `stages/stage7_deployment/README.md`

---

## Notes

- The empty `SKANN-SSL-Demo/` folder at root may still exist (locked by process). Delete manually after terminal restart.
- Demo requires model/tensor files after clone. See `stages/stage7_deployment/demo/README.md` for setup.
- Data policy: Do not commit files under `data/` or demo tensor folders.

---

## How to Run Demo

```bash
cd stages/stage7_deployment/demo
pip install -r requirements.txt
python skann_ssl_demo_v2.py
```
