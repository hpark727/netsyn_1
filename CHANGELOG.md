# Changelog

## 2026-02-27

### Summary
Refactored `MLP_pipeline.py` from pooled-feature MLP to a compact short-kernel temporal CNN pipeline focused on generalization under low-data conditions.

### Added
- Repeated cross-validation with `train_test_split` using `60% train / 40% test` for each round.
- Group-aware split attempt (`GroupShuffleSplit`) with automatic fallback to stratified split if group split cannot preserve both classes.
- Train-only normalization statistics (fit only on train folds, then applied to train/test).
- Compact short-convolution model (`Conv1d` kernels `3/5/3`) with masked global mean/max pooling.
- AdamW optimizer with weight decay regularization.
- Early stopping with configurable patience and minimum improvement threshold.
- Training-time augmentation for low-data robustness:
  - random time crop
  - random shift
  - feature dropout
  - Gaussian noise
- Test-time augmentation (TTA) prediction averaging across multiple passes.
- Fold-level and aggregate metrics (`sample accuracy`, `fold mean ± std`).
- CSV export for predictions: `cv_cnn_predictions.csv`.

### Changed
- Updated defaults to a generalization-oriented setup:
  - `TRAIN_SIZE=0.6`, `TEST_SIZE=0.4`
  - `PLOT_LIVE=False` for stable non-interactive runs
  - `LR=3e-4`, `WEIGHT_DECAY=1e-3`, `PATIENCE=10`
- Preserved repeated-CV workflow while replacing weak pooled-only representation with temporal sequence modeling.

### Debugging / Validation Performed
- Recompiled script: `python -m py_compile MLP_pipeline.py`.
- Executed full pipeline end-to-end in the project venv.
- Fixed runtime issues until clean completion:
  - adjusted Matplotlib config to writable local directory
  - reordered imports so `MPLCONFIGDIR` is set before importing `matplotlib`
  - disabled interactive plotting by default for headless stability
  - verified fold loop, early stop logic, TTA inference, and output CSV writing.
- Confirmed final rerun exits with code `0` and writes `cv_cnn_predictions.csv`.

## 2026-02-27 (L2 Regularization Tuning)

### Changed
- Updated optimizer regularization in `MLP_pipeline.py` to explicit L2 penalty:
  - `WEIGHT_DECAY` changed from `1e-3` to `1e-4`
  - optimizer changed from `AdamW` to `Adam(..., weight_decay=1e-4)` for L2-style parameter regularization
- Updated training log text to reflect `Adam (L2)`.

### Validation
- Recompiled: `python -m py_compile MLP_pipeline.py`.
- Ran full pipeline successfully with new L2 setting (no exceptions).
- Latest run summary: `sample_acc=0.8229`, `fold_mean=0.8229 +/- 0.1136`.
