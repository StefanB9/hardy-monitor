# Phase 1: Foundation — Detailed Implementation Plan

**Status:** Approved
**Parent:** `docs/plan/ml-model-upgrade-design.md` (Phase 1 of 6)
**Goal:** Restructure the ML module and add missing infrastructure. No algorithm changes.
**Invariant:** All existing tests pass after every step.

---

## Decisions (approved)

| # | Decision | Choice |
|---|----------|--------|
| Q1 | TimeSeriesSplit gap unit | Sample indices (caller converts hours → samples) |
| Q2 | MAPE zero targets | Skip zeros; return `None` if all targets are zero |
| Q3 | R² when SS_tot = 0 | Return `Option<f64>` (`None` when undefined) |
| Q4 | Feature module split | Free functions in submodules; `FeatureExtractor` calls them |
| Q5 | `calculate_mse` ownership | Move to `evaluation.rs`; `model.rs` imports from there |

---

## Step 1: Extract `MlConfig` to `ml/config.rs`

**Type:** Refactor (existing tests are safety net)

### Changes

**Create `src/ml/config.rs`:**
- Move `MlConfig` struct + `Default` impl from `ml/mod.rs`.
- Move the `serde::Deserialize` derive and `std::path::PathBuf` import.

**Modify `src/ml/mod.rs`:**
- Add `pub mod config;`
- Add `pub use config::MlConfig;`
- Remove the `MlConfig` struct, its `Default` impl, and the now-unused `PathBuf` import.
- Keep `use serde::Deserialize;` only if still needed (it won't be — remove it).

### Public API impact
- `ml::MlConfig` still resolves via re-export. No downstream breakage.
- `ml::training` imports `super::MlConfig` — still works via the re-export.

### Verification
```bash
cargo clippy --all-targets
cargo nextest run
```

### Tests to move
- `test_config_defaults` stays in `ml/mod.rs` tests (it tests `MlConfig::default()` which is
  still re-exported). Alternatively, move it to `config.rs` inline tests. Either is fine — move
  it to `config.rs` so tests live next to the code they test.

---

## Step 2: Create `ml/evaluation.rs` — metric functions

**Type:** TDD red-green

### New file: `src/ml/evaluation.rs`

Public functions (all take `predictions: &[f64], targets: &[f64]`):

```rust
/// Mean Squared Error. Returns `None` if inputs are empty or mismatched length.
pub fn mse(predictions: &[f64], targets: &[f64]) -> Option<f64>

/// Root Mean Squared Error.
pub fn rmse(predictions: &[f64], targets: &[f64]) -> Option<f64>

/// Mean Absolute Error.
pub fn mae(predictions: &[f64], targets: &[f64]) -> Option<f64>

/// Mean Absolute Percentage Error. Skips targets where |actual| < epsilon.
/// Returns `None` if no valid targets remain (all zero or empty).
pub fn mape(predictions: &[f64], targets: &[f64]) -> Option<f64>

/// Coefficient of determination (R²). Returns `None` if SS_tot ≈ 0
/// (all targets identical) or inputs invalid.
pub fn r_squared(predictions: &[f64], targets: &[f64]) -> Option<f64>
```

All functions return `Option<f64>` for uniform error handling.

### Modify `src/ml/mod.rs`
- Add `pub mod evaluation;`

### TDD sequence

**Red phase — write these tests first (all must fail initially):**

```
test_mse_simple                 — [10, 20, 30] vs [12, 18, 32] → 4.0
test_mse_perfect                — identical predictions and targets → 0.0
test_mse_empty                  — empty slices → None
test_mse_mismatched_lengths     — different lengths → None

test_rmse_simple                — sqrt of mse result
test_rmse_perfect               — 0.0

test_mae_simple                 — [10, 20, 30] vs [12, 18, 32] → 2.0
test_mae_perfect                — 0.0

test_mape_simple                — known values
test_mape_skips_zero_targets    — targets with zeros → skipped, rest computed
test_mape_all_zero_targets      — all zeros → None

test_r_squared_perfect          — identical → Some(1.0)
test_r_squared_mean_prediction  — all predictions = mean(targets) → Some(0.0)
test_r_squared_constant_targets — all targets same → None
test_r_squared_negative         — worse than mean → Some(negative value)
```

**Property-based tests (proptest, 1000 cases):**

```
prop_mse_non_negative           — MSE is always >= 0 for valid inputs
prop_rmse_non_negative           — RMSE is always >= 0
prop_mae_non_negative           — MAE is always >= 0
prop_mape_non_negative          — MAPE is always >= 0 when Some
prop_r_squared_perfect_self     — r²(x, x) == Some(1.0) when targets not constant
prop_metrics_symmetric_zero     — mse(x, x) == 0, mae(x, x) == 0
```

**Green phase:** Implement each function minimally to pass its tests.

**Refactor phase:** Extract shared validation (`validate_inputs` helper for length/empty checks).

---

## Step 3: Move `calculate_mse` from `model.rs` to `evaluation.rs`

**Type:** Refactor

### Changes

**Modify `src/ml/model.rs`:**
- Remove the private `fn calculate_mse(predictions, targets) -> f64`.
- Add `use super::evaluation;`
- Replace calls:
  - `calculate_mse(&predictions.to_vec(), targets)` → `evaluation::mse(&predictions.to_vec(), targets).unwrap_or(f64::MAX)`
  - `calculate_mse(&val_predictions, val_targets)` → `evaluation::mse(&val_predictions, val_targets).unwrap_or(f64::MAX)`
- Remove the `test_calculate_mse` test from `model.rs` (now covered by `evaluation.rs` tests).

### Verification
```bash
cargo clippy --all-targets
cargo nextest run
```

All existing model tests must still pass — the behavior is identical.

---

## Step 4: Convert `features.rs` → `features/` directory module

**Type:** Refactor (pure file move, no code changes)

### Filesystem changes
1. Create directory `src/ml/features/`
2. Move `src/ml/features.rs` → `src/ml/features/mod.rs` (contents unchanged)

### Verification
```bash
cargo clippy --all-targets
cargo nextest run
```

Everything compiles identically — Rust resolves `mod features` to either
`features.rs` or `features/mod.rs`.

---

## Step 5: Extract `cyclical.rs` from `features/mod.rs`

**Type:** Refactor + new property-based tests

### Create `src/ml/features/cyclical.rs`

Move from `features/mod.rs`:
```rust
fn cyclical_encode(value: f64, period: f64) -> (f64, f64)
```

Change visibility to `pub(super)` so `mod.rs` can call it.

Move tests from `features/mod.rs`:
- `test_cyclical_encoding_continuity`
- `test_cyclical_encoding_opposite`
- `test_cyclical_encoding_quarter`

### Modify `src/ml/features/mod.rs`
- Add `mod cyclical;`
- Replace the `cyclical_encode(...)` call in `FeatureExtractor::extract()` with
  `cyclical::cyclical_encode(...)`.
- Remove the moved function and tests.

### New property-based tests in `cyclical.rs`

```
prop_cyclical_sin_cos_unit_circle  — sin² + cos² ≈ 1.0 for all (value, period > 0)
prop_cyclical_range                — sin and cos always in [-1.0, 1.0]
prop_cyclical_period_invariance    — encode(v, p) == encode(v + p, p)
prop_cyclical_output_finite        — both outputs are always finite for finite inputs
```

### TDD note
Property tests are written first (red), then verified against the moved function (should be
green immediately since the code already works — but the tests are new coverage).

---

## Step 6: Extract `momentum.rs` from `features/mod.rs`

**Type:** Refactor (convert methods to free functions) + move tests

### Create `src/ml/features/momentum.rs`

Move from `FeatureExtractor` impl in `features/mod.rs` and convert to free functions:

```rust
use std::collections::VecDeque;
use chrono::{DateTime, Local, Utc};

/// Extract 1h average, 3h average, and linear trend from recent observations.
pub(super) fn extract_momentum(
    recent_data: &VecDeque<(DateTime<Utc>, f64)>,
) -> (f64, f64, f64)

/// Least-squares linear trend (slope × 60) over a value series.
pub(super) fn calculate_trend(values: &[f64]) -> f64

/// Today's average-so-far and yesterday's average from recent observations.
pub(super) fn extract_day_features(
    recent_data: &VecDeque<(DateTime<Utc>, f64)>,
    local_time: &DateTime<Local>,
) -> (f64, f64)
```

Move tests:
- `test_calculate_trend_increasing`
- `test_calculate_trend_decreasing`
- `test_calculate_trend_flat`
- `test_extract_momentum_empty`

### Modify `src/ml/features/mod.rs`

- Add `mod momentum;`
- In `FeatureExtractor::extract()`, replace:
  - `self.extract_momentum(recent_data)` → `momentum::extract_momentum(recent_data)`
  - `self.extract_day_features(recent_data, &local_time)` →
    `momentum::extract_day_features(recent_data, &local_time)`
- Remove the three method bodies and the moved tests.
- Remove `self.calculate_trend(...)` call (it's now internal to `momentum.rs`).

### New property-based tests in `momentum.rs`

```
prop_extract_momentum_defaults_on_empty  — empty VecDeque → (50.0, 50.0, 0.0)
prop_calculate_trend_zero_for_constant   — constant values → trend ≈ 0.0
prop_calculate_trend_finite              — trend is always finite for finite inputs
prop_calculate_trend_sign                — strictly increasing → positive, decreasing → negative
```

---

## Step 7: Add property-based tests for `PredictionFeatures`

**Type:** TDD red-green (new tests only)

### Location: `src/ml/features/mod.rs` (inline `#[cfg(test)]` block)

```
prop_to_vec_correct_length      — to_vec().len() == NUM_FEATURES for any valid PredictionFeatures
prop_all_features_finite        — all values in to_vec() are finite for finite inputs
prop_feature_names_match_count  — feature_names().len() == NUM_FEATURES (static, but good to assert)
prop_to_vec_deterministic       — to_vec() called twice returns identical results
```

### Implementation notes

- Use `proptest` with `ProptestConfig::with_cases(1000)`.
- Generate `PredictionFeatures` by generating each field independently:
  - `hour_sin`, `hour_cos`: `-1.0..=1.0`
  - `historical_avg`, `historical_std`: `0.0..=100.0`
  - `is_weekend`, `is_holiday`: `0.0` or `1.0` (prop_oneof)
  - `hours_ahead`: `0.0..=24.0`
  - etc.
- Implement an `Arbitrary`-like strategy via `prop_compose!` for reuse across tests.

---

## Step 8: Convert `training.rs` → `training/` directory module

**Type:** Refactor (pure file move, no code changes)

### Filesystem changes
1. Create directory `src/ml/training/`
2. Move `src/ml/training.rs` → `src/ml/training/mod.rs` (contents unchanged)

### Verification
```bash
cargo clippy --all-targets
cargo nextest run
```

`hardy_monitor::ml::training::train_model` path still resolves — no downstream breakage.

---

## Step 9: Create `training/cross_validation.rs` — `TimeSeriesSplit`

**Type:** TDD red-green

### New file: `src/ml/training/cross_validation.rs`

```rust
/// A single fold's train/validation index ranges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fold {
    /// Inclusive start..exclusive end of training indices.
    pub train_start: usize,
    pub train_end: usize,
    /// Inclusive start..exclusive end of validation indices.
    pub val_start: usize,
    pub val_end: usize,
}

/// Expanding-window time-series cross-validation splitter.
///
/// Produces `k` folds where each fold's training set is all data before
/// a cutoff point, a gap of `gap_samples` is skipped, and the validation
/// set is the next chunk.
#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    k: usize,
    gap_samples: usize,
}

impl TimeSeriesSplit {
    /// Create a new splitter.
    ///
    /// Returns `None` if `k < 2`.
    pub fn new(k: usize, gap_samples: usize) -> Option<Self>

    /// Generate fold index ranges for `n_samples` data points.
    ///
    /// Returns `None` if there aren't enough samples to fill all folds
    /// with at least 1 training and 1 validation sample each.
    pub fn split(&self, n_samples: usize) -> Option<Vec<Fold>>
}
```

### Algorithm

Expanding window with equal-sized validation folds:

```
total_usable = n_samples
val_size = total_usable / (k + 1)  // roughly
min_train = val_size                // at least as many train as val samples

Fold i (0-indexed):
  val_end   = n_samples - (k - 1 - i) * val_size
  val_start = val_end - val_size
  train_end = val_start - gap_samples
  train_start = 0
```

Edge cases:
- Return `None` if `n_samples` is too small for even 1 training + gap + 1 validation sample
  per fold.
- `gap_samples = 0` is allowed (no gap).

### Modify `src/ml/training/mod.rs`
- Add `pub mod cross_validation;`

### TDD sequence

**Red phase — unit tests:**

```
test_split_basic_4_folds        — 1000 samples, k=4, gap=0 → 4 folds, verify structure
test_split_with_gap             — 1000 samples, k=4, gap=50 → gap respected
test_split_minimum_k            — k=2 works
test_split_k_less_than_2        — k=1 → new() returns None
test_split_insufficient_samples — 10 samples, k=4, gap=5 → split() returns None
test_fold_train_before_val      — all folds: train_end <= val_start
test_fold_no_overlap            — no index in both train and val ranges
test_fold_expanding_train       — each successive fold has more training data
test_fold_gap_respected         — val_start - train_end >= gap_samples for all folds
test_fold_complete_coverage     — every index appears in exactly one validation fold
```

**Property-based tests (1000 cases):**

```
prop_no_train_val_overlap       — for all (n in 50..2000, k in 2..8, gap in 0..20):
                                  no validation index falls within any training range
prop_temporal_ordering          — train_end < val_start for every fold
prop_expanding_window           — fold[i].train_end <= fold[i+1].train_end
prop_gap_respected              — val_start - train_end >= gap_samples
prop_all_val_indices_covered    — union of all validation ranges covers all indices
                                  from first val_start to n_samples (no holes)
```

**Green phase:** Implement `TimeSeriesSplit` minimally.

**Refactor phase:** Ensure no allocations in split computation (index arithmetic only).

---

## Step 10: Add property-based tests for `confidence.rs`

**Type:** TDD red-green (new tests only)

### Location: `src/ml/confidence.rs` (inline `#[cfg(test)]` block)

```
prop_new_clamps_predicted_value  — PredictionWithConfidence::new() always produces
                                   predicted_value in [0.0, 100.0]
prop_new_clamps_confidence_score — confidence_score always in [0.0, 1.0]
prop_new_clamps_intervals        — confidence_low and confidence_high in [0.0, 100.0]
prop_interval_width_non_negative — interval_width() >= 0.0 (since both are clamped)
prop_is_valid_after_new          — PredictionWithConfidence::new() always produces is_valid()
                                   == true (the constructor clamps, so this should hold)
prop_method_confidence_range     — PredictionMethod::confidence() always in [0.0, 1.0]
                                   for MachineLearning variant with confidence in 0.0..=1.0
```

### Implementation notes

- `prop_is_valid_after_new` is the most important: it proves the constructor's clamping
  guarantees the invariant. Generate arbitrary f64 inputs (including negatives, >100, NaN
  excluded) and show `is_valid()` is always true.
- Note: `new()` clamps `confidence_low` and `confidence_high` independently, so
  `confidence_low <= predicted_value` might NOT hold after clamping if the raw low was above
  the raw predicted (since both get clamped to [0, 100] but the ordering isn't enforced).
  This is actually a bug in the current code — if `confidence_low > predicted_value` after
  clamping, `is_valid()` returns false. The property test will surface this if it exists.
  **Do not fix it in Phase 1** — just document the finding. The confidence overhaul in Phase 5
  will address it.

---

## File Summary

### Created (6 files)
| File | Step | LOC estimate |
|------|------|-------------|
| `src/ml/config.rs` | 1 | ~35 |
| `src/ml/evaluation.rs` | 2 | ~180 (incl. tests) |
| `src/ml/features/cyclical.rs` | 5 | ~80 (incl. tests) |
| `src/ml/features/momentum.rs` | 6 | ~160 (incl. tests) |
| `src/ml/training/cross_validation.rs` | 9 | ~250 (incl. tests) |
| `src/ml/features/mod.rs` | 4 | (moved from features.rs) |

### Modified (4 files)
| File | Steps | Nature of change |
|------|-------|-----------------|
| `src/ml/mod.rs` | 1, 2 | Remove MlConfig, add `mod config`, `mod evaluation`, re-exports |
| `src/ml/model.rs` | 3 | Replace private `calculate_mse` with `evaluation::mse()` import |
| `src/ml/features/mod.rs` | 5, 6, 7 | Extract functions to submodules, add property tests |
| `src/ml/training/mod.rs` | 8, 9 | Directory conversion, add `mod cross_validation` |

### Deleted (0 files)
No files deleted — `features.rs` and `training.rs` become `features/mod.rs` and
`training/mod.rs` respectively (move, not delete + create).

### Unchanged
| File | Why |
|------|-----|
| `src/ml/confidence.rs` | Only new tests added (Step 10) |
| `src/ml/persistence.rs` | No changes in Phase 1 |
| `src/lib.rs` | Re-exports `ml::*` — still works via re-exports in `ml/mod.rs` |
| `src/app.rs` | Uses `ml::training::train_model` — path unchanged |
| `Cargo.toml` | No new dependencies in Phase 1 |

---

## Execution Order & Parallelism

Steps can be partially parallelized:

```
Step 1 (config extract) ──────────────────────────┐
Step 2 (evaluation.rs) ───→ Step 3 (move mse) ────┤
Step 4 (features/ dir) ──→ Step 5 (cyclical) ─┐   │
                           Step 6 (momentum) ──┼───┤
                           Step 7 (feat props) ┘   │
Step 8 (training/ dir) ──→ Step 9 (cv split) ──────┤
Step 10 (confidence props) ────────────────────────┘
                                                    │
                                            Final verify:
                                       cargo fmt --all -- --check
                                       cargo clippy --all-targets
                                       cargo nextest run
```

Steps 1, 2, 4, 8, and 10 have no interdependencies and can be done in any order.

---

## Pre-merge Checklist

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --all-targets` — zero warnings
- [ ] `cargo nextest run` — all tests pass
- [ ] `cargo check --all-targets`
- [ ] All property-based tests use `ProptestConfig::with_cases(1000)`
- [ ] No `.unwrap()`, `.expect()`, `panic!()`, `todo!()`
- [ ] Error paths have `.context()` where applicable
- [ ] New public items have `///` doc comments
- [ ] `pub(super)` or `pub(crate)` visibility on internal items
- [ ] Existing public API paths unchanged (verified by downstream compile)
