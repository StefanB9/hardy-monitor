# Phase 5: Confidence Interval Overhaul — Implementation Plan

## Context

Phases 1–4 are complete and merged to `dev`. The ML pipeline has Random Forest with grid-searched hyperparameters, 4-fold time-series CV, and 22 prediction features. Current confidence intervals are heuristic — `±adjusted_std` with a sigmoid confidence score and linear horizon penalty — and have no relationship to actual model prediction error.

Phase 5 replaces the heuristic confidence system with model-derived (residual-based) prediction intervals calibrated from cross-validation residuals. Since smartcore v0.4 does NOT expose per-tree predictions (the `per_tree_predictions()` stub returns `None`), the Quantile Regression Forest approach (design doc §7.2) is unavailable. We implement the fallback approach (§7.3): residual-based calibration using empirical quantiles from CV validation residuals.

## Decisions (User-Approved)

1. **Interval coverage**: 80% prediction interval (10th/90th percentile). Narrower and more actionable for gym-goers than 95%.
2. **Residual collection strategy**: Separate pass — after grid search finds best config, run one additional CV pass with only the best config to collect per-prediction residuals. Avoids modifying `evaluate_config` and wasting memory on 143 non-best configs.
3. **Persistence**: Persist quantile data in `PersistedModel`, bump version 2→3. On restart, calibrated intervals work immediately. V2 models load with `quantiles=None` (heuristic fallback).
4. **Minimum residuals per slot**: 10 per (weekday, hour) slot. Below threshold → fall back to global quantiles. No CV data → fall back to heuristic.
5. **Horizon scaling**: Multiplicative widening, reduced from 0.15 to 0.10 per hour (residual-based intervals already capture more real uncertainty).
6. **Confidence score formula**: `(1.0 - interval_width / 100.0).clamp(0.1, 0.95)` — narrower interval = higher confidence.
7. **`interval_width()` negativity fix**: Swap low/high if inverted after clamping. Fixed as part of this phase.
8. **New `PredictionMethod` variant**: `RandomForest { confidence: f64, n_trees: usize }` for RF models with calibrated intervals.
9. **Fallback chain**: Slot-specific quantiles → Global quantiles → Heuristic (old `calculate_confidence` logic preserved as private method).

## Implementation Steps

### Step 0: Save plan to `docs/plan/`

Save this plan to `docs/plan/phase5-confidence-interval-overhaul.md`.

---

### Step 1: Fix `interval_width()` negativity + add `RandomForest` variant

**File:** `src/ml/confidence.rs`

**1a.** Modify `PredictionWithConfidence::new()` — after clamping both `confidence_low` and `confidence_high` to `[0, 100]`, swap them if `low > high`:

```rust
let clamped_low = confidence_low.clamp(0.0, 100.0);
let clamped_high = confidence_high.clamp(0.0, 100.0);
let (final_low, final_high) = if clamped_low <= clamped_high {
    (clamped_low, clamped_high)
} else {
    (clamped_high, clamped_low)
};
```

**1b.** Add `PredictionMethod::RandomForest { confidence: f64, n_trees: usize }` variant. Update `is_ml()` → `true`, `confidence()` → stored value.

**Tests (write first):**
- `test_new_swaps_inverted_interval` — pass `low=80, high=20`, verify `confidence_low <= confidence_high`
- `test_interval_width_always_non_negative` — edge cases where clamping causes inversion
- Strengthen `prop_interval_width_non_negative` — assert `width >= 0.0` (not just `is_finite()`)
- `test_prediction_method_random_forest_is_ml` — `RandomForest` returns `true` for `is_ml()`
- `test_prediction_method_random_forest_confidence` — returns stored confidence value
- Update `prop_method_confidence_range` — include `RandomForest` variant

---

### Step 2: Create `src/ml/residuals.rs` — core quantile types

**File:** `src/ml/residuals.rs` (NEW)

Types:
```rust
/// Quantile pair for a single (weekday, hour) slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotQuantiles {
    pub q_low: f64,   // 10th percentile residual (typically negative)
    pub q_high: f64,  // 90th percentile residual (typically positive)
    pub count: usize,
}

/// Residual-based quantile lookup table for confidence intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualQuantiles {
    slot_quantiles: HashMap<(u32, u32), SlotQuantiles>,  // (weekday, hour)
    global_quantiles: SlotQuantiles,
    min_residuals_per_slot: usize,
}
```

Constants:
```rust
pub const LOWER_PERCENTILE: f64 = 10.0;
pub const UPPER_PERCENTILE: f64 = 90.0;
pub const MIN_RESIDUALS_PER_SLOT: usize = 10;
pub const HORIZON_SCALING_FACTOR: f64 = 0.10;
pub const MAX_INTERVAL_WIDTH: f64 = 100.0;
```

Public methods:
- `ResidualQuantiles::from_residuals(residuals: &[(u32, u32, f64)]) -> Option<Self>` — builds quantiles from (weekday, hour, residual) triples. Returns `None` if empty.
- `get_quantiles(&self, weekday: u32, hour: u32) -> &SlotQuantiles` — slot-specific if count >= 10, else global.
- `compute_confidence_interval(&self, predicted: f64, weekday: u32, hour: u32, hours_ahead: i64) -> (f64, f64, f64)` — returns `(low, high, score)` with horizon scaling.
- `slot_quantiles(&self) -> &HashMap<(u32, u32), SlotQuantiles>` — for persistence serialization.
- `global_quantiles(&self) -> &SlotQuantiles` — for persistence serialization.

Private helper:
- `compute_quantile(sorted: &[f64], percentile: f64) -> f64` — linear interpolation.

Confidence interval logic:
```rust
let quantiles = self.get_quantiles(weekday, hour);
let horizon_factor = 1.0 + (hours_ahead.max(1) - 1) as f64 * HORIZON_SCALING_FACTOR;
let low = (predicted + quantiles.q_low * horizon_factor).clamp(0.0, 100.0);
let high = (predicted + quantiles.q_high * horizon_factor).clamp(0.0, 100.0);
// Swap if inverted after clamping
let (final_low, final_high) = if low <= high { (low, high) } else { (high, low) };
let width = final_high - final_low;
let score = (1.0 - width / MAX_INTERVAL_WIDTH).clamp(0.1, 0.95);
(final_low, final_high, score)
```

**Tests (write first):**
- `test_compute_quantile_simple` — known sorted values, verify 10th/90th
- `test_compute_quantile_single_value` — returns that value
- `test_compute_quantile_two_values` — interpolation
- `test_from_residuals_empty` — returns `None`
- `test_from_residuals_single_slot` — builds quantiles for one slot
- `test_from_residuals_multiple_slots` — per-slot quantiles are distinct
- `test_get_quantiles_falls_back_to_global` — slot < 10 residuals → global
- `test_get_quantiles_uses_slot_when_sufficient` — slot >= 10 → slot-specific
- `test_confidence_interval_basic` — predicted=50, verify low/high/score
- `test_confidence_interval_horizon_scaling` — further hours → wider intervals
- `test_confidence_interval_clamps` — extreme values clamp to [0, 100]
- `test_confidence_score_narrow_interval` — narrow → high confidence
- `test_confidence_score_wide_interval` — wide → low confidence
- `prop_quantiles_ordering` — proptest: q_low <= q_high
- `prop_confidence_interval_ordered` — proptest: low <= high always
- `prop_confidence_score_in_range` — proptest: score in [0.1, 0.95]
- `prop_horizon_widens_interval` — proptest: hours_ahead=3 wider than hours_ahead=1

---

### Step 3: Add `collect_cv_residuals()` to hyperparameter module

**File:** `src/ml/training/hyperparameter.rs`

New function that re-runs CV with the best config and collects per-prediction residuals:

```rust
/// Collect per-prediction CV residuals from validation folds.
///
/// Returns (weekday, hour, residual) triples where residual = actual - predicted.
/// Weekday/hour extracted from feature's `raw_weekday`/`raw_hour` fields.
pub fn collect_cv_residuals(
    config: &HyperparameterSet,
    features: &[PredictionFeatures],
    targets: &[f64],
    folds: &[Fold],
) -> Result<Vec<(u32, u32, f64)>, TrainingError>
```

Logic per fold:
1. Train model with config on training slice
2. Predict validation slice
3. For each val sample: `residual = actual - predicted`, extract `raw_weekday` and `raw_hour` (truncated to `u32`)
4. Collect `(weekday, hour, residual)` triples

Pre-allocate result vector to total validation samples across folds.

**Tests (write first):**
- `test_collect_cv_residuals_basic` — 500 features, 3 folds, verify count = total val samples
- `test_collect_cv_residuals_weekday_hour_range` — all weekdays 0..7, hours 0..24
- `test_collect_cv_residuals_finite` — all residuals are finite
- `prop_residual_count_equals_val_samples` — proptest: total = sum of fold val lengths

---

### Step 4: Integrate residual collection into training pipeline

**File:** `src/ml/training/mod.rs`

**4a.** Add `residual_quantiles: Option<ResidualQuantiles>` to `TrainingResult`.

**4b.** Update `build_training_result()` signature to accept `Option<ResidualQuantiles>`, propagate to `TrainingResult`.

**4c.** Update `train_rf_with_tuning()` — after grid search finds best params:
1. Call `collect_cv_residuals(&best_params, features, targets, &folds)`
2. Build `ResidualQuantiles::from_residuals(&residuals)`
3. Return as 4th tuple element: `(model, Some(cv_scores), Some(best_params), residual_quantiles)`

**4d.** Update `compute_lr_cv_scores()` to also collect residuals and return `Option<ResidualQuantiles>` alongside `CrossValidationScores`. Change return type to `Option<(CrossValidationScores, Option<ResidualQuantiles>)>`.

**4e.** Update `train_model_sync()` to thread residual quantiles through both paths.

**Tests:**
- `test_train_rf_with_tuning_produces_residual_quantiles` — after CV, `residual_quantiles` is `Some`
- `test_train_rf_no_tuning_no_residual_quantiles` — without CV, `residual_quantiles` is `None`
- `test_train_lr_with_cv_produces_residual_quantiles` — LR path with CV also produces quantiles
- `test_train_model_sync_insufficient_for_cv_no_quantiles` — fallback path produces `None`

---

### Step 5: Wire `ResidualQuantiles` into `OccupancyPredictor`

**File:** `src/ml/mod.rs`

**5a.** Add field `residual_quantiles: Option<ResidualQuantiles>` to `OccupancyPredictor`.

**5b.** Add `pub fn set_residual_quantiles(&mut self, quantiles: Option<ResidualQuantiles>)`.

**5c.** Replace `calculate_confidence()` implementation:
- If `self.residual_quantiles` is `Some` → delegate to `quantiles.compute_confidence_interval(predicted, weekday, hour, hours_ahead)`
- If `None` → use existing heuristic logic (preserved as `calculate_confidence_heuristic()`)

**5d.** Update `ml_predict()` — use `PredictionMethod::RandomForest { confidence, n_trees }` when model is RF backend, `MachineLearning { confidence }` for LR. Getting `n_trees` requires adding a `pub fn n_trees(&self) -> Option<usize>` to `TrainedModel`.

**Tests (write first):**
- `test_predictor_uses_residual_quantiles` — set quantiles, verify interval from quantiles
- `test_predictor_falls_back_without_quantiles` — no quantiles, verify heuristic used
- `test_predictor_random_forest_method` — RF model + quantiles → `PredictionMethod::RandomForest`
- `test_predictor_ml_method_for_lr` — LR model → `PredictionMethod::MachineLearning`
- `test_set_residual_quantiles` — setting and clearing

---

### Step 6: Update persistence for residual quantiles

**File:** `src/ml/persistence.rs`

**6a.** Bump `CURRENT_VERSION` to 3.

**6b.** Add `SerializedSlotQuantiles`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedSlotQuantiles {
    pub weekday: u32,
    pub hour: u32,
    pub q_low: f64,
    pub q_high: f64,
    pub count: usize,
}
```

**6c.** Add fields to `PersistedModel`:
```rust
pub residual_quantiles: Option<Vec<SerializedSlotQuantiles>>,
pub global_quantiles: Option<(f64, f64, usize)>,  // (q_low, q_high, count)
```

**6d.** Update `PersistedModel::new()` to accept quantile data.

**6e.** Version handling: V2 models (without quantile fields) will deserialize with `residual_quantiles = None` / `global_quantiles = None` via Serde defaults. The existing `version > CURRENT_VERSION` check already rejects future versions.

**Tests:**
- `test_persisted_model_v3_roundtrip` — save+load with quantile data
- `test_persisted_model_v3_without_quantiles` — save+load with `None` quantiles
- `test_persisted_model_version_bumped` — `CURRENT_VERSION == 3`
- `test_serialized_slot_quantiles` — serialization roundtrip

---

### Step 7: Wire into `app.rs` and GUI

**File:** `src/app.rs`

In `Message::MlTrainingCompleted` handler (line 603), after `set_model()`:
```rust
self.data.predictor.set_residual_quantiles(training.residual_quantiles);
```

**File:** `src/views/ml_predictions.rs`

Add match arm for `PredictionMethod::RandomForest { .. }` at line 150:
```rust
PredictionMethod::RandomForest { .. } => ("RF", style::ACCENT_CYAN),
```

**File:** `src/ml/mod.rs`

Register module: `pub mod residuals;` and re-export `pub use residuals::ResidualQuantiles;`.

---

### Step 8: Verify everything compiles and passes

```bash
cargo fmt --all -- --check
cargo clippy --all-targets
cargo check --all-targets
cargo nextest run
cargo nextest run --no-default-features
```

Verify:
- `interval_width()` always non-negative
- `PredictionMethod::RandomForest` variant works end-to-end
- Residual quantiles computed from CV and stored in `TrainingResult`
- `OccupancyPredictor` uses quantiles when available, heuristic when not
- `PersistedModel` v3 roundtrip works, v2 compat (quantiles = None)
- All existing tests pass with updated types

## Files Modified

| File | Change |
|------|--------|
| `src/ml/residuals.rs` | **NEW** — `ResidualQuantiles`, `SlotQuantiles`, quantile computation |
| `src/ml/confidence.rs` | Fix `interval_width()` negativity, add `RandomForest` variant |
| `src/ml/training/hyperparameter.rs` | Add `collect_cv_residuals()` function |
| `src/ml/training/mod.rs` | `TrainingResult.residual_quantiles`, wire through training paths |
| `src/ml/mod.rs` | `OccupancyPredictor` residual quantiles field, new confidence calc, register module |
| `src/ml/model/mod.rs` | Add `n_trees()` method to `TrainedModel` |
| `src/ml/persistence.rs` | Version 3, `SerializedSlotQuantiles`, persist quantile data |
| `src/app.rs` | Pass residual quantiles to predictor after training |
| `src/views/ml_predictions.rs` | Match arm for `PredictionMethod::RandomForest` |

## Key Reusable Code

- `evaluate_config()` (`training/hyperparameter.rs`) — same train/predict-per-fold pattern, reused for `collect_cv_residuals()`
- `raw_weekday` / `raw_hour` fields in `PredictionFeatures` — direct slot lookup keys
- `FeatureExtractor::get_slot_std()` (`features/mod.rs`) — pattern for slot-based lookup, same (weekday, hour) key structure
- `SerializedSlotStats` (`persistence.rs`) — pattern for per-slot serialization, reused for `SerializedSlotQuantiles`
- `PredictionWithConfidence::new()` clamping — existing pattern, enhanced with swap logic

## Implementation Order

Step 1 (confidence.rs) and Step 2 (residuals.rs) are independent. Step 3 depends on Step 2 types only conceptually. Steps 4–5 depend on Steps 2–3. Step 6 depends on Step 2. Step 7 depends on Steps 5–6.

Practical order: **0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8**
