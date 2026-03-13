# Phase 3: Training Pipeline Upgrade — Implementation Plan

## Context

Phase 1 (foundation restructuring) and Phase 2 (Random Forest core) are complete and merged to `dev`. The ML module now has `ModelBackend` enum dispatch (LR | RF), `ModelBuilder` with `train()`/`train_rf()` methods, and `TimeSeriesSplit` for expanding-window CV. However, training still uses hardcoded hyperparameters with a simple 80/20 validation split — no cross-validation, no grid search, no model selection.

Phase 3 adds: hyperparameter grid search, cross-validated training orchestration, `MlAlgorithm` config for model selection, and `spawn_blocking` for CPU-intensive work. After this phase, the default training pipeline will use Random Forest with grid-searched hyperparameters and 4-fold time-series CV.

## Decisions (User-Approved)

1. `max_features` grid: `[4, 8, None]` → 144 configs
2. Add `max_features(Option<usize>)` builder method to `ModelBuilder`
3. `feature_importance`: `Option<Vec<(String, f64)>>` in `TrainingResult`, always `None` for now
4. `oob_error`: `Option<f64>` in `TrainingResult`, always `None` for now
5. `cv_gap_hours` in `MlConfig`, converted to `gap_samples` at runtime from data density
6. Extract `TrainingDataPreparer` to `training/data_prep.rs`
7. `spawn_blocking` inside `train_model()` (callers unchanged)

## Implementation Steps

### Step 1: Add `max_features()` to `ModelBuilder`

**File:** `src/ml/model/mod.rs`

Add `max_features: Option<usize>` field to `ModelBuilder` (default: `None`), add builder method, wire it into `train_rf()` where it currently hardcodes `max_features: None`.

```rust
// In ModelBuilder struct:
max_features: Option<usize>,  // default: None

// Builder method:
#[must_use]
pub fn max_features(mut self, max: Option<usize>) -> Self {
    self.max_features = max;
    self
}

// In train_rf(), change:
max_features: self.max_features,  // was: None
```

Tests:
- `test_model_builder_max_features` — setter works, default is None
- `test_train_rf_with_max_features` — RF training respects max_features value
- Existing ModelBuilder tests still pass

---

### Step 2: Add `MlAlgorithm` enum to `ml/config.rs`

**File:** `src/ml/config.rs`

```rust
#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
pub enum MlAlgorithm {
    #[default]
    RandomForest,
    LinearRegression,
}
```

Add new fields to `MlConfig`:
```rust
pub struct MlConfig {
    // Existing fields unchanged...

    // New:
    pub algorithm: MlAlgorithm,     // Default: RandomForest
    pub cv_folds: usize,            // Default: 4
    pub cv_gap_hours: i64,          // Default: 24
    pub tune_hyperparameters: bool, // Default: true
}
```

Update `Default` impl with new field defaults.

**File:** `src/config.rs`

Update `AppConfig` defaults to include the new ML fields if needed (the `config` crate + `Deserialize` with `#[serde(default)]` should handle missing keys in TOML).

Tests:
- `test_ml_algorithm_default` — default is RandomForest
- `test_ml_config_new_fields_default` — cv_folds=4, cv_gap_hours=24, tune=true
- `test_ml_config_deserialize_without_new_fields` — old configs still parse (new fields use defaults)
- `test_ml_algorithm_deserialize` — "RandomForest" and "LinearRegression" parse correctly

---

### Step 3: Add `FoldScores`, `CrossValidationScores` types

**File:** `src/ml/training/cross_validation.rs`

Add scoring types alongside the existing `Fold`/`TimeSeriesSplit`:

```rust
/// Per-fold metric values with aggregate statistics.
#[derive(Debug, Clone)]
pub struct FoldScores {
    pub per_fold: Vec<f64>,
    pub mean: f64,
    pub std_dev: f64,
}

impl FoldScores {
    pub fn from_scores(scores: Vec<f64>) -> Self {
        // Compute mean and std_dev from the vec
    }
}

/// Complete cross-validation results across all metrics.
#[derive(Debug, Clone)]
pub struct CrossValidationScores {
    pub mse: FoldScores,
    pub rmse: FoldScores,
    pub mae: FoldScores,
    pub r_squared: FoldScores,
}
```

Note: Skip `mape` in CV scores — MAPE is fragile with near-zero targets common in early-morning gym data.

Tests:
- `test_fold_scores_from_scores` — mean and std_dev computed correctly
- `test_fold_scores_single_value` — single-fold case (std_dev = 0)
- `test_fold_scores_empty` — graceful handling
- Property test: `prop_fold_scores_mean_in_range` — mean is between min and max of inputs

---

### Step 4: Add `HyperparameterSet` and grid generation

**File:** `src/ml/training/hyperparameter.rs` (new file)

```rust
/// A single hyperparameter configuration for grid search.
#[derive(Debug, Clone)]
pub struct HyperparameterSet {
    pub n_trees: usize,
    pub max_depth: usize,           // 0 = unlimited
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
}

impl HyperparameterSet {
    /// Generate the default grid of hyperparameter configurations.
    ///
    /// Grid: n_trees × max_depth × min_samples_leaf × max_features
    /// = 4 × 4 × 3 × 3 = 144 configurations
    pub fn default_grid() -> Vec<Self> {
        let n_trees_values = [100, 200, 300, 500];
        let max_depth_values = [8, 12, 16, 0]; // 0 = unlimited
        let min_samples_leaf_values = [2, 5, 10];
        let max_features_values: [Option<usize>; 3] = [Some(4), Some(8), None];

        let mut grid = Vec::with_capacity(144);
        for &n_trees in &n_trees_values {
            for &max_depth in &max_depth_values {
                for &min_samples_leaf in &min_samples_leaf_values {
                    for &max_features in &max_features_values {
                        grid.push(Self {
                            n_trees,
                            max_depth,
                            min_samples_leaf,
                            max_features,
                        });
                    }
                }
            }
        }
        grid
    }
}

/// Result of a grid search: best hyperparameters + their CV scores.
#[derive(Debug, Clone)]
pub struct GridSearchResult {
    pub best_params: HyperparameterSet,
    pub best_cv_scores: CrossValidationScores,
    pub configs_evaluated: usize,
}
```

Add a function to evaluate a single hyperparameter config across CV folds:

```rust
/// Evaluate one hyperparameter config across all CV folds.
///
/// Returns the CrossValidationScores for this config.
pub fn evaluate_config(
    config: &HyperparameterSet,
    features: &[PredictionFeatures],
    targets: &[f64],
    folds: &[Fold],
) -> Result<CrossValidationScores, TrainingError>
```

And the grid search orchestrator:

```rust
/// Run grid search: evaluate all configs, return best by mean CV MSE.
pub fn grid_search(
    features: &[PredictionFeatures],
    targets: &[f64],
    folds: &[Fold],
) -> Result<GridSearchResult, TrainingError>
```

Tests:
- `test_default_grid_size` — 144 configurations
- `test_default_grid_values` — spot-check specific configs exist
- `test_evaluate_config_basic` — single config produces valid scores
- `test_grid_search_selects_best` — best has lowest mean MSE
- `test_grid_search_insufficient_data` — proper error
- Property test: `prop_grid_search_best_mse_le_all` — best MSE ≤ all other MSEs

---

### Step 5: Extract `TrainingDataPreparer` to `training/data_prep.rs`

**File:** `src/ml/training/data_prep.rs` (new file, extracted from `training/mod.rs`)

Move `TrainingDataPreparer` struct and its `impl` block (including `new()` and `prepare()`). Update imports.

**File:** `src/ml/training/mod.rs`

Add `pub(crate) mod data_prep;` and update `train_model()`/`train_model_sync()` to use `data_prep::TrainingDataPreparer`.

Tests: Existing `TrainingDataPreparer` tests move to `data_prep.rs`. All existing tests pass unchanged.

---

### Step 6: Add gap conversion helper

**File:** `src/ml/training/data_prep.rs`

Add a helper to estimate samples-per-hour from training data:

```rust
/// Estimate the number of samples per hour from the training data.
///
/// Uses the first and last timestamps plus total sample count to derive
/// average data density. Returns at least 1 to avoid division by zero.
pub fn estimate_samples_per_hour(logs: &[OccupancyLog]) -> usize
```

This will be used to convert `cv_gap_hours` → `gap_samples` in the orchestrator.

Tests:
- `test_estimate_samples_per_hour_minute_data` — ~60 sph for minute-resolution
- `test_estimate_samples_per_hour_hourly_data` — ~1 sph for hourly data
- `test_estimate_samples_per_hour_empty` — returns 1
- `test_estimate_samples_per_hour_single` — returns 1

---

### Step 7: Update `TrainingResult` with new fields

**File:** `src/ml/training/mod.rs`

```rust
#[derive(Debug, Clone)]
pub struct TrainingResult {
    // Existing:
    pub model: TrainedModel,
    pub feature_extractor: FeatureExtractor,
    pub persisted: PersistedModel,

    // New:
    pub cv_scores: Option<CrossValidationScores>,
    pub best_hyperparameters: Option<HyperparameterSet>,
    pub feature_importance: Option<Vec<(String, f64)>>,
    pub oob_error: Option<f64>,
}
```

All new fields are `Option` — `None` when training uses the simple path (e.g., insufficient data for CV).

Update `ModelSummary` in `persistence.rs` if needed to store the algorithm name via `model.model_type()` (already done in Phase 2).

Tests: Existing `train_model_sync` tests updated to check new fields are `Some`/`None` as expected.

---

### Step 8: Rewrite `train_model()` and `train_model_sync()` orchestration

**File:** `src/ml/training/mod.rs`

**New orchestration for `train_model_sync()`:**

```rust
pub fn train_model_sync(
    logs: &[OccupancyLog],
    baseline: &[HourlyAverage],
    schedule: &GymSchedule,
    config: &MlConfig,
) -> Result<TrainingResult, TrainingError> {
    // 1. Prepare data
    let preparer = TrainingDataPreparer::new(config.clone());
    let (features, targets) = preparer.prepare(logs, baseline, schedule)?;

    // 2. Compute gap_samples from data density
    let sph = estimate_samples_per_hour(logs);
    let gap_samples = (config.cv_gap_hours as usize) * sph;

    // 3. Branch on algorithm
    let (model, cv_scores, best_params) = match config.algorithm {
        MlAlgorithm::RandomForest => {
            if config.tune_hyperparameters {
                // Grid search with CV
                let splitter = TimeSeriesSplit::new(config.cv_folds, gap_samples);
                let folds = splitter.and_then(|s| s.split(features.len()));

                match folds {
                    Some(folds) => {
                        let result = grid_search(&features, &targets, &folds)?;
                        // Retrain on full data with best params
                        let builder = ModelBuilder::new()
                            .n_trees(result.best_params.n_trees)
                            .max_depth(result.best_params.max_depth)
                            .min_samples_leaf(result.best_params.min_samples_leaf)
                            .max_features(result.best_params.max_features);
                        let model = builder.train_rf(&features, &targets)?;
                        (model, Some(result.best_cv_scores), Some(result.best_params))
                    }
                    None => {
                        // Not enough data for CV — train with defaults
                        let model = ModelBuilder::new()
                            .n_trees(100).max_depth(10)
                            .min_samples_leaf(2)
                            .train_rf(&features, &targets)?;
                        (model, None, None)
                    }
                }
            } else {
                // No tuning — train RF with defaults
                let model = ModelBuilder::new()
                    .n_trees(100).max_depth(10)
                    .min_samples_leaf(2)
                    .train_rf(&features, &targets)?;
                (model, None, None)
            }
        }
        MlAlgorithm::LinearRegression => {
            // LR path: CV for scores only, no grid search
            let builder = ModelBuilder::new().ridge_lambda(1e-3);
            let model = builder.train_with_validation(&features, &targets, 0.2)?;

            // Optionally run CV to produce scores
            let cv_scores = compute_lr_cv_scores(
                &features, &targets, config, logs,
            );
            (model, cv_scores, None)
        }
    };

    // 4. Build PersistedModel and TrainingResult
    // ... (same slot_stats + ModelSummary construction as before)
}
```

**`train_model()` (async) wraps with `spawn_blocking`:**

```rust
pub async fn train_model(
    db: &Database,
    clock: &dyn Clock,
    schedule: &GymSchedule,
    config: &MlConfig,
) -> Result<TrainingResult, TrainingError> {
    // 1. Async DB fetches (stay on async runtime)
    let logs = db.get_history_range(start, end).await...;
    let baseline = db.get_averages_range(start, end).await...;

    // 2. CPU-intensive training on blocking thread pool
    let schedule = schedule.clone();
    let config = config.clone();
    tokio::task::spawn_blocking(move || {
        train_model_sync(&logs, &baseline, &schedule, &config)
    })
    .await
    .map_err(|e| TrainingError::FitError(format!("Task join error: {e}")))?
}
```

Tests:
- `test_train_model_sync_rf_with_cv` — RF path produces cv_scores and best_params
- `test_train_model_sync_lr_path` — LR path produces model with algorithm=LR
- `test_train_model_sync_rf_no_tuning` — tune=false skips grid search
- `test_train_model_sync_insufficient_for_cv` — falls back gracefully
- Existing `test_train_model_sync` tests updated/split per algorithm

---

### Step 9: LR cross-validation helper

**File:** `src/ml/training/mod.rs`

```rust
/// Run cross-validation for linear regression (no grid search).
fn compute_lr_cv_scores(
    features: &[PredictionFeatures],
    targets: &[f64],
    config: &MlConfig,
    logs: &[OccupancyLog],
) -> Option<CrossValidationScores> {
    let sph = estimate_samples_per_hour(logs);
    let gap_samples = (config.cv_gap_hours as usize) * sph;
    let splitter = TimeSeriesSplit::new(config.cv_folds, gap_samples)?;
    let folds = splitter.split(features.len())?;

    let builder = ModelBuilder::new().ridge_lambda(1e-3);
    // For each fold: train LR, predict on val, compute metrics
    // Collect into CrossValidationScores
}
```

Tests:
- `test_lr_cv_scores_basic` — produces valid scores
- `test_lr_cv_scores_insufficient_data` — returns None

---

### Step 10: Verify everything compiles and passes

Run the full verification suite:
```bash
cargo fmt --all -- --check
cargo clippy --all-targets
cargo check --all-targets
cargo nextest run
cargo nextest run --no-default-features  # core-only
```

Verify:
- `TrainedModel` public API unchanged: `predict()`, `predict_batch()`, `info()`
- `OccupancyPredictor` compiles without changes (uses only `TrainedModel::predict()`)
- `app.rs` compiles — `TrainingResult` still derives `Clone`, `Message::MlTrainingCompleted` works
- `views/ml_predictions.rs` unchanged
- All 332+ tests pass (existing + new)

## Files Modified

| File | Change |
|------|--------|
| `src/ml/model/mod.rs` | Add `max_features` field + builder method to `ModelBuilder` |
| `src/ml/config.rs` | Add `MlAlgorithm` enum, `cv_folds`, `cv_gap_hours`, `tune_hyperparameters` to `MlConfig` |
| `src/ml/training/mod.rs` | Rewrite orchestration: algorithm dispatch, grid search integration, `spawn_blocking`, updated `TrainingResult` |
| `src/ml/training/cross_validation.rs` | Add `FoldScores`, `CrossValidationScores` types |
| `src/ml/training/hyperparameter.rs` | **New** — `HyperparameterSet`, `GridSearchResult`, `default_grid()`, `grid_search()`, `evaluate_config()` |
| `src/ml/training/data_prep.rs` | **New** — extracted `TrainingDataPreparer`, `estimate_samples_per_hour()` |
| `src/config.rs` | Possibly update defaults for new MlConfig fields |

## Key Reusable Code

- `TimeSeriesSplit::split()` (`training/cross_validation.rs`) — generates fold indices for CV
- `ModelBuilder::train_rf()` (`model/mod.rs`) — RF training via builder, used inside grid search
- `evaluation::{mse, rmse, mae, r_squared}` (`evaluation.rs`) — metric computation per fold
- `features_to_dense_matrix()` (`model/random_forest.rs`) — feature conversion for RF
- `TrainingDataPreparer::prepare()` (`training/data_prep.rs`) — feature extraction from logs

## Verification

1. `cargo fmt --all -- --check`
2. `cargo clippy --all-targets` — zero warnings
3. `cargo nextest run` — all tests pass
4. `cargo nextest run --no-default-features` — core-only tests pass
5. `cargo check --all-targets` — type-check everything
6. Manually verify `TrainingResult` fields in test output: `cv_scores` is `Some` with RF+tuning, `None` without
7. Verify grid search produces 144 evaluated configs in test
