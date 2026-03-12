# Phase 2: Random Forest Core — Implementation Plan

## Context

Phase 1 (foundation restructuring) is complete and merged to `dev`. The ML module now has clean separation: `config.rs`, `evaluation.rs`, `features/` directory, `training/` directory with cross-validation. The current model system is a monolithic `model.rs` containing `TrainedModel` (wrapping linfa `FittedLinearRegression`), `ModelBuilder`, and `TrainingError`.

Phase 2 adds a Random Forest Regressor via `smartcore` alongside the existing Linear Regression, using enum dispatch. The training pipeline continues to default to LR; RF model selection is deferred to Phase 3.

## Design Decisions (User-Approved)

1. **`TrainedModel` stays as public API** — wraps internal `ModelBackend` enum. Zero breaking changes.
2. **`coefficients()`/`intercept()` move to `LinearRegressionModel` only** — removed from `TrainedModel`.
3. **RF stub methods added now** — `feature_importance() -> Option<Vec<f64>>`, `per_tree_predictions() -> Option<Vec<f64>>` returning `None`.
4. **Training defaults to LR** — RF model selection deferred to Phase 3 (`ModelType` config).
5. **zstd deferred to Phase 6** — only `smartcore` added in Phase 2.
6. **`max_depth` stays `usize`** — converted to `Option<u16>` internally at the smartcore boundary.

## Implementation Steps

### Step 1: Add smartcore dependency

**File:** `Cargo.toml`

Add `smartcore` as optional, gui-gated, with `default-features = false`:

```toml
smartcore = { version = "0.4", default-features = false, optional = true }
```

Update gui feature:
```toml
gui = ["dep:approx", "dep:bincode", "dep:iced", "dep:image", "dep:linfa", "dep:linfa-linear", "dep:muda", "dep:ndarray", "dep:notify-rust", "dep:smartcore", "dep:tray-icon"]
```

### Step 2: Convert `model.rs` to `model/` directory module

Create directory structure:
```
src/ml/model/
├── mod.rs              (ModelBackend enum, TrainedModel wrapper, ModelBuilder, TrainingError)
├── linear.rs           (LinearRegressionModel — extracted from current model.rs)
└── random_forest.rs    (RandomForestModel — new)
```

### Step 3: Implement `LinearRegressionModel` in `model/linear.rs`

Extract LR-specific code from current `model.rs`:

```rust
pub struct LinearRegressionModel {
    model: linfa_linear::FittedLinearRegression<f64>,
}
```

Methods:
- `train(features, targets, fit_intercept, ridge_lambda) -> Result<Self, TrainingError>` — LR-specific training logic (ridge augmentation, linfa fit)
- `predict(&self, features: &[f64]) -> Option<f64>` — single sample, takes flat feature vec
- `predict_batch(&self, feature_matrix: &Array2<f64>) -> Vec<f64>` — batch prediction
- `coefficients(&self) -> &Array1<f64>` — LR-specific, stays here
- `intercept(&self) -> f64` — LR-specific, stays here
- `compute_training_mse(&self, x: &Array2<f64>, targets: &[f64]) -> f64`

Tests:
- `test_lr_train_success` — basic training works
- `test_lr_predict_single` — single prediction returns value
- `test_lr_predict_batch` — batch prediction correct length
- `test_lr_coefficients_length` — coefficients match NUM_FEATURES
- `test_lr_ridge_handles_singular` — ridge regularization recovers from singular data
- Property test: `prop_lr_predictions_finite` — all predictions are finite for valid inputs

### Step 4: Implement `RandomForestModel` in `model/random_forest.rs`

```rust
use smartcore::{
    ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters},
    linalg::basic::matrix::DenseMatrix,
};

pub struct RandomForestModel {
    model: RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>,
}
```

Methods:
- `train(features_matrix: &DenseMatrix<f64>, targets: &[f64], params: &RfHyperparameters) -> Result<Self, TrainingError>`
- `predict(&self, features: &[f64]) -> Option<f64>` — wraps single-row DenseMatrix predict
- `predict_batch(&self, features_matrix: &DenseMatrix<f64>) -> Vec<f64>`
- `feature_importance(&self) -> Option<Vec<f64>>` — returns `None` (smartcore v0.4 limitation)
- `per_tree_predictions(&self) -> Option<Vec<f64>>` — returns `None` (smartcore v0.4 limitation)

Helper struct for RF hyperparameters:
```rust
pub struct RfHyperparameters {
    pub n_trees: usize,        // default: 100
    pub max_depth: Option<u16>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,  // default: sqrt(n_features)
}
```

DenseMatrix conversion (at the smartcore boundary):
- `features_to_dense_matrix(features: &[PredictionFeatures]) -> Result<DenseMatrix<f64>, TrainingError>` — uses `PredictionFeatures::to_vec()` → `Vec<Vec<f64>>` → `DenseMatrix::from_2d_vec()`
- `single_to_dense_matrix(features: &[f64]) -> Result<DenseMatrix<f64>, TrainingError>` — single sample

Tests:
- `test_rf_train_success` — basic RF training works with sufficient data
- `test_rf_predict_single` — single prediction returns value
- `test_rf_predict_batch` — batch prediction correct length
- `test_rf_feature_importance_returns_none` — stub returns None
- `test_rf_per_tree_predictions_returns_none` — stub returns None
- `test_dense_matrix_conversion` — features_to_dense_matrix correct dimensions
- `test_rf_train_insufficient_data` — fails gracefully with small datasets
- Property test: `prop_rf_predictions_finite` — all predictions are finite for valid inputs
- Property test: `prop_dense_matrix_dimensions` — matrix dimensions match input

### Step 5: Implement `ModelBackend` enum and update `TrainedModel` in `model/mod.rs`

```rust
pub(crate) enum ModelBackend {
    LinearRegression(linear::LinearRegressionModel),
    RandomForest(random_forest::RandomForestModel),
}
```

`TrainedModel` becomes:
```rust
pub struct TrainedModel {
    backend: ModelBackend,
    pub training_mse: f64,
    pub validation_mse: Option<f64>,
    pub training_samples: usize,
    pub created_at: DateTime<Utc>,
}
```

Public methods on `TrainedModel` (dispatch to backend):
- `predict(&self, features: &PredictionFeatures) -> Option<f64>` — unchanged signature
- `predict_batch(&self, features: &[PredictionFeatures]) -> Vec<f64>` — unchanged signature
- `info(&self) -> String` — includes model type name
- `model_type(&self) -> &'static str` — returns `"LinearRegression"` or `"RandomForest"`
- `feature_importance(&self) -> Option<Vec<f64>>` — delegates to RF, returns None for LR

Removed from `TrainedModel`:
- `coefficients()` — moved to `LinearRegressionModel`
- `intercept()` — moved to `LinearRegressionModel`

The `new()` constructor changes to accept `ModelBackend` instead of `FittedLinearRegression`.

Tests:
- `test_trained_model_lr_predict` — LR backend predict works through TrainedModel
- `test_trained_model_rf_predict` — RF backend predict works through TrainedModel
- `test_trained_model_info_lr` — info() includes "LinearRegression"
- `test_trained_model_info_rf` — info() includes "RandomForest"
- `test_trained_model_model_type` — model_type() returns correct strings
- `test_trained_model_feature_importance_lr` — returns None for LR
- `test_trained_model_feature_importance_rf` — returns None for RF (stub)
- All existing `ModelBuilder` tests still pass

### Step 6: Update `ModelBuilder` in `model/mod.rs`

`ModelBuilder` stores both LR and RF parameters:

```rust
pub struct ModelBuilder {
    fit_intercept: bool,
    ridge_lambda: f64,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    n_trees: usize,           // new, default: 100
}
```

Methods:
- Existing builder methods unchanged: `fit_intercept()`, `ridge_lambda()`, `max_depth()`, `min_samples_split()`, `min_samples_leaf()`
- New: `n_trees(mut self, n: usize) -> Self`
- `train()` — builds LR (same as current behavior)
- `train_rf()` — builds RF using smartcore
- `train_with_validation()` — unchanged (still builds LR)
- `train_rf_with_validation()` — RF version with validation split

`max_depth` conversion for RF: `if self.max_depth == 0 { None } else { Some(u16::try_from(self.max_depth).unwrap_or(u16::MAX)) }`
Note: This saturating conversion avoids `unwrap()` — values > 65535 clamp to u16::MAX.

Tests:
- `test_model_builder_n_trees` — n_trees setter works
- `test_train_rf_success` — RF training through builder
- `test_train_rf_with_validation` — RF validation split works
- `test_train_rf_insufficient_data` — proper error
- Existing LR builder tests continue to pass

### Step 7: Update `ModelSummary` in `persistence.rs`

Update `ModelSummary.model_type` usage in `training/mod.rs` to use `TrainedModel::model_type()`:

```rust
ModelSummary {
    model_type: model.model_type().to_string(),
    max_depth: Some(10),
    feature_importance: model.feature_importance(),
}
```

No structural changes to `PersistedModel` — model weights are still not persisted (deferred to Phase 6).

### Step 8: Update `training/mod.rs`

Update `train_model()` and `train_model_sync()` to use `model.model_type()` for `ModelSummary`. Both still build LR by default (Phase 3 adds model selection).

### Step 9: Update `ml/mod.rs`

- Update `mod model;` declaration (now a directory module)
- Re-exports remain: `pub use model::TrainedModel;`
- Add re-export: `pub use model::ModelBuilder;` (if not already re-exported)
- `OccupancyPredictor` unchanged — it only uses `TrainedModel::predict()` which is stable

### Step 10: Verify all existing tests pass

Run `cargo nextest run` to confirm zero regressions.

## Files Modified

| File | Change |
|------|--------|
| `Cargo.toml` | Add `smartcore` optional dep, add to gui feature |
| `src/ml/model.rs` → `src/ml/model/mod.rs` | Extract to directory module, `ModelBackend` enum, updated `TrainedModel` |
| `src/ml/model/linear.rs` | **New** — `LinearRegressionModel` extracted from old `model.rs` |
| `src/ml/model/random_forest.rs` | **New** — `RandomForestModel` with smartcore RF |
| `src/ml/mod.rs` | Module declaration update (model directory) |
| `src/ml/training/mod.rs` | Use `model.model_type()` in `ModelSummary` |
| `src/ml/persistence.rs` | No structural changes (just consumed differently) |

## Key Reusable Code

- `PredictionFeatures::to_vec()` (`src/ml/features/mod.rs:41`) — used for both LR (→ ndarray) and RF (→ DenseMatrix) conversion
- `evaluation::mse()` (`src/ml/evaluation.rs:4`) — used in both LR and RF training MSE calculation
- `TrainingError` enum — shared by both backends
- `create_test_features()` helper in current `model.rs:231` tests — reuse for RF tests

## Verification

1. `cargo check --all-targets` — type-check everything
2. `cargo nextest run` — all tests pass (existing + new)
3. `cargo nextest run --no-default-features` — core-only tests still pass (smartcore is gui-gated)
4. Verify `TrainedModel` public API is unchanged: `predict()`, `predict_batch()`, `info()` signatures stable
5. Verify `OccupancyPredictor`, `app.rs`, `views/ml_predictions.rs` compile without changes
