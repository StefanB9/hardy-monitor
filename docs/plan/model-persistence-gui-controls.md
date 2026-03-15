# Full Model Persistence + GUI Training Controls

## Context

The current persistence layer (`PersistedModel` v4) saves only metadata — training stats, residual quantiles, CV scores, hyperparameters. The actual model weights (smartcore `RandomForestRegressor`, linfa `FittedLinearRegression`) are **never serialized**, so every app restart requires a full retrain. Additionally, training is fully automatic (triggered every fetch cycle when stale), giving the user no control over when training happens.

**Goal:** Persist actual model weights so the app loads a working model on startup. Give the user explicit GUI control over training instead of auto-triggering it.

## Part 1: Model Serialization

### Step 1 — Enable smartcore serde feature

**File:** `Cargo.toml` (workspace root)

```toml
# Change:
smartcore = { version = "0.4.9", default-features = false }
# To:
smartcore = { version = "0.4.9", default-features = false, features = ["serde"] }
```

This enables `Serialize`/`Deserialize` derives on `RandomForestRegressor` and related types.

### Step 2 — Refactor `LinearRegressionModel` to own its coefficients

**File:** `crates/hardy-gui/src/ml/model/linear.rs`

Replace `FittedLinearRegression<f64>` wrapper with raw coefficient storage:

```rust
pub(crate) struct LinearRegressionModel {
    coefficients: Array1<f64>,
    intercept: f64,
}
```

- `train()` still uses linfa, but extracts `.params().clone()` and `.intercept()` at the end
- `predict()` / `predict_batch()` become `x.dot(&coefficients) + intercept`
- Add `pub(crate) fn from_coefficients(coefficients: Array1<f64>, intercept: f64) -> Self`
- Existing `coefficients()` and `intercept()` getters stay the same
- `compute_training_mse()` uses the same dot-product logic

**Tests:**
- `test_lr_from_coefficients_roundtrip` — train, extract, reconstruct, verify predictions match
- All existing LR tests must still pass (identical math)

### Step 3 — Add serialize/deserialize to `RandomForestModel`

**File:** `crates/hardy-gui/src/ml/model/random_forest.rs`

Add two methods:
- `pub(crate) fn serialize(&self) -> Result<Vec<u8>>` — bincode-serialize the inner `RandomForestRegressor`
- `pub(crate) fn from_serialized(bytes: &[u8], n_trees: usize) -> Result<Self>` — deserialize, wrap in `Arc`

**Tests:**
- `test_rf_serialize_roundtrip` — train small RF, serialize, deserialize, verify predictions match
- `test_rf_deserialize_invalid_bytes` — graceful error on corrupt input

### Step 4 — Add `serialize_weights()` to `TrainedModel`

**File:** `crates/hardy-gui/src/ml/model/mod.rs`

```rust
impl TrainedModel {
    pub(crate) fn serialize_weights(&self) -> Option<SerializedModelWeights> { ... }
}
```

Matches on `self.backend` to produce `SerializedModelWeights::RandomForest(bytes)` or `SerializedModelWeights::LinearRegression { coefficients, intercept }`.

## Part 2: Persistence v5

### Step 5 — Add `SerializedModelWeights` and bump to v5

**File:** `crates/hardy-gui/src/ml/persistence.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializedModelWeights {
    RandomForest(Vec<u8>),
    LinearRegression {
        coefficients: Vec<f64>,
        intercept: f64,
    },
}
```

Add to `PersistedModel`:
- `pub model_weights: Option<SerializedModelWeights>` field
- Bump `CURRENT_VERSION` to `5`
- Update `PersistedModel::new()` to accept `model_weights` parameter

Add reconstruction:
- `pub fn to_trained_model(&self) -> Result<TrainedModel, PersistenceError>` — reconstructs a functional `TrainedModel` from `model_weights` using the `from_coefficients` / `from_serialized` constructors
- Add `PersistenceError::ModelWeightsInvalid(String)` variant

**Backward compat:** `model_weights: Option<...>` deserializes as `None` from v4 files.

**Tests:**
- `test_v5_roundtrip_with_rf_weights` — save+load RF weights
- `test_v5_roundtrip_with_lr_weights` — save+load LR weights
- `test_v4_file_loads_as_v5_no_weights` — v4 compat, `model_weights` is `None`
- `test_to_trained_model_rf` — reconstruct working RF from persisted
- `test_to_trained_model_lr` — reconstruct working LR from persisted
- `test_to_trained_model_no_weights_returns_error`

### Step 6 — Serialize weights in training pipeline

**File:** `crates/hardy-gui/src/ml/training/mod.rs`

Update `build_training_result()` to call `model.serialize_weights()` and pass the result to `PersistedModel::new()`.

**Tests:**
- `test_build_training_result_includes_weights`

## Part 3: GUI Training Controls

### Step 7 — Load full model on startup

**File:** `crates/hardy-gui/src/app.rs` (in `HardyMonitorApp::new()`)

Replace current metadata-only loading:
- Remove `is_stale()` guard — always load if file exists
- Call `persisted.to_trained_model()` to reconstruct actual model
- Call `predictor.set_model(model, trained_at)` so ML predictions work immediately
- Fall through to metadata loading (quantiles, training info) as before
- Log warnings on weight reconstruction failure (still load metadata)

### Step 8 — Remove auto-training trigger

**File:** `crates/hardy-gui/src/app.rs` (in `handle_fetch_completed()`)

Remove the block at lines ~1041-1052 that checks `needs_retraining()` and auto-triggers `train_ml_model()`. Training is now user-initiated only.

### Step 9 — Add new Message variant and handler

**File:** `crates/hardy-gui/src/app.rs`

Add to `Message` enum:
```rust
TrainModelRequested,
```

Add handler in `update()`:
```rust
Message::TrainModelRequested => {
    if self.data.ml_training_in_progress {
        return Task::none();
    }
    self.data.ml_training_in_progress = true;
    Self::train_ml_model(...)
}
```

Same logic as the old auto-trigger, but only fires on user action.

### Step 10 — Add Train/Retrain button and staleness hint to ML predictions view

**File:** `crates/hardy-gui/src/views/ml_predictions.rs`

Update `build_status_card()`:

| State | Display | Button |
|-------|---------|--------|
| No model, not training | "No model loaded" | **Train Model** |
| No model, training | "Training initial model..." | *(disabled)* |
| Has model, not training | "Active (Algorithm) - Trained Xh ago" + staleness hint if old | **Retrain Model** |
| Has model, retraining | "Active - Retraining..." | *(disabled)* |

**Staleness hint:** When `hours_since_training >= retrain_interval_hours`, show subtle text: "Consider retraining for improved accuracy"

**Props changes:** Add `retrain_interval_hours: i64` and `model_trained_at: Option<DateTime<Utc>>` to the view props (or compute age in the view from existing `ml_last_trained`).

## File Change Summary

| File | Change |
|------|--------|
| `Cargo.toml` (root) | Add `features = ["serde"]` to smartcore |
| `model/linear.rs` | Refactor to own coefficients; add `from_coefficients()` |
| `model/random_forest.rs` | Add `serialize()` / `from_serialized()` |
| `model/mod.rs` | Add `serialize_weights()` on `TrainedModel` |
| `persistence.rs` | `SerializedModelWeights`, v5, `to_trained_model()` |
| `training/mod.rs` | Pass serialized weights to `PersistedModel::new()` |
| `app.rs` | Startup loading, remove auto-train, add `TrainModelRequested` |
| `views/ml_predictions.rs` | Train/Retrain button, staleness hint |

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo nextest run --workspace
cargo check --workspace --all-targets
```

Manual: launch GUI, verify model loads from disk on startup, predictions display immediately, Train/Retrain button works, model persists across restarts.
