# ML Integration Plan

## Executive Summary

The `src/ml/` module is fully implemented — linear regression on 16 features, confidence
intervals, fallback to historical averages, async training pipeline — but is completely
disconnected from the app. It is not exported from `lib.rs`, its Cargo dependencies are
undeclared, and nothing calls it.

This plan wires it into the existing architecture with minimal disruption. The analytics
module is untouched; ML predictions **supplement** the existing 2-hour baseline lookup
rather than replace it.

---

## Scope

| In scope | Out of scope |
|---|---|
| Declare missing Cargo dependencies | Changing the ML algorithm |
| Export ML types from `lib.rs` | Adding model persistence (weights can't be serialised via linfa — retrain on restart) |
| Add `MlConfig` to `AppConfig` | Daemon-mode training (GUI only for now) |
| Integrate training as an async `Task` | New views beyond confidence bands and model status |
| Replace dashboard predictions with ML predictions + confidence intervals | Property-based testing of ML features |
| Add model-status card to insights view | |
| Add structured `tracing` to all new paths | |

---

## Step 1 — Declare Missing Dependencies in `Cargo.toml`

The ML module imports `linfa`, `linfa-linear`, `ndarray`, `approx`, and `bincode`. None
are declared. Add them with `default-features = false` per codebase convention.

```toml
approx      = { version = "0.5",  default-features = false }
bincode     = { version = "1",    default-features = false }
linfa       = { version = "0.7",  default-features = false }
linfa-linear = { version = "0.7", default-features = false }
ndarray     = { version = "0.16", default-features = false }
```

Use `cargo add` with `--no-default-features` for each. Check `cargo tree -d` afterwards
to confirm no duplicate `ndarray` versions (linfa already pulls it transitively — the
versions must align or the build will fail).

**Risk:** linfa 0.7 requires `ndarray 0.16`. If another transitive dep pins a conflicting
version, resolve via `[patch.crates-io]` or by choosing a compatible linfa version.
Run `cargo build` immediately after this step to surface any version conflicts early.

---

## Step 2 — Add `MlConfig` to `AppConfig`

### 2a — `src/ml/mod.rs`: derive `Deserialize` on `MlConfig`

`MlConfig` currently derives `Debug, Clone`. Add `serde::Deserialize` so the `config`
crate can populate it from `config.toml`.

```rust
// Before
#[derive(Debug, Clone)]
pub struct MlConfig { ... }

// After
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MlConfig { ... }
```

### 2b — `src/config.rs`: add `ml: MlConfig` field and defaults

```rust
pub struct AppConfig {
    // ... existing fields ...
    pub ml: MlConfig,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let builder = Config::builder()
            // ... existing defaults ...
            .set_default("ml.enabled", true)?
            .set_default("ml.training_window_days", 56)?
            .set_default("ml.retrain_interval_hours", 24)?
            .set_default("ml.prediction_horizon_hours", 6)?
            .set_default("ml.min_samples_for_training", 500)?
            .set_default("ml.model_path", None::<String>)?
            .set_default("ml.fallback_on_error", true)?;
    }
}
```

### 2c — `config.toml`: document the section

```toml
[ml]
enabled = true
training_window_days = 56
retrain_interval_hours = 24
prediction_horizon_hours = 6
min_samples_for_training = 500
# model_path =          # optional: path to persist model metadata
fallback_on_error = true
```

---

## Step 3 — Integrate `TrainingError` into `AppError`

Library code must use `AppError`. `TrainingError` currently lives in `src/ml/model.rs`
as a standalone enum. Add a variant to `AppError` in `src/error.rs`:

```rust
// src/error.rs
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    // ... existing variants ...

    /// ML model training failed.
    #[error("ML training failed: {0}")]
    MlTraining(String),
}
```

Add a `From<ml::model::TrainingError>` impl — either in `src/error.rs` (with a `use`
guard) or in `src/ml/model.rs`:

```rust
impl From<ml::model::TrainingError> for AppError {
    fn from(e: ml::model::TrainingError) -> Self {
        AppError::MlTraining(e.to_string())
    }
}
```

This lets `?` propagate `TrainingError` through functions that return
`Result<_, AppError>`.

---

## Step 4 — Export ML Types from `lib.rs`

Add a `pub mod ml;` declaration and re-export the types that views and `app.rs` will
reference. Gate nothing — the ML module has no GUI-specific code.

```rust
// src/lib.rs
pub mod ml;

pub use ml::{
    MlConfig,
    OccupancyPredictor,
    confidence::PredictionWithConfidence,
    confidence::PredictionMethod,
    training::TrainingResult,
};
```

---

## Step 5 — Extend App State and Messages

### 5a — `MonitorState` additions (`src/app.rs`)

```rust
struct MonitorState {
    // ... existing fields ...

    /// ML predictor — holds trained model, feature extractor, and rolling
    /// observation buffer. Initialised with config; model starts as None
    /// until training completes.
    ml_predictor: OccupancyPredictor,

    /// ML predictions for the next N hours (N = config.ml.prediction_horizon_hours).
    /// Empty until the first training completes. Falls back to historical average
    /// per-slot inside OccupancyPredictor when ML model is absent.
    ml_predictions: Vec<PredictionWithConfidence>,

    /// True while a training task is in flight. Used to show a spinner in the
    /// insights view and to prevent concurrent training tasks.
    ml_training_in_progress: bool,
}
```

**Allocation note:** `OccupancyPredictor` holds a `VecDeque<(DateTime<Utc>, f64)>` capped
at 180 entries (fixed bound). `Vec<PredictionWithConfidence>` holds at most
`prediction_horizon_hours` (≤ 6) entries. Both are small.

### 5b — `Message` additions

```rust
pub enum Message {
    // ... existing variants ...

    /// Trigger ML model training. Sent on startup (after baseline loads)
    /// and when `OccupancyPredictor::needs_retraining()` is true.
    TrainMlModel,

    /// Returned by the async training task. Contains the result or an error.
    MlTrainingCompleted(Result<TrainingResult, AppError>),

    /// Returned when ML predictions have been refreshed (after new data or
    /// after training completes). Inner vec may be empty on error.
    MlPredictionsUpdated(Vec<PredictionWithConfidence>),
}
```

### 5c — `HardyMonitorApp::new()` initialisation

```rust
// In new():
ml_predictor: OccupancyPredictor::new(config.ml.clone()),
ml_predictions: Vec::new(),
ml_training_in_progress: false,
```

---

## Step 6 — Training Integration

### 6a — Trigger on `PredictionBaselineLoaded`

Baseline data is loaded when the app starts and on view switches. When it arrives,
feed it to the predictor and decide whether to train:

```rust
// In handle for Message::PredictionBaselineLoaded:
self.data.ml_predictor.update_baseline(&averages);

if self.config.ml.enabled
    && !self.data.ml_training_in_progress
    && self.data.ml_predictor.needs_retraining(&*self.clock)
{
    tasks.push(Task::perform(
        async { Message::TrainMlModel },
        std::convert::identity,
    ));
}
```

### 6b — `Message::TrainMlModel` handler

Sets the in-progress flag and spawns the async training task. Training is expensive
(DB query + matrix operations); it must **not** run on the Tokio executor directly.
The training pipeline already uses `tokio::task::spawn_blocking` internally — verify
this is the case in `ml/training.rs`; if not, wrap in `spawn_blocking`.

```rust
Message::TrainMlModel => {
    if self.data.ml_training_in_progress || !self.config.ml.enabled {
        return Task::none();
    }
    self.data.ml_training_in_progress = true;

    let db = self.db.clone();
    let clock = self.clock.clone();
    let schedule = self.schedule.clone();
    let config = self.config.ml.clone();

    tracing::debug!("starting ML model training");

    Task::perform(
        async move {
            ml::training::train_model(&db, &*clock, &schedule, &config)
                .await
                .map_err(AppError::from)
        },
        Message::MlTrainingCompleted,
    )
}
```

### 6c — `Message::MlTrainingCompleted` handler

```rust
Message::MlTrainingCompleted(result) => {
    self.data.ml_training_in_progress = false;
    match result {
        Ok(training_result) => {
            let now = self.clock.now_utc();
            self.data.ml_predictor.set_model(training_result.model, now);

            // Immediately refresh feature extractor with trained slot stats
            self.data.ml_predictor.update_baseline(&self.data.prediction_baseline);

            tracing::info!(
                training_samples = training_result.model.training_samples,
                training_mse     = training_result.model.training_mse,
                validation_mse   = ?training_result.model.validation_mse,
                "ML model trained successfully"
            );

            // Refresh predictions now that a model is available
            Task::perform(
                async { Message::MlPredictionsUpdated(vec![]) },
                std::convert::identity,
            )
            // See §7 for how MlPredictionsUpdated triggers a re-predict
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "ML training failed, continuing with baseline fallback"
            );
            Task::none()
        }
    }
}
```

### 6d — Periodic retraining check

Add a check inside the existing `Message::Tick` handler, after the UI tick logic:

```rust
// In Message::Tick:
if self.config.ml.enabled
    && !self.data.ml_training_in_progress
    && self.data.ml_predictor.needs_retraining(&*self.clock)
{
    return Task::perform(
        async { Message::TrainMlModel },
        std::convert::identity,
    );
}
```

---

## Step 7 — Prediction Integration

### 7a — Feed observations to the predictor

In the `FetchCompleted` success path, after recording the new occupancy:

```rust
// After writing to DB, before notification check:
self.data.ml_predictor.add_observation(self.clock.now_utc(), percentage);
```

This maintains the rolling 180-sample window inside the predictor.

### 7b — Refresh ML predictions

After any event that should update predictions (new fetch, training complete, baseline
load), call a shared helper:

```rust
fn refresh_ml_predictions(&self) -> Task<Message> {
    if !self.config.ml.enabled {
        return Task::none();
    }
    // predict() is synchronous and cheap (matmul on ≤6 rows × 16 cols).
    // No spawn_blocking needed; run inline inside Task::perform.
    let predictor = self.data.ml_predictor.clone();  // OccupancyPredictor: Clone
    let baseline  = self.data.prediction_baseline.clone();
    let schedule  = self.schedule.clone();
    let clock     = self.clock.clone();

    Task::perform(
        async move {
            let preds = predictor.predict(&baseline, &schedule, &*clock);
            Message::MlPredictionsUpdated(preds)
        },
        std::convert::identity,
    )
}
```

Call `refresh_ml_predictions()` from:
- `handle_fetch_completed` success path
- `Message::MlTrainingCompleted` success path
- `Message::PredictionBaselineLoaded` success path

### 7c — `Message::MlPredictionsUpdated` handler

```rust
Message::MlPredictionsUpdated(preds) => {
    tracing::debug!(
        prediction_count = preds.len(),
        using_ml = preds.first().map(|p| p.method.is_ml()).unwrap_or(false),
        "ML predictions updated"
    );
    self.data.ml_predictions = preds;
    Task::none()
}
```

---

## Step 8 — Dashboard UI Updates

The dashboard already displays `predictions: &[(DateTime<Utc>, f64)]` in the history
chart. Extend `DashboardProps` to carry ML predictions:

```rust
// src/views/dashboard.rs
pub struct DashboardProps<'a> {
    // ... existing fields ...
    pub ml_predictions: &'a [PredictionWithConfidence],
    pub ml_training_in_progress: bool,
}
```

### 8a — Confidence band rendering

Where the prediction line is drawn in the history chart widget, add shaded confidence
bands using the `confidence_low` and `confidence_high` fields. Use a distinct colour
with reduced opacity (e.g., the primary accent colour at ~30% alpha) to indicate
uncertainty.

If `method.is_ml()` is false (historical fallback), render with a dashed line instead
of solid to signal lower confidence visually.

### 8b — "Model training" indicator

When `ml_training_in_progress` is true, show a small status chip below the prediction
chart: `"Updating model…"`. This disappears once `MlTrainingCompleted` arrives.

### 8c — Pass props in `app.rs`

```rust
// In the DashboardProps construction:
ml_predictions: &self.data.ml_predictions,
ml_training_in_progress: self.data.ml_training_in_progress,
```

---

## Step 9 — Insights View Updates

Add a **Model Status card** to `InsightsProps` and the insights view.

### 9a — `InsightsProps` extension

```rust
pub struct InsightsProps<'a> {
    // ... existing fields ...
    pub ml_enabled: bool,
    pub ml_has_model: bool,
    pub ml_training_in_progress: bool,
    pub ml_last_trained: Option<DateTime<Utc>>,
    pub ml_training_mse: Option<f64>,
    pub ml_validation_mse: Option<f64>,
}
```

### 9b — Model Status card layout

```
┌─ Prediction Model ───────────────────────────────┐
│  Status:   ● Ready  (or  ○ Training…  /  ◌ No data) │
│  Last trained:  2 hours ago                        │
│  Validation error:  4.3%                           │
│  Method:   Machine learning  (or  Historical avg)  │
└────────────────────────────────────────────────────┘
```

Display colour-coded status: green = ready, amber = training, grey = not enough data.

### 9c — Pass props in `app.rs`

```rust
// In the ViewMode::Insights arm of view():
ml_enabled: self.config.ml.enabled,
ml_has_model: self.data.ml_predictor.has_model(),
ml_training_in_progress: self.data.ml_training_in_progress,
ml_last_trained: self.data.ml_predictor.last_training(),
ml_training_mse: self.data.ml_predictor.model_mse(),    // add accessor
ml_validation_mse: self.data.ml_predictor.validation_mse(), // add accessor
```

---

## Step 10 — Add `tracing` Instrumentation

All new paths must follow the structured-logging rules from `CLAUDE.md`.

| Location | Call | Fields |
|---|---|---|
| `Message::TrainMlModel` handler | `debug!` | `needs_retraining`, `has_model` |
| `Message::MlTrainingCompleted` Ok | `info!` | `training_samples`, `training_mse`, `validation_mse` |
| `Message::MlTrainingCompleted` Err | `warn!` | `error = %e` |
| `Message::MlPredictionsUpdated` | `debug!` | `prediction_count`, `using_ml` |
| `refresh_ml_predictions` | `debug!` | trigger source |
| `OccupancyPredictor::predict` (in ml/) | `debug!` | `horizon_hours`, `has_model` |
| `train_model` (in ml/training.rs) | `info!` / `debug!` | `sample_count`, `window_days` |

No `tracing::info!` inside per-prediction loops — use `trace!` there.

---

## Step 11 — Testing

### 11a — ML Config loading test

Add to `src/config.rs` tests:

```rust
#[test]
fn test_ml_config_defaults() {
    let config = AppConfig::load().expect("Config should load");
    assert!(config.ml.training_window_days > 0);
    assert!(config.ml.min_samples_for_training > 0);
    assert!(config.ml.fallback_on_error);
}
```

### 11b — Prediction fallback test

Add to `tests/app_logic.rs` (uses `MockNotifier` pattern):

```rust
#[tokio::test]
async fn test_ml_predictions_fall_back_when_no_model() {
    // Build a predictor with no trained model
    let predictor = OccupancyPredictor::new(MlConfig { enabled: true, ..Default::default() });
    let clock = MockClock::new(fixed_utc_time());

    // predict() should return historical fallback, not panic or error
    let preds = predictor.predict(&[], &test_schedule(), &clock);
    // With empty baseline, all predictions should be the default fallback (50%)
    assert!(preds.iter().all(|p| !p.method.is_ml()));
}
```

### 11c — Training error propagation test

```rust
#[test]
fn test_training_error_converts_to_app_error() {
    let err = ml::model::TrainingError::InsufficientData(10);
    let app_err = AppError::from(err);
    assert!(matches!(app_err, AppError::MlTraining(_)));
}
```

---

## Implementation Order

This order minimises integration risk at each step — each step compiles and passes tests
before the next begins.

| Step | Files changed | Risk | Blocker for |
|---|---|---|---|
| 1 Cargo deps | `Cargo.toml` | Medium (version conflicts) | All |
| 2 MlConfig | `src/ml/mod.rs`, `src/config.rs`, `config.toml` | Low | 5, 6, 7 |
| 3 Error type | `src/error.rs`, `src/ml/model.rs` | Low | 6 |
| 4 lib.rs exports | `src/lib.rs` | Low | 5, 8, 9 |
| 5 App state & messages | `src/app.rs` | Low | 6, 7, 8, 9 |
| 6 Training | `src/app.rs` | Medium | 7 |
| 7 Predictions | `src/app.rs` | Low | 8 |
| 8 Dashboard UI | `src/views/dashboard.rs`, `src/widgets/history_chart.rs` | Medium | — |
| 9 Insights UI | `src/views/insights.rs`, `src/app.rs` | Low | — |
| 10 Tracing | All modified files | Low | — |
| 11 Tests | `src/config.rs`, `tests/app_logic.rs` | Low | — |

---

## Known Constraints & Trade-offs

### Model persistence is partial

`linfa`'s `FittedLinearRegression` does not implement `Serialize`. `ml/persistence.rs`
saves only metadata (MSE, slot statistics, feature importance) — **not the model weights**.
On every restart, if `needs_retraining()` is true (no `last_training` timestamp), the
model is re-trained from the DB. With 56 days of data at 60-second intervals this is
~80,640 rows — training takes a few seconds on modern hardware.

The fallback to historical averages covers the startup window until training completes.
This is acceptable behaviour and is already handled by `OccupancyPredictor::predict`.

### `OccupancyPredictor` must be `Clone`

`Task::perform` closures require `'static` captured values. `OccupancyPredictor` holds a
`VecDeque` and a `HashMap` — both are `Clone`. Check that `TrainedModel` (which holds the
linfa model) also implements `Clone`; if it does not, wrap it in `Arc<TrainedModel>`.

### Feature scope: GUI only

ML training is triggered from the Iced `update()` loop. Daemon mode has no app state and
will continue to collect data without running the predictor. If training in daemon mode is
wanted later, it would be a separate background task in `run_daemon()` — out of scope here.

### `unsafe_code = "forbid"` — linfa compatibility

Linfa itself contains `unsafe` in its internals (ndarray SIMD paths). Since the lint is
`forbid` on **this crate's code** only (not transitive dependencies), this is not an issue.
Confirm with `cargo clippy` after adding deps.
