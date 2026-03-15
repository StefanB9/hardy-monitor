# Fix: Stale Lag Features in Extrapolation

## Context

During **training** (`data_prep.rs:41-49`), `recent_window` slides forward with each log, so every training sample sees realistic, evolving lag features. During **prediction** (`mod.rs:162-184`), `self.recent_data` is frozen — the same VecDeque is passed unchanged to every iteration. Seven features (`recent_avg_1h`, `recent_avg_3h`, `recent_avg_6h`, `recent_trend`, `day_avg_so_far`, `prev_day_avg`, `occupancy_volatility`) are identical across all 24 prediction hours, causing flat-line output instead of dynamic curves.

The fix is an **autoregressive prediction loop**: clone `recent_data` into a working buffer, inject each prediction back before computing the next.

## Changes

**Single file:** `crates/hardy-gui/src/ml/mod.rs`

No changes needed to `FeatureExtractor::extract()`, momentum functions, `fallback_predict()`, or confidence calculation — they already accept `recent_data` as a parameter.

### 1. `predict()` — autoregressive loop (lines 162-184)

- Clone `self.recent_data` into `working_buffer` at the start
- Pass `&working_buffer` to `predict_single()` instead of letting it use `self.recent_data`
- After each prediction, push `(target_time, predicted_value)` into `working_buffer` (with capacity eviction matching `add_observation()`)
- `self.recent_data` remains unchanged

### 2. `predict_single()` — add `recent_data` parameter (lines 186-200)

```rust
fn predict_single(
    &self,
    target_time: DateTime<Utc>,
    hours_ahead: i64,
    baseline: &[HourlyAverage],
    schedule: &GymSchedule,
    recent_data: &VecDeque<(DateTime<Utc>, f64)>,  // NEW
) -> PredictionWithConfidence
```

Forward `recent_data` to `ml_predict()`.

### 3. `ml_predict()` — add `recent_data` parameter (lines 202-243)

```rust
fn ml_predict(
    &self,
    target_time: DateTime<Utc>,
    hours_ahead: i64,
    baseline: &[HourlyAverage],
    schedule: &GymSchedule,
    recent_data: &VecDeque<(DateTime<Utc>, f64)>,  // NEW
) -> Option<PredictionWithConfidence>
```

Use `recent_data` instead of `&self.recent_data` in the `feature_extractor.extract()` call.

### 4. Update existing tests

Tests that call `ml_predict()` directly need the new parameter:
- `test_predictor_uses_residual_quantiles` (line 509)
- `test_predictor_falls_back_without_quantiles` (line 545)
- `test_predictor_random_forest_method` (line 573)
- `test_predictor_ml_method_for_lr` (line 603)

Pass `&VecDeque::new()` — these tests don't exercise autoregressive behavior.

## New Tests

| Test | Purpose |
|------|---------|
| `test_predict_autoregressive_features_differ_across_hours` | Core regression test: populate `recent_data` with a clear trend, predict 6h, assert predictions are NOT all identical |
| `test_predict_working_buffer_does_not_mutate_recent_data` | Verify `self.recent_data` unchanged after `predict()` |
| `test_predict_empty_recent_data_still_works` | Edge case: empty buffer, predictions still produced with defaults |
| `test_predict_single_receives_recent_data_parameter` | Two different buffers (high/low) produce different predictions for same target |
| `prop_predict_values_always_in_valid_range` | All predictions in [0, 100] for arbitrary recent_data |
| `prop_predict_preserves_recent_data` | recent_data unchanged for arbitrary inputs |

## Design Notes

- **Clone cost**: 360 entries x 24 bytes = ~8.6 KB — negligible
- **Granularity mismatch**: Training has ~60 points/hour, predictions inject 1/hour. Acceptable — momentum functions handle sparse data gracefully and this is strictly better than frozen features
- **Fallback injection**: When ML falls back to historical average, that value is still injected into the working buffer so subsequent ML predictions see evolving features
- **Closed-hour gaps**: Skipped hours leave gaps in the buffer. Momentum functions filter by time window (not index), so this is handled naturally

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo nextest run --workspace
cargo check --workspace --all-targets
```
