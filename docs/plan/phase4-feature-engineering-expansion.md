# Phase 4: Feature Engineering Expansion — Implementation Plan

## Context

Phases 1–3 are complete and merged to `dev`. The ML pipeline now has Random Forest with grid-searched hyperparameters and 4-fold time-series CV. `PredictionFeatures` currently has 16 features covering cyclical time encoding, historical baselines, momentum (1h/3h), day-level averages, context flags (weekend/holiday), and seasonal encoding.

Phase 4 adds 6 new features (16→22) to improve RF prediction accuracy: `raw_hour`, `raw_weekday`, `time_to_close`, `occupancy_volatility`, `recent_avg_6h`, and `prev_week_same_slot`. These give RF direct numeric splits (raw hour/weekday), schedule awareness (time to close), stability signal (volatility), broader trend context (6h average), and autoregressive signal (previous week's value).

## Decisions (User-Approved)

1. **`prev_week_same_slot` sourcing**: Post-hoc override — `extract()` sets default to `historical_avg`, `data_prep.rs` overrides by binary-searching logs for the value at `timestamp - 7 days`. `OccupancyPredictor` keeps the `historical_avg` fallback. No signature change to `extract()`.
2. **`occupancy_volatility` window**: 1-hour window (std dev of values within last 1h). Complements existing `recent_avg_1h`.
3. **`time_to_close` when closed**: Clamp to `0.0` when `current_hour >= close_hour`.
4. **`recent_data` window expansion**: 180→360 (supports 6h rolling average at 1-min resolution).
5. **Grid search `max_features`**: `[4, 8, None]` → `[5, 11, None]` (sqrt(22)≈5, 22/2=11). Small grid: `[4, None]` → `[5, None]`.
6. **`PersistedModel` version**: Bump 1→2 (old 16-feature models rejected on load).

## Implementation Steps

### Step 1: Add `extract_volatility()` to `momentum.rs`

**File:** `src/ml/features/momentum.rs`

Add function computing population std dev of values within the last 1 hour of `recent_data`:

```rust
pub(super) fn extract_volatility(recent_data: &VecDeque<(DateTime<Utc>, f64)>) -> f64
```

Logic: filter entries within last 1h (same window as `recent_avg_1h` in `extract_momentum`), compute population std dev. Return `0.0` if fewer than 2 data points.

**Tests (write first):**
- `test_extract_volatility_empty` — returns 0.0
- `test_extract_volatility_constant` — returns 0.0 for identical values
- `test_extract_volatility_varying` — returns correct positive value
- `prop_extract_volatility_non_negative` — proptest: always >= 0.0
- `prop_extract_volatility_finite` — proptest: always finite

---

### Step 2: Add `extract_avg_6h()` to `momentum.rs`

**File:** `src/ml/features/momentum.rs`

Add function computing 6-hour rolling average from `recent_data`:

```rust
pub(super) fn extract_avg_6h(recent_data: &VecDeque<(DateTime<Utc>, f64)>) -> f64
```

Logic: filter entries within last 6 hours, compute mean. Return `50.0` (neutral default) if no data in window.

**Tests (write first):**
- `test_extract_avg_6h_empty` — returns 50.0
- `test_extract_avg_6h_with_data` — returns correct average
- `prop_extract_avg_6h_finite` — proptest: always finite

---

### Step 3: Expand `PredictionFeatures` struct + update `extract()`

**File:** `src/ml/features/mod.rs`

Add 6 new fields to `PredictionFeatures`:

```rust
pub raw_hour: f64,              // Direct hour (0–23), RF splits on boundaries
pub raw_weekday: f64,           // Direct weekday (0–6)
pub time_to_close: f64,         // Hours until gym closes, clamped >= 0
pub occupancy_volatility: f64,  // Std dev of 1h window
pub recent_avg_6h: f64,         // 6-hour rolling average
pub prev_week_same_slot: f64,   // Occupancy from same slot 7 days ago
```

Update `NUM_FEATURES` from 16 to 22. Update `to_vec()` with 6 new elements appended. Update `feature_names()` with 6 new names.

Update `FeatureExtractor::extract()`:
- Rename `_schedule` → `schedule`
- Compute `raw_hour = f64::from(hour)` and `raw_weekday = f64::from(weekday)`
- Compute `time_to_close`: `schedule.get_close_hour(local_time.date_naive())` minus current hour, clamped to `0.0`
- Call `momentum::extract_volatility(recent_data)` for `occupancy_volatility`
- Call `momentum::extract_avg_6h(recent_data)` for `recent_avg_6h`
- Set `prev_week_same_slot = historical_avg` (default; overridden in data_prep.rs)

Update `arb_prediction_features()` proptest strategy: split into 3 groups (8+8+6) to stay within proptest's 12-element tuple limit. Add appropriate ranges for new fields:
- `raw_hour`: `0.0..=23.0`
- `raw_weekday`: `0.0..=6.0`
- `time_to_close`: `0.0..=17.0`
- `occupancy_volatility`: `0.0..=50.0`
- `recent_avg_6h`: `0.0..=100.0`
- `prev_week_same_slot`: `0.0..=100.0`

**Tests (update existing):**
- `test_features_to_vec` — update struct literal with 6 new fields, verify length 22
- `test_feature_names_count` — passes via `NUM_FEATURES` (no change needed)
- All proptests pass via `NUM_FEATURES` constant
- `test_extract_time_to_close_weekday` — verify for known weekday scenario
- `test_extract_time_to_close_clamped` — when hour >= close_hour, result is 0.0
- `test_extract_raw_hour_and_weekday` — verify values match expected
- `test_extract_volatility_integrated` — volatility is finite and non-negative
- `test_extract_prev_week_defaults_to_historical` — without weekly data, uses historical_avg

---

### Step 4: Update all `create_test_features()` helpers across the codebase

**Files (4 files):**
- `src/ml/model/mod.rs`
- `src/ml/model/linear.rs`
- `src/ml/model/random_forest.rs`
- `src/ml/training/hyperparameter.rs`

Each has a `create_test_features(n)` function that constructs `PredictionFeatures` with named fields. Add the 6 new fields with varied realistic values:

```rust
raw_hour: f64::from((i as u32) % 24),
raw_weekday: f64::from((i as u32) % 7),
time_to_close: 5.0 + (t % 12.0),
occupancy_volatility: 2.0 + (t % 10.0),
recent_avg_6h: 42.0 + ((t * 0.9) % 28.0),
prev_week_same_slot: 38.0 + (t % 35.0),
```

**No new tests** — existing tests exercise the full 22-feature shape after this change.

---

### Step 5: Expand `recent_data` window capacity (180→360)

**Files:**
- `src/ml/training/data_prep.rs` — change `VecDeque::with_capacity(180)` to 360 and eviction threshold `>= 180` to `>= 360` (lines 41, 46)
- `src/ml/mod.rs` — change capacity in `OccupancyPredictor::new()` (line 34) and eviction in `add_observation()` (line 60) from 180 to 360

**Tests:**
- Existing tests pass unchanged (they just have a larger window now)

---

### Step 6: Add `lookup_prev_week()` and wire into `data_prep.rs`

**File:** `src/ml/training/data_prep.rs`

Add helper function to binary-search logs for the value closest to `timestamp - 7 days`:

```rust
fn lookup_prev_week(logs: &[OccupancyLog], target_time: DateTime<Utc>) -> Option<f64>
```

Logic: compute `target = target_time - 7 days`, binary search `logs` by timestamp for the entry nearest to `target`. Return `Some(percentage)` if found within a reasonable tolerance (e.g., within 1 hour), otherwise `None`.

Update the training loop in `prepare()` to override `prev_week_same_slot`:

```rust
for log in logs {
    // ... existing window management ...
    let mut feature = feature_extractor.extract(timestamp, 0, &recent_window, baseline, schedule);
    feature.prev_week_same_slot = lookup_prev_week(logs, timestamp)
        .unwrap_or(feature.historical_avg);
    features.push(feature);
    targets.push(log.percentage);
}
```

**Tests:**
- `test_lookup_prev_week_found` — with 2+ weeks of logs, returns value from 7 days ago
- `test_lookup_prev_week_not_found` — with only 1 day of logs, returns None
- `test_training_preparer_prev_week_values` — verify features include non-default prev_week values when logs span 2+ weeks

---

### Step 7: Update hyperparameter grid `max_features` values

**File:** `src/ml/training/hyperparameter.rs`

Update `default_grid()`: `max_features_values` from `[Some(4), Some(8), None]` to `[Some(5), Some(11), None]`
Update `small_grid()`: `max_features_values` from `[Some(4), None]` to `[Some(5), None]`

Update proptest configs that use hardcoded `Some(4)` to `Some(5)`.

**Tests (update):**
- `test_default_grid_values` — first config's max_features is `Some(5)`, last is `None`
- `test_default_grid_size` — still 144 (same dimensions)
- `test_small_grid_size` — still 16

---

### Step 8: Bump `PersistedModel` version

**File:** `src/ml/persistence.rs`

Change `CURRENT_VERSION: u32 = 1` to `CURRENT_VERSION: u32 = 2`.

**Tests (update):**
- `test_persisted_model_creation` — assert version == 2

---

### Step 9: Verify everything compiles and passes

```bash
cargo fmt --all -- --check
cargo clippy --all-targets
cargo check --all-targets
cargo nextest run
cargo nextest run --no-default-features
```

Verify:
- `NUM_FEATURES == 22` everywhere
- `to_vec()` returns exactly 22 elements
- `feature_names()` returns exactly 22 names
- All existing tests pass with updated struct literals
- RF training works with 22-feature matrix
- LR training works with 22-feature Array2
- Grid search runs with updated max_features values
- Old models rejected (version 2 > 1)

## Files Modified

| File | Change |
|------|--------|
| `src/ml/features/momentum.rs` | Add `extract_volatility()` and `extract_avg_6h()` |
| `src/ml/features/mod.rs` | Add 6 fields, update NUM_FEATURES/to_vec/feature_names, update extract(), update proptest strategy |
| `src/ml/mod.rs` | Expand OccupancyPredictor window 180→360 |
| `src/ml/training/data_prep.rs` | Expand window 180→360, add `lookup_prev_week()`, override `prev_week_same_slot` in training loop |
| `src/ml/model/mod.rs` | Update `create_test_features()` helper |
| `src/ml/model/linear.rs` | Update `create_test_features()` helper |
| `src/ml/model/random_forest.rs` | Update `create_test_features()` helper |
| `src/ml/training/hyperparameter.rs` | Update `create_test_features()` helper, update max_features grid [5,11,None] |
| `src/ml/persistence.rs` | Bump CURRENT_VERSION 1→2 |

## Key Reusable Code

- `momentum::extract_momentum()` (`features/momentum.rs`) — pattern for 1h/3h window filtering, reuse for volatility/6h
- `schedule.get_close_hour(date)` (`schedule.rs:73`) — already handles weekday/weekend/holiday
- `historical_avg` fallback pattern in `extract()` (`features/mod.rs:166-176`) — same fallback for `prev_week_same_slot`
- `cyclical::cyclical_encode()` (`features/cyclical.rs`) — existing, no changes needed

## Implementation Order

Steps 1–2 are independent (new functions, no callers yet). Step 3 is the breaking change — must be done together with Step 4 (test helpers) for compilation. Steps 5–8 are independent once Step 3+4 are done.

Practical order: **1 → 2 → 3+4 (atomically) → 5 → 6 → 7 → 8 → 9**
