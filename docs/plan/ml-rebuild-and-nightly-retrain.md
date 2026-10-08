# ML Rebuild and Nightly Retraining

## Problems found (audit of `hardy-gui/src/ml`)

1. **Target leakage.** `TrainingDataPreparer` pushes the current reading into
   the lag window *before* extracting features, so `recent_avg_1h/3h/6h`,
   trend and volatility contain the target. `hours_ahead` is always 0: the
   model learns a nowcast ("now ≈ last hour") and CV scores look excellent.
2. **Train/serve skew.** At prediction time the lag features come from an
   autoregressive loop that injects one prediction per hour into ~60 real
   readings per hour, so forecasts barely move. `prev_week_same_slot` uses
   real data in training but the slot average when predicting;
   `prev_day_avg` can never be computed (6 h window) and is always 50;
   `week_of_year` cannot be learned from 8 weeks of data.
3. **Warm-up / stale buffer.** The predictor's lag buffer is filled only by
   the GUI's per-minute fetch (stamped with `now`, re-adding the same reading
   if the daemon is down) and starts empty on every launch.
4. **Saved models don't reload.** `MAX_DECOMPRESSED_SIZE` is 1 MB; a tuned
   random forest is far larger, so loading after restart fails.
5. **Overconfident intervals.** Residual quantiles come from leaky
   horizon-0 fits, widened by a fixed factor per hour.
6. **Too heavy to run nightly.** Default grid: 48 configs × 4 folds × up to
   500 unlimited-depth trees.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Model storage | Database (`ml_models`), last 3 kept; shared by daemon and GUI |
| ML scope | Rebuild as direct multi-horizon forecasting |
| Training cost | Nightly: one fit with last best hyperparameters. Weekly (Sunday night): small grid |
| GUI "Train" button | Requests a retrain from the daemon via the database |

## Phase 1 — `hardy-ml` crate (mechanical)

Move `hardy-gui/src/ml/*` into a new library crate `crates/hardy-ml`
(depends on `hardy-core`; ML dependencies move from `hardy-gui`). Daemon and
GUI depend on it. Boundary: core ← ml ← daemon / gui. No behaviour change;
all existing tests move with the code. CLAUDE.md project structure updated.

## Phase 2 — Direct multi-horizon forecasting

**Samples.** Anchors `t` every 15 minutes while the gym is open, over measured
rows only. For each horizon `h ∈ 1..=prediction_horizon_hours` whose target
`t + h` is inside opening hours and has a reading (±5 min), one sample:
features from data **≤ t only**, target = reading at `t + h`.

**Features** (`FEATURE_VERSION = 2`; models with another version are
ignored):

- Target calendar: hour-of-day sin/cos (fractional), weekday sin/cos,
  weekend, holiday, minutes until closing — all in gym time.
- Horizon: `hours_ahead`.
- Slot profile of the target: historical mean/std for its (weekday, hour).
- State at `t`: current value, 1 h and 3 h means, 1 h trend (%/h), today's
  mean so far, and deviation of the current value from its own slot mean
  ("is today busier than usual").
- Same slot yesterday and a week ago (falls back to the slot mean).

Dropped: `week_of_year` (unlearnable from 8 weeks), `prev_day_avg`,
autoregressive injection.

**Prediction.** `Forecaster::forecast(&History, now, &GymSchedule)` uses the
same feature code on the last 8 days of measured readings from the database,
so daemon and GUI produce identical forecasts with no warm-up. `History` is a
sorted series with binary-search lookups.

**Confidence.** 10th/90th percentile of out-of-fold residuals **per
horizon** (no ad-hoc widening).

**Quality gate.** Hold out the last 7 days: fit on earlier data (slot
profile also from earlier data only), compute MAE on the holdout for the
model and for the slot-average baseline. The candidate is accepted only if it
beats the baseline; it is then refit on all data. Both MAEs are stored with
the model and shown in the GUI.

**Model size.** Hyperparameters are capped (≤ 100 trees, depth ≤ 12,
min leaf ≥ 10); the decompression limit becomes 64 MB, and a test asserts a
model trained on a realistic synthetic year stays well below it.

**Weekly grid.** `n_trees ∈ {50, 100}`, `max_depth ∈ {8, 12}`,
`min_samples_leaf ∈ {10, 20}`, 3 time-series folds with a 24 h gap.

## Phase 3 — Storage, nightly retrain, GUI

**Migration `ml_models`** (reversible): `id`, `trained_at`,
`feature_version`, `algorithm`, `training_samples`, `holdout_mae`,
`baseline_mae`, `tuned` (bool), `model` (bytea: zstd/bincode
`PersistedModel`). Only the newest three rows are kept.
**Migration `ml_state`** (single row): `retrain_requested_at`,
`last_attempt_at`, `last_error`.

**Daemon.**

- Retrains when any of: no model for the current feature version (also at
  startup), retrain requested, or the gym closed ≥ 15 min ago and the latest
  model is from before today's closing. Sunday nights run the grid search;
  other nights reuse the latest model's hyperparameters.
- Training runs on a blocking task; the fetch loop keeps collecting data.
  One training at a time; failures are recorded in `ml_state.last_error`
  and logged, and the previous model stays in use.

**GUI.**

- Loads the newest model from the database at start and whenever a newer
  one appears (checked each data refresh); forecasts with the shared
  `Forecaster` on history loaded from the database (8 days at start, then
  incremental).
- "Train model" sets `retrain_requested_at`; the view shows "Retrain
  requested…", the last error if any, and the model's holdout MAE vs
  baseline MAE. The local training, cancel and file load/save paths are
  removed; `ml.model_path` is no longer used.

## Tests

- Feature extraction never reads data after `t` (property test over random
  series and anchors); sample builder targets/horizons; history lookups.
- Forecaster: same features in training and prediction for the same anchor;
  forecasts only for open target hours.
- Quality gate: accepts a model on a learnable synthetic pattern, rejects a
  model worse than the baseline.
- Persistence: bytes round-trip, version mismatch rejected, size bound.
- DB: model save/load/prune to 3, `ml_state` request/clear/error
  (`TestDatabase`); migrations up/down.
- Daemon scheduling decision as a pure function (closing, Sunday tuning,
  request flag, missing model) with property tests.
