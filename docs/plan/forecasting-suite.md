# Forecasting suite

Every model on the list below runs in production: the daemon fits them
nightly, benchmarks them on a rolling origin over the last four weeks, picks
the winner and publishes every model's live forecast. The GUI shows the
winner by default, lets the user switch or overlay models, and shows a
leaderboard. Delivered in seven phases, one PR each.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Scope | All models in production, not only an offline benchmark |
| Language | Pure Rust; best crate per model, judged by training/prediction speed and correctness (compile time does not matter) |
| Grid | 15-minute slots around the clock (96/day, 672/week); closed slots are 0; every forecast is clipped to 0–100 and forced to 0 when closed |
| Minimum history | Each model trains as soon as it has the technical minimum to fit; the leaderboard shows how much it trained on |
| Default forecast | Benchmark winner; the user can switch to or overlay any model |
| Nightly budget | Unlimited overnight (while the gym is closed) |
| Benchmark | Rolling origin, hourly, over the last 28 days |
| Refits in the backtest | Daily for statistical and ML models, weekly for neural nets (which roll forward on new data in between) |
| Results | Database tables + GUI leaderboard on the Model & Data page |
| Live forecasts | Computed by the daemon and written to a table; the GUI only reads rows (no ML crates or model loading in the GUI) |
| Current pipeline | Absorbed and retired: today's baseline becomes "weekly profile + decaying offset", ridge and RF become tabular models, the benchmark replaces the quality gate; `ml_models` / `ml_state` are dropped once phase 1 works |
| anofox-forecast | Accepted with guards: exact version pin, fits isolated on their own thread, output gate, golden tests; hand-port any model that misbehaves |

## Architecture (hardy-ml)

### Series grid

`Grid` — a dense `Vec<f64>` of 15-minute slots from a gym-local midnight,
built from readings (slot mean of the readings in it; open slots without a
reading are interpolated linearly, up to one hour, else left as missing and
handled by each model's fit), with a parallel `open: Vec<bool>` from
`GymSchedule`. Calendar helpers: slot of day, slot of week, holiday flag.

### Model interface

```rust
pub enum ModelId { Naive, MovingAverage, /* … one per model … */ }

pub trait Forecaster: Send {
    fn id(&self) -> ModelId;
    /// Smallest history the model can be fitted on.
    fn min_history(&self) -> TimeDelta;
    /// Fit on everything up to `until` (exclusive).
    fn fit(&mut self, data: &TrainData<'_>, until: DateTime<Utc>) -> Result<(), MlError>;
    /// Point forecasts for the next `steps` slots after `origin`, conditioned
    /// on data up to `origin` without refitting.
    fn predict(&self, data: &TrainData<'_>, origin: DateTime<Utc>, steps: usize)
        -> Result<Vec<f64>, MlError>;
    fn to_bytes(&self) -> Result<Vec<u8>, MlError>;
}
```

`TrainData` gives both views every model needs: the `Grid` (series models)
and the existing leak-free `History` / feature extraction (tabular models,
one direct model per horizon bucket). A registry builds every model with its
id; `ModelId` is stored as a short stable string in the database.

Conditioning on new data without refitting: models with state (ETS, ARIMA,
Kalman, RNNs) re-run their filter/encoder with fixed parameters up to the
origin. Where a crate cannot do that, the model refits at each origin if a
fit is fast enough (measured in its phase); otherwise its forecast from the
day's fit is used, and the leaderboard notes it.

### Benchmark

- Origins: every full hour during opening hours plus the hour before
  opening, over the last 28 days that have data.
- Each model is fitted per backtest day on data before that day (neural nets
  per backtest week) and predicts 96 steps from each origin of that day.
- Scored against actual slot values (open slots only), per horizon bucket
  ≤ 1 h, 1–3 h, 3–24 h, and overall: MAE, RMSE, MASE (scale: weekly
  seasonal naive, falling back to daily seasonal naive while there is less
  than a week), interval coverage (80 %) and pinball loss (10 % / 90 %).
- Stored with the training days and the number of origins behind each score.
- A model that lacks its minimum history at a backtest day skips that day; a
  model with too few scored origins (< 24) is listed but cannot win.

### Intervals

One method for every model: split-conformal — the 10 % / 90 % quantiles of
the model's own backtest errors per horizon bucket, added to its point
forecast, clipped to 0–100. Until a model has backtest errors, it has no
interval (the GUI shows the point line only).

### Ensembles

Built from the members' backtest forecasts, using for each backtest day only
origins before that day (no leakage): simple mean, median, inverse-MAE
weighted, horizon-dependent (best member per bucket), stacking (non-negative
least squares on member forecasts, per bucket). Members: every non-ensemble
model with a full backtest.

### Selection

The winner per horizon bucket is the eligible model with the lowest MAE in
that bucket; ties keep the previous winner. The default GUI forecast stitches
the winners of each bucket (i.e. the horizon-dependent choice), labelled as
such.

### Safety

- Each fit runs on its own thread with `catch_unwind`; a panic or error marks
  that model failed for the night (stored with the message) and the rest
  continue.
- Output gate: a forecast containing a non-finite value or anything outside
  −50…150 before clipping marks the fit failed (catches exploding ARIMA).
- ARIMA family: AR/MA roots checked for stationarity/invertibility after fit.

## Models

Minimum history is the technical minimum; "day" = one day with data.

### Phase 1 — baselines (hand-written)

| Model | Minimum |
|---|---|
| Persistence / naive | 1 reading |
| Recent moving average (last hour) | 1 reading |
| Damped local trend (last hour's slope, damped) | 2 readings |
| Daily seasonal naive | 1 day |
| Weekly seasonal naive | 7 days |
| Weekly seasonal mean | 7 days |
| Weekly seasonal median | 7 days |
| Recency-weighted weekly mean | 7 days |

### Phase 2 — profile models (hand-written)

| Model | Minimum |
|---|---|
| Weekly profile + current offset | 7 days |
| Weekly profile + decaying offset (today's baseline) | 1 day (falls back to weekday-type profile, as today) |
| Weekly profile + residual autoregression | 7 days |
| Similar-day / nearest-neighbour | 2 days |
| Daily-profile PCA + regression | 3 days |

### Phase 3 — ensembles

Simple mean, median, validation-weighted, horizon-dependent, stacking.

### Phase 4 — statistical, hand-written

| Model | Implementation | Minimum |
|---|---|---|
| Simple exponential smoothing | argmin Nelder–Mead on SSE | 1 day |
| Holt's linear trend | argmin | 1 day |
| Damped Holt | argmin | 1 day |
| Holt–Winters (additive, m = 96 and m = 672) | hand-written, argmin | 2 seasons |
| Autoregression | OLS on lags (faer) | 1 day |
| Theta | hand-written (Assimakopoulos & Nikolopoulos 2000) on the deseasonalised series | 2 days |
| Structural state-space / Kalman | hand-written local linear trend + trigonometric seasonal states, MLE via argmin (Durbin & Koopman 2012) | 2 days |

### Phase 5 — statistical, anofox-forecast + stlrs

| Model | Implementation | Minimum |
|---|---|---|
| Seasonal ETS / AutoETS | anofox `ETS`, `AutoETS` (m = 672, falls back to 96) | 2 seasons |
| ARIMA / AutoARIMA | anofox `AutoARIMA` | 1 day |
| SARIMA | anofox `SARIMA` (m = 96) | 2 days |
| STL + ETS, STL + ARIMA | stlrs `Stl` (period 96) + anofox on the remainder | 2 days |
| MSTL + ETS, MSTL + ARIMA | stlrs `Mstl` (96, 672) + anofox | 14 days |
| TBATS / AutoTBATS | anofox | 2 days |
| MFLES | anofox | 2 days |
| AutoTheta | anofox | 2 days |
| Fourier regression + ARIMA errors (moved from phase 4) | Fourier OLS (faer) + AutoARIMA on residuals | 7 days |

Persisted: the selected inner model (the `Auto*` search objects and the MSTL
forecaster do not serialise), plus the stlrs seasonal components.

### Phase 6 — regression and ML (tabular, direct per horizon bucket)

| Model | Implementation | Minimum |
|---|---|---|
| Ridge (lag/calendar features) | hand-written on faer (Cholesky) | 1 day |
| Elastic-net | hand-written coordinate descent (glmnet) | 1 day |
| GAM / spline regression | hand-written P-splines on faer, λ by GCV | 2 days |
| Random forest | smartcore 0.6 (existing) | 1 day |
| Gradient-boosted trees | forust-ml 0.7, stored as JSON inside the postcard envelope (its postcard round-trip is broken) | 1 day |
| Support-vector regression | smartcore SVR on the most recent ≤ 4 000 samples (kernel SVR is O(n²)) | 1 day |
| Quantile regression | hand-written linear quantile regression (IRLS), 10/50/90 % | 1 day |

### Phase 7 — neural networks (burn 0.22, `flex` CPU backend)

| Model | Input | Minimum |
|---|---|---|
| MLP | lag/calendar features | 1 day |
| Temporal convolutional network | last 672 slots | 8 days |
| GRU, LSTM | last 96 slots + calendar | 2 days |
| N-BEATS, N-HiTS | last 672 slots | 8 days |
| PatchTST | last 672 slots, patches of 16 | 8 days |

Multi-output heads (96 steps). Adam, early stopping on the last day before
the fit date, fixed seed. Weights saved with burn's burnpack store (`store`
feature; 0.22 dropped `into_bytes` and the backend type parameter — the
device picks the backend at runtime). The `cpu` backend is not enabled
(build-time LLVM download, slow here); `simd` / `rayon` measured no faster
on this machine, so they stay off.

## Database

New migrations (created with `cargo sqlx migrate add -r`):

- `forecast_models` — one row per model: `model` (id), `fitted_at`,
  `data_until`, `training_days`, `status` (`fitted` / `waiting` / `failed`),
  `message`, `bytes` (zstd-compressed artifact). Replaces `ml_models`.
- `benchmark_runs` — `id`, `started_at`, `finished_at`, `origins`.
- `benchmark_scores` — `run_id`, `model`, `bucket`, `mae`, `rmse`, `mase`,
  `coverage`, `pinball`, `origins`, `training_days`; plus the per-bucket
  conformal quantiles.
- `forecast_selection` — winner per bucket for the latest run.
- `live_forecasts` — `model`, `made_at`, `target`, `value`, `low`, `high`;
  replaced per model on each publish (every new reading, at most every 15
  minutes).
- `forecast_requests` — replaces `ml_state` (GUI's "Retrain now").
- `forecast_log` keeps logging the selected forecast and gains a `model`
  column for accuracy tracking.

`ml_models` and `ml_state` are dropped in phase 1 once the new tables carry
the baseline and its replacements.

## Daemon

- Nightly after closing: fit every model, run the benchmark, store scores and
  selection, fit the production models on all data, publish live forecasts.
  Models run in parallel (rayon) except neural nets, which run one at a time.
- During opening hours: on each new reading (at most every 15 minutes),
  re-predict every fitted model and replace its `live_forecasts` rows.
- A failed model never blocks the rest; its status and message are stored.

## GUI

- Chart: the selected forecast (stitched winners) with its band; a model
  picker to switch, and an overlay of up to five other models.
- Model & Data page: leaderboard (per bucket and overall, sortable), with
  status per model (fitted / waiting for data — N more days / failed —
  message), training days and last fit.
- Removes hardy-ml model loading from the GUI; the GUI reads rows only.

## Dependencies (workspace, all `default-features = false`)

| Crate | Version | Phase | Why |
|---|---|---|---|
| argmin, argmin-math (`vec`) | 0.11, 0.5 | 4 | Nelder–Mead / L-BFGS fits (already in the lock file via linfa) |
| faer | 0.24 | 4/6 | Pure-Rust linear algebra (ndarray-linalg needs LAPACK) |
| anofox-forecast (`serde`) | =0.16.0 | 5 | AutoETS, ARIMA, SARIMA, TBATS, MFLES, AutoTheta — no other Rust crate has them |
| stlrs | 0.4 | 5 | f64 STL / MSTL |
| forust-ml | 0.7 | 6 | Fastest, most accurate pure-Rust boosted trees measured |
| burn (`std`, `autodiff`, `optim`, `flex`, `store`) | =0.22.0 | 7 | Every layer built in, pure-Rust CPU backend, weights to bytes |

serde_json moves into hardy-ml for forust-ml's artifacts. linfa / linfa-linear
are removed once ridge moves to faer (phase 6). Research timings and the
rejected alternatives are in the session record of this plan's discussion.

## Tests (written first, every phase)

- Each model: reference values on small fixed series (hand-computed or from
  statsforecast / statsmodels, stored as fixtures), minimum-history
  behaviour, round-trip through bytes, closed slots forced to 0.
- Grid, benchmark origins, leakage (a model never sees data at or after the
  origin — proptest), scoring metrics, conformal intervals, ensemble weights
  without leakage, selection ties — proptest with 1000 cases where it is
  pipeline logic.
- Panic isolation and the output gate with deliberately broken models.
- Database: every new query with `TestDatabase`.
- Daemon: the nightly run and live publishing with `MockClock`.
- anofox golden tests guard against behaviour changes on upgrade.

## Phases

1. Grid, interface, registry, benchmark, conformal intervals, safety,
   new tables, the 8 baselines, nightly run + live publishing in the daemon,
   GUI reading `live_forecasts` and a basic leaderboard; retire the old
   pipeline. Done in phase 1 as well, so nothing is lost meanwhile: the old
   profile baseline ("weekly profile + decaying offset") and the old ridge /
   random forest (hourly, interpolated to slots) run as suite models until
   phases 2 and 6 replace them, and per-bucket selection drives the default
   forecast. `ml_state` stays (it coordinates the nightly run).
2. The other profile models, GUI model picker and overlay. As built: the
   offset holds until the next closing; the residual AR(4) is shrunk to
   Σ|φ| < 1 (stationary); similar days = the 3 nearest earlier days of the
   same kind (weekday / weekend-or-holiday) by RMSE over today so far; the
   PCA model keeps up to 3 components and updates the weekday-mean scores
   with today's slots as a Bayesian ridge (prior variance = eigenvalue,
   noise 9 pts²). The picker ("Best per horizon" or any published model)
   also drives the quiet hour; up to five comparison lines, kept in memory
   only.
3. Ensembles. As built: five ensembles of the non-ensemble models, fitted
   per horizon bucket on the members' backtest forecasts (a backtest day's
   ensemble uses only earlier days, so its first day is not scored): mean
   and median of the 5 members with the lowest MAE, inverse-MAE weighted
   (all members), best member, and stacking (least squares with weights
   ≥ 0 summing to 1, solved by projected gradient on the simplex). A member
   counts in a bucket if it forecast at least 80 % of that bucket's scored
   steps; missing member values are skipped and the weights renormalised.
   Ensembles compete in the per-bucket selection like any model. Their
   final weights (fitted on every backtest day) are stored as their
   artifact in `forecast_models`, so no migration; the daemon combines the
   members' live forecasts with them.
4. Hand-written statistical models. As built: simple exponential smoothing,
   Holt, damped Holt, autoregression and Theta run on the deviation from the
   weekly profile (forecast = profile + model), Holt–Winters (m = 96 and
   672) and the structural Kalman model on occupancy itself. Closed slots
   are skipped (missing: time advances, nothing is updated). Smoothing
   parameters and state-space variances by Nelder–Mead (argmin) on the
   one-step errors; the AR order by AIC with OLS through faer. Each model
   stores its state at the end of the fit and filters forward from there to
   the origin with fixed parameters. Fourier regression + ARIMA errors moved
   to phase 5, where AutoARIMA arrives.
5. anofox + stlrs statistical models. As built: anofox (exact pin, no
   default features) cannot condition on new data, so every model runs its
   full search and fit on the last 56 days up to each origin (fits measured
   at ≤ 0.25 s on 8 weeks, AutoTBATS ≈ 1 s); `fit` only prepares the
   weekly profile, and nothing of anofox is persisted. Seasonal models see
   the full slot grid: closed slots 0, missing open slots filled with the
   weekly profile. AutoARIMA and the residual ARIMA of the Fourier model run
   on the open-slot deviation from the profile. Models: AutoETS (m = 672,
   96 under two weeks), AutoARIMA, SARIMA(1,0,1)(1,1,1)[96],
   STL[96] + AutoETS / AutoARIMA, MSTL[96, 672] + AutoETS / AutoARIMA
   (stlrs, seasonal part repeated from the last period), AutoTBATS[96, 672],
   MFLES[96, 672], AutoTheta[672], Fourier regression (faer) + AutoARIMA
   errors. Exploding fits are caught by the output gate; golden tests pin
   each model's output on a fixed series.
6. Regression and ML models. As built: every model learns the deviation
   from the weekly profile at the target slot from slot-level features
   (horizon, target calendar, profile value, recent deviations: last slot,
   1 h, 3 h, trend, today so far, same time yesterday and last week). One
   model per horizon bucket (≤ 1 h, 1–3 h, 3–24 h) with the horizon as a
   feature; training rows from hourly origins, every 4th step beyond 3 h
   (SVR on the newest 4 000). The new ridge and random forest take over the
   `ridge` / `random_forest` ids; the old pipeline (features, samples,
   model) and linfa go. Quantile regression fits the median only: its band
   is conformal like every other model's.
7. Neural networks. Agreed: like phases 4–6 the nets learn the deviation
   from the weekly profile (forecast = profile + output), with recent
   occupancy and deviation as inputs (scaled to 0–1, closed slots 0, missing
   open slots filled from the profile) and the target's calendar / profile
   as known covariates; the loss covers open target slots with a value
   only. Training windows: one per 15-minute origin over the last 56 days.
   Weights in burn's burnpack format. In the backtest a net is fitted on
   the first day of each week of backtest days and reused for the next six
   (its inputs roll forward), so a night fits each net five times. Weight
   initialisation is seeded under a lock, so parallel fits stay
   reproducible; dropout is off.

Each phase: fmt, `clippy --workspace --all-targets`, `check --workspace
--all-targets` (also `SQLX_OFFLINE=true`), `nextest --workspace`, `.sqlx`
regenerated, headless GUI screenshots where the GUI changed.
