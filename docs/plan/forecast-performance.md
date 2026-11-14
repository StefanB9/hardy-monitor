# Forecast performance page

A full page on how good the forecasts are: the default forecast over all
history, every model live and in the nightly backtest, with all standard
accuracy measures and visual breakdowns.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Placement | New sidebar tab "Forecasts"; the leaderboard and Live accuracy move there; Model & Data keeps repair and export |
| Sources | All four: default forecast 1 h ahead over all history (`forecast_history`), the hourly 1–6 h log with its baseline (`forecast_log`), the nightly benchmark runs, and a new per-model live log |
| Per-model log | Every full hour while open, one row per model (all models and the default forecast) with its 24 quarter-hour values to 6 h ahead and their band, as arrays; kept forever (~850 rows/day, ~40 MB/year) |
| Benchmark runs | All kept (was 14); the selection of each run is kept too |
| Period | Selector 7 d / 28 d / 90 d / All, shared by every sub-tab |
| Layout | Sub-tabs: Overview · Models · When & where · Intervals & bias · Misses & days |
| Headline | Typical error (MAE, pts), hit rate within ±5 pts, better than averages by %, band coverage; each with the change from the previous period |
| Measures | MAE, RMSE, bias, WAPE, MASE (vs the same slot a week earlier), skill vs plain averages, band coverage, band width, interval score (80 % band), hit rates within ±5 / ±10 pts |
| Leaderboard | Same table with a Backtest \| Live switch; live follows the period; a row opens the model's detail |
| Extra views | Lead-time curve, forecast vs actual, decision quality, backtest vs live |
| Promotion | PRs into `dev` only |

## Definitions

- Forecasts are scored against measured or smoothed readings only, as the
  Live accuracy card does today.
- **Plain averages** (the baseline): the mean reading of the same slot on
  the same weekday over the four weeks before, from whatever of those
  readings exist.
- **MASE**: MAE divided by the MAE of the reading one week earlier at the
  same slot, on the same targets.
- **Interval score** (Winkler, 80 % band): width + 10 × the distance by
  which the actual value lies outside the band.
- **Hit rate**: share of forecasts within ±5 (±10) points of the reading.

## PR 1: data and measures (core, ml, forecaster)

1. Migration: `model_forecast_log (made_at, model, forecast real[],
   low real[], high real[])`, element `i` at `made_at + 15 min × (i + 1)`,
   NULL for closed slots; `benchmark_selection (run_id, bucket, model)`.
2. The forecaster logs every model's live forecast on full hours, next to
   the existing default-forecast log (no extra computation).
3. `save_benchmark` keeps every run and stores the run's selection.
4. Queries: scored pairs of the default forecast history (with baseline and
   last-week value), per-model log error sums grouped by model and step and
   by model and day, benchmark history per run and model.
5. `hardy_core::accuracy`: error sums from pairs or from the database, and
   the measures computed from them; breakdowns by day, weekday × hour,
   occupancy level and lead time.

## PR 2: Forecasts tab, Overview, Models

- New tab with the period selector and sub-tabs; leaderboard and Live
  accuracy move from Model & Data.
- Overview: the four headline numbers with change from the previous
  period, error per day with a weekly line and the baseline, a table of
  every measure per source.
- Models: leaderboard with the Backtest | Live switch; a model's detail
  (nightly backtest trend, live error over time, lead-time curve, bias,
  nights selected); backtest-vs-live scatter.

## PR 3: When & where, Intervals & bias

- Error heatmap by weekday × hour (the Week tab's grid), error by
  occupancy level, lead-time curve (default forecast against chosen
  models), forecast vs actual plot with an error histogram.
- Bias by hour and weekday, over- vs under-forecast share, band coverage
  over time and by lead time against 80 %, band width and interval score.

## PR 4: Misses & days, decision quality

- Biggest misses; a click shows that day's forecast vs actual chart.
- Decision quality: whether the quiet hour the forecast suggested, and the
  Plan tab's best time, actually turned out quiet (hit rate and points
  lost), computed from the logs and readings.
