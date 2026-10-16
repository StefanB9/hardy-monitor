# Forecaster split

The daemon runs on a small cloud instance; the nightly fit + benchmark of
51 models (~17 min on 4 cores, neural nets need memory) is too much for it.
Forecasting moves to a separate long-running program on a stronger machine;
both share the database.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Cloud daemon | Fetches readings and runs the nightly data repair only (still the only process that migrates). All forecasting leaves it, including the hardy-ml dependency; no fallback switch |
| Local program | New long-running binary `hardy-forecaster`: one-time history backfill at start-up (or with `--backfill`), live forecasts of all models every 15 minutes from the current fits, accuracy log every full hour, full refit + benchmark nightly; catches up at start-up when the last run is over a day old |
| One at a time | A Postgres advisory lock: a second forecaster refuses to start |
| Schema | The forecaster verifies the schema like the GUI and waits while it is behind |
| History ("as it would have been") | Backtest over all days with data; each day uses fits, ensemble weights, intervals and per-horizon winners from earlier days only |
| History outputs | Accuracy log (default forecast 1–6 h ahead per hourly origin, band, slot-average baseline) and a new `forecast_history` table with the hour-ahead default forecast for every past slot (made at the last full hour at least an hour before) |
| Keeping it current | Each nightly run appends its newest benchmark day to both |
| Retention | The 90-day accuracy-log retention is removed |
| GUI | 7/30-day chart draws the past hour-ahead forecast line and band; a forecast older than 30 min is labelled "forecast from HH:MM — forecaster offline"; older than an hour the quiet hour falls back to slot averages |
| Progress | Terminal progress for the long stages (backfill, benchmark days, shared fits, final fits) and a status line between runs; hardy-ml reports progress through a small trait, the forecaster draws it with `indicatif` (updates throttled, negligible next to fitting); hidden when not a terminal; console logs printed above the bars |

## Delivery

1. Split: `hardy-forecaster`, daemon without forecasting, advisory lock,
   progress, retention removed, GUI stale handling.
2. History: backfill, `forecast_history` (migration + queries), nightly
   append, chart line.

## History details (agreed for part 2)

| Topic | Decision |
|---|---|
| Earlier days | Each past day's default model, intervals and ensemble weights come from **all** earlier days of the backfill (cumulative), not a rolling 28-day window |
| Comparison column | The accuracy log's comparison value is the same model live logging uses (profile with decaying offset), so backfilled and live rows mean the same |
| Which slots | Only open slots with a reading (the slots the benchmark scores) |
| Existing rows | Kept: backfill and nightly append never overwrite stored rows (live log rows win) |
| When | At start-up while `forecast_history` is empty, or with `--backfill` (fills gaps); runs in the background, live forecasts keep publishing, the nightly run waits for it |

## Dependencies

| Crate | Why | Alternatives |
|---|---|---|
| indicatif 0.18 (no default features) | Multi-line progress bars with throttled redraws, ETA, hides itself off-terminal | Hand-written `\r` lines (no multi-bar, no ETA), `kdam` / `pbr` (less maintained) |
