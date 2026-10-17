# Performance

A review of the GUI, the forecaster and the daemon for work that is done
more often than needed. Nothing changes behaviour: every view, value and
forecast stays the same (the forecaster's golden and suite tests guard
the forecasts bit for bit).

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Scope | GUI (no visible change), forecaster (bit-identical), daemon in closed hours |
| Behaviour changes | None: no incremental chart fetch, no trimming of the GUI's history, no chart thinning, no reuse of earlier backtest days |
| PRs | One per group, each with before/after numbers |
| Promotion | Not promoted to `main` as part of this work |

## Measured before

- GUI idle on the Now tab (debug build): 1.64 CPU-seconds per 30 s; the
  tray is polled every 50 ms, and each message rebuilds the whole view.
- Database: the biggest GUI queries take 10–24 ms on the server; the
  1.5–3.5 s "slow statement" warnings at GUI start were queueing in the
  client, not the database.

## PR 1: GUI

1. Tray and menu events arrive through a subscription instead of a 50 ms
   poll.
2. The chart's "now" marker is drawn in the per-frame layer; the chart
   cache is cleared only when its readings, forecast or overlays change.
   The value-axis maximum is computed when data is applied, not per frame.
3. Published forecasts and the suite status are reloaded only when a
   newer publication or benchmark exists (a cheap "latest" query first);
   identical data is not re-applied. `available_models` uses one set.
4. The slot profile is rebuilt only when history grows; the Data tab's
   source counts and the forecast's "made at" are kept in state.
5. Quiet-window and Plan forecast lookups by binary search / index;
   Insights' two queries run concurrently; the hourly-averages query
   converts each timestamp's time zone once.

## PR 2: forecaster

1. Easter computed directly instead of through a locked cache; the grid
   keeps open/closed per slot and slot walks use it.
2. Live publishing runs the models in parallel (order kept) inside
   `spawn_blocking`.
3. Backtest origins borrow the history instead of copying it.
4. Conformal intervals merge each day's sorted errors into kept sorted
   lists instead of re-sorting.
5. Weekly profile and tabular training rows built once per backtest day
   and shared.
6. Benchmark results written with UNNEST batches; no second publish when
   a nightly run ends on a quarter hour.

## PR 3: daemon

- The repair state is cached in the daemon, so its query runs only when a
  repair is due; the schema check runs only while the gym is open.
