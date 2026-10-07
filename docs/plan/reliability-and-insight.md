# Reliability, Auto Repair, Live Accuracy, Tray, Docs

Follows the open-improvements review (items 1, 2, 4, 7, 8, 12). Delivered as
four PRs in this order.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Migrations | **Daemon only.** The GUI never migrates; it verifies the schema. |
| Auto repair | **Yesterday + catch-up** of days missed while the daemon was down |
| Forecast accuracy | **Log real forecasts** hourly; compare with what happened |
| Tray | **Tooltip + coloured status dot** on the icon |
| Health alerts | Existing ntfy alert topic; after 5 failed minutes (configurable) while open; one message per outage plus "resumed" |
| Forecast log retention | 90 days |

## PR 1 — Daemon health and schema guard (items 1, 2)

**Schema version (hardy-core `db`).**
- `SchemaStatus { Current, DbOlder { db, app }, DbNewer { db, app } }` from
  the newest successful row in `_sqlx_migrations` vs the newest migration
  compiled into the binary.
- `Database::connect` takes `Migrations::Apply` (daemon) or
  `Migrations::Verify` (GUI). `DbNewer` is an `AppError::SchemaMismatch`
  (non-retryable, Display-tested) with a message naming what to update.
- Daemon: `DbNewer` at startup → clear error, ntfy message, exit (no endless
  retry). Every 10 minutes it re-checks; if the DB became newer it logs an
  error and notifies once, and keeps running (additive migrations may still
  work; the failure alert covers real breakage).
- GUI: on `DbOlder`/`DbNewer` it shows a full-window notice ("Database not
  upgraded yet — update and restart the daemon" / "This app is older than the
  database — update the app") with a Retry button, and runs no queries.

**Health alerts (hardy-core `health`).**
- `HealthMonitor` state machine fed with each cycle's outcome while open:
  after `notifications.health_after_minutes` (default 5) consecutive failed
  cycles → `Down { since, reason }` once; first success after that →
  `Recovered { gap }`. Closed hours neither count nor reset.
- Also covers startup: database unreachable for that long → notified.
- Published via `AlertService` (new `publish_health`), title
  "Hardy Monitor: data collection".
- Property tests: at most one `Down` per outage, `Recovered` only after
  `Down`, never while closed.

Limitation: a daemon that is not running at all cannot report itself; the
GUI's stale-data warning covers that case.

## PR 2 — Nightly automatic repair (item 4)

- Migration `repair_state` (single row: `repaired_through DATE`).
- Pure `repair_due(now, schedule, repaired_through) -> Option<(first, last)>`:
  after the last closing + 15 min, the days after `repaired_through` up to the
  day that just closed, capped at 30 days; first run = only that day.
  Proptested.
- Daemon runs it each tick before model maintenance (so the nightly retrain
  sees repaired data), records `repaired_through` on success, logs the
  summary. A failed repair is retried next tick, at most once per 30 min.
- Manual repair in Model & Data unchanged.

## PR 3 — Live forecast accuracy (item 7)

- Migration `forecast_log(made_at, target, horizon, model_id NULL, predicted,
  low, high, baseline)`, primary key `(made_at, horizon)`.
- Daemon keeps the current model in memory (loaded at start and after each
  training). At each full hour while open it builds the 8-day history,
  forecasts +1…+6 h with the model and with plain averages, and stores them.
  Rows older than 90 days are deleted nightly.
- Query: per day and per horizon, mean absolute error of model and averages
  against the measured reading at the target minute (missing actuals
  skipped). Integration-tested with `TestDatabase`.
- GUI Model & Data: "Live accuracy" card — last 14 days, model vs averages
  error per day (paired bars) and the overall numbers; empty state until the
  first forecasts are scored.

## PR 4 — Tray status and docs (items 8, 12)

- Tooltip from the same data as the sidebar: "23% · Quiet · next quiet hour
  14:00", "Closed · opens 06:00", or the stale-data warning. Updated on each
  poll and tick (ignored by Linux AppIndicator).
- Icon: base icon with a dot in the occupancy-level colour (grey when closed
  or stale). Pure `badge_icon(rgba, size, colour)` with unit tests; only
  re-rendered when the colour changes.
- CLAUDE.md: current structure (core `db/`, `alert/`, `ntfy`, `retry`,
  `health`; ml modules; migrations list), migrations ownership rule ("only
  the daemon migrates"), headless verification notes.

## Verification

Each PR: tests first, `cargo fmt --check`, `cargo clippy --workspace
--all-targets`, `cargo nextest run --workspace`; GUI changes screenshot-checked
headless (temporary Linux features, not committed).
