# Data Integrity and Daemon Robustness

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Duplicates | One row per UTC minute: inserts truncate to the minute, `UNIQUE (timestamp)`, `ON CONFLICT DO NOTHING` |
| Out-of-range readings | Rejected (not stored) and logged; DB `CHECK (percentage BETWEEN 0 AND 100)` |
| Provenance | `source` column; ML training uses measured rows only, charts show all |
| Daemon | Retry with backoff, startup DB retry, graceful shutdown, explicit pool limits |

## Migration (`cargo sqlx migrate add -r data_integrity`)

Existing production rows violate the new constraints, so the migration cleans
them first — **without deleting anything**:

1. Create `occupancy_logs_quarantine (id, timestamp, percentage, reason,
   quarantined_at)`.
2. Move rows with `percentage` outside `[0, 100]` (incl. NaN) to quarantine
   with reason `out_of_range`.
3. Per UTC minute keep the lowest `id` (the first-inserted row, which is the
   measured one — repair always runs later) and move the rest to quarantine
   with reason `duplicate_minute`.
4. Truncate remaining timestamps to the UTC minute
   (`date_trunc('minute', timestamp, 'UTC')`).
5. Add `source TEXT NOT NULL DEFAULT 'measured'` with
   `CHECK (source IN ('measured', 'interpolated', 'boundary', 'smoothed'))`.
6. Add `CHECK` constraints for minute alignment and percentage range, and
   replace the plain timestamp index with `UNIQUE (timestamp)`.

The down migration drops the constraints and column, restores the plain
index, moves quarantined rows back and drops the quarantine table. It cannot
restore the sub-minute seconds removed in step 4 (fetch-time jitter, no
analytic meaning).

### Provenance values

- `measured` — written by the daemon.
- `interpolated` — repair gap fill.
- `boundary` — repair open/close zero entries, including measured rows that
  repair zeroed at the boundary.
- `smoothed` — a measured row whose value repair changed (outlier filter or
  smoothing). Added beyond the original three values because a changed
  measured value is no longer a measurement; it is excluded from training
  like the other repaired rows. Repaired rows that are smoothed again keep
  their original source.

## Code changes

### hardy-core

- `db::DataSource` enum (`sqlx::Type`, text) and `OccupancyLog.source`.
- `DatabaseConfig` gains `max_connections` (default 5) and
  `acquire_timeout_secs` (default 10); `Database::connect(&DatabaseConfig)`
  uses `PgPoolOptions`. `Database::new(url)` stays as the default-options
  shortcut used by tests.
- `insert_record(ts, pct)` stores a measured reading at its minute slot and
  returns `Option<i64>` (`None` = slot already filled).
  `insert_with_source(ts, pct, source)` for repair. `insert_at_timestamp`
  is removed.
- `batch_insert(records, source)` becomes one `UNNEST` statement with
  `ON CONFLICT DO NOTHING`, returning the inserted count.
- `batch_update_percentage(updates, source_if_measured)` also re-labels
  measured rows; single-row `update_percentage` is replaced by it in repair
  (one statement per day instead of one per row).
- `GymResponse::occupancy_percentage` rejects non-finite or out-of-range
  values with `AppError::Validation`.
- `retry` module: `RetryPolicy` (validated constructor, exponential delay,
  capped) and `retry(policy, op)` that retries only `AppError::is_retryable`
  errors. Tested with paused tokio time.
- `AppError::from_anyhow_sqlx` classifies a context-wrapped `sqlx::Error`
  so DB failures can be retried.

### hardy-daemon

- Startup: connect with retry (retryable errors only, backoff capped at 60 s,
  interruptible by shutdown).
- Each tick: the minute slot is fixed at the tick; fetch and insert run with
  retry (3 attempts, 2 s → 4 s) under a timeout of one fetch interval.
- `Ctrl-C`/`SIGTERM` handled with `tokio::select!` at every await point; the
  pool is closed before exit. Dropping an in-flight insert is safe: a single
  `INSERT` commits atomically or not at all, and the unique minute slot makes
  a retry idempotent.
- New tokio feature for the daemon: `signal`.

### hardy-gui

- Uses `Database::connect(&config.database)`.
- ML training filters to `DataSource::Measured`.

## Tests

- Migration: integration test seeds pre-migration-style rows (duplicates,
  out-of-range, sub-minute) by running migrations up to the previous version,
  then applies the new one and checks rows, quarantine and constraints.
- DB: minute truncation, duplicate insert returns `None`, out-of-range insert
  rejected by the constraint, batch insert skips occupied slots, source
  round-trip and re-labelling.
- API: out-of-range / NaN / infinite rejected (existing tests that asserted
  acceptance are inverted).
- Retry: success after transient failures, no retry on non-retryable errors,
  attempts exhausted, delay growth and cap (proptest), constructor
  validation.
- ML: measured-only filter.
