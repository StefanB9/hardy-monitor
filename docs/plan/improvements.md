# Hardy Monitor — Improvement Plan

Status legend: `[ ]` pending · `[~]` in progress · `[x]` done

---

## Phase 0 — Test Isolation (Prerequisite for reliable testing)

> **Problem:** All database integration tests share `DATABASE_URL`, which points at the
> production database. Tests insert records without cleanup, assertions use `>= N` instead of
> exact counts because they expect foreign data, and concurrent nextest runs corrupt each
> other's results.

### 0-A — Add `Database::close()` · `src/db.rs`

- [x] Add `pub async fn close(self)` that calls `self.pool.close().await`
- Required so the test fixture can drain all connections before issuing `DROP DATABASE`.
  PostgreSQL refuses to drop a database with active connections; `PgPool::drop` schedules
  closure but does not await it.

```rust
/// Waits for all in-flight queries to finish and all connections to return,
/// then closes the pool. Call before `DROP DATABASE` in test teardown.
pub async fn close(self) {
    self.pool.close().await;
}
```

### 0-B — Create `tests/common/mod.rs` — `TestDatabase` fixture

- [x] New file: `tests/common/mod.rs`
- Reads `DATABASE_URL`, derives an admin URL by replacing the database name with `postgres`
- Generates a unique name: `hardy_test_{pid}_{nanos}` (nanosecond timestamp prevents
  collisions even across parallel nextest workers in the same binary)
- Creates the database via `CREATE DATABASE`, runs SQLx migrations, returns the handle
- `cleanup()` calls `db.close().await` then `DROP DATABASE IF EXISTS "..." WITH (FORCE)`
  (`WITH (FORCE)` requires PostgreSQL 13+ and terminates any stray connections)

**URL parsing strategy** (handles `postgres://`, `postgresql://`, query params):
```rust
fn replace_db_name(url: &str, new_db: &str) -> String {
    let (base, params) = url.split_once('?').unwrap_or((url, ""));
    let last_slash = base.rfind('/').expect("invalid PostgreSQL URL");
    let prefix = &base[..last_slash];
    if params.is_empty() { format!("{prefix}/{new_db}") }
    else                  { format!("{prefix}/{new_db}?{params}") }
}
```

**Leftover databases** — if a test panics before `cleanup()` is called, the database
remains. Clean up manually with:
```sql
SELECT 'DROP DATABASE "' || datname || '";'
FROM   pg_database
WHERE  datname LIKE 'hardy_test_%';
```

### 0-C — Rewrite `tests/database.rs`

- [x] Replace `require_db!()` / shared `DATABASE_URL` with `TestDatabase::new().await`
- [x] Add `tdb.cleanup().await` at the end of every test
- [x] Tighten assertions from `>= N` to exact counts (clean DB means deterministic counts)
- [x] Remove the `// Note: In a shared test database, there might be more records` comments

**Pattern for every test:**
```rust
#[tokio::test]
async fn test_insert_record() {
    let tdb = common::TestDatabase::new().await;

    let id = tdb.db.insert_record(Utc::now(), 50.0).await.expect("insert failed");
    assert!(id > 0);

    tdb.cleanup().await;
}
```

---

## Phase 1 — Configuration Correctness

> **Problem:** Invalid config values are silently accepted. The ntfy server URL is hardcoded.
> Missing cooldown means rapid threshold crossings spam notifications.

### 1-A — Add `AppConfig::validate()` · `src/config.rs`

- [x] Add `pub fn validate(&self) -> Result<()>` and call it at the end of `load()`
- Checks to enforce (with descriptive `AppError::Config` messages on failure):

| Field | Rule |
|---|---|
| `schedule.weekday.open_hour` | `< close_hour` |
| `schedule.weekend.open_hour` | `< close_hour` |
| `schedule.*.{open,close}_hour` | `<= 24` |
| `thresholds.low_occupancy_percent` | `< high_occupancy_percent` |
| `thresholds.*.occupancy_percent` | `0.0 ..= 100.0` |
| `notifications.threshold_percent` | `0.0 ..= 100.0` |
| `refresh.data_fetch_interval_secs` | `> 0` |
| `refresh.ui_interval_secs` | `> 0` |
| `analytics.prediction_window_days` | `> 0` |

### 1-B — Add `ntfy_server` to `NotificationConfig` · `src/config.rs`, `config.toml`

- [x] Add `ntfy_server: String` field with default `"https://ntfy.sh"`
- [x] Update `CombinedNotifier` to use `config.notifications.ntfy_server` instead of the
  hardcoded `"https://ntfy.sh"` string in `src/traits.rs:134`
- [x] Add the field to `config.toml` with a comment

```toml
[notifications]
enabled = true
threshold_percent = 30.0
ntfy_topic = "hardys-occupancy-1993"
ntfy_server = "https://ntfy.sh"    # override for self-hosted instances
cooldown_secs = 300                 # minimum seconds between notifications
```

### 1-C — Add `cooldown_secs` to `NotificationConfig` · `src/config.rs`

- [x] Add `cooldown_secs: u64` field, default `300` (5 minutes)
- [x] Add `last_notified_at: Option<Instant>` to `NotificationState` in `src/app.rs`
- [x] In `handle_fetch_completed`, gate notification dispatch on:
  ```rust
  let cooldown_elapsed = self.notifications.last_notified_at
      .map(|t| t.elapsed().as_secs() >= self.config.notifications.cooldown_secs)
      .unwrap_or(true);

  if self.notifications.enabled && is_below && !self.notifications.was_below_threshold
      && cooldown_elapsed
  {
      self.notifications.last_notified_at = Some(Instant::now());
      // dispatch notification task
  }
  ```

### 1-D — Fix misleading config priority comment · `src/config.rs:201`

- [x] Change "optional, lowest priority" to "overrides defaults; overridden by user config
  dir (`~/.config/hardy-monitor/config.toml`) and `HARDY__*` env vars"

### 1-E — Log warning on config dir fallback · `src/config.rs:165`

- [x] Replace silent `unwrap_or_else(|| PathBuf::from("."))` with:
  ```rust
  dirs::config_dir().unwrap_or_else(|| {
      tracing::warn!("could not determine OS config directory, skipping user config file");
      PathBuf::from(".")
  })
  ```

---

## Phase 2 — Notification Correctness

> **Problem:** Desktop notifications block the Tokio executor thread. The ntfy thread is
> untracked and uses a blocking HTTP client inside an async context. Errors are swallowed.

### 2-A — Fix blocking `notify_rust` call in async context · `src/traits.rs:126-128`

- [x] The `Notifier::notify` trait method must become `async fn notify`
- [x] Wrap `notify_rust::Notification::new().show()` in `tokio::task::spawn_blocking`:
  ```rust
  tokio::task::spawn_blocking(move || {
      notify_rust::Notification::new()
          .summary(title)
          .body(body)
          .appname("Hardy Monitor")
          .show()
  })
  .await
  .map_err(|e| anyhow::anyhow!("desktop notification task panicked: {e}"))??;
  ```

### 2-B — Replace `std::thread::spawn` for ntfy with `tokio::task::spawn` · `src/traits.rs:140`

- [x] Since `notify` is now async and called from an async context, use the async `reqwest`
  client (not `reqwest::blocking`) and `tokio::task::spawn` (or just `.await` directly since
  we're already in an async fn):
  ```rust
  if let Some(ref topic) = self.ntfy_topic {
      let url  = format!("{}/{topic}", self.ntfy_server);
      let body = format!("{title}\n{body}");
      let client = reqwest::Client::builder()
          .timeout(Duration::from_secs(10))
          .build()?;
      if let Err(e) = client.post(&url).body(body).send().await {
          tracing::warn!(error = %e, %url, "ntfy notification failed");
      }
  }
  ```

### 2-C — Log notification errors · `src/traits.rs`, `src/app.rs`

- [x] ntfy send failure: `tracing::warn!(error = %e, topic = %topic, "ntfy push failed")`
- [x] Desktop notification failure: `tracing::warn!(error = %e, "desktop notification failed")`
- [x] In `app.rs`: propagate the `Result` from `notifier.notify()` to the task result instead
  of discarding it with `let _ =`

### 2-D — Update all `Notifier` implementors to async · `src/traits.rs`

- [x] `SystemNotifier::notify` → async
- [x] `CombinedNotifier::notify` → async
- [x] `MockNotifier::notify` → async (trivial, just add `async`)
- [x] Update callers in `src/app.rs` to `.await` the result

---

## Phase 3 — Code Correctness

> **Problem:** Panicking `.unwrap()` calls in production paths, a misleading 0% occupancy
> reading when there is no data, and a non-atomic batch insert.

### 3-A — Fix `.unwrap()` on poisoned mutex in `MockClock` / `MockNotifier` · `src/traits.rs`

- [x] `self.utc_time.lock().unwrap()` × 2 → `lock().unwrap_or_else(|p| p.into_inner())`
- [x] `self.notifications.lock().unwrap()` × 5 → same pattern
- These types are in the main library (not `#[cfg(test)]`), so `unwrap_used = "deny"` applies.
  Mutex poisoning recovery with `into_inner()` is the idiomatic safe alternative.

### 3-B — Fix `.unwrap()` on `and_hms_opt` · `src/db.rs:245, 250`

- [x] `date.and_hms_opt(0, 0, 0).unwrap()` — panics when the local timezone has a DST gap
  exactly at midnight. Replace with:
  ```rust
  date.and_hms_opt(0, 0, 0)
      .context("failed to construct start-of-day time (possible DST gap)")?
  ```

### 3-C — Fix "no data" → `0.0` misrepresentation · `src/app.rs:971`

- [x] `Ok(None) => Message::FetchCompleted(Ok(0.0))` treats "database is empty" as 0%
  occupancy. The gauge shows 0% rather than a "no data" state.
- [x] Change `FetchCompleted(Result<f64, AppError>)` to `FetchCompleted(Result<Option<f64>, AppError>)`
- [x] Handle `None` in `handle_fetch_completed`: set `self.data.occupancy = None` and display
  a "Waiting for data…" state in the gauge instead of 0%

### 3-D — Make `batch_insert` atomic · `src/db.rs:281`

- [x] Current implementation: one `INSERT` per record in a loop, each auto-committed. A
  failure midway leaves a partial write.
- [x] Wrap in a transaction:
  ```rust
  pub async fn batch_insert(&self, records: Vec<(DateTime<Utc>, f64)>) -> Result<()> {
      let mut tx = self.pool.begin().await.context("failed to begin transaction")?;
      for (timestamp, percentage) in records {
          let ts = timestamp.to_rfc3339();
          sqlx::query!(
              "INSERT INTO occupancy_logs (timestamp, percentage) VALUES ($1, $2)",
              ts, percentage
          )
          .execute(&mut *tx)
          .await
          .context("failed to insert record in batch")?;
      }
      tx.commit().await.context("failed to commit batch insert")
  }
  ```

---

## Phase 4 — Code Quality

> **Problem:** Unstructured tracing calls (CLAUDE.md violation), duplicated CSV export logic,
> and string-based timestamp storage that forces a re-parse on every use.

### 4-A — Fix unstructured tracing calls · `src/main.rs`

- [x] All `tracing::*!` calls with format-string data fields must be converted to KV form:

```rust
// Before (forbidden):
tracing::info!("Waiting {} seconds until next full minute...", seconds_until_next_minute);
tracing::warn!("Timer drift detected: {}s off from minute boundary, re-syncing", drift);
tracing::info!("Starting fetch loop with interval: {} seconds", interval_secs);
tracing::info!("Recorded occupancy: {:.1}%", percentage);
tracing::info!("Schedule configured: weekday {}-{}, weekend {}-{}",
    config.schedule.weekday.open_hour, ...);

// After (required):
tracing::info!(wait_secs = seconds_until_next_minute, "waiting for next full minute");
tracing::warn!(drift_secs = drift, threshold_secs = DRIFT_THRESHOLD_SECS,
    "timer drift detected, re-syncing");
tracing::info!(interval_secs, "starting fetch loop");
tracing::info!(occupancy_pct = percentage, "recorded occupancy");
tracing::info!(
    weekday_open  = config.schedule.weekday.open_hour,
    weekday_close = config.schedule.weekday.close_hour,
    weekend_open  = config.schedule.weekend.open_hour,
    weekend_close = config.schedule.weekend.close_hour,
    "schedule configured"
);
```

### 4-B — Remove duplicated CSV export in `app.rs` · `src/app.rs:452-480`

- [x] `src/db.rs` already has `export_to_csv(output_dir, clock)`. The inline CSV write in
  `app.rs` is a full duplication of that logic and diverges independently.
- [x] Replace the `ExportCsv` handler body with a call to `self.db.export_to_csv(...)`.
- [x] Remove the dead code from `app.rs`.

### 4-C — Change `OccupancyLog::timestamp` from `String` to `DateTime<Utc>` · `src/db.rs`

- [x] SQLx supports `DateTime<Utc>` directly via the `chrono` feature (already enabled)
- [x] Change the field type, remove `datetime()` method, update all `query_as!` macros and
  any code that calls `.datetime()`
- [x] Time complexity impact: eliminates one RFC3339 parse per record per access — relevant
  in analytics paths that iterate over hundreds of `OccupancyLog` values

> **Note:** This is a larger refactor. Confirm the migration (`timestamp TEXT`) is not changed
> — only the Rust struct type changes; SQLx will coerce `TEXT` → `DateTime<Utc>` at query
> time via its `chrono` decoder.

---

## Cross-Cutting: Notification Settings Persistence

- [ ] `NotificationThresholdChanged` and `NotificationToggled` update in-memory state only.
  After a restart, config.toml values take over again.
- [ ] **Decision required:** either persist changes back to the user config file
  (`~/.config/hardy-monitor/config.toml`) on toggle, or add a UI label making it explicit
  that settings are session-only and the config file is the source of truth.

---

## Summary Table

| # | Phase | File(s) | Risk | Size |
|---|---|---|---|---|
| 0-A | Test isolation | `src/db.rs` | Low | XS |
| 0-B | Test isolation | `tests/common/mod.rs` (new) | Low | S |
| 0-C | Test isolation | `tests/database.rs` | Low | M |
| 1-A | Config | `src/config.rs` | Low | S |
| 1-B | Config | `src/config.rs`, `src/traits.rs`, `config.toml` | Low | S |
| 1-C | Config | `src/config.rs`, `src/app.rs` | Low | S |
| 1-D | Config | `src/config.rs` | Low | XS |
| 1-E | Config | `src/config.rs` | Low | XS |
| 2-A | Notifications | `src/traits.rs` | Medium | S |
| 2-B | Notifications | `src/traits.rs` | Medium | S |
| 2-C | Notifications | `src/traits.rs`, `src/app.rs` | Low | XS |
| 2-D | Notifications | `src/traits.rs`, `src/app.rs` | Medium | M |
| 3-A | Correctness | `src/traits.rs` | Low | XS |
| 3-B | Correctness | `src/db.rs` | Low | XS |
| 3-C | Correctness | `src/app.rs` | Medium | M |
| 3-D | Correctness | `src/db.rs` | Low | S |
| 4-A | Quality | `src/main.rs` | Low | S |
| 4-B | Quality | `src/app.rs`, `src/db.rs` | Low | S |
| 4-C | Quality | `src/db.rs`, callers | Medium | L |

**Recommended order of execution:** 0 → 1-A through 1-E → 2-A through 2-D → 3-A, 3-B → 3-C → 3-D → 4-A, 4-B → 4-C
