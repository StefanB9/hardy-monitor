# Dev-Mode Logging Plan

## Goal

Add structured `debug!` / `trace!` calls throughout the codebase to make the notification
state machine, data pipeline, and repair logic observable during development, with **zero
runtime cost in release builds** via compile-time log stripping.

---

## Step 1 — Enable Compile-Time Stripping in `Cargo.toml`

`tracing` supports profile-scoped `max_level_*` feature flags that instruct the macro
expander to replace log calls below the ceiling with `()` at compile time — no branches,
no string allocations, no overhead.

### Change

```toml
# Before
tracing = { version = "0.1.44", default-features = false, features = ["attributes", "std"] }

# After
tracing = { version = "0.1.44", default-features = false, features = [
    "attributes",
    "std",
    "release_max_level_info",   # strips debug! and trace! from release builds
] }
```

`release_max_level_info` means:
- **Debug builds**: `error!`, `warn!`, `info!`, `debug!`, `trace!` all compile in.
- **Release builds**: `error!`, `warn!`, `info!` compile in; `debug!` and `trace!` compile
  to a no-op (`()`). The optimizer removes them entirely — no branch, no format string.

No changes to `main.rs` or `EnvFilter` configuration are needed.

---

## Step 2 — `src/app.rs`: Notification State Machine

This is the highest-priority gap. The entire decision tree (threshold, cooldown, state
transition) is currently silent.

### 2a — `handle_fetch_completed()` (lines ~863–933)

Add a `debug!` block covering every condition in the notification guard:

```rust
// After computing is_below and cooldown_elapsed, before the if-block:
tracing::debug!(
    occupancy_pct = percentage,
    threshold_pct = self.notifications.threshold,
    is_below,
    was_below = self.notifications.was_below_threshold,
    notifications_enabled = self.notifications.enabled,
    cooldown_elapsed,
    "notification eligibility check"
);

// Inside the if-block, before spawning the task:
tracing::debug!(
    occupancy_pct = percentage,
    "dispatching notification"
);

// In the else path (when the guard fails), add a single debug explaining why:
// (either not below, already was below, cooldown active, or disabled — the above
//  debug log already carries all fields needed to reconstruct the reason)
```

Also add a debug on the `Ok(None)` branch (empty DB on first run):

```rust
tracing::debug!("fetch returned no data (database empty or first run)");
```

And on the error branch:

```rust
// The existing error path in handle_fetch_completed's Err arm
// already has no tracing — add:
tracing::debug!(error = %e, "fetch error, skipping notification check");
```

### 2b — `update()` — `Message::FetchAlignmentComplete` and `Message::FetchTick` (lines ~292–311)

Both arms have a `schedule.is_open()` check that silently skips the fetch. Add:

```rust
tracing::debug!(
    is_open = schedule_result,  // bool from is_open()
    "schedule check result"
);
```

### 2c — `update()` — `Message::TrayCheck` (lines ~430–456)

Window visibility toggle is silent. Add:

```rust
tracing::debug!(visible = self.ui.is_window_visible, "tray event toggled window visibility");
```

### 2d — `update()` — `Message::NotificationToggled` (lines ~370–374)

When the user toggles notifications from the UI, log the state change:

```rust
tracing::debug!(
    enabled,
    threshold_pct = self.notifications.threshold,
    "notification toggle changed"
);
```

---

## Step 3 — `src/main.rs`: Daemon Loop

### 3a — `run_daemon()` fetch loop (lines ~148–201)

The alignment check iteration is silent for most ticks. Add a trace (stripped in release):

```rust
tracing::trace!(iteration, "daemon loop tick");
```

The fetch path (when schedule is open) has no debug confirming it actually proceeds:

```rust
tracing::debug!("schedule open, proceeding with fetch");
```

### 3b — `fetch_and_store()` (lines ~218–224)

Add a debug between fetch and insert:

```rust
tracing::debug!(occupancy_pct = percentage, "fetched occupancy, storing to database");
```

---

## Step 4 — `src/repair.rs`: Repair Pipeline

The repair pipeline does the most complex multi-step work and has no internal logging.

### 4a — `repair_date_range()` (lines ~69–127)

```rust
// At entry, after computing the date list:
tracing::debug!(date_count = dates.len(), "starting repair job");

// On progress update:
tracing::debug!(
    repairs_made,
    date = %date,
    "date repaired"
);
```

### 4b — `repair_day()` (lines ~130–171)

```rust
tracing::debug!(date = %date, "repairing day");

// After each step, log the count returned:
tracing::debug!(cleaned = cleaned_count, "outside-hours records cleaned");
tracing::debug!(gaps_filled, "gaps filled");
tracing::debug!(smoothed_count, "records smoothed");
```

### 4c — `clean_outside_hours()` (lines ~175–215)

```rust
// When deleting a record:
tracing::trace!(timestamp = %record.timestamp, "deleted out-of-hours record");

// When zeroing a record:
tracing::trace!(timestamp = %record.timestamp, "zeroed boundary record");
```

Use `trace!` here (not `debug!`) — this fires per-record in a loop and would be noisy
even in debug builds except when actively debugging repair logic.

### 4d — `fill_gaps()` (lines ~278–345)

```rust
// When a fillable gap is found:
tracing::debug!(
    gap_minutes,
    from = %m1_timestamp,
    to = %m2_timestamp,
    "filling gap with interpolated records"
);

// When a gap is too large to fill:
tracing::debug!(
    gap_minutes,
    max = MAX_GAP_MINUTES,
    "gap too large to fill, skipping"
);
```

### 4e — `smooth_and_filter()` (lines ~218–272)

```rust
// When clamping an outlier:
tracing::trace!(original = value, clamped_to = clamped, "outlier clamped");

// When despiking:
tracing::trace!(timestamp = %ts, "despiked record");
```

Again `trace!` — per-record loops.

---

## Step 5 — `src/schedule.rs`: Schedule Checks

### 5a — `is_open()` (lines ~25–39)

```rust
tracing::debug!(
    is_holiday,
    is_weekend,
    hour = time.hour(),
    open_hour,
    close_hour,
    result = is_open,
    "schedule::is_open evaluated"
);
```

### 5b — `is_bavarian_holiday()` (lines ~90–134)

```rust
// On fixed holiday match:
tracing::trace!(month, day, "matched fixed Bavarian holiday");

// On Easter-relative holiday match:
tracing::trace!(date = %date, "matched Easter-relative holiday");
```

---

## Step 6 — `src/analytics.rs`: Analytics Decisions

These functions have `#[instrument]` but no internal branching logs.

### 6a — `determine_trend()` (lines ~486–510)

```rust
// On insufficient data:
tracing::debug!(sample_count, required = MIN_SAMPLES, "insufficient data for trend");

// On trend selected:
tracing::debug!(trend = ?selected_trend, avg_change, "trend determined");
```

### 6b — `generate_insights()` (lines ~703–884)

```rust
// When stats is None:
tracing::debug!("no stats available, skipping insight generation");

// On baseline comparison:
tracing::debug!(
    trend = ?trend,
    current_avg,
    baseline_avg,
    "generated baseline comparison insight"
);
```

### 6c — `find_quiet_windows()` (lines ~639–692)

```rust
// When a window opens:
tracing::trace!(hour, "quiet window opened");

// When a window closes:
tracing::trace!(hour, duration_hours, avg_occupancy, "quiet window closed and emitted");
```

---

## Step 7 — `src/api.rs` and `src/db.rs`

These are less critical but complete the picture.

### 7a — `fetch_occupancy()` (lines ~58–82)

```rust
// After successful HTTP response before status check:
tracing::debug!(status = %response.status(), "gym API response received");

// After successful JSON parse:
tracing::debug!(occupancy_pct = parsed_value, "gym API response parsed");
```

### 7b — `db::get_records_for_date()` (lines ~271–294)

```rust
tracing::debug!(
    date = %date,
    start_utc = %start,
    end_utc = %end,
    "date boundary computed for query"
);
```

### 7c — `db::export_to_csv()` (lines ~210–264)

```rust
// When streaming begins:
tracing::debug!(record_count, "starting CSV export stream");
```

---

## Summary Table

| File | New `debug!` calls | New `trace!` calls | Priority |
|---|---|---|---|
| `src/app.rs` | 6 | 0 | **High** — notification SM |
| `src/main.rs` | 2 | 1 | High — daemon loop |
| `src/repair.rs` | 6 | 4 | High — repair pipeline |
| `src/schedule.rs` | 1 | 2 | Medium |
| `src/analytics.rs` | 3 | 2 | Medium |
| `src/api.rs` | 2 | 0 | Low |
| `src/db.rs` | 2 | 0 | Low |
| **Total** | **22** | **9** | |

All `trace!` calls are in per-record loops; they will be noisier than `debug!` but only
visible when `RUST_LOG=hardy_monitor=trace` is set. Both are compiled out in release via
`release_max_level_info`.

---

## Implementation Order

1. `Cargo.toml` — add `release_max_level_info` (one-line change, no logic risk)
2. `src/app.rs` — notification state machine (highest diagnostic value)
3. `src/main.rs` — daemon loop (second-highest)
4. `src/repair.rs` — repair pipeline
5. `src/schedule.rs`, `src/analytics.rs`, `src/api.rs`, `src/db.rs` — remainder
