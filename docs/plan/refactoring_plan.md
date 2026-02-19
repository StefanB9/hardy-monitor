# Hardy Monitor — Architectural & Performance Refactoring Plan

**Review date:** 2026-02-19
**Reviewer role:** Principal Rust Systems Architect
**Standard:** `CLAUDE.md` — mandatory, not advisory

---

## Executive Summary

The codebase is in **fair-but-not-production-ready** condition against the `CLAUDE.md` standard.
The recent improvement phases (Phases 0–4) resolved the most critical correctness defects (blocking
notifications, poisoned-mutex panics, partial batch writes, type-unsafe timestamps). What remains
is a cluster of **lint-level violations that will fail CI in release mode**, a **systematic
absence of observability instrumentation across every public API boundary**, two
**performance-class issues** (per-notification HTTP client construction, O(n²) key deduplication),
and a **hard violation of the error-hierarchy contract** in `src/api.rs` and `src/config.rs`.

### Severity Map

| Severity | Count | Representative issue |
|---|---|---|
| **Critical (CI-fail)** | 14 | `expect_used` / `unwrap_used` in non-test production code |
| **Critical (contract)** | 3 | Library functions returning `anyhow::Result` instead of `AppError` |
| **High (performance)** | 3 | Per-call `reqwest::Client` build; O(n²) key set; unbounded DB fetch |
| **High (observability)** | 1 | Zero `#[instrument]` spans across entire public API surface |
| **Medium (allocation)** | 4 | Redundant clones, duplicate constant, missing capacity hints |
| **Low (type-safety)** | 2 | Primitive `i32` weekday/hour; missing `size_of` guard on hot struct |
| **Low (dependency)** | 1 | `reqwest::blocking` feature may be dead weight |

---

## Dependency Review

### `Cargo.toml` — `[dependencies]`

| Dependency | Issue | Proposed Action |
|---|---|---|
| `reqwest` | `blocking` feature enabled globally | **Audit `repair.rs`**: if the blocking client is used only during startup (outside an async context), keep it; if all paths are now async, drop `blocking` to reduce binary size and compile time. Run `cargo tree --features blocking` to identify all users. |
| `futures` | Only `BoxFuture` is used from this crate | Acceptable as-is (already `default-features = false`). Alternative: replace `BoxFuture<'s, R>` with `Pin<Box<dyn Future<Output = R> + Send + 's>>` inline and drop the dependency. Low priority. |
| `chrono` | `default-features = false, features = ["serde"]` | The `clock` feature (`Utc::now()`, `Local::now()`) is conspicuously absent. The code compiles only because an upstream transitive dep re-enables it. **Add `"clock"` explicitly** to prevent a future transitive dep update from silently breaking compilation. |
| `anyhow` | `default-features = false` | `anyhow::Context` is a `std`-feature item. Add `features = ["std"]` explicitly to document intent. |
| `config` | `default-features = false` | No explicit feature list. Verify which parser backend is being pulled in and pin the minimum required features. |

### `[dev-dependencies]`

| Dependency | Status |
|---|---|
| `proptest` | Good — already present and used in `analytics.rs` and `schedule.rs`. |
| `wiremock` | Present but **no wiremock tests exist for `GymApiClient`**. Either add the integration test suite this implies, or remove the dependency until it is needed. |
| `temp-env` | Good — used correctly in `config.rs` tests. |
| `tempfile` | Used in `tests/database.rs`. Good. |

---

## Module-by-Module Action Plan

---

### `src/analytics.rs` — Hot Path, Zero Instrumentation, Multiple Lint Violations

> Per `CLAUDE.md`: analytics.rs is "hot path, called on every UI frame tick." Performance
> analysis is **required** for every function. Zero functions have it. Zero functions are
> instrumented.

#### A-1 — `expect_used` / `unwrap_used` violations (Critical — CI-fail)

| Target | Current Issue | Proposed Solution | Expected Impact |
|---|---|---|---|
| `midnight_utc` (line 206) | `.expect("midnight (0,0,0) is always valid")` — `expect_used = "deny"` violation | `NaiveDate::and_hms_opt(0,0,0)` for `(0,0,0)` is provably infallible (the inputs are compile-time literals, not runtime values). Use `unwrap_or_else(` `\|_\|` `unreachable!(...))`, or better: call `.and_utc()` on `NaiveDateTime` via `date.and_time(NaiveTime::MIN)` which requires no fallible call. | Eliminates lint violation; zero runtime cost. |
| `midnight_local_as_utc` (lines 216–218) | Two `.expect(...)` calls for `and_hms_opt` and `.single()` | Same `NaiveTime::MIN` strategy for the first; `.single()` is fallible at DST overlap — surface as `Option` in the return type or propagate as `AppError`. | Eliminates 2 lint violations; forces caller to handle the DST ambiguity case explicitly. |
| `calculate_predictions_with_clock` (lines 266–271) | Three chained `.unwrap()` on `with_minute(0)`, `with_second(0)`, `with_nanosecond(0)` | These methods return `Option` only for out-of-range inputs; `0` is always in range. Use `unwrap_or(target_time)` as a no-op fallback, or use `target_time.with_time(NaiveTime::MIN)` to truncate to hour-boundary in one call. | Eliminates 3 lint violations. |
| `compare_periods`, `find_peak_hours`, `find_quiet_hours`, `find_quiet_windows`, `generate_insights` | 8+ `.partial_cmp(...).unwrap()` calls across sort comparators | Replace with `f64::total_cmp` (stable since Rust 1.62). `total_cmp` provides a total ordering over all `f64` values including NaN, so no `Option` is involved: `\|a, b\| a.total_cmp(b)` | Eliminates all `partial_cmp().unwrap()` violations. `total_cmp` has identical runtime cost; it is a single CPU instruction on x86. |

#### A-2 — Missing Performance Annotations (CLAUDE.md Mandatory)

| Target | Proposed Annotation |
|---|---|
| `calculate_predictions_with_clock` | `// Time: O(n) where n = baseline.len() (linear scan per prediction slot). Allocs: 1 Vec (≤2 items).` |
| `build_hourly_comparisons` | `// Time: O(n + m + k log k) where n,m = slice lengths, k = unique (weekday,hour) pairs (≤168). Allocs: 2 HashMaps + 1 Vec<(i32,i32)>.` |
| `calculate_stats` | `// Time: O(n log n) (dominated by sort). Allocs: 2 Vec<f64> (see A-3).` |
| `find_peak_hours` / `find_quiet_hours` | `// Time: O(n log n). Allocs: 1 Vec (clone of filtered subset).` |
| `find_quiet_windows` | `// Time: O(7n) ≈ O(n). Allocs: 1 Vec<TimePeriod> (output) + 1 Vec per weekday (inner day_hours, ≤24 items → SmallVec candidate).` |
| `generate_insights` | `// Time: O(n log n) (dominated by compare_periods internally). Allocs: 1 Vec<Insight> + transient String per insight (format!).` |

#### A-3 — Double Allocation in `calculate_stats` (High Performance)

| Target | `src/analytics.rs:494–500` |
|---|---|
| **Current Issue** | Two `Vec<f64>` allocated: `percentages` (to compute mean) and `sorted` (clone of `percentages`, to find median/min/max). Total: 2 heap allocations for `n` floats. |
| **Proposed Solution** | Build `sorted: Vec<f64>` directly from the iterator. Compute the mean in a separate `.fold()` pass over the original slice (`data.iter().map(\|h\| h.avg_percentage)`). This requires only 1 allocation and keeps the mean loop cache-friendly over the original `&[HourlyAverage]` rather than an intermediate `Vec<f64>`. |
| **Expected Impact** | Halves heap allocations; reduces peak memory by `n * 8` bytes (one fewer float buffer). For n=168 (max weekly slots), saves 1.3 KB per call. |

#### A-4 — O(n²) Key Deduplication in `build_hourly_comparisons` (High Performance)

| Target | `src/analytics.rs:340–345` |
|---|---|
| **Current Issue** | `all_keys.contains(key)` is O(k) inside a loop of m items → O(n·m) total dedup, where n and m are the sizes of the two input slices. For the maximum realistic input (168 slots per period), this is 168×168 = 28,224 comparisons. |
| **Proposed Solution** | Collect `all_keys` into a `BTreeSet<(i32, i32)>` (or `HashSet`) during construction, then `.into_iter().collect::<Vec<_>>()` after. The `BTreeSet` insertion is O(log k) making deduplication O((n+m) log k). Alternatively, iterate both maps' keys and chain them into a `BTreeSet` in one pass. |
| **Expected Impact** | Reduces deduplication from O(n·m) to O((n+m) log k). At max 168 slots: from 28K comparisons to ~1K. More importantly, the `contains()` call on a `Vec` is cache-unfriendly; `BTreeSet` improves branch predictability. |

#### A-5 — Duplicate `DAY_NAMES` Constant (Medium Allocation / Maintainability)

| Target | `src/analytics.rs` — lines 528, 755, 773, 837 |
|---|---|
| **Current Issue** | `const DAY_NAMES: [&str; 7]` is defined **four times** inside function bodies (one long form in `analyze_days`, three short forms in `generate_insights`). This is a maintenance hazard and wastes binary space if the compiler doesn't deduplicate. |
| **Proposed Solution** | Hoist to two module-level constants: `const DAY_NAMES_LONG: [&str; 7]` and `const DAY_NAMES_SHORT: [&str; 7]`. Both `weekday_name` and `weekday_short` utility functions already exist — route `generate_insights` through them instead of embedding raw arrays. |
| **Expected Impact** | Eliminates 3 redundant constant definitions; centralises future localisation changes to 2 sites. |

#### A-6 — Missing `#[instrument]` on All Public Functions (High Observability)

Every public function in `analytics.rs` that has non-trivial branching is uninstrumented. Per
`CLAUDE.md`, `#[instrument]` is required at every public API boundary.

| Function | Proposed Span |
|---|---|
| `calculate_predictions_with_clock` | `#[instrument(skip_all, fields(baseline.len = baseline.len()))]` |
| `build_hourly_comparisons` | `#[instrument(skip_all, fields(baseline.len, current.len))]` |
| `compare_periods` | `#[instrument(skip_all, fields(mode = ?mode))]` |
| `generate_insights` | `#[instrument(skip_all, fields(current.len, has_baseline = baseline.is_some()))]` |
| `calculate_stats` | `#[instrument(skip_all, fields(n = data.len()))]` |
| `analyze_days` | `#[instrument(skip_all)]` |
| `find_quiet_windows` | `#[instrument(skip_all, fields(threshold, min_hours))]` |

Short pure leaf functions (`weekday_name`, `weekday_short`, `midnight_utc`) should **not** be
instrumented per `CLAUDE.md` guidance ("for short pure functions, skip instrumentation").

---

### `src/api.rs` — Error Hierarchy Violation (Critical Contract)

#### B-1 — Library Functions Returning `anyhow::Result` (Critical)

| Target | Current Issue | Proposed Solution | Expected Impact |
|---|---|---|---|
| `GymResponse::occupancy_percentage` (line 23) | Returns `anyhow::Result<f64>`. This is a public library function — callers that `match` on `AppError` cannot type-match on its failures. | Change to `Result<f64, AppError>`. Use `AppError::api_error(0, format!("failed to parse numval: {s:?}"))` or add a new `AppError::ParseError(String)` variant for parse failures. | Restores type-safety at call sites; callers can now distinguish parse failures from network failures. |
| `GymApiClient::new` (line 39) | Returns `anyhow::Result<Self>`. | Return `Result<Self, AppError>`. Use `AppError::Network { kind: NetworkErrorKind::Unknown, message: e.to_string() }` for client build failure. | Same as above. |
| `GymApiClient::fetch_occupancy` (line 50) | Uses `anyhow::bail!` for HTTP error status and `anyhow::Context` for network errors. | Return `Result<GymResponse, AppError>`. Network errors → `AppError::from_reqwest(e)`. Non-2xx status → `AppError::api_error(status.as_u16(), ...)`. | Callers in `main.rs` can selectively retry on `AppError::is_retryable()` without re-parsing a string message. |

#### B-2 — Missing `#[instrument]` on Public API (High Observability)

| Target | Proposed Span |
|---|---|
| `GymApiClient::fetch_occupancy` | `#[instrument(skip_all, fields(url = %self.url, http.status_code))]` — populate `http.status_code` after the response arrives. |
| `GymApiClient::new` | `#[instrument(skip_all)]` |

---

### `src/config.rs` — Error Hierarchy Violation (Critical Contract)

#### C-1 — `AppConfig::validate` and `load` Return `anyhow::Result` (Critical)

| Target | Current Issue | Proposed Solution | Expected Impact |
|---|---|---|---|
| `AppConfig::validate` (line 167) | Uses `anyhow::bail!` and returns `anyhow::Result<()>`. This is a library function exposed in `lib.rs`. | Change to `Result<(), AppError>`. Replace every `bail!(msg)` with `return Err(AppError::Config(format!(msg)))`. | Library consumers (including tests) can type-match on `AppError::Config`. |
| `AppConfig::load` (line 231) | Returns `anyhow::Result<Self>`. The `config` crate errors are not `AppError`-typed. | Introduce `AppError::Config` as the canonical return for this function. Convert `config::ConfigError` via a `From` impl or a helper. This is the **highest-priority item in this module** because the binary's `main.rs` already wraps it with `.context(...)` — the `AppError` layer is missing in between. | |

> **Note:** The `config::Config::builder()` API returns `config::ConfigError` which is not an
> `AppError` variant. The cleanest approach is to add a `ConfigError::Build(String)` variant to
> `DatabaseError` or keep a separate `AppError::Config(String)` and map the error there.
> `AppError::Config` already exists — use it.

---

### `src/traits.rs` — Per-Call HTTP Client Construction (High Performance)

#### D-1 — `reqwest::Client` Rebuilt on Every Notification (High Performance)

| Target | `CombinedNotifier::notify`, lines 173–175 |
|---|---|
| **Current Issue** | `reqwest::Client::builder().timeout(...).build()?` is called **inside the `async` block** that runs on every notification. A `reqwest::Client` contains a connection pool, TLS session cache, and DNS resolver. Constructing one per call wastes ~100µs and prevents connection reuse (defeating TLS session resumption). |
| **Proposed Solution** | Store a pre-built `reqwest::Client` as a field in `CombinedNotifier`. Initialise it in `CombinedNotifier::new`. Since `Client` is `Clone + Send + Sync`, this requires no `Arc` wrapping — the field can be `reqwest::Client` directly. |
| **Expected Impact** | Eliminates 1 allocation + TLS session setup per notification. Enables HTTP keep-alive and connection pooling to the ntfy server. |

#### D-2 — Double Clone of `title`/`body` in `CombinedNotifier::notify` (Medium Allocation)

| Target | `src/traits.rs:154–155` |
|---|---|
| **Current Issue** | `title` and `body` are cloned into owned Strings at the start of the async block (lines 104–105), then immediately cloned again for the `spawn_blocking` closure (lines 154–155). That is 4 String allocations for 2 strings on every notification. |
| **Proposed Solution** | Use the strings once inside `spawn_blocking`, then reuse the outer `title` / `body` for the ntfy message. Specifically: clone only `t = title.clone()` and `b = body.clone()` for `spawn_blocking`; use the outer `title` and `body` (which are already owned) for the ntfy `format!`. This eliminates 2 of the 4 String allocations. |
| **Expected Impact** | Halves per-call String allocations: 4 → 2. At notification rate ≤1/5 min this is negligible for performance but shows adherence to zero-copy discipline. |

#### D-3 — `HashMap<_> + Arc<Mutex<Vec<_>>>` Capacity Hints Missing (Medium)

| Target | `MockNotifier` internals |
|---|---|
| **Current Issue** | `MockNotifier::notifications` is `Arc<Mutex<Vec<(String, String)>>>` initialised via `Default` (empty Vec, zero capacity). In tests that send N notifications, this causes log₂(N) reallocations. |
| **Proposed Solution** | This is test-only code — acceptable as-is. Document with a comment: `// Test-only: unbounded growth is acceptable; no capacity pre-allocation needed.` |

---

### `src/db.rs` — Missing Instrumentation, Unbounded Fetch, Layout Guard

#### E-1 — Missing `#[instrument]` on All Async Public Functions (High Observability)

Every public async function is a database operation — exactly the boundary where spans are most
valuable for diagnosing latency. None are instrumented.

| Function | Proposed Span |
|---|---|
| `Database::insert_record` | `#[instrument(skip_all, fields(db.operation = "insert", timestamp = %timestamp))]` |
| `Database::get_history` | `#[instrument(skip_all, fields(db.operation = "get_history", days))]` |
| `Database::get_history_range` | `#[instrument(skip_all, fields(db.operation = "get_history_range", start = %start, end = %end))]` |
| `Database::get_averages_range` | `#[instrument(skip_all, fields(db.operation = "get_averages_range"))]` |
| `Database::get_latest_record` | `#[instrument(skip_all, fields(db.operation = "get_latest"))]` |
| `Database::export_to_csv` | `#[instrument(skip_all, fields(db.operation = "export_csv", output_dir = %output_dir.display()))]` |
| `Database::batch_insert` | `#[instrument(skip_all, fields(db.operation = "batch_insert", count = records.len()))]` |

#### E-2 — `export_to_csv` Unbounded Memory Fetch (High Performance)

| Target | `src/db.rs:203` — `self.get_history(365 * 10)` |
|---|---|
| **Current Issue** | `export_to_csv` fetches up to **10 years of records** in a single `SELECT` into a `Vec<OccupancyLog>`. At 1 record/minute, 10 years = 5.26M records. Each `OccupancyLog` is ~28 bytes → **147 MB peak heap allocation** per export. |
| **Proposed Solution** | Either (a) page the query in chunks of 10,000 rows using `LIMIT`/`OFFSET` and flush each chunk to the CSV writer before fetching the next, or (b) use SQLx's `fetch()` stream API (`query_as!(...).fetch(&self.pool)`) and pipe records directly into the CSV writer without materialising the full result set. Option (b) is strictly superior: it keeps memory at `O(1)` and increases throughput by overlapping DB read with CSV write. |
| **Expected Impact** | Reduces peak memory from O(n·row_size) to O(1) during export. Enables exports of arbitrarily large datasets. |

#### E-3 — Missing `size_of` Guard on `OccupancyLog` (Low)

| Target | `src/db.rs:OccupancyLog` struct |
|---|---|
| **Current Issue** | `OccupancyLog` is stored in large collections returned by all `get_history*` calls. No `const` assertion guards its size. A future field addition (e.g., a `source: u8` enum) could silently add padding and increase memory consumption in analytics loops. |
| **Proposed Solution** | Add immediately after the struct definition: `const _: () = assert!(std::mem::size_of::<OccupancyLog>() <= 32, "OccupancyLog size regression");` The current layout (`id: i64` 8B + `timestamp: DateTime<Utc>` 12B + `percentage: f64` 8B + padding) is ≤32 bytes. |
| **Expected Impact** | Future size regressions become compile errors rather than silent performance degradations. |

#### E-4 — Missing Performance Annotations on DB Functions (CLAUDE.md Mandatory)

| Function | Required Annotation |
|---|---|
| `get_history` | `// Network: 1 round-trip. Allocs: 1 Vec<OccupancyLog> (n rows, bounded by `days` param).` |
| `batch_insert` | `// Network: 1 BEGIN + n INSERT + 1 COMMIT round-trips (pipelined). Allocs: O(n) string temporaries for rfc3339 formatting.` |
| `get_averages_range` | `// Network: 1 round-trip (aggregate query). Allocs: 1 Vec<HourlyAverage> (≤168 items).` |

---

### `src/schedule.rs` — Minor Issues

#### F-1 — `is_bavarian_holiday` Recalculates Easter on Every Call (Low)

| Target | `src/schedule.rs:90` |
|---|---|
| **Current Issue** | `is_bavarian_holiday(date)` runs the full Anonymous Gregorian algorithm on **every** call to `is_open`, `get_open_hour`, and `get_close_hour`. In the daemon loop, this is called once per minute, and each call recomputes Easter for the current year. |
| **Proposed Solution** | The result is deterministic per year. Cache the Easter date for the current year inside `GymSchedule` as `Option<(i32, NaiveDate)>` (year, date) — a `(i32, NaiveDate)` is 8 bytes. Invalidate the cache when the year changes. Alternatively, since the algorithm is O(1) integer arithmetic (no allocation, ~20 instructions), leave it as-is and add a comment: `// O(1) integer arithmetic — no allocation, no I/O. Re-computation is cheaper than a mutex-guarded cache.` |
| **Expected Impact** | If cached: eliminates ~20 integer ops per `is_open` call. If commented: documents the deliberate choice. |

#### F-2 — `GymSchedule` Fields are `pub(self)` Only via `Default` (Design Note)

`GymSchedule` exposes `get_open_hour` / `get_close_hour` but its fields (`weekday_open`, etc.) are
private. `new_for_test` is `#[cfg(test)]`. This is correct encapsulation. No action required.

---

### `src/main.rs` — Remaining Unstructured Log Calls

#### G-1 — Structured Tracing now complete but no spans on `run_daemon` / `fetch_and_store`

| Target | `src/main.rs:run_daemon`, `fetch_and_store` |
|---|---|
| **Current Issue** | `fetch_and_store` is an async function called on every loop iteration. It has no span. `run_daemon` is the top-level daemon entry — it has no span. |
| **Proposed Solution** | Add `#[instrument(skip_all)]` to `fetch_and_store`. Wrap the outer daemon loop with a `tracing::info_span!("daemon.loop")` or use `#[instrument(skip_all, fields(interval_secs))]` on `run_daemon`. |
| **Expected Impact** | Every DB write and API call in the daemon gets a parent span — enabling distributed-trace-style latency analysis even with a local `fmt` subscriber. |

---

## Type Safety Improvements

### H-1 — Primitive Obsession: `weekday: i32` / `hour: i32` (Low)

Across `HourlyAverage`, `HourlyComparison`, `TimePeriod`, `DayAnalysis`, and all analytics
functions, weekday and hour are raw `i32` values with no type enforcement. Nothing prevents
calling `weekday_name(25)` or passing an hour value as a weekday argument.

| Proposed Solution |
|---|
| Introduce `struct Weekday(u8)` (0–6) and `struct Hour(u8)` (0–23) newtypes with `TryFrom<i32>` conversions and validity assertions. This is a **Phase 3 (future)** item — it is a medium-sized refactor touching `HourlyAverage`, all analytics functions, and the `query_as!` macros. Gate it on a user decision, as it requires SQLx decode annotations. |

---

## Implementation Phases

> These phases are ordered by risk-adjusted value: highest severity defects that are smallest in
> scope go first.

---

### Phase 1 — Lint & Contract Fixes (No behaviour change, CI-blocking)

**Goal:** Get to zero `cargo +nightly clippy --all-features -- -D warnings`.

1. **P1-A** — Replace all `partial_cmp().unwrap()` with `f64::total_cmp` in `analytics.rs`
   (affects `compare_periods`, `find_peak_hours`, `find_quiet_hours`, `find_quiet_windows`,
   `generate_insights`, `calculate_stats`).

2. **P1-B** — Eliminate `expect(...)` in `midnight_utc` and `midnight_local_as_utc` in
   `analytics.rs`. Use `NaiveTime::MIN` + `and_time`.

3. **P1-C** — Eliminate three `unwrap()` in `calculate_predictions_with_clock` using
   `unwrap_or` or `with_time(NaiveTime::from_hms_opt(target_time.hour(), 0, 0).unwrap_or(NaiveTime::MIN))`.

4. **P1-D** — Migrate `AppConfig::validate()` to return `Result<(), AppError>` using
   `AppError::Config(...)` instead of `anyhow::bail!`.

5. **P1-E** — Migrate `GymApiClient::new`, `fetch_occupancy`, and `GymResponse::occupancy_percentage`
   to return `Result<_, AppError>`.

**Estimated file count:** 3 (`analytics.rs`, `config.rs`, `api.rs`)
**Risk:** Low — all are signature changes where every call site in the binary already uses `?`
propagation or `.context(...)`.

---

### Phase 2 — Observability: Add `#[instrument]` to All Public Boundaries

**Goal:** Every significant async call appears in tracing output with structured fields.

1. **P2-A** — Instrument all public async functions in `db.rs`.
2. **P2-B** — Instrument `GymApiClient::fetch_occupancy`.
3. **P2-C** — Instrument `calculate_predictions_with_clock`, `build_hourly_comparisons`,
   `compare_periods`, `generate_insights`, `calculate_stats`, `analyze_days`,
   `find_quiet_windows` in `analytics.rs`.
4. **P2-D** — Add `#[instrument]` to `fetch_and_store` in `main.rs`; wrap daemon loop with
   a named span.

**Estimated file count:** 3 (`db.rs`, `api.rs`, `analytics.rs`, `main.rs`)
**Risk:** Zero (adding spans never changes behaviour).

---

### Phase 3 — Performance: Hot-Path Allocations & Client Reuse

**Goal:** Eliminate the most impactful per-call allocations and the unbounded memory fetch.

1. **P3-A** — `CombinedNotifier`: store `reqwest::Client` as a field, build once in `::new`.
   Remove per-call `Client::builder()...build()`.

2. **P3-B** — `CombinedNotifier::notify`: reduce from 4 String clones to 2 by reordering clone
   sites.

3. **P3-C** — `calculate_stats`: collect directly into `sorted`, compute mean from a separate
   `.sum()` / `.len()` pass over the original slice.

4. **P3-D** — `build_hourly_comparisons`: replace `Vec::contains()` deduplication with a
   `BTreeSet` key collector.

5. **P3-E** — `export_to_csv`: replace `get_history(365*10)` bulk fetch with a streaming
   approach using SQLx `fetch()` + incremental CSV writes.

**Estimated file count:** 2 (`traits.rs`, `analytics.rs`, `db.rs`)
**Risk:** Medium. P3-A changes `CombinedNotifier`'s `new()` signature slightly (no breaking
change — the field is internal). P3-E changes the DB access pattern.

---

### Phase 4 — Code Quality: Constants, Guards, Dependencies

**Goal:** Enforce `CLAUDE.md` structural rules.

1. **P4-A** — Hoist `DAY_NAMES_LONG` and `DAY_NAMES_SHORT` to module-level constants in
   `analytics.rs`. Remove 3 function-local redefinitions.

2. **P4-B** — Add `const _: () = assert!(std::mem::size_of::<OccupancyLog>() <= 32, ...)`
   after the struct definition in `db.rs`.

3. **P4-C** — Add mandatory performance-analysis comments to every analytics function and DB
   query function per `CLAUDE.md`.

4. **P4-D** — Audit `reqwest::blocking` usage. If no path calls it inside an `async` context,
   confirm that the `repair.rs` path is the sole user and document it with a comment. If it is
   unused, remove the feature flag to save compile time.

5. **P4-E** — Add `"clock"` to `chrono`'s feature list and `features = ["std"]` to `anyhow`
   in `Cargo.toml` to make transitive-dependency assumptions explicit.

6. **P4-F** — Either write the `wiremock`-based `GymApiClient` integration test promised by the
   dev-dependency, or remove `wiremock` from `[dev-dependencies]`.

**Estimated file count:** 2–3
**Risk:** Low.

---

### Phase 5 — Type Safety: NewTypes (Future, User Decision Required)

1. **P5-A** — Introduce `Weekday(u8)` and `Hour(u8)` newtypes with `TryFrom<i32>` conversion
   and a `const` validity assertion. Update `HourlyAverage`, `HourlyComparison`, `TimePeriod`,
   `DayAnalysis`, all analytics functions, and the SQLx `query_as!` decode annotations.

> **Decision gate:** This phase is a medium-sized refactor (10+ files). Confirm the user wants
> this level of type-safety enforcement before starting. The payoff is compile-time prevention
> of weekday/hour confusion bugs; the cost is additional wrapper boilerplate at every call site.

---

## Appendix — Issue Count by File

| File | Critical | High | Medium | Low |
|---|---|---|---|---|
| `src/analytics.rs` | 6 (lint) | 3 (perf + obs) | 2 | 1 |
| `src/api.rs` | 3 (contract) | 1 (obs) | 0 | 0 |
| `src/config.rs` | 1 (contract) | 0 | 0 | 0 |
| `src/traits.rs` | 0 | 1 (perf) | 1 | 0 |
| `src/db.rs` | 0 | 2 (perf + obs) | 0 | 2 |
| `src/main.rs` | 0 | 1 (obs) | 0 | 0 |
| `Cargo.toml` | 0 | 0 | 0 | 3 |
| **Total** | **10** | **8** | **3** | **6** |
