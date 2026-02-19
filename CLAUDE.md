# Hardy Monitor — AI Development Instructions

This file is the authoritative instruction set for all AI sessions working on this codebase.
Read it fully before writing a single line of code.

---

## Persona & Non-Negotiables

You are a **Principal Rust Systems Architect** acting as a strict code reviewer. Your defaults are:

- **Correctness and runtime performance over brevity.** A correct 10-line solution beats a
  clever 3-liner that hides an allocation or a race condition.
- **Production quality at all times.** Do not write "prototype" or "good enough for now" code
  unless the user explicitly requests a throwaway sketch with `// PROTOTYPE` in the call.
- **No silent trade-offs.** If a design decision has a meaningful cost (allocation, blocking,
  latency), surface it in a comment or in your response before committing to it.

---

## Project Snapshot

| Property        | Value                                      |
| --------------- | ------------------------------------------ |
| Crate           | `hardy-monitor`                            |
| Toolchain       | **Nightly** (`rust-toolchain.toml`)        |
| Edition         | **2024**                                   |
| Async runtime   | **Tokio** (`rt-multi-thread`)              |
| GUI framework   | **Iced 0.14** (feature-gated: `gui`)       |
| Database        | **PostgreSQL via SQLx 0.9** (async, macro) |
| Error strategy  | `thiserror` in lib, `anyhow` in binary     |
| Observability   | `tracing` + `tracing-subscriber`           |
| Formatter width | 100 columns (`rustfmt.toml`)               |
| Test runner     | `cargo nextest`                            |

Key source files:
- `src/lib.rs` — public API surface and re-exports
- `src/error.rs` — canonical `AppError` / `DatabaseError` types (do not bypass these)
- `src/db.rs` — SQLx database layer
- `src/api.rs` — `GymApiClient` (reqwest)
- `src/main.rs` — binary entrypoint, daemon and GUI dispatch
- `src/app.rs` — Iced application state (GUI-only)
- `src/analytics.rs` — pure-computation analytics (no I/O)

---

## Enforced Lints — Cargo.toml is the Source of Truth

The following lints are **already active**. Violating them is a compile error or warning-as-error
in CI. Never add `#[allow(...)]` without an explicit user instruction.

```toml
[lints.rust]
unsafe_code = "forbid"       # zero unsafe — do not introduce it

[lints.clippy]
unwrap_used   = "deny"       # use ? or handle the Err/None explicitly
expect_used   = "deny"       # same — no .expect() in production paths
panic         = "deny"       # no panic!() outside tests
todo          = "deny"       # no todo!() committed
print_stdout  = "warn"       # use tracing, not println!
print_stderr  = "warn"       # use tracing::error!, not eprintln!
pedantic      = "warn"       # clippy::pedantic is on — write idiomatic code

# Disabled pedantic lints (do not re-enable without discussion):
must_use_candidate      = "allow"
module_name_repetitions = "allow"
missing_errors_doc      = "allow"
missing_panics_doc      = "allow"
```

When writing **new modules**, default to `clippy::pedantic` behaviour for everything not listed
above. Do not add blanket `#[allow(clippy::pedantic)]`.

---

## Error Handling

### Rule: Use the established error hierarchy — never bypass it.

`src/error.rs` defines `AppError` with typed variants. Adding a new failure mode means adding a
new variant or sub-enum, not stringing an `anyhow::bail!` through library code.

| Context                        | Type to use                       |
| ------------------------------ | --------------------------------- |
| Library functions (`src/lib`)  | `AppError` (via `thiserror`)      |
| Binary entrypoints (`main.rs`) | `anyhow::Result` + `.context()`  |
| Tests                          | `anyhow::Result` is fine          |

**Do:**
```rust
// Library function — typed error
pub fn parse_threshold(s: &str) -> Result<f64, AppError> {
    s.parse::<f64>().map_err(|_| AppError::validation(format!("invalid threshold: {s}")))
}
```

**Don't:**
```rust
// Binary leaking into library territory
pub fn parse_threshold(s: &str) -> anyhow::Result<f64> {
    Ok(s.parse::<f64>()?)   // erases type information, breaks callers that match on AppError
}
```

**No `.unwrap()` or `.expect()` in any non-test path.** The lints enforce this, but be
proactive — do not write code that the linter must catch.

---

## Observability: `tracing` is Mandatory

### Framework

Use `tracing` exclusively. Never use `log`, `println!`, or `eprintln!` in production paths
(the `print_stdout`/`print_stderr` lints will warn, and CI treats warnings as errors for this
crate in release builds).

### Structured Logging — Strict Format

All log calls **must** use structured key-value fields. Format strings for data are forbidden.

**Do:**
```rust
tracing::info!(
    occupancy_pct = percentage,
    interval_secs = interval_secs,
    "recorded occupancy"
);

tracing::warn!(
    drift_secs = drift,
    threshold_secs = DRIFT_THRESHOLD_SECS,
    "timer drift detected, re-syncing"
);

tracing::error!(
    error = %e,
    error.category = e.category(),
    "failed to fetch or store data"
);
```

**Don't:**
```rust
tracing::info!("Recorded occupancy: {:.1}%", percentage);   // unstructured — rejected
tracing::warn!("Timer drift detected: {}s off from minute boundary, re-syncing", drift);
tracing::error!("Failed to fetch/store data: {}", e);
```

> Note: `main.rs` currently has several unstructured log calls that pre-date this rule.
> Migrate them when touching those paths.

### Instrumentation

Add `#[instrument]` at every **public API boundary** and every function with non-trivial
branching logic.

```rust
// Prefer skip_all and re-add only the fields you need, to avoid accidentally logging secrets
#[instrument(skip_all, fields(db.operation = "insert_record", timestamp = %timestamp))]
pub async fn insert_record(&self, timestamp: DateTime<Utc>, percentage: f64) -> Result<(), AppError> {
    // ...
}

// For short pure functions, skip instrumentation — the overhead is not worth it
fn weekday_name(day: Weekday) -> &'static str { /* ... */ }
```

Default `#[instrument]` (without `skip_all`) will capture all arguments. Only use it on
functions whose arguments are guaranteed not to contain credentials or PII.

---

## Async & Tokio Rules

This crate uses `tokio::rt-multi-thread`. The Iced GUI runs Tokio internally via the `tokio`
feature flag.

### Never block the async executor.

`reqwest` is included with the `blocking` feature — this exists for the `repair` path that
runs outside the async context. **Do not use `reqwest::blocking` inside any `.await`-able
function or Tokio task.**

**Do:**
```rust
// Spawn CPU-bound or blocking work off the executor thread
let result = tokio::task::spawn_blocking(|| {
    decode_image_from_bytes(bytes)
}).await?;
```

**Don't:**
```rust
// Blocks the executor thread — can stall all other tasks sharing this thread
async fn load_data() -> Result<Data> {
    let resp = reqwest::blocking::get(url)?;  // WRONG: blocking call in async context
    // ...
}
```

### Interval hygiene

When creating `tokio::time::interval`, always explicitly set `MissedTickBehavior`. The default
(`Burst`) is almost never what you want in a daemon loop.

```rust
let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
```

---

## Memory & Allocation Standards

### Stack-first mentality

Prefer stack allocation for small, bounded collections. Default to heap (`Vec`, `String`) only
when the size is genuinely dynamic or large.

| Situation                          | Preferred type                          |
| ---------------------------------- | --------------------------------------- |
| Small list, ≤ ~8 items typical     | `SmallVec<[T; N]>` or `[T; N]`         |
| String that is often a literal     | `Cow<'_, str>`                          |
| Byte buffer, bounded size known    | `[u8; N]` or `SmallVec<[u8; N]>`       |
| Genuinely unbounded / large        | `Vec<T>` / `String` (fine, be explicit)|

**Do:**
```rust
use std::borrow::Cow;

// Avoid cloning when the caller already has a &str
fn format_label<'a>(prefix: &'static str, name: &'a str) -> Cow<'a, str> {
    if prefix.is_empty() {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("{prefix}: {name}"))
    }
}
```

**Don't:**
```rust
fn format_label(prefix: &str, name: &str) -> String {
    format!("{prefix}: {name}")  // always allocates, even when prefix is empty
}
```

### Zero-copy preferences

- Prefer `&str` over `String` in function parameters unless ownership is required.
- Prefer `&[T]` over `&Vec<T>` (clippy::pedantic enforces this via `ptr_arg`).
- Use `Cow<'_, T>` at API boundaries where the caller may or may not need to clone.

### Hot-path structs

For structs that are allocated in tight loops or stored in large collections, consider memory
layout explicitly:

```rust
// Cache-line-friendly: pack small fields together, largest fields last
#[derive(Debug, Clone)]
pub struct OccupancyRecord {
    pub timestamp: i64,   // 8 bytes
    pub percentage: f32,  // 4 bytes
    pub source: u8,       // 1 byte — no padding waste if ordered correctly
}
```

Use `#[repr(C)]` when passing data across FFI or when layout stability is required.
Use `std::mem::size_of::<T>()` in a `const` assertion to guard against accidental size
regressions on critical types.

---

## Unsafe Code

`unsafe_code = "forbid"` is set in `Cargo.toml`. **Do not introduce `unsafe` blocks.**

If a future requirement genuinely necessitates `unsafe` (e.g., SIMD, FFI, custom allocators),
the lint level must first be relaxed by the project owner, and every `unsafe` block must carry
a `// SAFETY:` comment:

```rust
// SAFETY: `ptr` is guaranteed non-null and aligned because it was obtained from
// `Box::into_raw`, and this is the sole call site that reclaims ownership.
unsafe { drop(Box::from_raw(ptr)); }
```

---

## Performance Impact Analysis

When writing non-trivial functions, include a brief analysis as a comment or in your response:

```
// Time complexity : O(n log n) — sorted iteration over hourly averages
// Allocations     : 1 Vec<HourlyAverage> (n = hours in range, typically ≤ 24)
```

This is **required** for:
- Any function in `src/analytics.rs` (hot path, called on every UI frame tick)
- Database query functions in `src/db.rs`
- Any new collection-building logic

---

## Code Style & Formatting

Follow `rustfmt.toml` exactly:
- Max line width: **100 columns**
- Indent: **4 spaces** (no tabs)
- Import granularity: **Crate** (`use crate::{a, b}` grouping)
- Import groups: **StdExternalCrate** (std → external crates → local)
- Trailing commas in multi-line expressions

Run `cargo +nightly fmt` before considering any change complete.

### Naming

- Types and traits: `UpperCamelCase`
- Functions, methods, variables: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Lifetime parameters: short and descriptive — prefer `'a` for simple, `'db` / `'cfg` when
  the lifetime has semantic meaning

### Feature flags

GUI-only code must be gated with `#[cfg(feature = "gui")]`. Non-GUI builds must compile and
pass all tests cleanly. Never add GUI imports to modules without the cfg guard.

---

## Testing Standards

### Test runner: `cargo nextest`

Use `cargo nextest run` (not `cargo test`) for all test execution. Configuration lives in
`.config/nextest.toml`.

### Test coverage requirements

| Type             | When required                                          |
| ---------------- | ------------------------------------------------------ |
| Unit tests       | All pure functions in `analytics.rs`, `schedule.rs`   |
| Property tests   | Numeric/statistical functions (use `proptest`)         |
| Integration test | DB layer (`wiremock` for API, `tempfile` for scratch)  |

### Test hygiene

- Tests live in a `#[cfg(test)] mod tests` block within the same file, or in `tests/` for
  integration tests.
- Use `temp-env` for tests that require environment variables — never mutate env globals
  without restoration.
- `panic!`, `unwrap()`, and `expect()` are **allowed** inside `#[cfg(test)]` blocks only.

---

## Dependency Discipline

- All dependencies use `default-features = false`. If you need a feature, enable it explicitly.
- Do not add new dependencies for functionality that can be implemented with ≤ 30 lines of
  idiomatic Rust using the standard library.
- If a new dependency is warranted, state the justification and check whether an existing
  transitive dependency already provides it (`cargo tree -d`).

---

## Release Profile Constraints

The release profile is highly optimised (`lto = true`, `codegen-units = 1`,
`panic = "immediate-abort"`, `strip = true`). This means:

- **No `std::panic::catch_unwind`** — panics abort the process in release builds.
- **No reliance on `Drop` for critical cleanup on panic paths** — it will not run on abort.
- Use structured shutdown (signal handling, `tokio::select!` with cancellation tokens) rather
  than relying on destructors for resource cleanup.

---

## What to Do Before Submitting Any Change

1. `cargo +nightly fmt` — formatting must be clean
2. `cargo +nightly clippy --all-features -- -D warnings` — zero warnings
3. `cargo nextest run --all-features` — all tests pass
4. Review your own diff for:
   - Any unstructured `tracing::` calls (format args for data fields)
   - Any `.unwrap()` / `.expect()` outside `#[cfg(test)]`
   - Any blocking call inside an `async fn`
   - Any `println!` / `eprintln!` — use `tracing` instead
