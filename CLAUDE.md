# Hardy Monitor Engineering Standards

Gym occupancy monitoring application built in Rust. Fetches real-time occupancy data, stores in PostgreSQL, analyzes historical patterns, predicts future occupancy via ML, and provides an iced desktop GUI with system tray. Correctness-critical data pipeline.

## Project Structure

Cargo workspace with 4 members:

```
hardy-monitor/                         (workspace root)
├── Cargo.toml                         [workspace] manifest — all dependency versions here
├── migrations/                        sqlx migrations (single table: occupancy_logs)
├── .sqlx/                             sqlx offline query cache
│
├── crates/
│   ├── hardy-core/                    (library — shared by daemon and GUI)
│   │   ├── src/
│   │   │   ├── lib.rs                 Module declarations + re-exports
│   │   │   ├── accuracy.rs            Summaries of logged forecast accuracy
│   │   │   ├── analytics.rs           OccupancyStats, insights, trend analysis, predictions
│   │   │   ├── api.rs                 GymApiClient (reqwest HTTP)
│   │   │   ├── config.rs              AppConfig, MlConfig, MlAlgorithm (TOML + env var)
│   │   │   ├── db.rs                  Database (sqlx PgPool). OccupancyLog, HourlyAverage
│   │   │   ├── error.rs               AppError, NetworkErrorKind, DatabaseError (thiserror)
│   │   │   ├── health.rs              HealthMonitor: outage / resumed messages for the daemon
│   │   │   ├── repair/                DataRepairer (gap filling, outliers, smoothing); nightly.rs: automatic nightly repair
│   │   │   ├── schedule.rs            GymSchedule, Bavarian holiday detection
│   │   │   └── traits.rs              Clock, Notifier, SystemClock, MockClock, MockNotifier
│   │   └── tests/                     Integration tests
│   │       ├── api.rs                 wiremock-based API client tests
│   │       ├── database.rs            PostgreSQL tests (TestDatabase isolation)
│   │       ├── app_logic.rs           MockClock/MockNotifier behavior tests
│   │       └── common/mod.rs          TestDatabase helper
│   │
│   ├── hardy-ml/                      (library — forecasting, shared by daemon and GUI)
│   │   └── src/                       features, training, models (linfa/smartcore), persistence
│   │
│   ├── hardy-daemon/                  (binary — headless fetch loop)
│   │   └── src/
│   │       ├── main.rs                Startup, fetch loop, fetch_and_store
│   │       ├── connect.rs             Database connect with retry, health and schema checks
│   │       ├── upkeep.rs              Nightly repair + training, schema check, forecast logging
│   │       ├── forecasts.rs           Hourly forecast log (accuracy tracking)
│   │       └── logging.rs             tracing setup (console / rotated file)
│   │
│   └── hardy-gui/                     (binary + library — iced desktop GUI)
│       ├── assets/icon.png
│       └── src/
│           ├── main.rs                Entry point, tray icon, iced runner
│           ├── lib.rs                 Module declarations
│           ├── app/                   HardyMonitorApp, Message (mod), update, view (shell), tasks (DB loads)
│           ├── style.rs               Design tokens: colours, occupancy scale, type/spacing scales
│           ├── forecasting.rs         Forecast state: history, loaded model, forecasts, quiet window
│           ├── quiet_window.rs        Next quiet hour (forecast, else slot averages)
│           ├── time_range.rs          ChartRange / AnalyticsRange and the instants they cover
│           ├── freshness.rs           Whether the newest reading is live or stale
│           ├── alerts.rs              Alert controls (shared settings, desktop popups)
│           ├── notifier.rs            SystemNotifier, CombinedNotifier
│           ├── widgets/               Canvas widgets: gauge, heatmap (WeekGrid), history_chart
│           └── views/                 now/, week, insights, model_data; components/ (cards, buttons, …)
```

**Dependency boundary:** core ← ml ← daemon / gui. Core has zero GUI or ML dependencies; `hardy-ml` depends only on core plus ML crates; both binaries depend on core and ml. GUI-only crates stay in `hardy-gui`. Never reverse.

## Planning Process

**Planning is interactive, not autonomous.** When asked to plan a step or feature:

1. Research the codebase and external dependencies. Present findings.
2. **Ask questions** before finalizing the plan — surface ambiguities, trade-offs, API changes, and design decisions that need the user's input.
3. Only finalize and save the plan after the user has reviewed and approved the approach.
4. Save the final plan to `docs/plan/` before implementation begins.

Do not silently make architectural decisions. If the implementation plan document conflicts with the current codebase state (e.g., outdated dependency versions, changed APIs), flag the discrepancy and ask how to proceed.

## Quick Reference

```bash
cargo nextest run --workspace                 # All workspace tests (use nextest, not cargo test)
cargo nextest run -p hardy-core               # Core-only tests
cargo clippy --workspace --all-targets        # Zero warnings required (includes tests)
cargo check --workspace --all-targets         # Type-check everything including tests
cargo fmt --all -- --check                    # Format check
cargo build -p hardy-daemon                   # Build daemon only
cargo build -p hardy-gui                      # Build GUI only
```

**Always use `cargo nextest run` instead of `cargo test`.** Nextest is the project's test runner.

**`--all-targets` is mandatory** for `clippy` and `check`. Lints must pass in tests and examples — not just lib/bin targets. Test code follows the same quality standards as production code.

## Test-Driven Development

**Strict red-green-refactor. No exceptions.**

1. **Red** — Write a failing test first. Run it. Confirm it fails for the right reason.
2. **Green** — Write the minimum code to make the test pass. Nothing more.
3. **Refactor** — Clean up while all tests stay green.

### Test Requirements

| Context | Requirement |
|---------|------------|
| Every public function | At least one unit test |
| Data pipeline / analytics logic | Property-based tests (`proptest`, 1000+ cases) |
| Async flows | Integration test with `#[tokio::test]` + `tokio::time::timeout` |
| API client changes | `wiremock`-based integration tests |
| Database queries | Integration test with `TestDatabase` isolation |
| `thiserror` enum variants | Test verifying each variant's Display output |

### Test Conventions

- **Naming:** `test_<unit>_<scenario>` (e.g., `test_database_rejects_duplicate_timestamp`)
- **Location:** `#[cfg(test)] mod tests` inline in the source file. Integration tests in `crates/hardy-core/tests/`.
- **Assertions:** `assert_eq!`, `assert!`, `prop_assert!`, `assert_relative_eq!` (approx)
- **Errors:** Tests return `anyhow::Result<()>` with `.context()` for diagnostics.
- **Quality:** Test code follows the same lint rules as production code. No `.unwrap()`, `.expect()`, or `panic!()` in tests — use `?` with `anyhow::Result` or `anyhow::bail!`.
- **Proptest config:** `#![proptest_config(ProptestConfig::with_cases(1000))]`
- **Database tests:** Use `TestDatabase` from `crates/hardy-core/tests/common/mod.rs` for isolated per-test databases.
- **Time mocking:** Use `MockClock` with `set_time()` / `advance()` instead of real time.
- **Notification mocking:** Use `MockNotifier` with `notification_count()`.

## Performance Rules

### Mandatory

- **Pre-allocate when size is known.** `String::with_capacity()`, `Vec::with_capacity()`.
- **`rust_decimal` or integer types for financial/percentage math where precision matters.** Use `f64` only where approximate values are acceptable (ML features, display percentages).
- **No allocation in tight loops.** Data processing and analytics paths should minimize allocations.

### Avoid

- `clone()` when a borrow suffices
- `Box<dyn Trait>` when monomorphic generics work
- `String` for fixed-vocabulary identifiers — use enums or newtypes
- `Arc<Mutex<T>>` when channels or `RwLock` suffice

## Error Handling

- **Public APIs:** Return `anyhow::Result<T>`. Add `.context("msg")` or `.with_context(|| format!(...))` (lazy) on every `?`.
- **Domain errors:** `thiserror` enums for recoverable, matchable errors (`AppError`, `DatabaseError`, `NetworkErrorKind`).
- **Retryable errors:** Use `AppError::is_retryable()` for network timeouts and pool exhaustion.
- **Forbidden everywhere (including tests):** `.unwrap()`, `.expect()`, `panic!()`, `todo!()` — enforced by lints with `--all-targets`.
- **Fallible constructors:** Return `Result<Self>` not `Self`. Validate inputs at construction.

## Coding Standards

### Lints (Cargo.toml, applied to all targets)

```
unsafe_code       = "forbid"
unwrap_used       = "deny"
expect_used       = "deny"
panic             = "deny"
todo              = "deny"
print_stdout      = "warn"
print_stderr      = "warn"
clippy::pedantic  = "warn" (priority -1)
```

These lints apply to **all code** — lib, bin, tests. Run `cargo clippy --all-targets` to verify.

### Formatting (`rustfmt.toml`)

- `max_width = 100`
- `imports_granularity = "Crate"`
- `group_imports = "StdExternalCrate"`
- `wrap_comments = true`
- `format_code_in_doc_comments = true`
- Edition 2024

### Import Order

```rust
use std::...;

use anyhow::...;       // external crates
use serde::...;

use crate::...;        // local
use super::...;
```

One blank line between groups.

### Visibility

Minimum necessary. `pub(super)` or `pub(crate)` for internal types. Private fields by default.

### Module Organization

- One responsibility per file.
- Split at ~300 lines. Convert to directory module (`mod.rs` + submodules) at ~500 lines.
- Directory modules: `mod.rs` is the coordinator (struct, trait impl, re-exports). Submodules own specific concerns.
- GUI-specific modules live in `hardy-gui`, core modules in `hardy-core`. No feature gates needed.

### Comments

- `///` doc comments on all public items.
- `// SAFETY:` for correctness reasoning (cancellation safety, invariant guarantees, why a conversion is sound).
- No trivial comments restating what code does. Comments explain *why*.

### Type Safety

- Domain types: `OccupancyLog`, `HourlyAverage`, `AppConfig`, `GymResponse`. No raw primitives in public APIs where a domain type exists.
- Trait abstractions: `Clock` and `Notifier` for testability. Always accept trait bounds, not concrete types, in functions that need time or notifications.

### Dependencies

**Rule 1: Always Disable Default Features.** Always explicitly set `default-features = false` when declaring dependencies. Only enable the specific features required.

**Rule 2: Minimum Features Enabled.** Strictly limit feature opt-ins to the absolute bare minimum required for the code to compile and run. Never use blanket features like `features = ["full"]`. This keeps compile times fast, binary sizes small, and the attack surface minimal.

**Rule 3: Deps live in the narrowest crate.** GUI-only dependencies belong in `hardy-gui/Cargo.toml`, ML-only ones in `hardy-ml/Cargo.toml`, never in `hardy-core`. All versions are defined in the workspace root `[workspace.dependencies]`.

**General:**
- New dependencies require justification: what problem, why this crate, what alternatives were considered.
- Prefer zero-cost abstractions over convenience crates.

## Database & sqlx

- **Migrations:** Use `cargo sqlx migrate add -r <name>` to create reversible migration files. Run from the project root.
- **Running migrations:** Only the daemon migrates: it connects with `Migrations::Apply`, the GUI with `Migrations::Verify` and waits (full-window notice) until `Database::schema_status()` is `Current`. A build older than the database refuses to run (`AppError::SchemaTooNew`). `Database::new()` (tests) applies migrations. `DATABASE_URL` is read from `.env`.
- **Offline cache:** After adding or changing queries, regenerate with `cargo sqlx prepare --workspace` from the project root. Commit the `.sqlx/` directory.
- **Compile-time checked queries:** Use `sqlx::query!` and `sqlx::query_as!` — never raw string queries without compile-time verification.
- **Never hand-create migration files.** Always use the `cargo sqlx migrate add` command so timestamps are generated correctly.
- **Test isolation:** Integration tests use `TestDatabase` (creates a unique `hardy_test_*` database per test, dropped on cleanup).

## Tracing & Observability

- **`#[instrument]`** on all public `async fn`. Use `skip(self)` or `skip_all` with explicit `fields(...)`.
- **Field syntax:** `%value` for Display, `?value` for Debug, bare primitives for Copy types.
- **Levels:** `error!` = failures requiring attention. `warn!` = degraded states. `info!` = lifecycle events. `debug!` = protocol detail.
- **Default filter:** `hardy_core=DEBUG,hardy_daemon=DEBUG` / `hardy_gui=DEBUG` (debug), `INFO` (release). Noisy crates (fontdb, wgpu, naga, iced) filtered out in GUI.
- **Override:** `RUST_LOG` env var via `EnvFilter::try_from_default_env()`.
- **Release logging:** Writes to `logs/hardy-monitor.log` with daily rotation via `tracing-appender`.

## Git Workflow

No CI, no hooks. All verification is the developer's responsibility before merging.

### Branch Structure

```
main          Stable release branch. Always clean: compiles, all tests pass, zero warnings.
  └── dev     Integration branch. Features merge here first via PR. Periodically merged to main.
       ├── feature/<name>   New functionality
       ├── fix/<name>       Bug fixes
       ├── refactor/<name>  Code restructuring
       ├── test/<name>      Test additions/improvements
       └── docs/<name>      Documentation changes
```

### Branch Lifecycle

1. **Create** a feature branch from `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/<name>
   ```

2. **Work** on the branch. Make atomic commits (each compiles, passes tests, zero warnings).

3. **Push** the branch and open a PR targeting `dev`:
   ```bash
   git push -u origin feature/<name>
   ```

4. **Verify** locally before merging (no CI — you must run these):
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets
   cargo nextest run --workspace
   ```

5. **Merge** the PR into `dev` via merge commit (no squash — preserve commit history).

6. **Delete** the feature branch after merge.

7. **Promote** `dev` to `main` when a stable milestone is reached:
   ```bash
   git checkout main
   git merge dev
   git push origin main
   ```

### Branch Rules

- **Never commit directly to `main` or `dev`.** All changes go through feature branches + PRs.
- **Never force-push** to `main` or `dev`.
- **No WIP commits** on `dev` or `main`. Feature branches may have WIP commits but clean them up before PR.
- **New commits over amends** — preserve history.
- **Delete merged branches** to keep the branch list clean.

### Commit Messages

```
<type>: <imperative summary, max 72 chars>

<optional body: explains WHY, not what. Motivation, trade-offs, context.>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

### Pull Requests

All PRs target `dev` unless it is a `dev` → `main` promotion.

#### PR Format

```
Title: <type>: <short description> (under 72 chars)

## Summary
- Bullet points of what changed and why

## Test Plan
- [ ] Tests written first (red-green-refactor)
- [ ] cargo fmt --all -- --check passes
- [ ] cargo nextest run passes
- [ ] cargo clippy --all-targets clean
- [ ] cargo check --all-targets clean
```

#### Pre-Merge Checklist

Run locally before every merge — there is no CI to catch failures:

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets` — zero warnings
- [ ] `cargo nextest run --workspace` — all tests pass
- [ ] `cargo check --workspace --all-targets` — all targets type-check
- [ ] Tests written first (TDD evidence in commit history)
- [ ] No `.unwrap()`, `.expect()`, `panic!()`, `todo!()`
- [ ] Error paths have `.context()`
- [ ] New public items have doc comments
- [ ] Minimum visibility applied
