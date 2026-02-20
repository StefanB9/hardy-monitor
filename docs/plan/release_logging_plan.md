# Release Logging Plan

## Goal

In **release builds**, replace stdout logging with a non-blocking rolling file appender so
that I/O never blocks the async executor or the GUI render thread. In **debug builds**,
keep the existing console output unchanged.

---

## Step 1 — Add `tracing-appender` to `Cargo.toml`

`tracing-appender` is not currently in the dependency tree.

```toml
tracing-appender = { version = "0.2", default-features = false }
```

`default-features = false` drops the optional `parking_lot` dependency; the standard
`std::sync::Mutex` used as fallback is fine for a background I/O thread.

---

## Step 2 — Architecture: `setup_logging()` in `main.rs`

Extract all logging initialisation into a single `setup_logging` function. It returns
`Option<tracing_appender::non_blocking::WorkerGuard>`:

- **Debug builds** (`cfg(debug_assertions)`): initialise console subscriber, return `None`.
- **Release builds** (`cfg(not(debug_assertions))`): initialise file subscriber, return
  `Some(guard)`.

The guard **must** be bound to a named variable in `main()` and held until program exit.
Dropping it early flushes and closes the background writer — any buffered log lines still
in flight at the time of drop would be lost.

```rust
// main.rs
fn main() -> Result<()> {
    let args = Args::parse();
    let _log_guard = setup_logging(&args); // held alive for the entire process lifetime
    // ...
}
```

The leading `_` suppresses the unused-variable warning while keeping the value alive
(a plain `_` binding would drop it immediately).

---

## Step 3 — `setup_logging` implementation

### Signature

```rust
fn setup_logging(args: &Args) -> Option<tracing_appender::non_blocking::WorkerGuard>
```

### Debug build branch — unchanged console behaviour

```rust
#[cfg(debug_assertions)]
{
    // Identical to current setup: honour RUST_LOG if set, else hardcoded defaults.
    #[cfg(feature = "gui")]
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else if args.daemon {
        EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .parse_lossy("hardy_monitor=debug")
    } else {
        EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .parse_lossy("hardy_monitor=debug,fontdb=error,wgpu=warn,naga=warn")
    };

    #[cfg(not(feature = "gui"))]
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .parse_lossy("hardy_monitor=debug")
    };

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    return None;
}
```

### Release build branch — non-blocking file appender

```rust
#[cfg(not(debug_assertions))]
{
    // Daily rolling file: logs/hardy-monitor.YYYY-MM-DD
    let file_appender = tracing_appender::rolling::daily("logs", "hardy-monitor.log");

    // Non-blocking channel — file I/O is moved to a background thread.
    // The returned WorkerGuard must be kept alive; dropping it shuts the thread down.
    let (non_blocking_writer, guard) = tracing_appender::non_blocking(file_appender);

    // In release, RUST_LOG is still honoured for production debugging.
    // Default: INFO for everything (debug! and trace! are already compiled out
    // by the release_max_level_info feature flag from the logging plan).
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .parse_lossy("hardy_monitor=info")
    };

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_writer(non_blocking_writer)
                .with_ansi(false)    // no terminal escape codes in log files
                .with_target(false), // module path adds noise without value in files
        )
        .with(filter)
        .init();

    return Some(guard);
}
```

---

## Step 4 — `main()` call site

```rust
fn main() -> Result<()> {
    let args = Args::parse();
    let _log_guard = setup_logging(&args);
    // ... rest of main unchanged
}
```

The existing inline logging initialisation block (lines ~64–84) is replaced entirely by
this call.

---

## Step 5 — Log directory

The appender writes to `logs/` relative to the **working directory** at process start.

- **Daemon mode**: typically launched from the project root or a service directory —
  `./logs/` is predictable and easy to find.
- **GUI mode**: launched from wherever the user double-clicks the binary. `./logs/` may
  end up in an unexpected location.

If the GUI release log location matters, the `logs/` path can be replaced with a
platform-appropriate directory:

```rust
// Alternative: write to %LOCALAPPDATA%\hardy-monitor\logs\ (Windows)
//              or ~/.local/share/hardy-monitor/logs/ (Linux/macOS)
let log_dir = dirs::data_local_dir()
    .unwrap_or_else(|| std::path::PathBuf::from("."))
    .join("hardy-monitor")
    .join("logs");
```

This is optional and can be deferred. `"logs"` is sufficient for now.

---

## Step 6 — Interaction with `release_max_level_info`

The `release_max_level_info` feature (planned in `logging_plan.md`) compiles out all
`debug!` and `trace!` calls in release. This means the file appender will only ever
receive `info!`, `warn!`, and `error!` events in release — the filter and the appender
have no work to do for the stripped levels. The two features are complementary.

---

## Step 7 — `tracing_appender` imports

Add to the import block in `main.rs`:

```rust
#[cfg(not(debug_assertions))]
use tracing_appender;
```

Or import the specific items used:

```rust
#[cfg(not(debug_assertions))]
use tracing_appender::non_blocking::WorkerGuard;
```

---

## Summary of Changes

| File | Change |
|---|---|
| `Cargo.toml` | Add `tracing-appender = { version = "0.2", default-features = false }` |
| `src/main.rs` | Replace inline logging init with `setup_logging(&args)` call + implement function |

No other files need to change. The rest of the codebase is unaffected — it only calls
`tracing::info!` etc. and is agnostic to which subscriber backend is active.

---

## What You Get

| Context | Output | Format | Level floor |
|---|---|---|---|
| `cargo run` (debug) | stdout | ANSI, with target | DEBUG (or `RUST_LOG`) |
| `cargo run --release` | `logs/hardy-monitor.YYYY-MM-DD` | plain text, no target | INFO (or `RUST_LOG`) |
| Release + `RUST_LOG=hardy_monitor=warn` | file | plain text | WARN |
