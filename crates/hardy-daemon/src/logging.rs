//! Log output: console in debug builds, a daily-rotated file in release
//! (plus the console with `--console`).

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Debug builds always log to the console; `--console` changes nothing.
#[cfg(debug_assertions)]
pub(crate) fn setup_logging(_console: bool) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
            .parse_lossy("hardy_core=debug,hardy_daemon=debug")
    };

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    None
}

/// Release builds log to a daily-rotated file and, with `--console`, to
/// the console as well.
#[cfg(not(debug_assertions))]
#[allow(clippy::unnecessary_wraps)] // the debug build's signature: it has no guard
pub(crate) fn setup_logging(console: bool) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let file_appender = tracing_appender::rolling::daily("logs", "hardy-monitor.log");
    let (non_blocking_writer, guard) = tracing_appender::non_blocking(file_appender);

    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
            .parse_lossy("hardy_core=info,hardy_daemon=info")
    };

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_writer(non_blocking_writer)
                .with_ansi(false)
                .with_target(false),
        )
        .with(console.then(fmt::layer))
        .with(filter)
        .init();

    Some(guard)
}
