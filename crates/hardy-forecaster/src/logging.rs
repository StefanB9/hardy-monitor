//! Log output: a daily-rotated file in release builds; the console (above
//! the progress bars) in debug builds or with `--console`.

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

use crate::progress::AboveBars;

/// Default filter when `RUST_LOG` is not set.
#[cfg(debug_assertions)]
const DEFAULT_FILTER: &str = "hardy_core=debug,hardy_ml=debug,hardy_forecaster=debug";
#[cfg(not(debug_assertions))]
const DEFAULT_FILTER: &str = "hardy_core=info,hardy_ml=info,hardy_forecaster=info";

/// Sets up logging; keep the returned guard alive so the file is flushed.
pub(crate) fn setup_logging(
    console: bool,
    terminal: AboveBars,
) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
            .parse_lossy(DEFAULT_FILTER)
    };
    let console = (console || cfg!(debug_assertions)).then(|| fmt::layer().with_writer(terminal));
    let (file, guard) = if cfg!(debug_assertions) {
        (None, None)
    } else {
        let appender = tracing_appender::rolling::daily("logs", "hardy-forecaster.log");
        let (writer, guard) = tracing_appender::non_blocking(appender);
        (
            Some(
                fmt::layer()
                    .with_writer(writer)
                    .with_ansi(false)
                    .with_target(false),
            ),
            Some(guard),
        )
    };
    tracing_subscriber::registry()
        .with(file)
        .with(console)
        .with(filter)
        .init();
    guard
}
