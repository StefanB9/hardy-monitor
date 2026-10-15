//! The forecaster: runs the forecasting suite against the shared database,
//! on a machine strong enough for it. It publishes every model's forecast
//! every quarter hour while the gym is open, logs the default forecast
//! every full hour, and refits and benchmarks every model each night.
//! Readings come from the daemon, which also migrates the database.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod forecasts;
mod logging;
mod progress;
mod status;

use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use hardy_core::{
    cli::Options,
    config::AppConfig,
    db::{Database, Migrations, SchemaStatus, minute_slot},
    schedule::GymSchedule,
};
use hardy_ml::{progress::Progress, service::ForecastService};
use indicatif::MultiProgress;

use crate::progress::Terminal;

/// How long to wait before checking the schema again while the daemon has
/// not migrated the database yet.
const SCHEMA_RETRY: Duration = Duration::from_secs(60);

fn main() -> Result<()> {
    let options = Options::parse("hardy-forecaster", std::env::args().skip(1))?;
    let terminal = Arc::new(Terminal::new(MultiProgress::new()));
    let _log_guard = logging::setup_logging(options.console, terminal.log_writer());
    let config = AppConfig::load().context("Failed to load configuration")?;
    let runtime = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
    let result = runtime.block_on(run(&config, &terminal));
    terminal.close();
    // A nightly run may still be computing on a blocking thread; it has
    // stored nothing yet and reruns at the next start.
    runtime.shutdown_background();
    if let Err(e) = &result {
        tracing::error!(error = %format!("{e:#}"), "forecaster stopped");
    }
    result
}

async fn run(config: &AppConfig, terminal: &Arc<Terminal>) -> Result<()> {
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    terminal.status("connecting to the database…".to_string());
    let database = Database::connect(&config.database, Migrations::Verify)
        .await
        .context("Failed to connect to the database")?;
    if !wait_for_schema(&database, terminal, shutdown.as_mut()).await? {
        return Ok(());
    }
    let Some(_lock) = database.lock_forecaster().await? else {
        bail!("another hardy-forecaster is already running against this database");
    };

    let schedule = GymSchedule::new(&config.schedule);
    let log_hours = u32::try_from(config.ml.prediction_horizon_hours)
        .context("ml.prediction_horizon_hours out of range")?;
    let progress: Arc<dyn Progress> = terminal.clone();
    let mut service = ForecastService::new(&config.ml, schedule.clone()).with_progress(progress);
    terminal.status("loading the stored models…".to_string());
    if let Err(e) = service.load(&database).await {
        tracing::warn!(error = %format!("{e:#}"), "could not read the stored forecast models");
    }
    tracing::info!("forecaster started");

    // Fresh forecasts at once rather than at the next quarter hour.
    terminal.status("publishing forecasts…".to_string());
    publish(
        &database,
        &mut service,
        &schedule,
        minute_slot(Utc::now()),
        log_hours,
        true,
    )
    .await;
    loop {
        // After the tick, so a run that just started shows at once.
        terminal.status(status::line(&service, &schedule, Utc::now()));
        tokio::select! {
            () = sleep_until_next_minute() => {}
            () = &mut shutdown => break,
        }
        let now = Utc::now();
        let was_running = service.is_running();
        if let Err(e) = service.tick(&database, now).await {
            tracing::warn!(error = %format!("{e:#}"), "forecast run failed");
        }
        if was_running && !service.is_running() {
            terminal.clear_run();
        }
        publish(
            &database,
            &mut service,
            &schedule,
            minute_slot(now),
            log_hours,
            false,
        )
        .await;
    }
    tracing::info!("forecaster stopped");
    database.close().await;
    Ok(())
}

/// Publishes forecasts at a quarter hour while the gym is open (or always
/// with `force`). Failures are logged only.
async fn publish(
    database: &Database,
    service: &mut ForecastService,
    schedule: &GymSchedule,
    slot: chrono::DateTime<Utc>,
    log_hours: u32,
    force: bool,
) {
    if !force && !forecasts::should_publish(schedule, slot) {
        return;
    }
    match forecasts::publish(database, service, slot, log_hours).await {
        Ok(logged) => tracing::debug!(%slot, logged, "published forecasts"),
        Err(e) => tracing::warn!(error = %format!("{e:#}"), "failed to publish forecasts"),
    }
}

/// Waits until the database schema matches this build; `false` if
/// shutdown came first. A newer schema is an error: update the forecaster.
async fn wait_for_schema(
    database: &Database,
    terminal: &Terminal,
    mut shutdown: std::pin::Pin<&mut impl std::future::Future<Output = ()>>,
) -> Result<bool> {
    loop {
        match database.schema_status().await? {
            SchemaStatus::Current => return Ok(true),
            SchemaStatus::DbNewer { db, app } => {
                bail!("the database schema ({db}) is newer than this forecaster ({app}); update it")
            }
            SchemaStatus::DbOlder { db, app } => {
                tracing::warn!(?db, app, "waiting for the daemon to migrate the database");
                terminal.status("waiting for the daemon to migrate the database…".to_string());
                tokio::select! {
                    () = tokio::time::sleep(SCHEMA_RETRY) => {}
                    () = &mut shutdown => return Ok(false),
                }
            }
        }
    }
}

async fn sleep_until_next_minute() {
    let into_minute = Utc::now().timestamp().rem_euclid(60);
    let wait = u64::try_from(60 - into_minute).unwrap_or(60);
    tokio::time::sleep(Duration::from_secs(wait)).await;
}

/// Completes on Ctrl-C or, on Unix, SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!(error = %e, "failed to listen for Ctrl-C");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to listen for SIGTERM");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => tracing::info!("received Ctrl-C"),
        () = terminate => tracing::info!("received SIGTERM"),
    }
}
