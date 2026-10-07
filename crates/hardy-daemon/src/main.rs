#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod connect;

use std::time::Duration;

use anyhow::{Context, Result};
use hardy_core::{
    AppError, RetryPolicy,
    alert::AlertService,
    api::GymApiClient,
    config::AppConfig,
    db::{Database, SchemaStatus, minute_slot},
    health::{HealthEvent, HealthMonitor},
    retry,
    schedule::GymSchedule,
};
use hardy_ml::maintenance::ModelMaintenance;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[cfg(debug_assertions)]
fn setup_logging() -> Option<tracing_appender::non_blocking::WorkerGuard> {
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

#[cfg(not(debug_assertions))]
fn setup_logging() -> Option<tracing_appender::non_blocking::WorkerGuard> {
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
        .with(filter)
        .init();

    Some(guard)
}

const DRIFT_THRESHOLD_SECS: i64 = 5;
/// Fetch cycles between schema version checks (about ten minutes).
const SCHEMA_CHECK_ITERATIONS: u64 = 10;
const ALIGNMENT_CHECK_ITERATIONS: u64 = 60;

/// Per-tick retries: three attempts, waiting 2 s then 4 s.
const FETCH_ATTEMPTS: u32 = 3;
const FETCH_RETRY_INITIAL: Duration = Duration::from_secs(2);
const FETCH_RETRY_MAX: Duration = Duration::from_secs(4);

/// Startup connection retries continue until shutdown, backing off to 60 s.
const CONNECT_RETRY_INITIAL: Duration = Duration::from_secs(1);
const CONNECT_RETRY_MAX: Duration = Duration::from_secs(60);

/// Head-room left in each fetch interval so a slow cycle never overlaps the
/// next tick.
const CYCLE_HEADROOM: Duration = Duration::from_secs(5);

/// How long an in-flight fetch may continue after a shutdown request; stays
/// below Docker's default 10 s stop timeout.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

fn main() -> Result<()> {
    let _log_guard = setup_logging();

    let config = AppConfig::load().context("Failed to load configuration")?;

    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;

    rt.block_on(run(&config))
}

async fn run(config: &AppConfig) -> Result<()> {
    tracing::info!("Starting Hardy Monitor in daemon mode");

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    let schedule = GymSchedule::new(&config.schedule);
    let alerts = AlertService::new(
        &config.notifications,
        &config.network,
        schedule.clone(),
        chrono::Utc::now(),
    )?;
    let mut health = HealthMonitor::new(config.notifications.health_after_minutes);

    let connect_policy = RetryPolicy::new(u32::MAX, CONNECT_RETRY_INITIAL, CONNECT_RETRY_MAX)?;
    tracing::info!("Connecting to database...");
    let Some(database) = connect::connect_database(
        &config.database,
        &connect_policy,
        &schedule,
        &alerts,
        &mut health,
        shutdown.as_mut(),
    )
    .await
    .context("Failed to connect to database")?
    else {
        tracing::info!("shutdown requested before the database connected");
        return Ok(());
    };
    tracing::info!("Database connected successfully");

    let api_client = GymApiClient::new(config.gym.api_url.clone(), &config.network)?;
    tracing::info!(
        timezone = %schedule.timezone(),
        weekday_open = config.schedule.weekday.open_hour,
        weekday_close = config.schedule.weekday.close_hour,
        weekend_open = config.schedule.weekend.open_hour,
        weekend_close = config.schedule.weekend.close_hour,
        "schedule configured"
    );

    let fetch_policy = RetryPolicy::new(FETCH_ATTEMPTS, FETCH_RETRY_INITIAL, FETCH_RETRY_MAX)?;
    let worker = Worker {
        api_client: &api_client,
        database: &database,
        schedule: &schedule,
        fetch_policy,
        cycle_budget: Duration::from_secs(config.refresh.data_fetch_interval_secs)
            .saturating_sub(CYCLE_HEADROOM)
            .max(Duration::from_secs(1)),
    };

    let mut alerts = alerts;
    tracing::info!(
        alert_topic = config.notifications.ntfy_topic.is_some(),
        control_topic = config.notifications.control_topic.is_some(),
        "alerts configured"
    );

    let mut models = ModelMaintenance::new(config.ml.clone(), schedule.clone());
    if let Err(e) = models.load_previous(&database).await {
        tracing::warn!(error = %e, "could not read the stored model");
    }

    fetch_loop(
        &worker,
        &mut alerts,
        &mut health,
        &mut models,
        config.refresh.data_fetch_interval_secs,
        shutdown.as_mut(),
    )
    .await;

    tracing::info!("closing database pool");
    database.close().await;
    tracing::info!("daemon stopped");
    Ok(())
}

/// Everything a fetch cycle needs, borrowed for the daemon's lifetime.
struct Worker<'a> {
    api_client: &'a GymApiClient,
    database: &'a Database,
    schedule: &'a GymSchedule,
    fetch_policy: RetryPolicy,
    cycle_budget: Duration,
}

/// Runs fetch cycles aligned to full minutes until `shutdown` completes.
async fn fetch_loop(
    worker: &Worker<'_>,
    alerts: &mut AlertService,
    health: &mut HealthMonitor,
    models: &mut ModelMaintenance,
    interval_secs: u64,
    mut shutdown: std::pin::Pin<&mut impl std::future::Future<Output = ()>>,
) {
    tokio::select! {
        () = wait_for_minute_alignment() => {}
        () = &mut shutdown => {
            tracing::info!("shutdown requested during startup alignment");
            return;
        }
    }
    tracing::info!(interval_secs, "starting fetch loop");

    let mut interval = new_interval(interval_secs);
    let mut iteration_count: u64 = 0;
    let mut schema_reported = false;

    loop {
        tokio::select! {
            _ = interval.tick() => {}
            () = &mut shutdown => {
                tracing::info!("shutdown requested");
                return;
            }
        }
        iteration_count += 1;

        if iteration_count.is_multiple_of(ALIGNMENT_CHECK_ITERATIONS) && realign_if_drifted().await
        {
            interval = new_interval(interval_secs);
            interval.tick().await;
        }

        if iteration_count.is_multiple_of(SCHEMA_CHECK_ITERATIONS) {
            check_schema(worker.database, alerts, &mut schema_reported).await;
        }

        // The slot is fixed when the tick fires, so retries store the reading
        // in the minute it belongs to.
        let slot = minute_slot(chrono::Utc::now());

        // Phone commands are handled even while the gym is closed, so alerts
        // can be armed ahead of time.
        tokio::select! {
            result = alerts.process_commands(worker.database, chrono::Utc::now()) => {
                if let Err(e) = result {
                    tracing::warn!(error = %e, "failed to process phone commands");
                }
            }
            () = &mut shutdown => {
                tracing::info!("shutdown requested");
                return;
            }
        }

        // Starts or collects background training; nightly runs happen while
        // the gym is closed, so this runs before the closed check.
        if let Err(e) = models.tick(worker.database, chrono::Utc::now()).await {
            tracing::warn!(error = %e, "model maintenance failed");
        }

        if !worker.schedule.is_open(&slot) {
            tracing::debug!(
                gym_time = %slot.with_timezone(&worker.schedule.timezone()).format("%H:%M"),
                "gym is closed, skipping fetch"
            );
            continue;
        }

        let cycle = fetch_and_store(worker, slot);
        tokio::pin!(cycle);
        let first = tokio::select! {
            result = tokio::time::timeout(worker.cycle_budget, &mut cycle) => Some(result),
            () = &mut shutdown => None,
        };
        let (outcome, stop) = if let Some(result) = first {
            (result, false)
        } else {
            tracing::info!("shutdown requested, finishing in-flight fetch");
            // SAFETY: if the grace period also expires, the cycle is dropped.
            // A single INSERT commits atomically or not at all and the minute
            // slot is unique, so no partial or duplicate row can result.
            (tokio::time::timeout(SHUTDOWN_GRACE, &mut cycle).await, true)
        };
        let stored_percentage = match &outcome {
            Ok(Ok(Stored::Inserted(percentage))) => Some(*percentage),
            _ => None,
        };
        let health_event = match &outcome {
            Ok(Ok(_)) => health.success(slot),
            Ok(Err(e)) => health.failure(slot, &e.to_string()),
            Err(_) => health.failure(slot, "fetch cycle timed out"),
        };
        log_outcome(slot, outcome);
        if stop {
            return;
        }
        if let Some(event) = health_event {
            tokio::select! {
                () = alerts.publish_health(&event) => {}
                () = &mut shutdown => {
                    tracing::info!("shutdown requested");
                    return;
                }
            }
        }

        if let Some(percentage) = stored_percentage {
            tokio::select! {
                result = alerts.process_reading(worker.database, percentage, slot) => {
                    if let Err(e) = result {
                        tracing::warn!(error = %e, "failed to evaluate alert");
                    }
                }
                () = &mut shutdown => {
                    tracing::info!("shutdown requested");
                    return;
                }
            }
        }
    }
}

/// Reports once if a newer build migrated the database while this daemon
/// runs. It keeps running: additive migrations may still work, and real
/// breakage shows up as a health outage.
async fn check_schema(database: &Database, alerts: &AlertService, reported: &mut bool) {
    match database.schema_status().await {
        Ok(SchemaStatus::DbNewer { db, app }) if !*reported => {
            tracing::error!(
                db,
                app,
                "database schema is newer than this daemon; update it"
            );
            alerts
                .publish_health(&HealthEvent::SchemaAhead { db, app })
                .await;
            *reported = true;
        }
        Ok(_) => {}
        Err(e) => tracing::warn!(error = %e, "could not check the database schema"),
    }
}

fn new_interval(interval_secs: u64) -> tokio::time::Interval {
    let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    interval
}

/// Re-aligns to the next full minute if the timer drifted; returns whether it
/// did.
async fn realign_if_drifted() -> bool {
    let seconds_into_minute = chrono::Utc::now().timestamp() % 60;
    let drift = if seconds_into_minute <= 30 {
        seconds_into_minute
    } else {
        60 - seconds_into_minute
    };

    if drift > DRIFT_THRESHOLD_SECS {
        tracing::warn!(
            drift_secs = drift,
            threshold_secs = DRIFT_THRESHOLD_SECS,
            "timer drift detected, re-syncing"
        );
        wait_for_minute_alignment().await;
        true
    } else {
        tracing::debug!(
            drift_secs = drift,
            threshold_secs = DRIFT_THRESHOLD_SECS,
            "alignment check passed"
        );
        false
    }
}

fn log_outcome(
    slot: chrono::DateTime<chrono::Utc>,
    outcome: Result<Result<Stored, AppError>, tokio::time::error::Elapsed>,
) {
    match outcome {
        Ok(Ok(Stored::Inserted(percentage))) => {
            tracing::info!(%slot, occupancy_pct = percentage, "recorded occupancy");
        }
        Ok(Ok(Stored::SlotTaken(percentage))) => {
            tracing::debug!(%slot, occupancy_pct = percentage, "minute slot already filled");
        }
        Ok(Err(AppError::Validation(reason))) => {
            tracing::warn!(%slot, %reason, "rejected invalid reading");
        }
        Ok(Err(e)) => {
            tracing::error!(%slot, error = %e, "failed to fetch/store data");
        }
        Err(_) => {
            tracing::error!(%slot, "fetch cycle timed out");
        }
    }
}

/// Completes on Ctrl-C or, on Unix, SIGTERM (sent by `docker stop`).
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

async fn wait_for_minute_alignment() {
    let now = chrono::Utc::now();
    let seconds_until_next_minute = 60 - (now.timestamp() % 60);
    if seconds_until_next_minute > 0 && seconds_until_next_minute < 60 {
        tracing::info!(
            wait_secs = seconds_until_next_minute,
            "waiting for next full minute"
        );
        let sleep_secs = seconds_until_next_minute.try_into().unwrap_or(0);
        tokio::time::sleep(Duration::from_secs(sleep_secs)).await;
    }
}

/// What happened to a fetched reading.
enum Stored {
    Inserted(f64),
    SlotTaken(f64),
}

#[tracing::instrument(skip_all, fields(%slot))]
async fn fetch_and_store(
    worker: &Worker<'_>,
    slot: chrono::DateTime<chrono::Utc>,
) -> Result<Stored, AppError> {
    let response = retry(&worker.fetch_policy, "fetch_occupancy", || {
        worker.api_client.fetch_occupancy()
    })
    .await?;
    let percentage = response.occupancy_percentage()?;

    let inserted = retry(&worker.fetch_policy, "insert_record", || async {
        worker
            .database
            .insert_record(slot, percentage)
            .await
            .map_err(|e| AppError::from_anyhow_sqlx(&e, "insert_record"))
    })
    .await?;

    Ok(if inserted.is_some() {
        Stored::Inserted(percentage)
    } else {
        Stored::SlotTaken(percentage)
    })
}
