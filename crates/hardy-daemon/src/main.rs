#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::time::Duration;

use anyhow::{Context, Result};
use hardy_core::{api, config::AppConfig, db, schedule::GymSchedule};
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
const ALIGNMENT_CHECK_ITERATIONS: u64 = 60;

fn main() -> Result<()> {
    let _log_guard = setup_logging();

    let config = AppConfig::load().context("Failed to load configuration")?;

    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;

    rt.block_on(async {
        tracing::info!("Starting Hardy Monitor in daemon mode");

        tracing::info!("Connecting to database...");
        let database = db::Database::new(&config.database.url).await?;
        tracing::info!("Database connected successfully");

        let api_client = api::GymApiClient::new(config.gym.api_url.clone(), &config.network)?;
        tracing::info!("API client initialized");

        let schedule = GymSchedule::new(&config.schedule);
        tracing::info!(
            weekday_open = config.schedule.weekday.open_hour,
            weekday_close = config.schedule.weekday.close_hour,
            weekend_open = config.schedule.weekend.open_hour,
            weekend_close = config.schedule.weekend.close_hour,
            "schedule configured"
        );

        wait_for_minute_alignment().await;

        let interval_secs = config.refresh.data_fetch_interval_secs;
        tracing::info!(interval_secs, "starting fetch loop");

        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut iteration_count: u64 = 0;

        loop {
            interval.tick().await;
            iteration_count += 1;

            if iteration_count.is_multiple_of(ALIGNMENT_CHECK_ITERATIONS) {
                let now = chrono::Utc::now();
                let seconds_into_minute = now.timestamp() % 60;
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
                    interval = tokio::time::interval(Duration::from_secs(interval_secs));
                    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    interval.tick().await;
                } else {
                    tracing::debug!(
                        drift_secs = drift,
                        threshold_secs = DRIFT_THRESHOLD_SECS,
                        "alignment check passed"
                    );
                }
            }

            let now_local = chrono::Local::now();
            if !schedule.is_open(&now_local) {
                tracing::debug!(
                    time = %now_local.format("%H:%M"),
                    "gym is closed, skipping fetch"
                );
                continue;
            }

            match fetch_and_store(&api_client, &database).await {
                Ok(percentage) => {
                    tracing::info!(occupancy_pct = percentage, "recorded occupancy");
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to fetch/store data");
                }
            }
        }
    })
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

#[tracing::instrument(skip_all)]
async fn fetch_and_store(api_client: &api::GymApiClient, database: &db::Database) -> Result<f64> {
    let response = api_client.fetch_occupancy().await?;
    let percentage = response.occupancy_percentage()?;
    let timestamp = chrono::Utc::now();
    database.insert_record(timestamp, percentage).await?;
    Ok(percentage)
}
