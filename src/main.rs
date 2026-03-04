use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result};
use clap::Parser;
use hardy_monitor::{api, config::AppConfig, db, schedule::GymSchedule};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[cfg(feature = "gui")]
mod app;
#[cfg(feature = "gui")]
mod views;

#[cfg(feature = "gui")]
use {
    crate::app::{HardyMonitorApp, Message},
    hardy_monitor::{CombinedNotifier, SystemClock},
    image::GenericImageView,
    muda::{Menu, MenuItem, PredefinedMenuItem},
    tray_icon::{Icon, TrayIconBuilder},
};

#[derive(Parser, Debug)]
#[command(name = "hardy-monitor")]
#[command(about = "Gym occupancy monitor - daemon or GUI mode")]
struct Args {
    #[arg(long)]
    daemon: bool,
}

#[cfg(feature = "gui")]
async fn load_icon_async() -> Option<iced::window::Icon> {
    tokio::task::spawn_blocking(|| {
        let bytes = include_bytes!("../assets/icon.png");
        let img = image::load_from_memory(bytes).ok()?;
        let (width, height) = img.dimensions();
        let rgba = img.into_rgba8().into_raw();
        iced::window::icon::from_rgba(rgba, width, height).ok()
    })
    .await
    .ok()
    .flatten()
}

#[cfg(feature = "gui")]
async fn load_tray_icon_async() -> Option<Icon> {
    tokio::task::spawn_blocking(|| {
        let bytes = include_bytes!("../assets/icon.png");
        let img = image::load_from_memory(bytes).ok()?;
        let (width, height) = img.dimensions();
        let rgba = img.into_rgba8().into_raw();
        Icon::from_rgba(rgba, width, height).ok()
    })
    .await
    .ok()
    .flatten()
}

#[cfg(debug_assertions)]
fn setup_logging(args: &Args) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    // DEBUG, noisy GPU/font crates suppressed so the terminal stays readable
    #[cfg(feature = "gui")]
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else if args.daemon {
        EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
            .parse_lossy("hardy_monitor=debug")
    } else {
        EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
            .parse_lossy("hardy_monitor=debug,fontdb=error,wgpu=warn,naga=warn")
    };

    #[cfg(not(feature = "gui"))]
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
            .parse_lossy("hardy_monitor=debug")
    };

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    None
}

#[cfg(not(debug_assertions))]
fn setup_logging(_args: &Args) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let file_appender = tracing_appender::rolling::daily("logs", "hardy-monitor.log");
    let (non_blocking_writer, guard) = tracing_appender::non_blocking(file_appender);

    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
            .parse_lossy("hardy_monitor=info")
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

fn main() -> Result<()> {
    let args = Args::parse();

    let _log_guard = setup_logging(&args);

    let config = AppConfig::load().context("Failed to load configuration")?;
    let config = Arc::new(config);

    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;

    if args.daemon {
        run_daemon(rt, config)
    } else {
        #[cfg(feature = "gui")]
        {
            run_gui(rt, config)
        }
        #[cfg(not(feature = "gui"))]
        {
            anyhow::bail!("GUI mode not available. Build with --features gui or run with --daemon")
        }
    }
}

const DRIFT_THRESHOLD_SECS: i64 = 5;
const ALIGNMENT_CHECK_ITERATIONS: u64 = 60;

fn run_daemon(rt: tokio::runtime::Runtime, config: Arc<AppConfig>) -> Result<()> {
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
        tokio::time::sleep(Duration::from_secs(seconds_until_next_minute as u64)).await;
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

#[cfg(feature = "gui")]
fn run_gui(rt: tokio::runtime::Runtime, config: Arc<AppConfig>) -> Result<()> {
    let (database, icon, tray_icon_data) = rt.block_on(async {
        tracing::info!("Connecting to database...");
        let database = db::Database::new(&config.database.url).await?;
        tracing::info!("Database connected successfully");

        let (icon, tray_icon_data) = tokio::join!(load_icon_async(), load_tray_icon_async());

        Ok::<_, anyhow::Error>((database, icon, tray_icon_data))
    })?;

    let tray_icon_data = tray_icon_data.context("Failed to load tray icon")?;
    let window_width = config.window.width;
    let window_height = config.window.height;

    let app = iced::application(
        move || {
            let tray_menu = Menu::new();
            let show_item = MenuItem::with_id("show", "Show/Hide", true, None);

            let quit_item = MenuItem::with_id("quit", "Quit", true, None);

            if let Err(e) = tray_menu
                .append_items(&[&show_item, &PredefinedMenuItem::separator(), &quit_item])
            {
                tracing::error!("Failed to build menu: {e}");
            }

            let tray_icon = TrayIconBuilder::new()
                .with_menu(Box::new(tray_menu))
                .with_tooltip("Hardy's Gym Monitor")
                .with_icon(tray_icon_data.clone())
                .build()
                .map_err(|e| tracing::error!("Failed to build tray icon: {e}"))
                .ok();

            let notifier = CombinedNotifier::new(
                config.notifications.ntfy_topic.clone(),
                config.notifications.ntfy_server.clone(),
            );

            HardyMonitorApp::new(
                database.clone(),
                tray_icon,
                config.clone(),
                Arc::new(SystemClock),
                Arc::new(notifier),
            )
        },
        update,
        view,
    )
    .title("Hardy's Gym Monitor")
    .subscription(subscription)
    .theme(theme)
    .window(iced::window::Settings {
        size: iced::Size::new(window_width, window_height),
        icon,
        exit_on_close_request: true,
        ..Default::default()
    })
    .antialiasing(true);

    app.run().context("Failed to run application")?;

    Ok(())
}

#[cfg(feature = "gui")]
fn update(app: &mut HardyMonitorApp, message: Message) -> iced::Task<Message> {
    app.update(message)
}

#[cfg(feature = "gui")]
fn view(app: &HardyMonitorApp) -> iced::Element<'_, Message> {
    app.view()
}

#[cfg(feature = "gui")]
fn subscription(app: &HardyMonitorApp) -> iced::Subscription<Message> {
    app.subscription()
}

#[cfg(feature = "gui")]
fn theme(app: &HardyMonitorApp) -> iced::Theme {
    app.theme()
}
