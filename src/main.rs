#[cfg(feature = "gui")]
mod app;
#[cfg(feature = "gui")]
mod views;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use hardy_monitor::{api, config::AppConfig, db, schedule::GymSchedule};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[cfg(feature = "gui")]
use hardy_monitor::{CombinedNotifier, SystemClock};
#[cfg(feature = "gui")]
use image::GenericImageView;
#[cfg(feature = "gui")]
use muda::{Menu, MenuItem, PredefinedMenuItem};
#[cfg(feature = "gui")]
use tray_icon::{Icon, TrayIconBuilder};

#[cfg(feature = "gui")]
use crate::app::{HardyMonitorApp, Message};

#[derive(Parser, Debug)]
#[command(name = "hardy-monitor")]
#[command(about = "Gym occupancy monitor - daemon or GUI mode")]
struct Args {
    /// Run in daemon mode (headless data collector)
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

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    #[cfg(feature = "gui")]
    let filter = if args.daemon {
        EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
            .parse_lossy("hardy_monitor=debug")
    } else {
        EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
            .parse_lossy("hardy_monitor=debug,fontdb=error,wgpu=warn,naga=warn")
    };

    #[cfg(not(feature = "gui"))]
    let filter = EnvFilter::builder()
        .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
        .parse_lossy("hardy_monitor=debug");

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

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

/// Maximum allowed drift in seconds before re-syncing
const DRIFT_THRESHOLD_SECS: i64 = 5;

/// Re-check alignment every N iterations (e.g., every hour if interval is 60s)
const ALIGNMENT_CHECK_ITERATIONS: u64 = 60;

/// Run in daemon mode - headless data collection
fn run_daemon(rt: tokio::runtime::Runtime, config: Arc<AppConfig>) -> Result<()> {
    rt.block_on(async {
        tracing::info!("Starting Hardy Monitor in daemon mode");

        // Connect to database
        tracing::info!("Connecting to database...");
        let database = db::Database::new(&config.database.url).await?;
        tracing::info!("Database connected successfully");

        // Create API client
        let api_client = api::GymApiClient::new(config.gym.api_url.clone(), &config.network)?;
        tracing::info!("API client initialized");

        // Create schedule for working hours check
        let schedule = GymSchedule::new(&config.schedule);
        tracing::info!("Schedule configured: weekday {}-{}, weekend {}-{}",
            config.schedule.weekday.open_hour, config.schedule.weekday.close_hour,
            config.schedule.weekend.open_hour, config.schedule.weekend.close_hour);

        // Wait until the next full minute before starting
        wait_for_minute_alignment().await;

        // Main fetch loop - fetch exactly at each full minute
        let interval_secs = config.refresh.data_fetch_interval_secs;
        tracing::info!("Starting fetch loop with interval: {} seconds", interval_secs);

        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut iteration_count: u64 = 0;

        loop {
            interval.tick().await;
            iteration_count += 1;

            // Periodically re-verify time alignment to prevent drift
            if iteration_count % ALIGNMENT_CHECK_ITERATIONS == 0 {
                let now = chrono::Utc::now();
                let seconds_into_minute = now.timestamp() % 60;
                let drift = if seconds_into_minute <= 30 {
                    seconds_into_minute
                } else {
                    60 - seconds_into_minute
                };

                if drift > DRIFT_THRESHOLD_SECS {
                    tracing::warn!(
                        "Timer drift detected: {}s off from minute boundary, re-syncing",
                        drift
                    );
                    wait_for_minute_alignment().await;
                    // Reset the interval after re-alignment
                    interval = tokio::time::interval(Duration::from_secs(interval_secs));
                    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    // Tick once immediately to consume the first instant tick
                    interval.tick().await;
                } else {
                    tracing::debug!("Alignment check passed: {}s drift (threshold: {}s)", drift, DRIFT_THRESHOLD_SECS);
                }
            }

            // Skip fetching when gym is closed
            let now_local = chrono::Local::now();
            if !schedule.is_open(&now_local) {
                tracing::debug!("Gym is closed at {}, skipping fetch", now_local.format("%H:%M"));
                continue;
            }

            match fetch_and_store(&api_client, &database).await {
                Ok(percentage) => {
                    tracing::info!("Recorded occupancy: {:.1}%", percentage);
                }
                Err(e) => {
                    tracing::error!("Failed to fetch/store data: {}", e);
                }
            }
        }
    })
}

/// Wait until the next full minute boundary
async fn wait_for_minute_alignment() {
    let now = chrono::Utc::now();
    let seconds_until_next_minute = 60 - (now.timestamp() % 60);
    if seconds_until_next_minute > 0 && seconds_until_next_minute < 60 {
        tracing::info!(
            "Waiting {} seconds until next full minute...",
            seconds_until_next_minute
        );
        tokio::time::sleep(Duration::from_secs(seconds_until_next_minute as u64)).await;
    }
}

/// Fetch current occupancy and store in database
async fn fetch_and_store(
    api_client: &api::GymApiClient,
    database: &db::Database,
) -> Result<f64> {
    let response = api_client.fetch_occupancy().await?;
    let percentage = response.occupancy_percentage()?;
    let timestamp = chrono::Utc::now();
    database.insert_record(timestamp, percentage).await?;
    Ok(percentage)
}

/// Run in GUI mode - desktop application (read-only, no API fetching)
#[cfg(feature = "gui")]
fn run_gui(rt: tokio::runtime::Runtime, config: Arc<AppConfig>) -> Result<()> {
    // Load database and icons asynchronously (non-blocking)
    let (database, icon, tray_icon_data) = rt.block_on(async {
        tracing::info!("Connecting to database...");
        let database = db::Database::new(&config.database.url).await?;
        tracing::info!("Database connected successfully");

        // Load icons in parallel using spawn_blocking for non-blocking decode
        let (icon, tray_icon_data) = tokio::join!(load_icon_async(), load_tray_icon_async());

        Ok::<_, anyhow::Error>((database, icon, tray_icon_data))
    })?;

    let tray_icon_data = tray_icon_data.context("Failed to load tray icon")?;
    let window_width = config.window.width;
    let window_height = config.window.height;

    let app = iced::application(
        move || {
            // --- CRITICAL: Use .with_id() so we can identify clicks in app.rs ---
            let tray_menu = Menu::new();

            // ID "show" matches the check in app.rs
            let show_item = MenuItem::with_id("show", "Show/Hide", true, None);

            // ID "quit" matches the check in app.rs
            let quit_item = MenuItem::with_id("quit", "Quit", true, None);

            tray_menu
                .append_items(&[&show_item, &PredefinedMenuItem::separator(), &quit_item])
                .expect("Failed to build menu");

            let tray_icon = TrayIconBuilder::new()
                .with_menu(Box::new(tray_menu))
                .with_tooltip("Hardy's Gym Monitor")
                .with_icon(tray_icon_data.clone())
                .build()
                .expect("Failed to build tray icon");

            let notifier = CombinedNotifier::new(config.notifications.ntfy_topic.clone());

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
        // Prevent closing via 'X' button (it will minimize instead)
        exit_on_close_request: false,
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
