#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::sync::Arc;

use anyhow::{Context, Result};
use hardy_core::{alert::AlertRules, config::AppConfig, db};
use hardy_gui::{
    app::{HardyMonitorApp, Message},
    notifier::SystemNotifier,
    tray::{Tray, TrayBase},
};
use image::GenericImageView;
use muda::{Menu, MenuItem, PredefinedMenuItem};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};
use tray_icon::{Icon, TrayIconBuilder};

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

const TRAY_ICON_SIZE: u32 = 64;

/// The tray icon's pixels; kept so the status dot can be drawn onto them.
async fn load_tray_icon_async() -> Option<TrayBase> {
    tokio::task::spawn_blocking(|| {
        let bytes = include_bytes!("../assets/icon.png");
        // Trays show 16–32 px; a small base keeps redrawing the dot cheap.
        let img = image::load_from_memory(bytes).ok()?.resize(
            TRAY_ICON_SIZE,
            TRAY_ICON_SIZE,
            image::imageops::FilterType::Lanczos3,
        );
        let (width, height) = img.dimensions();
        Some(TrayBase {
            rgba: img.into_rgba8().into_raw(),
            width,
            height,
        })
    })
    .await
    .ok()
    .flatten()
}

/// Tray icon with its menu; `None` (logged) if the platform refuses it.
fn build_tray(base: &TrayBase) -> Option<Tray> {
    let menu = Menu::new();
    let show_item = MenuItem::with_id("show", "Show/Hide", true, None);
    let quit_item = MenuItem::with_id("quit", "Quit", true, None);
    if let Err(e) = menu.append_items(&[&show_item, &PredefinedMenuItem::separator(), &quit_item]) {
        tracing::error!("Failed to build menu: {e}");
    }
    let icon = Icon::from_rgba(base.rgba.clone(), base.width, base.height)
        .map_err(|e| tracing::error!("Failed to decode tray icon: {e}"))
        .ok()?;
    TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("Hardy's Gym Monitor")
        .with_icon(icon)
        .build()
        .map_err(|e| tracing::error!("Failed to build tray icon: {e}"))
        .ok()
        .map(|tray| Tray::new(tray, base.clone()))
}

#[cfg(debug_assertions)]
fn setup_logging() -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::INFO.into())
            .parse_lossy("hardy_core=debug,hardy_gui=debug,fontdb=error,wgpu=warn,naga=warn")
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
            .parse_lossy("hardy_core=info,hardy_gui=info")
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
    let _log_guard = setup_logging();

    let config = AppConfig::load().context("Failed to load configuration")?;
    let config = Arc::new(config);

    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;

    let (database, icon, tray_icon_data) = rt.block_on(async {
        tracing::info!("Connecting to database...");
        // The daemon owns the schema; the GUI only checks it (see app).
        let database = db::Database::connect(&config.database, db::Migrations::Verify).await?;
        tracing::info!("Database connected successfully");

        let (icon, tray_icon_data) = tokio::join!(load_icon_async(), load_tray_icon_async());

        Ok::<_, anyhow::Error>((database, icon, tray_icon_data))
    })?;

    let tray_base = tray_icon_data.context("Failed to load tray icon")?;
    let alert_rules = AlertRules::new(
        config.notifications.cooldown_secs,
        config.notifications.opening_grace_minutes,
        config.notifications.windows.clone(),
    )
    .context("Invalid alert configuration")?;
    let window_width = config.window.width;
    let window_height = config.window.height;

    let app = iced::application(
        move || {
            let tray = build_tray(&tray_base);

            // Phone alerts are sent by the daemon; the GUI only shows
            // desktop popups.
            let notifier = SystemNotifier;

            HardyMonitorApp::new(
                database.clone(),
                tray,
                config.clone(),
                Arc::new(hardy_core::SystemClock),
                Arc::new(notifier),
                alert_rules.clone(),
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

fn update(app: &mut HardyMonitorApp, message: Message) -> iced::Task<Message> {
    app.update(message)
}

fn view(app: &HardyMonitorApp) -> iced::Element<'_, Message> {
    app.view()
}

fn subscription(app: &HardyMonitorApp) -> iced::Subscription<Message> {
    app.subscription()
}

fn theme(app: &HardyMonitorApp) -> iced::Theme {
    app.theme()
}
