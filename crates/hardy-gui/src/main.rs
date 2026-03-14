#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::sync::Arc;

use anyhow::{Context, Result};
use hardy_core::{config::AppConfig, db};
use hardy_gui::{
    app::{HardyMonitorApp, Message},
    notifier::CombinedNotifier,
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

            if let Err(e) =
                tray_menu.append_items(&[&show_item, &PredefinedMenuItem::separator(), &quit_item])
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
                Arc::new(hardy_core::SystemClock),
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
