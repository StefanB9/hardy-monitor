//! Application state, messages and the iced entry points.

mod data;
mod maintenance;
mod message;
mod sidebar;
mod tasks;
mod update;
mod view;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::{
    accuracy::AccuracySummary,
    alert::AlertRules,
    analytics::{DayAnalysis, Insight, OccupancyStats, TrendDirection},
    config::AppConfig,
    db::{Database, OccupancyLog},
    error::AppError,
    repair::RepairSummary,
    schedule::GymSchedule,
    traits::{Clock, Notifier},
};
use iced::{Subscription, Task, Theme, widget::canvas::Cache, window};
pub use message::{Message, RepairPreset, ViewMode};

pub use crate::time_range::{AnalyticsRange, ChartRange};
use crate::{
    alerts::AlertControls,
    forecasting::Forecasting,
    freshness::{Freshness, freshness},
    style::{self, OccupancyLevel},
    tray::{Tray, TrayStatus, quiet_label, rgba8, tooltip_text},
    views::{opening::opening_status, schema_notice::SchemaGate},
    widgets::heatmap::WeekGrid,
};

struct RepairState {
    start_date: String,
    end_date: String,
    is_running: bool,
    last_result: Option<Result<RepairSummary, AppError>>,
}

struct MonitorState {
    occupancy: Option<f64>,
    /// Readings of the chart range.
    history: Vec<OccupancyLog>,
    /// When the GUI last polled the database.
    last_update: Option<DateTime<Utc>>,
    /// When the newest stored reading was taken.
    latest_reading_at: Option<DateTime<Utc>>,
    /// Hourly averages of the heatmap range.
    week_grid: WeekGrid,
    week_days: Vec<DayAnalysis>,
    insights: Vec<Insight>,
    stats: Option<OccupancyStats>,
    peak_hours: Vec<(i32, i32, f64)>,
    quiet_hours: Vec<(i32, i32, f64)>,
    trend: Option<TrendDirection>,
    forecasting: Forecasting,
    /// Live accuracy of the daemon's logged forecasts.
    accuracy: Option<AccuracySummary>,
}

const LOADING_DEBOUNCE_MS: u64 = 200;

struct UiState {
    is_loading: bool,
    loading_started_at: Option<Instant>,
    is_poll_aligned: bool,
    chart_cache: Cache,
    gauge_cache: Cache,
    heatmap_cache: Cache,
    heatmap_tooltip_cache: Cache,
    current_view: ViewMode,
    analytics_range: AnalyticsRange,
    chart_range: ChartRange,
    custom_start: String,
    custom_end: String,
    is_window_visible: bool,
}

/// The desktop application.
pub struct HardyMonitorApp {
    db: Arc<Database>,
    config: Arc<AppConfig>,
    schedule: GymSchedule,
    clock: Arc<dyn Clock>,
    notifier: Arc<dyn Notifier>,
    tray: Option<Tray>,
    error: Option<AppError>,

    data: MonitorState,
    ui: UiState,
    alerts: AlertControls,
    export_status: Option<String>,
    repair: RepairState,
    /// No queries run until the schema matches this build.
    schema: SchemaGate,
}

impl HardyMonitorApp {
    pub fn new(
        db: Database,
        tray: Option<Tray>,
        config: Arc<AppConfig>,
        clock: Arc<dyn Clock>,
        notifier: Arc<dyn Notifier>,
        alert_rules: AlertRules,
    ) -> (Self, Task<Message>) {
        let db = Arc::new(db);
        let now = clock.now_utc();
        let schedule = GymSchedule::new(&config.schedule);
        let gym_today = now.with_timezone(&schedule.timezone()).date_naive();
        let today_str = gym_today.format("%Y-%m-%d").to_string();
        let week_ago_str = (gym_today - TimeDelta::days(6))
            .format("%Y-%m-%d")
            .to_string();

        let horizon_hours = u32::try_from(config.ml.prediction_horizon_hours).unwrap_or(6);
        let grace = TimeDelta::minutes(i64::from(config.notifications.opening_grace_minutes));
        let app = Self {
            db,
            schedule,
            clock,
            notifier,
            tray,
            error: None,
            data: MonitorState {
                occupancy: None,
                history: Vec::new(),
                last_update: None,
                latest_reading_at: None,
                week_grid: WeekGrid::default(),
                week_days: Vec::new(),
                insights: Vec::new(),
                stats: None,
                peak_hours: Vec::new(),
                quiet_hours: Vec::new(),
                trend: None,
                forecasting: Forecasting::new(horizon_hours, grace),
                accuracy: None,
            },
            ui: UiState {
                is_loading: false,
                loading_started_at: None,
                is_poll_aligned: false,
                chart_cache: Cache::new(),
                gauge_cache: Cache::new(),
                heatmap_cache: Cache::new(),
                heatmap_tooltip_cache: Cache::new(),
                current_view: ViewMode::default(),
                analytics_range: AnalyticsRange::default(),
                chart_range: ChartRange::default(),
                custom_start: week_ago_str.clone(),
                custom_end: today_str.clone(),
                is_window_visible: true,
            },
            alerts: AlertControls::new(alert_rules),
            export_status: None,
            repair: RepairState {
                start_date: week_ago_str,
                end_date: today_str,
                is_running: false,
                last_result: None,
            },
            config,
            schema: SchemaGate::Checking,
        };

        let initial = app.check_schema();

        let seconds_to_next_minute = 60 - now.timestamp() % 60;
        let alignment = Task::perform(
            async move {
                tokio::time::sleep(Duration::from_secs(seconds_to_next_minute.cast_unsigned()))
                    .await;
            },
            |()| Message::FetchAlignmentComplete,
        );

        (app, Task::batch([initial, alignment]))
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let ui_interval = Duration::from_secs(self.config.refresh.ui_interval_secs);
        let data_interval = Duration::from_secs(self.config.refresh.data_fetch_interval_secs);
        let tray_interval = Duration::from_millis(self.config.refresh.tray_poll_interval_ms);

        let mut subs = vec![iced::time::every(ui_interval).map(|_| Message::Tick)];
        if self.ui.is_poll_aligned {
            subs.push(iced::time::every(data_interval).map(|_| Message::FetchTick));
        }
        subs.push(iced::time::every(tray_interval).map(|_| Message::TrayCheck));
        subs.push(iced::event::listen_with(|event, _status, _window_id| {
            if let iced::Event::Window(window::Event::CloseRequested) = event {
                Some(Message::WindowCloseRequested)
            } else {
                None
            }
        }));
        Subscription::batch(subs)
    }

    #[allow(clippy::unused_self)]
    pub fn theme(&self) -> Theme {
        Theme::Dark
    }

    /// Everything loaded once the schema is known to match.
    fn start(&self) -> Task<Message> {
        Task::batch([
            self.load_alert_settings(),
            self.load_forecast_history(),
            self.load_model_status(),
            self.load_chart_history(),
            self.load_analytics(),
            self.load_insights_data(),
            self.load_accuracy(),
            self.fetch_latest(),
        ])
    }

    /// Summarises the current state in the tray tooltip and status dot.
    fn update_tray(&mut self) {
        if self.tray.is_none() {
            return;
        }
        let now = self.clock.now_utc();
        let tz = self.schedule.timezone();
        let warning = self.freshness().warning(tz);
        let level = self.data.occupancy.map(|p| {
            let level = OccupancyLevel::from_percentage(
                p,
                self.config.thresholds.low_occupancy_percent,
                self.config.thresholds.high_occupancy_percent,
            );
            (p, level)
        });
        let quiet = self
            .data
            .forecasting
            .quiet_window()
            .map(|w| quiet_label(w.start, now, tz));
        let opening = opening_status(now, &self.schedule);
        let tooltip = tooltip_text(&TrayStatus {
            occupancy: level.map(|(p, l)| (p, l.label())),
            quiet_hour: quiet.as_deref(),
            warning: warning.as_deref(),
            opening: &opening,
        });
        let color = match level {
            Some((_, l)) if warning.is_none() => l.color(),
            _ => style::TEXT_TERTIARY,
        };
        if let Some(tray) = &mut self.tray {
            tray.update(tooltip, rgba8(color));
        }
    }

    /// Whether the newest reading is live; unknown until the first poll.
    fn freshness(&self) -> Freshness {
        if self.data.last_update.is_none() {
            return Freshness::Live;
        }
        freshness(
            self.data.latest_reading_at,
            self.clock.now_utc(),
            &self.schedule,
        )
    }

    fn start_loading(&mut self) {
        if !self.ui.is_loading {
            self.ui.is_loading = true;
            self.ui.loading_started_at = Some(Instant::now());
        }
    }

    fn stop_loading(&mut self) {
        self.ui.is_loading = false;
        self.ui.loading_started_at = None;
    }

    fn should_show_loading(&self) -> bool {
        self.ui.is_loading
            && self.ui.loading_started_at.is_none_or(|started| {
                started.elapsed().as_millis() >= u128::from(LOADING_DEBOUNCE_MS)
            })
    }
}
