//! Application state, messages and the iced entry points.

mod tasks;
mod update;
mod view;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::{
    alert::{AlertDuration, AlertRules, AlertSettings},
    analytics::{DayAnalysis, Insight, OccupancyStats, TrendDirection},
    config::AppConfig,
    db::{Database, HourlyAverage, MlState, ModelInfo, OccupancyLog},
    error::AppError,
    repair::RepairSummary,
    schedule::GymSchedule,
    traits::{Clock, Notifier},
};
use hardy_ml::ModelArtifact;
use iced::{Subscription, Task, Theme, widget::canvas::Cache, window};
use tray_icon::TrayIcon;

pub use crate::time_range::{AnalyticsRange, ChartRange};
use crate::{alerts::AlertControls, forecasting::Forecasting, widgets::heatmap::WeekGrid};

/// The four top-level views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    #[default]
    Now,
    Week,
    Insights,
    ModelData,
}

impl ViewMode {
    /// All views in sidebar order.
    pub const ALL: [ViewMode; 4] = [
        ViewMode::Now,
        ViewMode::Week,
        ViewMode::Insights,
        ViewMode::ModelData,
    ];

    pub fn title(self) -> &'static str {
        match self {
            ViewMode::Now => "Now",
            ViewMode::Week => "Week",
            ViewMode::Insights => "Insights",
            ViewMode::ModelData => "Model & Data",
        }
    }

    pub fn icon(self) -> &'static str {
        match self {
            ViewMode::Now => "◉",
            ViewMode::Week => "▦",
            ViewMode::Insights => "✦",
            ViewMode::ModelData => "⚙",
        }
    }
}

/// Date presets for Data Repair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairPreset {
    Last7Days,
    Last30Days,
    AllData,
}

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
    last_update: Option<DateTime<Utc>>,
    /// Hourly averages of the heatmap range.
    week_grid: WeekGrid,
    week_days: Vec<DayAnalysis>,
    insights: Vec<Insight>,
    stats: Option<OccupancyStats>,
    peak_hours: Vec<(i32, i32, f64)>,
    quiet_hours: Vec<(i32, i32, f64)>,
    trend: Option<TrendDirection>,
    forecasting: Forecasting,
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
    _tray_icon: Option<TrayIcon>,
    error: Option<AppError>,

    data: MonitorState,
    ui: UiState,
    alerts: AlertControls,
    export_status: Option<String>,
    repair: RepairState,
}

/// Everything the application reacts to.
#[derive(Debug, Clone)]
pub enum Message {
    Tick,
    FetchTick,
    FetchAlignmentComplete,
    RefreshNow,

    FetchCompleted(Result<Option<f64>, AppError>),
    HistoryLoaded(Result<Vec<OccupancyLog>, AppError>),
    AnalyticsLoaded(Result<Vec<HourlyAverage>, AppError>),
    InsightsDataLoaded {
        current: Result<Vec<HourlyAverage>, AppError>,
        baseline: Result<Vec<HourlyAverage>, AppError>,
    },

    NotificationThresholdChanged(f64),
    NotificationThresholdReleased,
    NotificationToggled(bool),
    NotificationDurationSelected(AlertDuration),
    AlertSettingsLoaded(Result<AlertSettings, AppError>),
    NotificationSent,

    SwitchView(ViewMode),
    SwitchAnalyticsRange(AnalyticsRange),
    ChartRangeSelected(ChartRange),
    CustomStartChanged(String),
    CustomEndChanged(String),
    ApplyCustomRange,

    ExportCsv,
    ExportCompleted(Result<String, AppError>),
    ClearExportStatus,
    TrayCheck,
    WindowCloseRequested,

    RepairStartDateChanged(String),
    RepairEndDateChanged(String),
    RepairPresetSelected(RepairPreset),
    StartRepairJob,
    RepairCompleted(Result<RepairSummary, AppError>),

    /// New rows for the forecasting history.
    ForecastHistoryLoaded(Result<Vec<OccupancyLog>, AppError>),
    /// Newest stored model and training state.
    ModelStatusLoaded(Result<(Option<ModelInfo>, MlState), AppError>),
    ModelLoaded(Result<(ModelInfo, Arc<ModelArtifact>), AppError>),
    TrainModelRequested,
}

impl HardyMonitorApp {
    pub fn new(
        db: Database,
        tray_icon: Option<TrayIcon>,
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
            _tray_icon: tray_icon,
            error: None,
            data: MonitorState {
                occupancy: None,
                history: Vec::new(),
                last_update: None,
                week_grid: WeekGrid::default(),
                week_days: Vec::new(),
                insights: Vec::new(),
                stats: None,
                peak_hours: Vec::new(),
                quiet_hours: Vec::new(),
                trend: None,
                forecasting: Forecasting::new(horizon_hours, grace),
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
        };

        let initial = Task::batch([
            app.load_alert_settings(),
            app.load_forecast_history(),
            app.load_model_status(),
            app.load_chart_history(),
            app.load_analytics(),
            app.load_insights_data(),
            app.fetch_latest(),
        ]);

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
