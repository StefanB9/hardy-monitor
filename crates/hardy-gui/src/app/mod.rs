//! Application state, messages and the iced entry points.

mod data;
mod data_tab;
mod loads;
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
pub(crate) use data_tab::TABLE_ID as DATA_TABLE_ID;
use hardy_core::{
    accuracy::AccuracySummary,
    alert::AlertRules,
    analytics::{DayAnalysis, Insight, OccupancyStats, SlotRange, TrendDirection},
    config::AppConfig,
    db::{Database, ForecastHistoryEntry, OccupancyLog},
    error::AppError,
    repair::RepairSummary,
    schedule::GymSchedule,
    traits::{Clock, Notifier},
};
use hardy_ml::suite::Family;
use iced::{Subscription, Task, Theme, widget::canvas::Cache, window};
pub use loads::RequestId;
use loads::{Errors, LatestOnly, PendingLoads};
pub use message::{ChartData, Message, RepairPreset, ViewMode};
use muda::MenuEvent;
use tray_icon::TrayIconEvent;

pub use crate::time_range::{AnalyticsRange, ChartRange};
use crate::{
    alerts::AlertControls,
    data_table::DataPeriod,
    forecasting::Forecasting,
    freshness::{Freshness, freshness},
    leaderboard::Sort,
    plan_settings,
    planner::Planner,
    style::{self, OccupancyLevel},
    time_range::Step,
    tray::{Tray, TrayStatus, menu_action, quiet_label, rgba8, tooltip_text, tray_action},
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
    /// The past hour-ahead forecast in the chart range.
    past_forecasts: Vec<ForecastHistoryEntry>,
    /// When the GUI last polled the database.
    last_update: Option<DateTime<Utc>>,
    /// When the newest stored reading was taken.
    latest_reading_at: Option<DateTime<Utc>>,
    /// Hourly averages of the heatmap range.
    week_grid: WeekGrid,
    week_days: Vec<DayAnalysis>,
    insights: Vec<Insight>,
    stats: Option<OccupancyStats>,
    /// Busiest and quietest ranges of the Insights period.
    busy_ranges: Vec<SlotRange>,
    quiet_ranges: Vec<SlotRange>,
    /// Busiest and quietest ranges of the heatmap's period.
    week_busy: Vec<SlotRange>,
    week_quiet: Vec<SlotRange>,
    trend: Option<TrendDirection>,
    forecasting: Forecasting,
    /// Live accuracy of the daemon's logged forecasts.
    accuracy: Option<AccuracySummary>,
    /// Oldest reading; the chart does not step back past it.
    first_reading: Option<DateTime<Utc>>,
    /// Readings of the Data tab's period.
    data_rows: Vec<OccupancyLog>,
    /// The Plan tab's choices and days.
    plan: Planner,
}

struct UiState {
    pending: PendingLoads,
    chart_requests: LatestOnly,
    analytics_requests: LatestOnly,
    data_requests: LatestOnly,
    is_poll_aligned: bool,
    chart_cache: Cache,
    gauge_cache: Cache,
    heatmap_cache: Cache,
    heatmap_tooltip_cache: Cache,
    /// The Plan tab's day curve.
    plan_cache: Cache,
    current_view: ViewMode,
    analytics_range: AnalyticsRange,
    chart_range: ChartRange,
    /// How the forecast models table is sorted and filtered (this session
    /// only).
    leaderboard_sort: Sort,
    leaderboard_family: Option<Family>,
    /// Periods back from the current one (preset ranges).
    chart_offset: u32,
    custom_start: String,
    custom_end: String,
    /// The Data tab: its period, how many periods back, and the table's
    /// scroll offset and visible height.
    data_period: DataPeriod,
    data_offset: u32,
    data_scroll: (f32, f32),
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
    errors: Errors,

    data: MonitorState,
    ui: UiState,
    alerts: AlertControls,
    export_status: Option<String>,
    repair: RepairState,
    /// No queries run until the schema matches this build.
    schema: SchemaGate,
}

impl HardyMonitorApp {
    /// Creates the app and its startup tasks (schema check, first loads).
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
            errors: Errors::default(),
            data: MonitorState {
                occupancy: None,
                history: Vec::new(),
                past_forecasts: Vec::new(),
                last_update: None,
                latest_reading_at: None,
                week_grid: WeekGrid::default(),
                week_days: Vec::new(),
                insights: Vec::new(),
                stats: None,
                busy_ranges: Vec::new(),
                quiet_ranges: Vec::new(),
                week_busy: Vec::new(),
                week_quiet: Vec::new(),
                trend: None,
                forecasting: Forecasting::new(horizon_hours, grace),
                accuracy: None,
                first_reading: None,
                data_rows: Vec::new(),
                plan: Planner::new(plan_settings::default_path()),
            },
            ui: UiState {
                pending: PendingLoads::default(),
                chart_requests: LatestOnly::default(),
                analytics_requests: LatestOnly::default(),
                data_requests: LatestOnly::default(),
                is_poll_aligned: false,
                chart_cache: Cache::new(),
                gauge_cache: Cache::new(),
                heatmap_cache: Cache::new(),
                heatmap_tooltip_cache: Cache::new(),
                plan_cache: Cache::new(),
                current_view: ViewMode::default(),
                analytics_range: AnalyticsRange::default(),
                chart_range: ChartRange::default(),
                leaderboard_sort: Sort::default(),
                leaderboard_family: None,
                chart_offset: 0,
                data_period: DataPeriod::default(),
                data_offset: 0,
                data_scroll: (0.0, 0.0),
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

    /// UI tick, aligned data polling, tray clicks and window-close events.
    pub fn subscription(&self) -> Subscription<Message> {
        let ui_interval = Duration::from_secs(self.config.refresh.ui_interval_secs);
        let data_interval = Duration::from_secs(self.config.refresh.data_fetch_interval_secs);

        let mut subs = vec![iced::time::every(ui_interval).map(|_| Message::Tick)];
        if self.ui.is_poll_aligned {
            subs.push(iced::time::every(data_interval).map(|_| Message::FetchTick));
        }
        subs.push(Subscription::run(tray_events));
        subs.push(iced::event::listen_with(|event, status, _window_id| {
            use iced::keyboard::{self, key::Named};
            match event {
                iced::Event::Window(window::Event::CloseRequested) => {
                    Some(Message::WindowCloseRequested)
                }
                // Keys a focused text field consumed (date inputs) are not
                // steps.
                iced::Event::Keyboard(keyboard::Event::KeyPressed {
                    key: keyboard::Key::Named(named),
                    ..
                }) if status == iced::event::Status::Ignored => match named {
                    Named::ArrowLeft => Some(Message::ChartStep(Step::Back)),
                    Named::ArrowRight => Some(Message::ChartStep(Step::Forward)),
                    _ => None,
                },
                _ => None,
            }
        }));
        Subscription::batch(subs)
    }

    /// The app always uses the dark theme.
    #[allow(clippy::unused_self)]
    pub fn theme(&self) -> Theme {
        Theme::Dark
    }

    /// Everything loaded once the schema is known to match.
    fn start(&mut self) -> Task<Message> {
        Task::batch([
            self.load_alert_settings(),
            self.load_forecast_history(),
            self.load_suite_status(),
            self.load_published_forecast(),
            self.load_chart_history(),
            self.load_analytics(),
            self.load_insights_data(),
            self.load_plan_averages(),
            self.load_accuracy(),
            self.load_first_reading(),
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

    /// Whether database loads have been pending long enough to say so.
    fn is_updating(&self) -> bool {
        self.ui.pending.is_visible(Instant::now())
    }
}

/// Tray clicks and menu entries as messages, sent when they happen (no
/// polling: every message rebuilds the view).
fn tray_events() -> iced::futures::channel::mpsc::UnboundedReceiver<Message> {
    let (tx, rx) = iced::futures::channel::mpsc::unbounded();
    let menu_tx = tx.clone();
    TrayIconEvent::set_event_handler(Some(move |event: TrayIconEvent| {
        if let Some(action) = tray_action(&event) {
            let _ = tx.unbounded_send(Message::Tray(action));
        }
    }));
    MenuEvent::set_event_handler(Some(move |event: MenuEvent| {
        if let Some(action) = menu_action(&event.id.0) {
            let _ = menu_tx.unbounded_send(Message::Tray(action));
        }
    }));
    rx
}
