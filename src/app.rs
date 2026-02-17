use std::{path::PathBuf, sync::Arc, time::{Duration, Instant}};

use chrono::{DateTime, Datelike, Duration as ChronoDuration, Local, NaiveDate, Utc};
use hardy_monitor::{
    analytics::{
        self, ComparisonMode, DayAnalysis, Insight, OccupancyStats, TrendDirection, analyze_days,
        calculate_stats, compare_periods, find_peak_hours, find_quiet_hours, generate_insights,
        midnight_local_as_utc, midnight_utc,
    },
    config::AppConfig,
    db::{Database, HourlyAverage, OccupancyLog},
    error::AppError,
    repair::DataRepairer,
    schedule::GymSchedule,
    style,
    traits::{Clock, Notifier},
};
use crate::views::{
    self,
    DashboardProps, DataRepairProps, InsightsProps, WeeklyPatternProps,
};
use iced::{
    Alignment, Border, Color, Element, Length, Shadow, Subscription, Task, Theme, Vector,
    widget::{Space, button, canvas::Cache, column, container, row, stack, text},
    window,
};
use muda::MenuEvent;
use tray_icon::{TrayIcon, TrayIconEvent};

// --- STATE STRUCTS ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    #[default]
    Dashboard,
    WeeklyPattern,
    Insights,
    DataRepair,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnalyticsRange {
    #[default]
    ThisWeek,
    Last2Weeks,
    Last4Weeks,
    Last8Weeks,
}

use hardy_monitor::repair::{RepairProgress, RepairSummary};

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
    progress: Option<RepairProgress>,
    last_result: Option<Result<RepairSummary, AppError>>,
}

struct MonitorState {
    occupancy: Option<f64>,
    history: Vec<OccupancyLog>,
    last_update: Option<DateTime<Utc>>,
    analytics_data: Vec<HourlyAverage>,
    best_time_today: Option<(i32, f64)>,
    prediction_baseline: Vec<HourlyAverage>,
    predictions: Vec<(DateTime<Utc>, f64)>,
    // Insights data
    insights: Vec<Insight>,
    stats: Option<OccupancyStats>,
    day_analysis: Vec<DayAnalysis>,
    peak_hours: Vec<(i32, i32, f64)>,
    quiet_hours: Vec<(i32, i32, f64)>,
    trend: Option<TrendDirection>,
    baseline_for_comparison: Vec<HourlyAverage>,
}

/// Threshold before showing "Updating..." indicator (prevents flickering)
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
    history_start_date: String,
    history_end_date: String,
    history_days_preset: Option<i64>,
    is_window_visible: bool,
}

struct NotificationState {
    threshold: f64,
    enabled: bool,
    was_below_threshold: bool,
}

struct ExportState {
    status: Option<String>,
}

pub struct HardyMonitorApp {
    db: Arc<Database>,
    config: Arc<AppConfig>,
    schedule: GymSchedule,
    clock: Arc<dyn Clock>,
    notifier: Arc<dyn Notifier>,
    _tray_icon: TrayIcon,
    error: Option<AppError>,

    // Grouped State
    data: MonitorState,
    ui: UiState,
    notifications: NotificationState,
    export: ExportState,
    repair: RepairState,
}

#[derive(Debug, Clone)]
pub enum Message {
    Tick,
    FetchTick,
    FetchAlignmentComplete,
    RefreshNow,
    ChartInteraction, // Mapped from widget interaction

    // Data Results
    FetchCompleted(Result<f64, AppError>),
    HistoryLoaded(Result<Vec<OccupancyLog>, AppError>),
    AnalyticsLoaded(Result<Vec<HourlyAverage>, AppError>),
    PredictionBaselineLoaded(Result<Vec<HourlyAverage>, AppError>),
    InsightsDataLoaded {
        current: Result<Vec<HourlyAverage>, AppError>,
        baseline: Result<Vec<HourlyAverage>, AppError>,
    },

    // Notifications
    NotificationThresholdChanged(f64),
    NotificationToggled(bool),
    NotificationSent,

    // Navigation & View
    SwitchView(ViewMode),
    SwitchAnalyticsRange(AnalyticsRange),
    HistoryStartDateChanged(String),
    HistoryEndDateChanged(String),
    HistoryPresetSelected(i64),
    ApplyDateRange,

    // Export & System
    ExportCsv,
    ExportCompleted(Result<String, AppError>),
    ClearExportStatus,
    TrayCheck,
    WindowCloseRequested,

    // Data Repair Page
    RepairStartDateChanged(String),
    RepairEndDateChanged(String),
    RepairPresetSelected(RepairPreset),
    StartRepairJob,
    #[allow(dead_code)]
    RepairProgress(RepairProgress),
    RepairCompleted(Result<RepairSummary, AppError>),
}

impl HardyMonitorApp {
    pub fn new(
        db: Database,
        tray_icon: TrayIcon,
        config: Arc<AppConfig>,
        clock: Arc<dyn Clock>,
        notifier: Arc<dyn Notifier>,
    ) -> (Self, Task<Message>) {
        let db = Arc::new(db);
        let now = clock.now_utc();
        let today_str = now.date_naive().format("%Y-%m-%d").to_string();
        let tomorrow_str = (now.date_naive() + ChronoDuration::days(1))
            .format("%Y-%m-%d")
            .to_string();

        let schedule = GymSchedule::new(&config.schedule);

        let app = Self {
            db: db.clone(),
            config: config.clone(),
            schedule,
            clock: clock.clone(),
            notifier,
            _tray_icon: tray_icon,
            error: None,
            data: MonitorState {
                occupancy: None,
                history: Vec::new(),
                last_update: None,
                analytics_data: Vec::new(),
                best_time_today: None,
                prediction_baseline: Vec::new(),
                predictions: Vec::new(),
                insights: Vec::new(),
                stats: None,
                day_analysis: Vec::new(),
                peak_hours: Vec::new(),
                quiet_hours: Vec::new(),
                trend: None,
                baseline_for_comparison: Vec::new(),
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
                history_start_date: today_str.clone(),
                history_end_date: tomorrow_str.clone(),
                history_days_preset: Some(1),
                is_window_visible: true,
            },
            notifications: NotificationState {
                threshold: config.notifications.threshold_percent,
                enabled: config.notifications.enabled,
                was_below_threshold: false,
            },
            export: ExportState { status: None },
            repair: RepairState {
                start_date: today_str.clone(),
                end_date: tomorrow_str,
                is_running: false,
                progress: None,
                last_result: None,
            },
        };

        let prediction_days = config.analytics.prediction_window_days;
        let clock_for_tasks = clock.clone();
        let initial_tasks = vec![
            Self::load_history(db.clone()),
            Self::load_analytics(
                db.clone(),
                AnalyticsRange::ThisWeek,
                clock_for_tasks.clone(),
            ),
            Self::load_prediction_baseline(db.clone(), prediction_days, clock_for_tasks),
        ];

        let seconds_to_next_minute = 60 - now.timestamp() % 60;
        let alignment_task = Task::perform(
            async move {
                tokio::time::sleep(Duration::from_secs(seconds_to_next_minute as u64)).await;
            },
            |_| Message::FetchAlignmentComplete,
        );

        (
            app,
            Task::batch([Task::batch(initial_tasks), alignment_task]),
        )
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Tick => {
                self.data.predictions =
                    analytics::calculate_predictions(&self.data.prediction_baseline);
                Task::none()
            }
            Message::ChartInteraction => Task::none(),
            Message::FetchAlignmentComplete => {
                self.ui.is_poll_aligned = true;
                if self.schedule.is_open(&self.clock.now_local()) {
                    self.start_loading();
                    Self::fetch_latest_from_db(self.db.clone())
                } else {
                    self.data.occupancy = None;
                    self.stop_loading();
                    Task::none()
                }
            }
            Message::FetchTick => {
                if self.schedule.is_open(&self.clock.now_local()) {
                    self.start_loading();
                    Self::fetch_latest_from_db(self.db.clone())
                } else {
                    self.data.occupancy = None;
                    self.stop_loading();
                    Task::none()
                }
            }
            Message::RefreshNow => {
                self.start_loading();
                self.error = None;
                let prediction_days = self.config.analytics.prediction_window_days;
                Task::batch([
                    Self::fetch_latest_from_db(self.db.clone()),
                    Self::load_history(self.db.clone()),
                    Self::load_analytics(
                        self.db.clone(),
                        self.ui.analytics_range,
                        self.clock.clone(),
                    ),
                    Self::load_prediction_baseline(
                        self.db.clone(),
                        prediction_days,
                        self.clock.clone(),
                    ),
                ])
            }
            Message::FetchCompleted(result) => self.handle_fetch_completed(result),
            Message::HistoryLoaded(result) => {
                if let Ok(logs) = result {
                    self.data.history = logs;
                    self.ui.chart_cache.clear();
                    self.data.predictions =
                        analytics::calculate_predictions(&self.data.prediction_baseline);
                } else if let Err(e) = result {
                    self.error = Some(e);
                }
                Task::none()
            }
            Message::AnalyticsLoaded(result) => {
                if let Ok(data) = result {
                    self.data.analytics_data = data;
                    self.ui.heatmap_cache.clear();
                    self.data.best_time_today =
                        analytics::find_best_time_today(&self.data.analytics_data);
                } else if let Err(e) = result {
                    self.error = Some(e);
                }
                Task::none()
            }
            Message::PredictionBaselineLoaded(result) => {
                if let Ok(data) = result {
                    self.data.prediction_baseline = data;
                    self.data.predictions =
                        analytics::calculate_predictions(&self.data.prediction_baseline);
                }
                Task::none()
            }
            Message::InsightsDataLoaded { current, baseline } => {
                self.handle_insights_data_loaded(current, baseline)
            }
            Message::NotificationThresholdChanged(val) => {
                self.notifications.threshold = val;
                Task::none()
            }
            Message::NotificationToggled(enabled) => {
                self.notifications.enabled = enabled;
                self.notifications.was_below_threshold =
                    self.data.occupancy.unwrap_or(100.0) < self.notifications.threshold;
                Task::none()
            }
            Message::NotificationSent => Task::none(),
            Message::SwitchView(mode) => {
                self.ui.current_view = mode;
                if mode == ViewMode::Insights {
                    // Load data for insights when switching to that view
                    Self::load_insights_data(self.db.clone(), self.clock.clone())
                } else {
                    Task::none()
                }
            }
            Message::SwitchAnalyticsRange(range) => {
                self.ui.analytics_range = range;
                self.ui.heatmap_cache.clear();
                Self::load_analytics(self.db.clone(), range, self.clock.clone())
            }
            Message::HistoryStartDateChanged(d) => {
                self.ui.history_start_date = d;
                self.ui.history_days_preset = None;
                Task::none()
            }
            Message::HistoryEndDateChanged(d) => {
                self.ui.history_end_date = d;
                self.ui.history_days_preset = None;
                Task::none()
            }
            Message::HistoryPresetSelected(days) => {
                self.ui.history_days_preset = Some(days);
                let now = self.clock.now_utc();
                let tomorrow = now.date_naive() + ChronoDuration::days(1);
                let start_date = tomorrow - ChronoDuration::days(days);
                self.ui.history_start_date = start_date.format("%Y-%m-%d").to_string();
                self.ui.history_end_date = tomorrow.format("%Y-%m-%d").to_string();
                Self::load_history_range(self.db.clone(), midnight_utc(start_date), now)
            }
            Message::ApplyDateRange => {
                if let (Some(s), Some(e)) = (
                    parse_date(&self.ui.history_start_date),
                    parse_date(&self.ui.history_end_date),
                ) {
                    let range_end = if s == e {
                        e + ChronoDuration::days(1)
                    } else {
                        e
                    };
                    Self::load_history_range(self.db.clone(), s, range_end)
                } else {
                    self.error = Some(AppError::validation("Invalid date format"));
                    Task::none()
                }
            }
            Message::WindowCloseRequested => {
                self.ui.is_window_visible = false;
                window::latest().and_then(|id| window::minimize(id, true))
            }
            Message::TrayCheck => {
                let mut tasks = Vec::new();
                let mut should_toggle = false;
                while let Ok(event) = TrayIconEvent::receiver().try_recv() {
                    if let TrayIconEvent::Click { .. } = event {
                        should_toggle = true;
                    }
                }
                while let Ok(event) = MenuEvent::receiver().try_recv() {
                    if event.id.0 == "quit" {
                        std::process::exit(0);
                    } else if event.id.0 == "show" {
                        should_toggle = true;
                    }
                }
                if should_toggle {
                    self.ui.is_window_visible = !self.ui.is_window_visible;
                    let target = self.ui.is_window_visible;
                    tasks.push(window::latest().and_then(move |id| {
                        if target {
                            Task::batch([window::minimize(id, false), window::gain_focus(id)])
                        } else {
                            window::minimize(id, true)
                        }
                    }));
                }
                Task::batch(tasks)
            }
            Message::ExportCsv => {
                self.start_loading();
                self.export.status = Some("Exporting...".to_string());
                let db = self.db.clone();
                let clock = self.clock.clone();
                Task::perform(
                    async move {
                        let logs = db
                            .get_history(365 * 10)
                            .await
                            .map_err(|e| AppError::from_anyhow_db(e, "get_history"))?;
                        let export_time = clock.now_utc();
                        let path =
                            tokio::task::spawn_blocking(move || -> Result<PathBuf, AppError> {
                                let mut path =
                                    dirs::download_dir().unwrap_or_else(|| PathBuf::from("."));
                                path.push(format!(
                                    "hardy_monitor_export_{}.csv",
                                    export_time.format("%Y%m%d_%H%M%S")
                                ));
                                let mut wtr = csv::Writer::from_path(&path)
                                    .map_err(|e| AppError::io(e.to_string()))?;
                                for log in logs {
                                    wtr.serialize(log)
                                        .map_err(|e| AppError::io(e.to_string()))?;
                                }
                                wtr.flush().map_err(|e| AppError::io(e.to_string()))?;
                                Ok(path)
                            })
                            .await
                            .map_err(|e| AppError::Unknown(e.to_string()))??;
                        Ok(path.to_string_lossy().to_string())
                    },
                    Message::ExportCompleted,
                )
            }
            Message::ExportCompleted(result) => {
                self.stop_loading();
                match result {
                    Ok(path) => self.export.status = Some(format!("Saved to {}", path)),
                    Err(e) => {
                        self.error = Some(e);
                        self.export.status = Some("Export failed".to_string());
                    }
                }
                Task::perform(
                    async {
                        tokio::time::sleep(Duration::from_secs(4)).await;
                    },
                    |_| Message::ClearExportStatus,
                )
            }
            Message::ClearExportStatus => {
                self.export.status = None;
                Task::none()
            }
            Message::RepairStartDateChanged(d) => {
                self.repair.start_date = d;
                Task::none()
            }
            Message::RepairEndDateChanged(d) => {
                self.repair.end_date = d;
                Task::none()
            }
            Message::RepairPresetSelected(preset) => {
                let now = self.clock.now_utc();
                let today = now.date_naive();
                match preset {
                    RepairPreset::Last7Days => {
                        let start = today - ChronoDuration::days(7);
                        self.repair.start_date = start.format("%Y-%m-%d").to_string();
                        self.repair.end_date = today.format("%Y-%m-%d").to_string();
                    }
                    RepairPreset::Last30Days => {
                        let start = today - ChronoDuration::days(30);
                        self.repair.start_date = start.format("%Y-%m-%d").to_string();
                        self.repair.end_date = today.format("%Y-%m-%d").to_string();
                    }
                    RepairPreset::AllData => {
                        // Set to a very early date
                        self.repair.start_date = "2020-01-01".to_string();
                        self.repair.end_date = today.format("%Y-%m-%d").to_string();
                    }
                }
                Task::none()
            }
            Message::StartRepairJob => {
                if self.repair.is_running {
                    return Task::none();
                }

                let start = match parse_date(&self.repair.start_date) {
                    Some(d) => d.date_naive(),
                    None => {
                        self.error = Some(AppError::validation("Invalid start date"));
                        return Task::none();
                    }
                };
                let end = match parse_date(&self.repair.end_date) {
                    Some(d) => d.date_naive(),
                    None => {
                        self.error = Some(AppError::validation("Invalid end date"));
                        return Task::none();
                    }
                };

                if start > end {
                    self.error = Some(AppError::validation(
                        "Start date must be before end date",
                    ));
                    return Task::none();
                }

                self.repair.is_running = true;
                self.repair.progress = None;
                self.repair.last_result = None;
                self.error = None;

                let db = self.db.clone();
                let schedule = self.schedule.clone();
                Task::perform(
                    async move {
                        let repairer = DataRepairer::new(db, schedule);
                        repairer.repair_date_range(start, end, None).await
                    },
                    |r| match r {
                        Ok(summary) => Message::RepairCompleted(Ok(summary)),
                        Err(e) => Message::RepairCompleted(Err(AppError::from_anyhow_db(e, "repair_date_range"))),
                    },
                )
            }
            Message::RepairProgress(progress) => {
                self.repair.progress = Some(progress);
                Task::none()
            }
            Message::RepairCompleted(result) => {
                self.repair.is_running = false;
                self.repair.last_result = Some(result);
                Task::none()
            }
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        let sidebar = self.view_sidebar();
        let content = match self.ui.current_view {
            ViewMode::Dashboard => views::dashboard::view(DashboardProps {
                occupancy: self.data.occupancy,
                history: &self.data.history,
                predictions: &self.data.predictions,
                best_time_today: self.data.best_time_today,
                chart_cache: &self.ui.chart_cache,
                gauge_cache: &self.ui.gauge_cache,
                schedule: &self.schedule,
                low_threshold: self.config.thresholds.low_occupancy_percent,
                high_threshold: self.config.thresholds.high_occupancy_percent,
                notification_enabled: self.notifications.enabled,
                notification_threshold: self.notifications.threshold,
                history_start_date: &self.ui.history_start_date,
                history_end_date: &self.ui.history_end_date,
                history_days_preset: self.ui.history_days_preset,
            }),
            ViewMode::WeeklyPattern => views::weekly_pattern::view(WeeklyPatternProps {
                analytics_data: &self.data.analytics_data,
                analytics_range: self.ui.analytics_range,
                heatmap_cache: &self.ui.heatmap_cache,
                heatmap_tooltip_cache: &self.ui.heatmap_tooltip_cache,
            }),
            ViewMode::Insights => views::insights::view(InsightsProps {
                trend: self.data.trend,
                stats: self.data.stats.as_ref(),
                peak_hours: &self.data.peak_hours,
                quiet_hours: &self.data.quiet_hours,
                day_analysis: &self.data.day_analysis,
                insights: &self.data.insights,
            }),
            ViewMode::DataRepair => views::data_repair::view(DataRepairProps {
                start_date: &self.repair.start_date,
                end_date: &self.repair.end_date,
                is_running: self.repair.is_running,
                progress: self.repair.progress.as_ref(),
                last_result: self.repair.last_result.as_ref(),
            }),
        };

        let main_area = container(column![
            self.view_header(),
            Space::new().height(20),
            content
        ])
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(30);

        let app_layout = row![sidebar, main_area]
            .width(Length::Fill)
            .height(Length::Fill);

        if let Some(msg) = &self.export.status {
            let toast = container(text(msg).size(14).color(style::TEXT_BRIGHT))
                .padding([12, 24])
                .style(|_| container::Style {
                    background: Some(style::BG_CARD.into()),
                    border: Border {
                        radius: 20.0.into(),
                        width: 1.0,
                        color: style::ACCENT_GREEN,
                    },
                    shadow: Shadow {
                        color: Color::from_rgba(0.0, 0.0, 0.0, 0.5),
                        offset: Vector::new(0.0, 4.0),
                        blur_radius: 10.0,
                    },
                    ..Default::default()
                });
            stack![
                app_layout,
                container(toast)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .align_x(Alignment::Center)
                    .padding(30)
            ]
            .into()
        } else {
            app_layout.into()
        }
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

    pub fn theme(&self) -> Theme {
        Theme::Dark
    }

    // --- VIEW COMPONENTS ---

    fn view_sidebar(&self) -> Element<'_, Message> {
        let sidebar_width = self.config.window.sidebar_width;

        let brand = column![
            text("HARDY")
                .size(32)
                .font(iced::font::Font::MONOSPACE)
                .color(style::ACCENT_BLUE),
            text("MONITOR").size(14).color(style::TEXT_MUTED),
        ];

        let nav_btn = |label: &str, mode: ViewMode| {
            let is_active = self.ui.current_view == mode;
            let bg = if is_active {
                style::ACCENT_BLUE
            } else {
                Color::TRANSPARENT
            };
            let txt = if is_active {
                style::BG_DARK
            } else {
                style::TEXT_MUTED
            };
            button(text(label.to_string()).color(txt).size(16))
                .on_press(Message::SwitchView(mode))
                .style(move |_, _| button::Style {
                    background: Some(bg.into()),
                    border: Border {
                        radius: 8.0.into(),
                        ..Default::default()
                    },
                    text_color: txt,
                    ..Default::default()
                })
                .width(Length::Fill)
                .padding(12)
        };

        container(column![
            brand,
            Space::new().height(40),
            nav_btn("Dashboard", ViewMode::Dashboard),
            Space::new().height(10),
            nav_btn("Weekly Heatmap", ViewMode::WeeklyPattern),
            Space::new().height(10),
            nav_btn("Insights", ViewMode::Insights),
            Space::new().height(10),
            nav_btn("Data Repair", ViewMode::DataRepair),
        ])
        .width(Length::Fixed(sidebar_width))
        .height(Length::Fill)
        .style(|_| container::Style {
            background: Some(style::BG_CARD.into()),
            border: Border {
                color: style::STROKE_DIM,
                width: 1.0,
                ..Default::default()
            },
            ..Default::default()
        })
        .padding(20)
        .into()
    }

    fn view_header(&self) -> Element<'_, Message> {
        let last_update = self
            .data
            .last_update
            .map(|t| t.with_timezone(&Local).format("%H:%M:%S").to_string())
            .unwrap_or_else(|| "--:--:--".to_string());

        let status = if self.should_show_loading() {
            row![
                text("Updating").size(14).color(style::TEXT_MUTED),
                text("...").size(14).color(style::ACCENT_BLUE)
            ]
            .spacing(5)
        } else if let Some(e) = &self.error {
            row![
                container(text("!").size(12).color(style::BG_DARK))
                    .padding([2, 6])
                    .style(|_| container::Style {
                        background: Some(style::ACCENT_RED.into()),
                        border: Border {
                            radius: 10.0.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }),
                text(e.to_string()).size(14).color(style::ACCENT_RED)
            ]
            .spacing(8)
            .align_y(Alignment::Center)
        } else {
            row![
                container(Space::new().width(8).height(8)).style(|_| container::Style {
                    background: Some(style::ACCENT_GREEN.into()),
                    border: Border {
                        radius: 4.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
                text(format!("Last Update: {}", last_update))
                    .size(14)
                    .color(style::TEXT_MUTED)
            ]
            .spacing(8)
            .align_y(Alignment::Center)
        };

        row![
            text(match self.ui.current_view {
                ViewMode::Dashboard => "Dashboard",
                ViewMode::WeeklyPattern => "Weekly Heatmap",
                ViewMode::Insights => "Insights",
                ViewMode::DataRepair => "Data Repair",
            })
            .size(28)
            .color(style::TEXT_BRIGHT),
            Space::new().width(Length::Fill),
            status,
            Space::new().width(10),
            button(text("↻").size(18))
                .on_press(Message::RefreshNow)
                .padding(10)
                .style(|_, _| button::Style {
                    background: Some(style::BG_CARD.into()),
                    text_color: style::TEXT_BRIGHT,
                    border: Border {
                        radius: 8.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                })
        ]
        .align_y(Alignment::Center)
        .into()
    }

    // --- LOADING STATE HELPERS ---

    /// Start loading state with debounce tracking
    fn start_loading(&mut self) {
        if !self.ui.is_loading {
            self.ui.is_loading = true;
            self.ui.loading_started_at = Some(Instant::now());
        }
    }

    /// Stop loading state and clear debounce tracking
    fn stop_loading(&mut self) {
        self.ui.is_loading = false;
        self.ui.loading_started_at = None;
    }

    /// Check if loading indicator should be visible (after debounce threshold)
    fn should_show_loading(&self) -> bool {
        if !self.ui.is_loading {
            return false;
        }
        match self.ui.loading_started_at {
            Some(started) => started.elapsed().as_millis() >= LOADING_DEBOUNCE_MS as u128,
            None => true, // Show if no timestamp (shouldn't happen, but safe default)
        }
    }

    // --- MESSAGE HANDLERS ---

    /// Handle successful or failed fetch completion
    fn handle_fetch_completed(&mut self, result: Result<f64, AppError>) -> Task<Message> {
        self.stop_loading();
        match result {
            Ok(percentage) => {
                self.data.occupancy = Some(percentage);
                self.data.last_update = Some(self.clock.now_utc());
                self.error = None;
                self.ui.gauge_cache.clear();

                // Update predictions
                self.data.predictions =
                    analytics::calculate_predictions(&self.data.prediction_baseline);

                // Notifications
                let is_below = percentage < self.notifications.threshold;

                // Always refresh history AND analytics on new data
                // This ensures the view is always up to date, including at hour marks
                let mut tasks = vec![
                    Self::load_history(self.db.clone()),
                    Self::load_analytics(
                        self.db.clone(),
                        self.ui.analytics_range,
                        self.clock.clone(),
                    ),
                ];

                if self.notifications.enabled
                    && is_below
                    && !self.notifications.was_below_threshold
                {
                    let notifier = self.notifier.clone();
                    tasks.push(Task::perform(
                        async move {
                            let _ = notifier.notify(
                                "Hardy's Gym Monitor",
                                &format!("Gym is empty! {:.0}%", percentage),
                            );
                        },
                        |_| Message::NotificationSent,
                    ));
                }
                self.notifications.was_below_threshold = is_below;
                Task::batch(tasks)
            }
            Err(e) => {
                self.error = Some(e);
                Task::none()
            }
        }
    }

    /// Handle loaded insights data and compute analytics
    fn handle_insights_data_loaded(
        &mut self,
        current: Result<Vec<HourlyAverage>, AppError>,
        baseline: Result<Vec<HourlyAverage>, AppError>,
    ) -> Task<Message> {
        if let Ok(current_data) = current {
            // Calculate statistics
            self.data.stats = calculate_stats(&current_data);

            // Analyze days
            self.data.day_analysis = analyze_days(&current_data);

            // Find peak and quiet hours
            self.data.peak_hours = find_peak_hours(&current_data, 5);
            self.data.quiet_hours = find_quiet_hours(&current_data, 5);

            // Generate insights with optional baseline comparison
            let baseline_opt = baseline.ok();
            if let Some(ref bl) = baseline_opt {
                self.data.baseline_for_comparison = bl.clone();
                let comparison = compare_periods(bl, &current_data, ComparisonMode::WeekOverWeek);
                self.data.trend = Some(comparison.overall_trend);
                self.data.insights = generate_insights(&current_data, Some(bl));
            } else {
                self.data.insights = generate_insights(&current_data, None);
                self.data.trend = None;
            }
        }
        Task::none()
    }

    // --- LOGIC HELPERS ---
    /// Fetch the latest occupancy record from the database (read-only, no API calls).
    fn fetch_latest_from_db(db: Arc<Database>) -> Task<Message> {
        Task::perform(
            async move {
                let record = db.get_latest_record().await?;
                Ok(record.map(|r| r.percentage))
            },
            |r: Result<Option<f64>, anyhow::Error>| match r {
                Ok(Some(v)) => Message::FetchCompleted(Ok(v)),
                Ok(None) => Message::FetchCompleted(Ok(0.0)), // No data yet
                Err(e) => Message::FetchCompleted(Err(AppError::from_anyhow_db(e, "get_latest_record"))),
            },
        )
    }

    fn load_history(db: Arc<Database>) -> Task<Message> {
        Task::perform(
            async move { db.get_history(1).await },
            |r: Result<Vec<OccupancyLog>, anyhow::Error>| {
                Message::HistoryLoaded(r.map_err(|e| AppError::from_anyhow_db(e, "get_history")))
            },
        )
    }

    fn load_history_range(db: Arc<Database>, s: DateTime<Utc>, e: DateTime<Utc>) -> Task<Message> {
        Task::perform(
            async move { db.get_history_range(s, e).await },
            |r: Result<Vec<OccupancyLog>, anyhow::Error>| {
                Message::HistoryLoaded(r.map_err(|e| AppError::from_anyhow_db(e, "get_history_range")))
            },
        )
    }

    fn load_analytics(
        db: Arc<Database>,
        range: AnalyticsRange,
        clock: Arc<dyn Clock>,
    ) -> Task<Message> {
        let now = clock.now_utc();
        let days_since_monday = now.weekday().num_days_from_monday() as i64;
        let this_week_start =
            midnight_utc(now.date_naive() - ChronoDuration::days(days_since_monday));
        let start = match range {
            AnalyticsRange::ThisWeek => this_week_start,
            AnalyticsRange::Last2Weeks => this_week_start - ChronoDuration::weeks(1),
            AnalyticsRange::Last4Weeks => this_week_start - ChronoDuration::weeks(3),
            AnalyticsRange::Last8Weeks => this_week_start - ChronoDuration::weeks(7),
        };
        Task::perform(
            async move { db.get_averages_range(start, now).await },
            |r: Result<Vec<HourlyAverage>, anyhow::Error>| {
                Message::AnalyticsLoaded(r.map_err(|e| AppError::from_anyhow_db(e, "get_averages_range")))
            },
        )
    }

    fn load_prediction_baseline(
        db: Arc<Database>,
        days: i64,
        clock: Arc<dyn Clock>,
    ) -> Task<Message> {
        let now = clock.now_utc();
        Task::perform(
            async move {
                db.get_averages_range(now - ChronoDuration::days(days), now)
                    .await
            },
            |r: Result<Vec<HourlyAverage>, anyhow::Error>| {
                Message::PredictionBaselineLoaded(r.map_err(|e| AppError::from_anyhow_db(e, "get_prediction_baseline")))
            },
        )
    }

    fn load_insights_data(db: Arc<Database>, clock: Arc<dyn Clock>) -> Task<Message> {
        let now = clock.now_utc();
        let days_since_monday = now.weekday().num_days_from_monday() as i64;
        let this_week_start =
            midnight_utc(now.date_naive() - ChronoDuration::days(days_since_monday));

        // Current period: last 4 weeks
        let current_start = this_week_start - ChronoDuration::weeks(3);
        // Baseline: 4 weeks before the current period (for comparison)
        let baseline_start = current_start - ChronoDuration::weeks(4);
        let baseline_end = current_start;

        let db_clone = db.clone();
        Task::perform(
            async move {
                let current = db.get_averages_range(current_start, now).await;
                let baseline = db_clone
                    .get_averages_range(baseline_start, baseline_end)
                    .await;
                (current, baseline)
            },
            |(current, baseline): (
                Result<Vec<HourlyAverage>, anyhow::Error>,
                Result<Vec<HourlyAverage>, anyhow::Error>,
            )| {
                Message::InsightsDataLoaded {
                    current: current.map_err(|e| AppError::from_anyhow_db(e, "get_insights_current")),
                    baseline: baseline.map_err(|e| AppError::from_anyhow_db(e, "get_insights_baseline")),
                }
            },
        )
    }
}

// --- HELPER FUNCTIONS ---
fn parse_date(s: &str) -> Option<DateTime<Utc>> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .ok()
        .map(midnight_local_as_utc)
}
