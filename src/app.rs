use std::{path::PathBuf, sync::Arc, time::Duration};

use chrono::{DateTime, Datelike, Duration as ChronoDuration, Local, NaiveDate, Utc};
use hardy_monitor::{
    analytics::{
        self, ComparisonMode, DayAnalysis, Insight, OccupancyStats, TrendDirection, analyze_days,
        calculate_stats, compare_periods, find_peak_hours, find_quiet_hours, generate_insights,
        midnight_local_as_utc, midnight_utc,
    },
    config::AppConfig,
    db::{Database, HourlyAverage, OccupancyLog},
    repair::DataRepairer,
    schedule::GymSchedule,
    style,
    traits::{Clock, Notifier},
    widgets::{gauge::GaugeWidget, heatmap::HeatmapWidget, history_chart::HistoryChart},
};
use iced::{
    Alignment, Border, Color, Element, Length, Shadow, Subscription, Task, Theme, Vector,
    widget::{
        Space, button,
        canvas::{Cache, Canvas},
        center, checkbox, column, container, row, scrollable, slider, stack, text, text_input,
    },
    window,
};
use muda::MenuEvent;
use thiserror::Error;
use tray_icon::{TrayIcon, TrayIconEvent};

/// Typed Application Errors
#[derive(Debug, Clone, Error)]
pub enum AppError {
    #[error("Network error: {0}")]
    Network(String),
    #[error("Database error: {0}")]
    Database(String),
    #[error("Data validation error: {0}")]
    Validation(String),
    #[error("IO error: {0}")]
    Io(String),
    #[error("Unexpected error: {0}")]
    Unknown(String),
}

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

struct UiState {
    is_loading: bool,
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
                    self.ui.is_loading = true;
                    Self::fetch_latest_from_db(self.db.clone())
                } else {
                    self.data.occupancy = None;
                    self.ui.is_loading = false;
                    Task::none()
                }
            }
            Message::FetchTick => {
                if self.schedule.is_open(&self.clock.now_local()) {
                    self.ui.is_loading = true;
                    Self::fetch_latest_from_db(self.db.clone())
                } else {
                    self.data.occupancy = None;
                    self.ui.is_loading = false;
                    Task::none()
                }
            }
            Message::RefreshNow => {
                self.ui.is_loading = true;
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
            Message::FetchCompleted(result) => {
                self.ui.is_loading = false;
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

                        // NEW: Always refresh history AND analytics on new data
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
                if let Ok(current_data) = current {
                    // Calculate statistics
                    self.data.stats = calculate_stats(&current_data);

                    // Analyze days
                    self.data.day_analysis = analyze_days(&current_data);

                    // Find peak and quiet hours
                    self.data.peak_hours = find_peak_hours(&current_data, 5);
                    self.data.quiet_hours = find_quiet_hours(&current_data, 5);

                    // Generate insights
                    let baseline_opt = baseline.ok();
                    if let Some(ref bl) = baseline_opt {
                        self.data.baseline_for_comparison = bl.clone();
                        let comparison =
                            compare_periods(bl, &current_data, ComparisonMode::WeekOverWeek);
                        self.data.trend = Some(comparison.overall_trend);
                        self.data.insights = generate_insights(&current_data, Some(bl));
                    } else {
                        self.data.insights = generate_insights(&current_data, None);
                        self.data.trend = None;
                    }
                }
                Task::none()
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
                    self.error = Some(AppError::Validation("Invalid date format".into()));
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
                self.ui.is_loading = true;
                self.export.status = Some("Exporting...".to_string());
                let db = self.db.clone();
                let clock = self.clock.clone();
                Task::perform(
                    async move {
                        let logs = db
                            .get_history(365 * 10)
                            .await
                            .map_err(|e| AppError::Database(e.to_string()))?;
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
                                    .map_err(|e| AppError::Io(e.to_string()))?;
                                for log in logs {
                                    wtr.serialize(log)
                                        .map_err(|e| AppError::Io(e.to_string()))?;
                                }
                                wtr.flush().map_err(|e| AppError::Io(e.to_string()))?;
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
                self.ui.is_loading = false;
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
                        self.error = Some(AppError::Validation("Invalid start date".into()));
                        return Task::none();
                    }
                };
                let end = match parse_date(&self.repair.end_date) {
                    Some(d) => d.date_naive(),
                    None => {
                        self.error = Some(AppError::Validation("Invalid end date".into()));
                        return Task::none();
                    }
                };

                if start > end {
                    self.error = Some(AppError::Validation(
                        "Start date must be before end date".into(),
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
                        Err(e) => Message::RepairCompleted(Err(AppError::Database(e.to_string()))),
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
            ViewMode::Dashboard => self.view_dashboard(),
            ViewMode::WeeklyPattern => self.view_weekly_pattern(),
            ViewMode::Insights => self.view_insights(),
            ViewMode::DataRepair => self.view_data_repair(),
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

        let status = if self.ui.is_loading {
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

    fn view_dashboard(&self) -> Element<'_, Message> {
        let low_threshold = self.config.thresholds.low_occupancy_percent;
        let high_threshold = self.config.thresholds.high_occupancy_percent;

        let gauge = Canvas::new(GaugeWidget {
            percentage: self.data.occupancy.unwrap_or(0.0),
            is_open: self.schedule.is_open(&Local::now()),
            low_threshold,
            high_threshold,
            cache: &self.ui.gauge_cache,
        })
        .width(Length::Fixed(220.0))
        .height(Length::Fixed(220.0));

        let is_checked = self.notifications.enabled;
        let active_rail = if is_checked {
            style::ACCENT_BLUE
        } else {
            style::TEXT_MUTED
        };
        let handle_bg = if is_checked {
            style::ACCENT_BLUE
        } else {
            style::TEXT_MUTED
        };
        let text_color = if is_checked {
            style::TEXT_BRIGHT
        } else {
            style::TEXT_MUTED
        };

        let slider_section: Element<'_, Message> = column![
            row![
                text("Threshold:").size(12).color(style::TEXT_MUTED),
                text(format!("{:.0}%", self.notifications.threshold))
                    .size(12)
                    .color(text_color)
            ]
            .spacing(5),
            slider(
                0.0..=60.0,
                self.notifications.threshold,
                Message::NotificationThresholdChanged
            )
            .step(5.0)
            .style(move |_: &Theme, _| slider::Style {
                rail: slider::Rail {
                    backgrounds: (active_rail.into(), style::BG_DARK.into()),
                    width: 4.0,
                    border: Border {
                        radius: 2.0.into(),
                        ..Default::default()
                    }
                },
                handle: slider::Handle {
                    shape: slider::HandleShape::Circle { radius: 8.0 },
                    background: handle_bg.into(),
                    border_width: 0.0,
                    border_color: Color::TRANSPARENT
                }
            })
        ]
        .spacing(5)
        .into();

        let notify_controls = column![
            row![
                checkbox(is_checked)
                    .on_toggle(Message::NotificationToggled)
                    .size(14)
                    .style(move |_theme, _status| checkbox::Style {
                        icon_color: style::TEXT_BRIGHT,
                        background: if is_checked {
                            style::ACCENT_BLUE.into()
                        } else {
                            style::BG_DARK.into()
                        },
                        border: Border {
                            radius: 4.0.into(),
                            width: 1.0,
                            color: style::STROKE_DIM
                        },
                        text_color: None,
                    }),
                text("Notify when empty").size(14).color(if is_checked {
                    style::TEXT_BRIGHT
                } else {
                    style::TEXT_MUTED
                })
            ]
            .spacing(8)
            .align_y(Alignment::Center),
            slider_section
        ]
        .spacing(10)
        .max_width(220);

        let current_card = card_container(column![
            text("Current Status").size(16).color(style::TEXT_MUTED),
            Space::new().height(10),
            center(gauge),
            Space::new().height(20),
            notify_controls
        ]);

        let rec_content = if let Some((hour, avg)) = self.data.best_time_today {
            column![
                text(format!("Best time on {}s", Local::now().format("%A")))
                    .size(16)
                    .color(style::TEXT_MUTED),
                Space::new().height(20),
                text(format!("{:02}:00", hour))
                    .size(36)
                    .color(style::ACCENT_CYAN),
                Space::new().height(10),
                container(
                    text(format!("~{:.0}% load", avg))
                        .size(14)
                        .color(style::BG_DARK)
                )
                .padding([6, 12])
                .style(|_| container::Style {
                    background: Some(style::ACCENT_CYAN.into()),
                    border: Border {
                        radius: 12.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                })
            ]
            .align_x(Alignment::Center)
        } else {
            column![
                text("Best Time Today").size(16).color(style::TEXT_MUTED),
                Space::new().height(20),
                text("Collecting Data...").color(style::TEXT_MUTED)
            ]
            .align_x(Alignment::Center)
        };

        let top_row = row![current_card, card_container(center(rec_content))]
            .spacing(20)
            .height(Length::Fixed(350.0));

        let controls = row![
            preset_btn("Today", 1, self.ui.history_days_preset),
            preset_btn("7D", 7, self.ui.history_days_preset),
            preset_btn("30D", 30, self.ui.history_days_preset),
            Space::new().width(20),
            styled_input(
                &self.ui.history_start_date,
                Message::HistoryStartDateChanged
            ),
            text("-").color(style::TEXT_MUTED),
            styled_input(&self.ui.history_end_date, Message::HistoryEndDateChanged),
            button(text("Go").size(12))
                .on_press(Message::ApplyDateRange)
                .padding([8, 12])
                .style(primary_btn_style),
            Space::new().width(10),
            button(text("Export CSV").size(12))
                .on_press(Message::ExportCsv)
                .padding([8, 12])
                .style(secondary_btn_style)
        ]
        .spacing(10)
        .align_y(Alignment::Center);

        // Use local time for chart boundaries so "Today" means local today
        let (chart_start, chart_end) = if let Some(days) = self.ui.history_days_preset {
            let local_today = Local::now().date_naive();
            let end_aligned = midnight_local_as_utc(local_today + ChronoDuration::days(1));
            let start_aligned = midnight_local_as_utc(local_today + ChronoDuration::days(1 - days));
            (start_aligned, end_aligned)
        } else {
            match (
                parse_date(&self.ui.history_start_date),
                parse_date(&self.ui.history_end_date),
            ) {
                (Some(s), Some(e)) => {
                    if s == e {
                        (s, s + ChronoDuration::days(1))
                    } else {
                        (s, e)
                    }
                }
                _ => (Utc::now() - ChronoDuration::days(1), Utc::now()),
            }
        };

        let chart = Canvas::new(HistoryChart {
            history: &self.data.history,
            predictions: &self.data.predictions,
            range_start: chart_start,
            range_end: chart_end,
            cache: &self.ui.chart_cache,
        })
        .width(Length::Fill)
        .height(Length::Fill);

        // CRITICAL FIX: Use Element::from() to help type inference
        let chart_element = Element::from(chart).map(|_| Message::ChartInteraction);

        column![
            top_row,
            card_container(column![
                row![
                    text("Occupancy Trends").size(16).color(style::TEXT_MUTED),
                    Space::new().width(Length::Fill),
                    controls
                ]
                .align_y(Alignment::Center),
                Space::new().height(20),
                chart_element
            ])
            .height(Length::Fill)
        ]
        .spacing(20)
        .into()
    }

    fn view_weekly_pattern(&self) -> Element<'_, Message> {
        let range_btn = |label: &str, range: AnalyticsRange| {
            let active = self.ui.analytics_range == range;
            button(text(label.to_string()).size(14))
                .on_press(Message::SwitchAnalyticsRange(range))
                .padding([8, 16])
                .style(move |_, _| {
                    if active {
                        primary_btn_style(&Theme::Dark, iced::widget::button::Status::Active)
                    } else {
                        secondary_btn_style(&Theme::Dark, iced::widget::button::Status::Active)
                    }
                })
        };
        let controls = row![
            range_btn("This Week", AnalyticsRange::ThisWeek),
            range_btn("Last 2 Weeks", AnalyticsRange::Last2Weeks),
            range_btn("Last 4 Weeks", AnalyticsRange::Last4Weeks),
            range_btn("Last 8 Weeks", AnalyticsRange::Last8Weeks)
        ]
        .spacing(10);

        let heatmap = Canvas::new(HeatmapWidget {
            data: &self.data.analytics_data,
            cache: &self.ui.heatmap_cache,
            tooltip_cache: &self.ui.heatmap_tooltip_cache,
        })
        .width(Length::Fill)
        .height(Length::Fill);

        // IMPORTANT: We need to capture the mouse interaction to trigger tooltips
        // FIX: Explicitly type the closure arg as `()` to satisfy compiler inference.
        let heatmap_element = Element::from(heatmap).map(|_: ()| Message::ChartInteraction);

        let legend_item = |color: Color, label: &str| {
            row![
                container(Space::new().width(12).height(12)).style(move |_| container::Style {
                    background: Some(color.into()),
                    border: Border {
                        radius: 2.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
                text(label.to_string()).size(12).color(style::TEXT_MUTED)
            ]
            .spacing(6)
            .align_y(Alignment::Center)
        };
        let legend = row![
            legend_item(style::ACCENT_GREEN, "Low"),
            legend_item(style::ACCENT_ORANGE, "Busy"),
            legend_item(style::ACCENT_RED, "Full")
        ]
        .spacing(15);

        let mut row_content = row![].spacing(15);
        for (idx, day_name) in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            .iter()
            .enumerate()
        {
            if let Some(b) = self
                .data
                .analytics_data
                .iter()
                .filter(|d| d.weekday == idx as i32)
                .min_by(|a, b| {
                    a.avg_percentage
                        .partial_cmp(&b.avg_percentage)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                row_content = row_content.push(
                    column![
                        text(day_name.to_string()).size(12).color(style::TEXT_MUTED),
                        text(format!("{:02}:00", b.hour))
                            .size(14)
                            .color(style::ACCENT_CYAN)
                    ]
                    .spacing(2),
                );
            }
        }

        card_container(column![
            row![
                text("Weekly Occupancy Heatmap")
                    .size(16)
                    .color(style::TEXT_MUTED),
                Space::new().width(Length::Fill),
                controls
            ]
            .align_y(Alignment::Center),
            Space::new().height(20),
            heatmap_element,
            Space::new().height(20),
            row![
                container(row_content),
                Space::new().width(Length::Fill),
                legend
            ]
            .align_y(Alignment::Center)
        ])
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    fn view_insights(&self) -> Element<'_, Message> {
        // Trend card
        let trend_card = {
            let (trend_icon, trend_text, trend_color) = match self.data.trend {
                Some(TrendDirection::Increasing) => ("📈", "Getting Busier", style::ACCENT_RED),
                Some(TrendDirection::Decreasing) => ("📉", "Getting Quieter", style::ACCENT_GREEN),
                Some(TrendDirection::Stable) => ("➡️", "Staying Stable", style::ACCENT_CYAN),
                Some(TrendDirection::Insufficient) | None => {
                    ("❓", "Collecting Data", style::TEXT_MUTED)
                }
            };

            card_container(column![
                text("Overall Trend").size(14).color(style::TEXT_MUTED),
                Space::new().height(15),
                row![
                    text(trend_icon).size(32),
                    Space::new().width(15),
                    column![
                        text(trend_text).size(20).color(trend_color),
                        text("vs previous 4 weeks")
                            .size(12)
                            .color(style::TEXT_MUTED),
                    ]
                ]
                .align_y(Alignment::Center)
            ])
            .width(Length::FillPortion(1))
        };

        // Statistics card
        let stats_card = if let Some(ref stats) = self.data.stats {
            let consistency = if stats.coefficient_of_variation < 0.3 {
                ("Very Predictable", style::ACCENT_GREEN)
            } else if stats.coefficient_of_variation < 0.5 {
                ("Moderately Predictable", style::ACCENT_ORANGE)
            } else {
                ("Highly Variable", style::ACCENT_RED)
            };

            card_container(column![
                text("Statistics").size(14).color(style::TEXT_MUTED),
                Space::new().height(15),
                row![
                    column![
                        text("Average").size(12).color(style::TEXT_MUTED),
                        text(format!("{:.1}%", stats.mean))
                            .size(24)
                            .color(style::TEXT_BRIGHT),
                    ],
                    Space::new().width(30),
                    column![
                        text("Range").size(12).color(style::TEXT_MUTED),
                        text(format!("{:.0}% - {:.0}%", stats.min, stats.max))
                            .size(18)
                            .color(style::TEXT_BRIGHT),
                    ],
                ]
                .align_y(Alignment::End),
                Space::new().height(15),
                row![
                    text("Consistency: ").size(12).color(style::TEXT_MUTED),
                    text(consistency.0).size(12).color(consistency.1),
                ]
            ])
            .width(Length::FillPortion(1))
        } else {
            card_container(column![
                text("Statistics").size(14).color(style::TEXT_MUTED),
                Space::new().height(20),
                text("Loading...").color(style::TEXT_MUTED),
            ])
            .width(Length::FillPortion(1))
        };

        // Peak hours card
        let peak_card = card_container(column![
            text("Busiest Times").size(14).color(style::TEXT_MUTED),
            Space::new().height(15),
            {
                let mut peak_col = column![].spacing(8);
                for (weekday, hour, pct) in self.data.peak_hours.iter().take(5) {
                    peak_col = peak_col.push(
                        row![
                            container(text(format!("{:.0}%", pct)).size(12).color(style::BG_DARK))
                                .padding([4, 8])
                                .style(|_| container::Style {
                                    background: Some(style::ACCENT_RED.into()),
                                    border: Border {
                                        radius: 4.0.into(),
                                        ..Default::default()
                                    },
                                    ..Default::default()
                                }),
                            Space::new().width(10),
                            text(format!(
                                "{} {:02}:00",
                                analytics::weekday_short(*weekday),
                                hour
                            ))
                            .size(14)
                            .color(style::TEXT_BRIGHT),
                        ]
                        .align_y(Alignment::Center),
                    );
                }
                peak_col
            }
        ])
        .width(Length::FillPortion(1));

        // Quiet hours card
        let quiet_card = card_container(column![
            text("Quietest Times").size(14).color(style::TEXT_MUTED),
            Space::new().height(15),
            {
                let mut quiet_col = column![].spacing(8);
                for (weekday, hour, pct) in self.data.quiet_hours.iter().take(5) {
                    quiet_col = quiet_col.push(
                        row![
                            container(text(format!("{:.0}%", pct)).size(12).color(style::BG_DARK))
                                .padding([4, 8])
                                .style(|_| container::Style {
                                    background: Some(style::ACCENT_GREEN.into()),
                                    border: Border {
                                        radius: 4.0.into(),
                                        ..Default::default()
                                    },
                                    ..Default::default()
                                }),
                            Space::new().width(10),
                            text(format!(
                                "{} {:02}:00",
                                analytics::weekday_short(*weekday),
                                hour
                            ))
                            .size(14)
                            .color(style::TEXT_BRIGHT),
                        ]
                        .align_y(Alignment::Center),
                    );
                }
                quiet_col
            }
        ])
        .width(Length::FillPortion(1));

        // Day analysis card
        let days_card = card_container(column![
            text("Daily Patterns").size(14).color(style::TEXT_MUTED),
            Space::new().height(15),
            {
                let mut days_row = row![].spacing(30); // Increased spacing
                for day in &self.data.day_analysis {
                    if day.sample_count > 0 {
                        // Increased multiplier for visibility in full-width view
                        let bar_height = (day.avg_occupancy * 1.5).max(5.0);
                        let color = if day.avg_occupancy < 40.0 {
                            style::ACCENT_GREEN
                        } else if day.avg_occupancy < 60.0 {
                            style::ACCENT_ORANGE
                        } else {
                            style::ACCENT_RED
                        };

                        days_row = days_row.push(
                            column![
                                container(
                                    Space::new()
                                        .width(30)
                                        .height(Length::Fixed(bar_height as f32))
                                )
                                .style(move |_| container::Style {
                                    background: Some(color.into()),
                                    border: Border {
                                        radius: 4.0.into(),
                                        ..Default::default()
                                    },
                                    ..Default::default()
                                }),
                                Space::new().height(8),
                                text(&day.day_name[..3]).size(12).color(style::TEXT_MUTED),
                                text(format!("{:.0}%", day.avg_occupancy))
                                    .size(12)
                                    .color(style::TEXT_BRIGHT),
                            ]
                            .align_x(Alignment::Center),
                        );
                    }
                }
                container(days_row)
                    .width(Length::Fill)
                    .align_x(Alignment::Center)
            }
        ])
        .width(Length::Fill);

        // Insights list
        let insights_card = card_container(column![
            text("Key Insights").size(14).color(style::TEXT_MUTED),
            Space::new().height(15),
            {
                let mut insights_col = column![].spacing(12);
                for insight in self.data.insights.iter().take(6) {
                    let importance_color = match insight.importance {
                        5 => style::ACCENT_GREEN,
                        4 => style::ACCENT_CYAN,
                        3 => style::ACCENT_ORANGE,
                        _ => style::TEXT_MUTED,
                    };

                    insights_col = insights_col.push(
                        container(column![
                            row![
                                container(
                                    text(format!("{}", insight.importance))
                                        .size(10)
                                        .color(style::BG_DARK)
                                )
                                .padding([2, 6])
                                .style(move |_| container::Style {
                                    background: Some(importance_color.into()),
                                    border: Border {
                                        radius: 8.0.into(),
                                        ..Default::default()
                                    },
                                    ..Default::default()
                                }),
                                Space::new().width(10),
                                text(&insight.title).size(14).color(style::TEXT_BRIGHT),
                            ]
                            .align_y(Alignment::Center),
                            Space::new().height(4),
                            text(&insight.description).size(12).color(style::TEXT_MUTED),
                        ])
                        .padding(12)
                        .style(|_| container::Style {
                            background: Some(style::BG_DARK.into()),
                            border: Border {
                                radius: 8.0.into(),
                                ..Default::default()
                            },
                            ..Default::default()
                        }),
                    );
                }

                if self.data.insights.is_empty() {
                    insights_col = insights_col.push(
                        text("No insights yet. Keep collecting data!")
                            .size(14)
                            .color(style::TEXT_MUTED),
                    );
                }

                insights_col
            }
        ])
        .width(Length::Fill);

        // Revised Layout using full width and columns
        let content = column![
            // Row 1: High Level Stats
            row![trend_card, stats_card]
                .spacing(20)
                .height(Length::Fixed(160.0)),
            Space::new().height(20),
            // Row 2: Daily Patterns (Full Width)
            days_card,
            Space::new().height(20),
            // Row 3: Hourly Analysis (Side by Side)
            row![peak_card, quiet_card].spacing(20),
            Space::new().height(20),
            // Row 4: Detailed Text Insights
            insights_card,
        ]
        .padding(10); // Add some internal padding

        // Wrap in scrollable to handle smaller screens or extensive data
        scrollable(content)
            .height(Length::Fill)
            .width(Length::Fill)
            .into()
    }

    fn view_data_repair(&self) -> Element<'_, Message> {
        let preset_btn = |label: &str, preset: RepairPreset| {
            button(text(label.to_string()).size(12))
                .on_press(Message::RepairPresetSelected(preset))
                .padding([6, 12])
                .style(secondary_btn_style)
        };

        let date_inputs = row![
            styled_input(&self.repair.start_date, Message::RepairStartDateChanged),
            text("to").color(style::TEXT_MUTED).size(14),
            styled_input(&self.repair.end_date, Message::RepairEndDateChanged),
        ]
        .spacing(10)
        .align_y(Alignment::Center);

        let presets = row![
            preset_btn("Last 7 days", RepairPreset::Last7Days),
            preset_btn("Last 30 days", RepairPreset::Last30Days),
            preset_btn("All data", RepairPreset::AllData),
        ]
        .spacing(10);

        let start_button = if self.repair.is_running {
            button(text("Running...").size(14))
                .padding([12, 24])
                .style(|_, _| button::Style {
                    background: Some(style::TEXT_MUTED.into()),
                    text_color: style::BG_DARK,
                    border: Border {
                        radius: 8.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                })
        } else {
            button(text("Start Repair").size(14))
                .on_press(Message::StartRepairJob)
                .padding([12, 24])
                .style(primary_btn_style)
        };

        let progress_section: Element<'_, Message> = if self.repair.is_running {
            if let Some(ref progress) = self.repair.progress {
                let pct = if progress.total_days > 0 {
                    (progress.processed_days as f32 / progress.total_days as f32) * 100.0
                } else {
                    0.0
                };
                column![
                    text(format!(
                        "Processing: {} (Day {} of {})",
                        progress.current_day,
                        progress.processed_days + 1,
                        progress.total_days
                    ))
                    .size(14)
                    .color(style::TEXT_MUTED),
                    Space::new().height(10),
                    container(
                        container(
                            Space::new()
                                .width(Length::FillPortion((pct as u16).max(1)))
                                .height(8)
                        )
                        .style(|_| container::Style {
                            background: Some(style::ACCENT_BLUE.into()),
                            border: Border {
                                radius: 4.0.into(),
                                ..Default::default()
                            },
                            ..Default::default()
                        })
                    )
                    .width(Length::Fill)
                    .style(|_| container::Style {
                        background: Some(style::BG_DARK.into()),
                        border: Border {
                            radius: 4.0.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    })
                ]
                .into()
            } else {
                text("Starting repair job...")
                    .size(14)
                    .color(style::TEXT_MUTED)
                    .into()
            }
        } else {
            Space::new().height(0).into()
        };

        let result_section: Element<'_, Message> = if let Some(ref result) = self.repair.last_result
        {
            match result {
                Ok(summary) => card_container(column![
                    text("Last Repair Results")
                        .size(16)
                        .color(style::ACCENT_GREEN),
                    Space::new().height(15),
                    row![
                        text("Days processed:").size(14).color(style::TEXT_MUTED),
                        Space::new().width(10),
                        text(summary.days_processed.to_string())
                            .size(14)
                            .color(style::TEXT_BRIGHT),
                    ],
                    Space::new().height(5),
                    row![
                        text("Gaps filled:").size(14).color(style::TEXT_MUTED),
                        Space::new().width(10),
                        text(summary.gaps_filled.to_string())
                            .size(14)
                            .color(style::ACCENT_CYAN),
                    ],
                    Space::new().height(5),
                    row![
                        text("Records zeroed:").size(14).color(style::TEXT_MUTED),
                        Space::new().width(10),
                        text(summary.records_zeroed.to_string())
                            .size(14)
                            .color(style::ACCENT_ORANGE),
                    ],
                    Space::new().height(5),
                    row![
                        text("End entries added:").size(14).color(style::TEXT_MUTED),
                        Space::new().width(10),
                        text(summary.end_entries_added.to_string())
                            .size(14)
                            .color(style::TEXT_BRIGHT),
                    ],
                ])
                .into(),
                Err(e) => card_container(column![
                    text("Repair Failed").size(16).color(style::ACCENT_RED),
                    Space::new().height(10),
                    text(e.to_string()).size(14).color(style::TEXT_MUTED),
                ])
                .into(),
            }
        } else {
            Space::new().height(0).into()
        };

        let description = column![
            text("Repair occupancy data by:")
                .size(14)
                .color(style::TEXT_MUTED),
            Space::new().height(8),
            row![
                text("•").color(style::ACCENT_CYAN),
                Space::new().width(8),
                text("Filling gaps up to 5 minutes with linear interpolation")
                    .size(13)
                    .color(style::TEXT_MUTED),
            ],
            Space::new().height(4),
            row![
                text("•").color(style::ACCENT_CYAN),
                Space::new().width(8),
                text("Setting values outside opening hours to 0")
                    .size(13)
                    .color(style::TEXT_MUTED),
            ],
            Space::new().height(4),
            row![
                text("•").color(style::ACCENT_CYAN),
                Space::new().width(8),
                text("Adding end-of-day closure entries")
                    .size(13)
                    .color(style::TEXT_MUTED),
            ],
        ];

        card_container(column![
            text("Select Date Range").size(16).color(style::TEXT_BRIGHT),
            Space::new().height(20),
            date_inputs,
            Space::new().height(15),
            presets,
            Space::new().height(25),
            description,
            Space::new().height(25),
            start_button,
            Space::new().height(20),
            progress_section,
            Space::new().height(20),
            result_section,
        ])
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
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
                Err(e) => Message::FetchCompleted(Err(AppError::Database(e.to_string()))),
            },
        )
    }

    fn load_history(db: Arc<Database>) -> Task<Message> {
        Task::perform(
            async move { db.get_history(1).await },
            |r: Result<Vec<OccupancyLog>, anyhow::Error>| {
                Message::HistoryLoaded(r.map_err(|e| AppError::Database(e.to_string())))
            },
        )
    }

    fn load_history_range(db: Arc<Database>, s: DateTime<Utc>, e: DateTime<Utc>) -> Task<Message> {
        Task::perform(
            async move { db.get_history_range(s, e).await },
            |r: Result<Vec<OccupancyLog>, anyhow::Error>| {
                Message::HistoryLoaded(r.map_err(|e| AppError::Database(e.to_string())))
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
                Message::AnalyticsLoaded(r.map_err(|e| AppError::Database(e.to_string())))
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
                Message::PredictionBaselineLoaded(r.map_err(|e| AppError::Database(e.to_string())))
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
                    current: current.map_err(|e| AppError::Database(e.to_string())),
                    baseline: baseline.map_err(|e| AppError::Database(e.to_string())),
                }
            },
        )
    }
}

// --- HELPER FUNCTIONS ---
fn card_container<'a>(
    content: impl Into<Element<'a, Message>>,
) -> container::Container<'a, Message> {
    container(content).padding(24).style(|_| container::Style {
        background: Some(style::BG_CARD.into()),
        border: Border {
            color: Color::TRANSPARENT,
            width: 0.0,
            radius: 16.0.into(),
        },
        shadow: Shadow {
            color: Color::from_rgba(0.0, 0.0, 0.0, 0.3),
            offset: Vector::new(0.0, 4.0),
            blur_radius: 10.0,
        },
        ..Default::default()
    })
}

fn styled_input(
    val: &str,
    on_change: impl Fn(String) -> Message + 'static,
) -> Element<'_, Message> {
    text_input("YYYY-MM-DD", val)
        .on_input(on_change)
        .padding(8)
        .width(Length::Fixed(110.0))
        .size(12)
        .style(|_, status| {
            let border_color = if matches!(status, iced::widget::text_input::Status::Focused { .. })
            {
                style::ACCENT_BLUE
            } else {
                style::STROKE_DIM
            };
            text_input::Style {
                background: style::BG_DARK.into(),
                border: Border {
                    color: border_color,
                    width: 1.0,
                    radius: 6.0.into(),
                },
                icon: style::TEXT_MUTED,
                placeholder: style::TEXT_MUTED,
                value: style::TEXT_BRIGHT,
                selection: style::ACCENT_BLUE,
            }
        })
        .into()
}

fn preset_btn(label: &str, days: i64, current: Option<i64>) -> Element<'_, Message> {
    let active = current == Some(days);
    button(text(label.to_string()).size(12))
        .on_press(Message::HistoryPresetSelected(days))
        .padding([6, 12])
        .style(move |_, _| {
            if active {
                primary_btn_style(&Theme::Dark, iced::widget::button::Status::Active)
            } else {
                secondary_btn_style(&Theme::Dark, iced::widget::button::Status::Active)
            }
        })
        .into()
}

fn primary_btn_style(_: &Theme, _: iced::widget::button::Status) -> button::Style {
    button::Style {
        background: Some(style::ACCENT_BLUE.into()),
        text_color: style::BG_DARK,
        border: Border {
            radius: 6.0.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn secondary_btn_style(_: &Theme, _: iced::widget::button::Status) -> button::Style {
    button::Style {
        background: Some(style::BG_DARK.into()),
        text_color: style::TEXT_BRIGHT,
        border: Border {
            radius: 6.0.into(),
            color: style::STROKE_DIM,
            width: 1.0,
        },
        ..Default::default()
    }
}

fn parse_date(s: &str) -> Option<DateTime<Utc>> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .ok()
        .map(midnight_local_as_utc)
}
