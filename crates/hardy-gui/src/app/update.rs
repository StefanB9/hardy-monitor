//! Message handling.

use std::time::Duration;

use chrono::{DateTime, Utc};
use hardy_core::{
    alert::Alert,
    analytics::{
        ComparisonMode, analyze_days, calculate_stats, compare_periods, find_peak_hours,
        find_quiet_hours, generate_insights,
    },
    db::HourlyAverage,
    error::AppError,
    repair::DataRepairer,
};
use iced::{Task, window};
use muda::MenuEvent;
use tray_icon::TrayIconEvent;

use super::{HardyMonitorApp, Message, RepairPreset, ViewMode};
use crate::{freshness::Freshness, time_range::parse_date, widgets::heatmap::WeekGrid};

impl HardyMonitorApp {
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Tick => {
                self.refresh_forecasts();
                // The "now" marker moves.
                self.ui.chart_cache.clear();
                Task::none()
            }
            Message::NotificationSent => Task::none(),
            Message::FetchAlignmentComplete => {
                self.ui.is_poll_aligned = true;
                self.poll()
            }
            Message::FetchTick => {
                // Settings may change from the phone at any time.
                Task::batch([self.poll(), self.load_alert_settings()])
            }
            Message::RefreshNow => {
                self.start_loading();
                self.error = None;
                Task::batch([
                    self.fetch_latest(),
                    self.load_chart_history(),
                    self.load_analytics(),
                    self.load_insights_data(),
                    self.load_model_status(),
                    self.load_forecast_history(),
                ])
            }
            Message::FetchCompleted(result) => self.handle_fetch_completed(result),
            Message::HistoryLoaded(result) => {
                match result {
                    Ok(logs) => {
                        self.data.history = logs;
                        self.ui.chart_cache.clear();
                    }
                    Err(e) => self.error = Some(e),
                }
                Task::none()
            }
            Message::AnalyticsLoaded(result) => {
                match result {
                    Ok(data) => self.set_week_data(&data),
                    Err(e) => self.error = Some(e),
                }
                Task::none()
            }
            Message::InsightsDataLoaded { current, baseline } => {
                self.handle_insights_data_loaded(current, baseline);
                Task::none()
            }
            Message::WindowCloseRequested => {
                self.ui.is_window_visible = false;
                window::latest().and_then(|id| window::minimize(id, true))
            }
            Message::TrayCheck => self.handle_tray(),
            Message::NotificationThresholdChanged(_)
            | Message::NotificationThresholdReleased
            | Message::NotificationToggled(_)
            | Message::NotificationDurationSelected(_)
            | Message::AlertSettingsLoaded(_) => self.update_alerts(message),
            Message::SwitchView(_)
            | Message::SwitchAnalyticsRange(_)
            | Message::ChartRangeSelected(_)
            | Message::CustomStartChanged(_)
            | Message::CustomEndChanged(_)
            | Message::ApplyCustomRange => self.update_navigation(message),
            Message::ExportCsv
            | Message::ExportCompleted(_)
            | Message::ClearExportStatus
            | Message::RepairStartDateChanged(_)
            | Message::RepairEndDateChanged(_)
            | Message::RepairPresetSelected(_)
            | Message::StartRepairJob
            | Message::RepairCompleted(_) => self.update_maintenance(message),
            Message::TrainModelRequested
            | Message::ForecastHistoryLoaded(_)
            | Message::ModelStatusLoaded(_)
            | Message::ModelLoaded(_) => self.update_model(message),
        }
    }

    /// Alert controls and their stored settings.
    fn update_alerts(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::NotificationThresholdChanged(val) => {
                self.alerts.drag_threshold(val);
                self.ui.chart_cache.clear();
                Task::none()
            }
            Message::NotificationThresholdReleased => {
                let change = self.alerts.release_threshold(self.clock.now_utc());
                self.save_alert_settings(change)
            }
            Message::NotificationToggled(enabled) => {
                let change = self
                    .alerts
                    .toggle(enabled, self.clock.now_utc(), &self.schedule);
                self.save_alert_settings(change)
            }
            Message::NotificationDurationSelected(duration) => {
                let change =
                    self.alerts
                        .select_duration(duration, self.clock.now_utc(), &self.schedule);
                self.save_alert_settings(change)
            }
            Message::AlertSettingsLoaded(result) => {
                match result {
                    Ok(settings) => {
                        self.alerts.set_settings(settings);
                        self.ui.chart_cache.clear();
                    }
                    Err(e) => self.error = Some(e),
                }
                Task::none()
            }
            _ => Task::none(),
        }
    }

    /// View, range and date selection.
    fn update_navigation(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::SwitchView(mode) => {
                self.ui.current_view = mode;
                // Reload on entry so long-running sessions stay current.
                match mode {
                    ViewMode::Week => self.load_analytics(),
                    ViewMode::Insights => self.load_insights_data(),
                    ViewMode::Now | ViewMode::ModelData => Task::none(),
                }
            }
            Message::SwitchAnalyticsRange(range) => {
                self.ui.analytics_range = range;
                self.load_analytics()
            }
            Message::ChartRangeSelected(range) => {
                self.ui.chart_range = range;
                self.ui.chart_cache.clear();
                self.load_chart_history()
            }
            Message::CustomStartChanged(d) => {
                self.ui.custom_start = d;
                Task::none()
            }
            Message::CustomEndChanged(d) => {
                self.ui.custom_end = d;
                Task::none()
            }
            Message::ApplyCustomRange => {
                let valid = self
                    .ui
                    .chart_range
                    .window(
                        self.clock.now_utc(),
                        &self.schedule,
                        (&self.ui.custom_start, &self.ui.custom_end),
                    )
                    .is_some();
                if valid {
                    self.ui.chart_cache.clear();
                    self.load_chart_history()
                } else {
                    self.error = Some(AppError::validation(
                        "Enter dates as YYYY-MM-DD, start before end",
                    ));
                    Task::none()
                }
            }
            _ => Task::none(),
        }
    }

    /// Export and data repair.
    fn update_maintenance(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ExportCsv => {
                self.start_loading();
                self.export_status = Some("Exporting…".to_string());
                self.export_csv()
            }
            Message::ExportCompleted(result) => {
                self.stop_loading();
                self.export_status = Some(match result {
                    Ok(path) => format!("Saved to {path}"),
                    Err(e) => {
                        self.error = Some(e);
                        "Export failed".to_string()
                    }
                });
                Task::perform(
                    async { tokio::time::sleep(Duration::from_secs(5)).await },
                    |()| Message::ClearExportStatus,
                )
            }
            Message::ClearExportStatus => {
                self.export_status = None;
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
                self.select_repair_preset(preset);
                Task::none()
            }
            Message::StartRepairJob => self.start_repair(),
            Message::RepairCompleted(result) => {
                self.repair.is_running = false;
                self.repair.last_result = Some(result);
                Task::batch([self.load_chart_history(), self.load_analytics()])
            }
            _ => Task::none(),
        }
    }

    /// Forecast history, model status and retraining.
    fn update_model(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::TrainModelRequested => {
                tracing::info!("model retraining requested");
                self.request_retrain()
            }
            Message::ForecastHistoryLoaded(result) => {
                match result {
                    Ok(logs) => {
                        self.data.forecasting.add_history(&logs);
                        self.refresh_forecasts();
                    }
                    Err(e) => self.error = Some(e),
                }
                Task::none()
            }
            Message::ModelStatusLoaded(result) => match result {
                Ok((latest, state)) => {
                    self.data.forecasting.set_state(state);
                    match latest {
                        Some(info) if self.data.forecasting.is_new_model(Some(&info)) => {
                            self.load_model(info)
                        }
                        _ => Task::none(),
                    }
                }
                Err(e) => {
                    self.error = Some(e);
                    Task::none()
                }
            },
            Message::ModelLoaded(result) => {
                match result {
                    Ok((info, artifact)) => {
                        tracing::info!(id = info.id, trained_at = %info.trained_at, "loaded model");
                        self.data.forecasting.set_model(&info, artifact);
                        self.refresh_forecasts();
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "could not load stored model");
                        self.error = Some(e);
                    }
                }
                Task::none()
            }
            _ => Task::none(),
        }
    }

    /// Fetches the latest reading while open; clears it while closed.
    fn poll(&mut self) -> Task<Message> {
        if self.schedule.is_open(&self.clock.now_utc()) {
            self.start_loading();
            self.fetch_latest()
        } else {
            self.data.occupancy = None;
            self.ui.gauge_cache.clear();
            self.stop_loading();
            Task::none()
        }
    }

    fn handle_fetch_completed(
        &mut self,
        result: Result<Option<(DateTime<Utc>, f64)>, AppError>,
    ) -> Task<Message> {
        self.stop_loading();
        let now = self.clock.now_utc();
        let (taken_at, percentage) = match result {
            Ok(Some(reading)) => reading,
            Ok(None) => {
                self.data.last_update = Some(now);
                self.error = None;
                return Task::none();
            }
            Err(e) => {
                self.error = Some(e);
                return Task::none();
            }
        };
        let is_new = self.data.latest_reading_at != Some(taken_at);
        self.data.latest_reading_at = Some(taken_at);
        self.data.last_update = Some(now);
        self.error = None;
        self.ui.gauge_cache.clear();

        // An old value must not look live or trigger alerts.
        let live = self.freshness() == Freshness::Live;
        self.data.occupancy = (live && self.schedule.is_open(&now)).then_some(percentage);
        if !live {
            tracing::warn!(%taken_at, "newest reading is stale");
        }
        if !is_new {
            return Task::none();
        }

        let mut tasks = vec![
            self.load_forecast_history(),
            self.load_model_status(),
            self.load_chart_history(),
        ];

        // Desktop popups only; phone alerts come from the daemon.
        if let Some(alert) = live
            .then(|| self.alerts.observe(percentage, now, &self.schedule))
            .flatten()
        {
            let notifier = self.notifier.clone();
            tasks.push(Task::perform(
                async move {
                    if let Err(e) = notifier.notify(Alert::TITLE, &alert.body()).await {
                        tracing::warn!(error = %e, "desktop notification failed");
                    }
                },
                |()| Message::NotificationSent,
            ));
        }
        Task::batch(tasks)
    }

    fn set_week_data(&mut self, data: &[HourlyAverage]) {
        self.data.week_grid = WeekGrid::new(data, &self.schedule);
        self.data.week_days = analyze_days(data);
        self.ui.heatmap_cache.clear();
        self.ui.heatmap_tooltip_cache.clear();
    }

    fn handle_insights_data_loaded(
        &mut self,
        current: Result<Vec<HourlyAverage>, AppError>,
        baseline: Result<Vec<HourlyAverage>, AppError>,
    ) {
        let current = match current {
            Ok(current) => current,
            Err(e) => {
                self.error = Some(e);
                return;
            }
        };
        self.data.stats = calculate_stats(&current);
        self.data.peak_hours = find_peak_hours(&current, 5);
        self.data.quiet_hours = find_quiet_hours(&current, 5);
        match baseline {
            Ok(baseline) => {
                let comparison = compare_periods(&baseline, &current, ComparisonMode::WeekOverWeek);
                self.data.trend = Some(comparison.overall_trend);
                self.data.insights = generate_insights(&current, Some(&baseline));
            }
            Err(e) => {
                tracing::warn!(error = %e, "insights baseline unavailable");
                self.data.trend = None;
                self.data.insights = generate_insights(&current, None);
            }
        }
    }

    fn handle_tray(&mut self) -> Task<Message> {
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
        if !should_toggle {
            return Task::none();
        }
        self.ui.is_window_visible = !self.ui.is_window_visible;
        let target = self.ui.is_window_visible;
        window::latest().and_then(move |id| {
            if target {
                Task::batch([window::minimize(id, false), window::gain_focus(id)])
            } else {
                window::minimize(id, true)
            }
        })
    }

    fn select_repair_preset(&mut self, preset: RepairPreset) {
        let today = self
            .clock
            .now_utc()
            .with_timezone(&self.schedule.timezone())
            .date_naive();
        let start = match preset {
            RepairPreset::Last7Days => today - chrono::TimeDelta::days(7),
            RepairPreset::Last30Days => today - chrono::TimeDelta::days(30),
            RepairPreset::AllData => chrono::NaiveDate::from_ymd_opt(2020, 1, 1).unwrap_or(today),
        };
        self.repair.start_date = start.format("%Y-%m-%d").to_string();
        self.repair.end_date = today.format("%Y-%m-%d").to_string();
    }

    fn start_repair(&mut self) -> Task<Message> {
        if self.repair.is_running {
            return Task::none();
        }
        let (Some(start), Some(end)) = (
            parse_date(&self.repair.start_date),
            parse_date(&self.repair.end_date),
        ) else {
            self.error = Some(AppError::validation("Enter dates as YYYY-MM-DD"));
            return Task::none();
        };
        if start > end {
            self.error = Some(AppError::validation("Start date must be before end date"));
            return Task::none();
        }

        self.repair.is_running = true;
        self.repair.last_result = None;
        self.error = None;

        let repairer = DataRepairer::new(self.db.clone(), self.schedule.clone());
        Task::perform(
            async move { repairer.repair_date_range(start, end, None).await },
            |r| Message::RepairCompleted(r.map_err(|e| AppError::from_anyhow_db(e, "repair"))),
        )
    }

    fn refresh_forecasts(&mut self) {
        self.data
            .forecasting
            .refresh(&self.schedule, self.clock.now_utc());
        self.ui.chart_cache.clear();
    }
}
