//! Message handling.

use hardy_core::{accuracy::summarize, db::SchemaStatus, error::AppError};
use iced::{Task, window};

use super::{HardyMonitorApp, Message, ViewMode, loads::ErrorSource};
use crate::{
    forecasting::ModelOption,
    time_range::{ChartRange, Step, can_step_back, shift_custom},
    tray::TrayAction,
    views::schema_notice::SchemaGate,
};

impl HardyMonitorApp {
    /// Handles one message and returns the follow-up tasks.
    pub fn update(&mut self, message: Message) -> Task<Message> {
        if message.finishes_load() {
            self.ui.pending.finish();
        }
        if self.schema != SchemaGate::Ready {
            return self.update_blocked(message);
        }
        match message {
            Message::Tick => {
                // The chart's "now" marker is drawn per frame, so the cached
                // chart is only redrawn if its lines changed.
                self.refresh_forecasts();
                self.update_tray();
                Task::none()
            }
            // A late schema check after start changes nothing.
            Message::NotificationSent | Message::SchemaChecked(_) | Message::RetrySchemaCheck => {
                Task::none()
            }
            Message::FetchAlignmentComplete => {
                self.ui.is_poll_aligned = true;
                self.poll()
            }
            Message::FetchTick => {
                // Settings may change from the phone at any time.
                Task::batch([self.poll(), self.load_alert_settings()])
            }
            Message::RefreshNow => self.refresh_all(),
            Message::FirstReadingLoaded(result) => {
                if let Some(first) = self.errors.record(ErrorSource::FirstReading, result) {
                    self.data.first_reading = first;
                }
                Task::none()
            }
            Message::FetchCompleted(result) => self.handle_fetch_completed(result),
            Message::HistoryLoaded(id, result) => {
                self.handle_chart_loaded(id, result);
                Task::none()
            }
            Message::AnalyticsLoaded(id, result) => {
                if self.ui.analytics_requests.is_current(id)
                    && let Some(data) = self.errors.record(ErrorSource::Week, result)
                {
                    self.set_week_data(&data);
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
            Message::Tray(action) => self.handle_tray(action),
            Message::NotificationThresholdChanged(_)
            | Message::NotificationThresholdReleased
            | Message::NotificationToggled(_)
            | Message::NotificationDurationSelected(_)
            | Message::AlertSettingsLoaded(_) => self.update_alerts(message),
            Message::SwitchView(_)
            | Message::SwitchAnalyticsRange(_)
            | Message::ChartRangeSelected(_)
            | Message::LeaderboardSorted(_)
            | Message::LeaderboardFiltered(_)
            | Message::CustomStartChanged(_)
            | Message::CustomEndChanged(_)
            | Message::ApplyCustomRange
            | Message::ChartStep(_) => self.update_navigation(message),
            Message::DataPeriodSelected(_)
            | Message::DataScrolled { .. }
            | Message::DataLoaded(..) => self.update_data(message),
            Message::ExportCsv
            | Message::ExportCompleted(_)
            | Message::ClearExportStatus
            | Message::RepairStartDateChanged(_)
            | Message::RepairEndDateChanged(_)
            | Message::RepairPresetSelected(_)
            | Message::StartRepairJob
            | Message::RepairCompleted(_) => self.update_maintenance(message),
            Message::AccuracyLoaded(_)
            | Message::ForecastRunRequested
            | Message::ForecastHistoryLoaded(_)
            | Message::SuiteStatusLoaded(_)
            | Message::PublishedForecastLoaded(_)
            | Message::ForecastChoiceSelected(_)
            | Message::OverlayAdded(_)
            | Message::OverlayRemoved(_) => self.update_model(message),
            Message::PlanAveragesLoaded(_)
            | Message::PlanDaySelected(_)
            | Message::PlanLengthSelected(_)
            | Message::PlanRangeChanged(..) => self.update_plan(message),
        }
    }

    /// The Plan tab: its averages and choices.
    fn update_plan(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::PlanAveragesLoaded(result) => {
                if let Some(averages) = self.errors.record(ErrorSource::Plan, result) {
                    self.data.plan.set_averages(averages);
                }
            }
            Message::PlanDaySelected(index) => self.data.plan.select_day(index),
            Message::PlanLengthSelected(length) => self.data.plan.set_length(length),
            Message::PlanRangeChanged(kind, end, hour) => {
                self.data.plan.set_range(kind, end, hour);
            }
            _ => return Task::none(),
        }
        self.refresh_plan();
        Task::none()
    }

    /// Reloads everything shown; the Data tab's readings only while it is
    /// open (a week is thousands of rows).
    fn refresh_all(&mut self) -> Task<Message> {
        self.errors.clear_all();
        let data = if self.ui.current_view == ViewMode::Data {
            self.load_data_table()
        } else {
            Task::none()
        };
        Task::batch([
            self.fetch_latest(),
            self.load_chart_history(),
            self.load_analytics(),
            self.load_insights_data(),
            self.load_plan_averages(),
            self.load_suite_status(),
            self.load_published_forecast(),
            self.load_forecast_history(),
            self.load_first_reading(),
            data,
        ])
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
                if let Some(settings) = self.errors.record(ErrorSource::AlertSettings, result) {
                    self.alerts.set_settings(settings);
                    self.ui.chart_cache.clear();
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
                    ViewMode::Data => self.load_data_table(),
                    ViewMode::Insights => self.load_insights_data(),
                    ViewMode::Plan => self.load_plan_averages(),
                    ViewMode::ModelData => self.load_accuracy(),
                    ViewMode::Now => Task::none(),
                }
            }
            Message::SwitchAnalyticsRange(range) => {
                self.ui.analytics_range = range;
                self.load_analytics()
            }
            Message::LeaderboardSorted(key) => {
                self.ui.leaderboard_sort = self.ui.leaderboard_sort.clicked(key);
                Task::none()
            }
            Message::LeaderboardFiltered(family) => {
                self.ui.leaderboard_family = family;
                Task::none()
            }
            Message::ChartRangeSelected(range) => {
                self.ui.chart_range = range;
                self.ui.chart_offset = 0;
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
                        0,
                    )
                    .is_some();
                if valid {
                    self.errors.clear(ErrorSource::ChartRangeInput);
                    self.ui.chart_cache.clear();
                    self.load_chart_history()
                } else {
                    self.errors.raise(
                        ErrorSource::ChartRangeInput,
                        AppError::validation("Enter dates as YYYY-MM-DD, start before end"),
                    );
                    Task::none()
                }
            }
            Message::ChartStep(step) => self.step_chart(step),
            _ => Task::none(),
        }
    }

    /// Shows the previous or next period, within the data and up to now.
    /// Arrow keys arrive from every view; the Now view's chart and the Data
    /// tab's table move.
    fn step_chart(&mut self, step: Step) -> Task<Message> {
        match self.ui.current_view {
            ViewMode::Now => {}
            ViewMode::Data => return self.step_data(step),
            ViewMode::Plan => {
                self.data.plan.step_day(step);
                self.ui.plan_cache.clear();
                return Task::none();
            }
            ViewMode::Week | ViewMode::Insights | ViewMode::ModelData => return Task::none(),
        }
        let now = self.clock.now_utc();
        let range = self.ui.chart_range;
        let Some((start, end)) = range.window(
            now,
            &self.schedule,
            (&self.ui.custom_start, &self.ui.custom_end),
            self.ui.chart_offset,
        ) else {
            return Task::none();
        };
        let allowed = match step {
            Step::Back => can_step_back(start, self.data.first_reading),
            Step::Forward => range.can_step_forward(self.ui.chart_offset, end, now),
        };
        if !allowed {
            return Task::none();
        }
        if range == ChartRange::Custom {
            let Some((first, last)) =
                shift_custom(&self.ui.custom_start, &self.ui.custom_end, step)
            else {
                return Task::none();
            };
            self.ui.custom_start = first.format("%Y-%m-%d").to_string();
            self.ui.custom_end = last.format("%Y-%m-%d").to_string();
        } else {
            self.ui.chart_offset = match step {
                Step::Back => self.ui.chart_offset.saturating_add(1),
                Step::Forward => self.ui.chart_offset.saturating_sub(1),
            };
        }
        self.ui.chart_cache.clear();
        self.load_chart_history()
    }

    /// Forecast history, the published forecasts, which of them is shown,
    /// model status, accuracy and run requests.
    fn update_model(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::AccuracyLoaded(result) => {
                if let Some(rows) = self.errors.record(ErrorSource::Accuracy, result) {
                    self.data.accuracy = summarize(&rows);
                }
                Task::none()
            }
            Message::ForecastRunRequested => {
                tracing::info!("forecast run requested");
                self.request_forecast_run()
            }
            Message::ForecastHistoryLoaded(result) => {
                if let Some(logs) = self.errors.record(ErrorSource::ForecastHistory, result) {
                    self.data.forecasting.add_history(&logs);
                    self.refresh_forecasts();
                }
                Task::none()
            }
            Message::SuiteStatusLoaded(result) => {
                if let Some(status) = self.errors.record(ErrorSource::SuiteStatus, result) {
                    self.data.forecasting.set_status(status);
                }
                Task::none()
            }
            Message::PublishedForecastLoaded(result) => {
                if let Some(forecast) = self.errors.record(ErrorSource::PublishedForecast, result) {
                    self.data.forecasting.set_published(forecast);
                    self.refresh_forecasts();
                }
                Task::none()
            }
            Message::ForecastChoiceSelected(choice) => {
                self.data.forecasting.choose(choice);
                self.refresh_forecasts();
                Task::none()
            }
            Message::OverlayAdded(ModelOption(id)) => {
                self.data.forecasting.add_overlay(id);
                self.refresh_forecasts();
                Task::none()
            }
            Message::OverlayRemoved(ModelOption(id)) => {
                self.data.forecasting.remove_overlay(id);
                self.refresh_forecasts();
                Task::none()
            }
            _ => Task::none(),
        }
    }

    /// Until the schema matches: only the schema check, the tray and the
    /// window work; nothing touches the database.
    fn update_blocked(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::SchemaChecked(result) => {
                self.schema = match result {
                    Ok(SchemaStatus::Current) => SchemaGate::Ready,
                    Ok(status) => SchemaGate::Mismatch(status),
                    Err(e) => SchemaGate::CheckFailed(e.to_string()),
                };
                if self.schema == SchemaGate::Ready {
                    tracing::info!("database schema matches; starting");
                    self.start()
                } else {
                    tracing::warn!(schema = ?self.schema, "database schema does not match");
                    Task::none()
                }
            }
            Message::RetrySchemaCheck => {
                self.schema = SchemaGate::Checking;
                self.check_schema()
            }
            Message::FetchAlignmentComplete => {
                self.ui.is_poll_aligned = true;
                Task::none()
            }
            Message::Tray(action) => self.handle_tray(action),
            Message::WindowCloseRequested => {
                self.ui.is_window_visible = false;
                window::latest().and_then(|id| window::minimize(id, true))
            }
            _ => Task::none(),
        }
    }

    fn handle_tray(&mut self, action: TrayAction) -> Task<Message> {
        if action == TrayAction::Quit {
            std::process::exit(0);
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
}
