//! Message handling.

use hardy_core::{accuracy::summarize, db::SchemaStatus, error::AppError};
use iced::{Task, window};
use muda::MenuEvent;
use tray_icon::TrayIconEvent;

use super::{HardyMonitorApp, Message, ViewMode, loads::ErrorSource};
use crate::views::schema_notice::SchemaGate;

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
                self.refresh_forecasts();
                // The "now" marker moves.
                self.ui.chart_cache.clear();
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
            Message::RefreshNow => {
                self.errors.clear_all();
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
            Message::HistoryLoaded(id, result) => {
                // A newer load for another range is on its way.
                if self.ui.chart_requests.is_current(id)
                    && let Some(logs) = self.errors.record(ErrorSource::Chart, result)
                {
                    self.data.history = logs;
                    self.ui.chart_cache.clear();
                }
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
            Message::AccuracyLoaded(result) => {
                if let Some(rows) = self.errors.record(ErrorSource::Accuracy, result) {
                    self.data.accuracy = summarize(&rows);
                }
                Task::none()
            }
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
                    ViewMode::Insights => self.load_insights_data(),
                    ViewMode::ModelData => self.load_accuracy(),
                    ViewMode::Now => Task::none(),
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
                if let Some(logs) = self.errors.record(ErrorSource::ForecastHistory, result) {
                    self.data.forecasting.add_history(&logs);
                    self.refresh_forecasts();
                }
                Task::none()
            }
            Message::ModelStatusLoaded(result) => {
                let Some((latest, state)) = self.errors.record(ErrorSource::ModelStatus, result)
                else {
                    return Task::none();
                };
                self.data.forecasting.set_state(state);
                match latest {
                    Some(info) if self.data.forecasting.is_new_model(Some(&info)) => {
                        self.load_model(info)
                    }
                    _ => Task::none(),
                }
            }
            Message::ModelLoaded(result) => {
                if let Err(e) = &result {
                    tracing::warn!(error = %e, "could not load stored model");
                }
                if let Some((info, artifact)) = self.errors.record(ErrorSource::Model, result) {
                    tracing::info!(id = info.id, trained_at = %info.trained_at, "loaded model");
                    self.data.forecasting.set_model(&info, artifact);
                    self.refresh_forecasts();
                }
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
            Message::TrayCheck => self.handle_tray(),
            Message::WindowCloseRequested => {
                self.ui.is_window_visible = false;
                window::latest().and_then(|id| window::minimize(id, true))
            }
            _ => Task::none(),
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
}
