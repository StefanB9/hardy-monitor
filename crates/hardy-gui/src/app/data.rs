//! Applying fetched and loaded data to the app state.

use chrono::{DateTime, Utc};
use hardy_core::{
    alert::Alert,
    analytics::{
        ComparisonMode, analyze_days, calculate_stats, compare_periods, find_peak_hours,
        find_quiet_hours, generate_insights,
    },
    db::HourlyAverage,
    error::AppError,
};
use iced::Task;

use super::{HardyMonitorApp, Message, loads::ErrorSource};
use crate::{freshness::Freshness, widgets::heatmap::WeekGrid};

impl HardyMonitorApp {
    /// Fetches the latest reading while open; clears it while closed.
    pub(super) fn poll(&mut self) -> Task<Message> {
        if self.schedule.is_open(&self.clock.now_utc()) {
            self.fetch_latest()
        } else {
            self.data.occupancy = None;
            self.ui.gauge_cache.clear();
            self.update_tray();
            Task::none()
        }
    }

    pub(super) fn handle_fetch_completed(
        &mut self,
        result: Result<Option<(DateTime<Utc>, f64)>, AppError>,
    ) -> Task<Message> {
        let now = self.clock.now_utc();
        let Some(reading) = self.errors.record(ErrorSource::LatestReading, result) else {
            return Task::none();
        };
        let Some((taken_at, percentage)) = reading else {
            self.data.last_update = Some(now);
            return Task::none();
        };
        let is_new = self.data.latest_reading_at != Some(taken_at);
        self.data.latest_reading_at = Some(taken_at);
        self.data.last_update = Some(now);
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
        self.update_tray();
        Task::batch(tasks)
    }

    pub(super) fn set_week_data(&mut self, data: &[HourlyAverage]) {
        self.data.week_grid = WeekGrid::new(data, &self.schedule);
        self.data.week_days = analyze_days(data);
        self.ui.heatmap_cache.clear();
        self.ui.heatmap_tooltip_cache.clear();
    }

    pub(super) fn handle_insights_data_loaded(
        &mut self,
        current: Result<Vec<HourlyAverage>, AppError>,
        baseline: Result<Vec<HourlyAverage>, AppError>,
    ) {
        let Some(current) = self.errors.record(ErrorSource::Insights, current) else {
            return;
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

    pub(super) fn refresh_forecasts(&mut self) {
        self.data
            .forecasting
            .refresh(&self.schedule, self.clock.now_utc());
        self.ui.chart_cache.clear();
    }
}
