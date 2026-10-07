//! Database loads and saves, each returning the message that delivers the
//! result.

use std::path::PathBuf;

use hardy_core::{
    alert::AlertSettings,
    db::{HorizonAccuracy, HourlyAverage, MlState, ModelInfo, OccupancyLog},
    error::AppError,
};
use hardy_ml::{ModelArtifact, features::FEATURE_VERSION};
use iced::Task;

use super::{HardyMonitorApp, Message};
use crate::time_range::AnalyticsRange;

/// Wraps a database error with the failing operation.
fn db_err(op: &'static str) -> impl Fn(anyhow::Error) -> AppError {
    move |e| AppError::from_anyhow_db(e, op)
}

impl HardyMonitorApp {
    pub(super) fn check_schema(&self) -> Task<Message> {
        let db = self.db.clone();
        Task::perform(async move { db.schema_status().await }, |r| {
            Message::SchemaChecked(r.map_err(db_err("schema_status")))
        })
    }

    pub(super) fn load_alert_settings(&self) -> Task<Message> {
        let db = self.db.clone();
        Task::perform(async move { db.get_alert_settings().await }, |r| {
            Message::AlertSettingsLoaded(r.map_err(db_err("get_alert_settings")))
        })
    }

    /// Saves a settings change and adopts the saved row; `None` = nothing to
    /// save.
    pub(super) fn save_alert_settings(
        &mut self,
        change: Option<Result<AlertSettings, AppError>>,
    ) -> Task<Message> {
        match change {
            None => Task::none(),
            Some(Err(e)) => {
                self.error = Some(e);
                Task::none()
            }
            Some(Ok(settings)) => {
                let db = self.db.clone();
                Task::perform(
                    async move {
                        db.save_alert_settings(&settings).await?;
                        Ok(settings)
                    },
                    |r: anyhow::Result<AlertSettings>| {
                        Message::AlertSettingsLoaded(r.map_err(db_err("save_alert_settings")))
                    },
                )
            }
        }
    }

    pub(super) fn fetch_latest(&self) -> Task<Message> {
        let db = self.db.clone();
        Task::perform(
            async move {
                Ok(db
                    .get_latest_record()
                    .await?
                    .map(|r| (r.timestamp, r.percentage)))
            },
            |r: anyhow::Result<Option<(chrono::DateTime<chrono::Utc>, f64)>>| {
                Message::FetchCompleted(r.map_err(db_err("get_latest_record")))
            },
        )
    }

    /// Readings for the selected chart range (up to now).
    pub(super) fn load_chart_history(&self) -> Task<Message> {
        let now = self.clock.now_utc();
        let Some((start, end)) = self.ui.chart_range.window(
            now,
            &self.schedule,
            (&self.ui.custom_start, &self.ui.custom_end),
        ) else {
            return Task::none();
        };
        let db = self.db.clone();
        let end = end.min(now);
        Task::perform(
            async move { db.get_history_range(start, end).await },
            |r: anyhow::Result<Vec<OccupancyLog>>| {
                Message::HistoryLoaded(r.map_err(db_err("get_history_range")))
            },
        )
    }

    /// Hourly averages for the heatmap range.
    pub(super) fn load_analytics(&self) -> Task<Message> {
        let now = self.clock.now_utc();
        let tz = self.schedule.timezone();
        let start = self.ui.analytics_range.start(now, tz);
        let db = self.db.clone();
        Task::perform(
            async move { db.get_averages_range(start, now, tz).await },
            |r: anyhow::Result<Vec<HourlyAverage>>| {
                Message::AnalyticsLoaded(r.map_err(db_err("get_averages_range")))
            },
        )
    }

    /// The last four weeks and the four before them, for insights.
    pub(super) fn load_insights_data(&self) -> Task<Message> {
        let now = self.clock.now_utc();
        let tz = self.schedule.timezone();
        let current_start = AnalyticsRange::Last4Weeks.start(now, tz);
        let baseline_start = current_start - chrono::TimeDelta::weeks(4);
        let db = self.db.clone();
        Task::perform(
            async move {
                let current = db.get_averages_range(current_start, now, tz).await;
                let baseline = db
                    .get_averages_range(baseline_start, current_start, tz)
                    .await;
                (current, baseline)
            },
            |(current, baseline)| Message::InsightsDataLoaded {
                current: current.map_err(db_err("get_insights_current")),
                baseline: baseline.map_err(db_err("get_insights_baseline")),
            },
        )
    }

    /// Scores of the daemon's logged forecasts over the last days.
    pub(super) fn load_accuracy(&self) -> Task<Message> {
        let now = self.clock.now_utc();
        let tz = self.schedule.timezone();
        let first_day = now.with_timezone(&tz).date_naive()
            - chrono::TimeDelta::days(crate::views::model_data::ACCURACY_DAYS - 1);
        let since = hardy_core::analytics::midnight_local_as_utc(first_day, tz);
        let db = self.db.clone();
        Task::perform(
            async move { db.forecast_accuracy(since, now, tz).await },
            |r: anyhow::Result<Vec<HorizonAccuracy>>| {
                Message::AccuracyLoaded(r.map_err(db_err("forecast_accuracy")))
            },
        )
    }

    /// Fetches readings newer than those already held for forecasting.
    pub(super) fn load_forecast_history(&self) -> Task<Message> {
        let db = self.db.clone();
        let now = self.clock.now_utc();
        let start = self.data.forecasting.history_fetch_start(now);
        Task::perform(
            async move { db.get_history_range(start, now).await },
            |r: anyhow::Result<Vec<OccupancyLog>>| {
                Message::ForecastHistoryLoaded(r.map_err(db_err("get_forecast_history")))
            },
        )
    }

    pub(super) fn load_model_status(&self) -> Task<Message> {
        let db = self.db.clone();
        Task::perform(
            async move {
                let latest = db.latest_model_info(FEATURE_VERSION).await?;
                let state = db.get_ml_state().await?;
                Ok((latest, state))
            },
            |r: anyhow::Result<(Option<ModelInfo>, MlState)>| {
                Message::ModelStatusLoaded(r.map_err(db_err("load_model_status")))
            },
        )
    }

    pub(super) fn load_model(&self, info: ModelInfo) -> Task<Message> {
        let db = self.db.clone();
        Task::perform(
            async move {
                let bytes = db.load_model(info.id).await.map_err(db_err("load_model"))?;
                let artifact = ModelArtifact::from_bytes(&bytes)
                    .map_err(|e| AppError::MlTraining(e.to_string()))?;
                Ok((info, std::sync::Arc::new(artifact)))
            },
            Message::ModelLoaded,
        )
    }

    pub(super) fn request_retrain(&self) -> Task<Message> {
        let db = self.db.clone();
        let now = self.clock.now_utc();
        Task::perform(
            async move {
                db.request_retrain(now).await?;
                Ok((None, db.get_ml_state().await?))
            },
            |r: anyhow::Result<(Option<ModelInfo>, MlState)>| {
                Message::ModelStatusLoaded(r.map_err(db_err("request_retrain")))
            },
        )
    }

    pub(super) fn export_csv(&self) -> Task<Message> {
        let db = self.db.clone();
        let clock = self.clock.clone();
        let output_dir = dirs::download_dir().unwrap_or_else(|| PathBuf::from("."));
        Task::perform(
            async move {
                let path = db
                    .export_to_csv(&output_dir, &*clock)
                    .await
                    .map_err(db_err("export_to_csv"))?;
                Ok(path.to_string_lossy().to_string())
            },
            Message::ExportCompleted,
        )
    }
}
