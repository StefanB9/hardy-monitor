//! Database loads and saves, each returning the message that delivers the
//! result.

use std::{path::PathBuf, time::Instant};

use hardy_core::{
    alert::AlertSettings,
    db::{HorizonAccuracy, HourlyAverage, LiveForecast, OccupancyLog},
    error::AppError,
};
use iced::Task;

use super::{ChartData, HardyMonitorApp, Message, loads::ErrorSource};
use crate::{forecasting::SuiteStatus, time_range::AnalyticsRange};

/// Wraps a database error with the failing operation.
pub(super) fn db_err(op: &'static str) -> impl Fn(anyhow::Error) -> AppError {
    move |e| AppError::from_anyhow_db(e, op)
}

impl HardyMonitorApp {
    /// Runs a database load, counted as pending until its result message
    /// (see [`Message::finishes_load`]) arrives.
    pub(super) fn load<T: Send + 'static>(
        &mut self,
        work: impl Future<Output = T> + Send + 'static,
        done: impl FnOnce(T) -> Message + Send + 'static,
    ) -> Task<Message> {
        self.ui.pending.begin(Instant::now());
        Task::perform(work, done)
    }

    pub(super) fn check_schema(&self) -> Task<Message> {
        let db = self.db.clone();
        Task::perform(async move { db.schema_status().await }, |r| {
            Message::SchemaChecked(r.map_err(db_err("schema_status")))
        })
    }

    pub(super) fn load_alert_settings(&mut self) -> Task<Message> {
        let db = self.db.clone();
        self.load(async move { db.get_alert_settings().await }, |r| {
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
                self.errors.raise(ErrorSource::AlertSettings, e);
                Task::none()
            }
            Some(Ok(settings)) => {
                let db = self.db.clone();
                self.load(
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

    pub(super) fn fetch_latest(&mut self) -> Task<Message> {
        let db = self.db.clone();
        self.load(
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

    /// Readings for the selected chart range (up to now), and for the week
    /// and month views the past hour-ahead forecast.
    pub(super) fn load_chart_history(&mut self) -> Task<Message> {
        let now = self.clock.now_utc();
        let Some((start, end)) = self.ui.chart_range.window(
            now,
            &self.schedule,
            (&self.ui.custom_start, &self.ui.custom_end),
            self.ui.chart_offset,
        ) else {
            return Task::none();
        };
        let db = self.db.clone();
        let end = end.min(now);
        let id = self.ui.chart_requests.issue();
        let past = self.ui.chart_range.shows_past_forecasts();
        self.load(
            async move {
                let readings = db.get_history_range(start, end).await?;
                let past_forecasts = if past {
                    db.forecast_history(start, end).await?
                } else {
                    Vec::new()
                };
                Ok(ChartData {
                    readings,
                    past_forecasts,
                })
            },
            move |r: anyhow::Result<ChartData>| {
                Message::HistoryLoaded(id, r.map_err(db_err("load_chart")))
            },
        )
    }

    /// The oldest reading, how far the chart may step back.
    pub(super) fn load_first_reading(&mut self) -> Task<Message> {
        let db = self.db.clone();
        self.load(async move { db.first_reading_time().await }, |r| {
            Message::FirstReadingLoaded(r.map_err(db_err("first_reading_time")))
        })
    }

    /// Hourly averages for the heatmap range.
    pub(super) fn load_analytics(&mut self) -> Task<Message> {
        let now = self.clock.now_utc();
        let tz = self.schedule.timezone();
        let start = self.ui.analytics_range.start(now, tz);
        let db = self.db.clone();
        let id = self.ui.analytics_requests.issue();
        self.load(
            async move { db.get_averages_range(start, now, tz).await },
            move |r: anyhow::Result<Vec<HourlyAverage>>| {
                Message::AnalyticsLoaded(id, r.map_err(db_err("get_averages_range")))
            },
        )
    }

    /// The last four weeks and the four before them, for insights.
    pub(super) fn load_insights_data(&mut self) -> Task<Message> {
        let now = self.clock.now_utc();
        let tz = self.schedule.timezone();
        let current_start = AnalyticsRange::Last4Weeks.start(now, tz);
        let baseline_start = current_start - chrono::TimeDelta::weeks(4);
        let db = self.db.clone();
        self.load(
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
    pub(super) fn load_accuracy(&mut self) -> Task<Message> {
        let now = self.clock.now_utc();
        let tz = self.schedule.timezone();
        let first_day = now.with_timezone(&tz).date_naive()
            - chrono::TimeDelta::days(crate::views::model_data::ACCURACY_DAYS - 1);
        let since = hardy_core::analytics::midnight_local_as_utc(first_day, tz);
        let db = self.db.clone();
        self.load(
            async move { db.forecast_accuracy(since, now, tz).await },
            |r: anyhow::Result<Vec<HorizonAccuracy>>| {
                Message::AccuracyLoaded(r.map_err(db_err("forecast_accuracy")))
            },
        )
    }

    /// Fetches readings newer than those already held for forecasting.
    pub(super) fn load_forecast_history(&mut self) -> Task<Message> {
        let db = self.db.clone();
        let now = self.clock.now_utc();
        let start = self.data.forecasting.history_fetch_start(now);
        self.load(
            async move { db.get_history_range(start, now).await },
            |r: anyhow::Result<Vec<OccupancyLog>>| {
                Message::ForecastHistoryLoaded(r.map_err(db_err("get_forecast_history")))
            },
        )
    }

    /// The forecast models' states, the newest benchmark and the selection.
    pub(super) fn load_suite_status(&mut self) -> Task<Message> {
        let db = self.db.clone();
        self.load(
            async move { suite_status(&db).await },
            |r: anyhow::Result<SuiteStatus>| {
                Message::SuiteStatusLoaded(r.map_err(db_err("load_suite_status")))
            },
        )
    }

    /// Every forecast the daemon published last (all models and the
    /// default), so switching models needs no reload.
    pub(super) fn load_published_forecast(&mut self) -> Task<Message> {
        let db = self.db.clone();
        let keys: Vec<String> = std::iter::once(hardy_ml::live::SELECTED)
            .chain(hardy_ml::suite::ModelId::ALL.iter().map(|id| id.key()))
            .map(ToString::to_string)
            .collect();
        self.load(
            async move { db.live_forecasts(&keys).await },
            |r: anyhow::Result<Vec<LiveForecast>>| {
                Message::PublishedForecastLoaded(r.map_err(db_err("live_forecasts")))
            },
        )
    }

    /// Asks the daemon for a forecast run now.
    pub(super) fn request_forecast_run(&mut self) -> Task<Message> {
        let db = self.db.clone();
        let now = self.clock.now_utc();
        self.load(
            async move {
                db.request_retrain(now).await?;
                suite_status(&db).await
            },
            |r: anyhow::Result<SuiteStatus>| {
                Message::SuiteStatusLoaded(r.map_err(db_err("request_forecast_run")))
            },
        )
    }

    pub(super) fn export_csv(&mut self) -> Task<Message> {
        let db = self.db.clone();
        let clock = self.clock.clone();
        let output_dir = dirs::download_dir().unwrap_or_else(|| PathBuf::from("."));
        self.load(
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

async fn suite_status(db: &hardy_core::Database) -> anyhow::Result<SuiteStatus> {
    Ok(SuiteStatus {
        records: db.model_records().await?,
        benchmark: db.latest_benchmark().await?,
        selection: db.forecast_selection().await?,
        state: db.get_ml_state().await?,
    })
}
