//! The GUI's forecasting state: recent history (for slot averages and the
//! trend arrow), the default forecast the daemon publishes, and the state of
//! its models and benchmark.

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::{
    GymSchedule,
    db::{BenchmarkRun, LiveForecast, MlState, ModelRecord, OccupancyLog},
};
use hardy_ml::{History, PredictionMethod, PredictionWithConfidence, SlotProfile, suite::ModelId};

use crate::{
    leaderboard::{self, Leaderboard},
    quiet_window::{QuietWindow, next_quiet_window},
};

/// History kept for the slot averages the quiet window falls back to.
pub(crate) const HISTORY_DAYS: i64 = 28;

/// What the daemon reports about its forecasting models.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SuiteStatus {
    /// Each model's fit state.
    pub records: Vec<ModelRecord>,
    /// The newest benchmark.
    pub benchmark: Option<BenchmarkRun>,
    /// Model key selected per horizon bucket code.
    pub selection: Vec<(i16, String)>,
    pub state: MlState,
}

/// Forecasting state for the GUI.
#[derive(Debug)]
pub(crate) struct Forecasting {
    history: History,
    status: SuiteStatus,
    leaderboard: Leaderboard,
    /// The published default forecast.
    published: Vec<LiveForecast>,
    max_hours_ahead: u32,
    /// No quiet window is suggested this soon after opening.
    opening_grace: TimeDelta,
    /// Published points after now (up to 24 hours).
    upcoming: Vec<PredictionWithConfidence>,
    /// How many of `upcoming` lie within the chart's horizon.
    charted: usize,
    quiet_window: Option<QuietWindow>,
}

impl Forecasting {
    pub(crate) fn new(max_hours_ahead: u32, opening_grace: TimeDelta) -> Self {
        Self {
            history: History::default(),
            leaderboard: leaderboard::build(&SuiteStatus::default()),
            status: SuiteStatus::default(),
            published: Vec::new(),
            max_hours_ahead,
            opening_grace,
            upcoming: Vec::new(),
            charted: 0,
            quiet_window: None,
        }
    }

    /// Start of the range to fetch next: everything after the newest reading
    /// held, or the full window on first load.
    pub(crate) fn history_fetch_start(&self, now: DateTime<Utc>) -> DateTime<Utc> {
        self.history
            .last_time()
            .map_or(now - TimeDelta::days(HISTORY_DAYS), |t| {
                t + TimeDelta::seconds(1)
            })
    }

    /// Adds newly fetched rows (only observed, only newer ones).
    pub(crate) fn add_history(&mut self, logs: &[OccupancyLog]) {
        if self.history.is_empty() {
            self.history = History::from_logs(logs);
        } else {
            self.history
                .extend(History::from_logs(logs).view().iter().collect::<Vec<_>>());
        }
    }

    pub(crate) fn set_status(&mut self, status: SuiteStatus) {
        self.leaderboard = leaderboard::build(&status);
        self.status = status;
    }

    pub(crate) fn set_published(&mut self, forecast: Vec<LiveForecast>) {
        self.published = forecast;
    }

    /// Recomputes the upcoming forecast and the next quiet window.
    pub(crate) fn refresh(&mut self, schedule: &GymSchedule, now: DateTime<Utc>) {
        self.upcoming = self
            .published
            .iter()
            .filter(|f| f.target > now)
            .map(|f| {
                let method = f
                    .source
                    .as_deref()
                    .and_then(ModelId::from_key)
                    .map_or(PredictionMethod::HistoricalAverage, PredictionMethod::Model);
                PredictionWithConfidence::new(
                    f.target,
                    f.value,
                    f.low.unwrap_or(f.value),
                    f.high.unwrap_or(f.value),
                    0.5,
                    method,
                )
            })
            .collect();
        let horizon = now + TimeDelta::hours(i64::from(self.max_hours_ahead));
        self.charted = self.upcoming.partition_point(|p| p.timestamp <= horizon);
        let profile = SlotProfile::from_history(self.history.view(), schedule.timezone());
        self.quiet_window =
            next_quiet_window(now, schedule, &self.upcoming, &profile, self.opening_grace);
    }

    /// The forecast within the chart's horizon.
    pub(crate) fn forecasts(&self) -> &[PredictionWithConfidence] {
        &self.upcoming[..self.charted]
    }

    pub(crate) fn quiet_window(&self) -> Option<&QuietWindow> {
        self.quiet_window.as_ref()
    }

    /// The measured reading closest to `t`, within five minutes.
    pub(crate) fn reading_near(&self, t: DateTime<Utc>) -> Option<f64> {
        self.history.view().value_near(t, TimeDelta::minutes(5))
    }

    /// Name of the model behind the next forecast value.
    pub(crate) fn forecast_source(&self) -> Option<&'static str> {
        self.upcoming.first().and_then(|p| match p.method {
            PredictionMethod::Model(id) => Some(id.name()),
            PredictionMethod::HistoricalAverage => None,
        })
    }

    /// The models and their benchmark, best first.
    pub(crate) fn leaderboard(&self) -> &Leaderboard {
        &self.leaderboard
    }

    /// Why the daemon's last run failed, if it did.
    pub(crate) fn last_error(&self) -> Option<&str> {
        self.status.state.last_error.as_deref()
    }

    /// A run was requested and the daemon has not started it yet.
    pub(crate) fn run_pending(&self) -> bool {
        let state = &self.status.state;
        state
            .retrain_requested_at
            .is_some_and(|requested| state.last_attempt_at.is_none_or(|a| a < requested))
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use hardy_core::db::DataSource;

    use super::*;

    fn at(minute: i64) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2024, 6, 17, 8, 0, 0)
            .single()
            .unwrap_or_default()
            + TimeDelta::minutes(minute)
    }

    fn log(id: i64, minute: i64, source: DataSource) -> OccupancyLog {
        OccupancyLog {
            id,
            timestamp: at(minute),
            percentage: 30.0,
            source,
        }
    }

    fn published(minute: i64, value: f64) -> LiveForecast {
        LiveForecast {
            model: "selected".to_string(),
            target: at(minute),
            made_at: at(0),
            value,
            low: Some(value - 4.0),
            high: None,
            source: Some("naive".to_string()),
        }
    }

    #[test]
    fn test_history_fetch_is_incremental() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        let now = at(100);
        assert_eq!(
            f.history_fetch_start(now),
            now - TimeDelta::days(HISTORY_DAYS)
        );

        f.add_history(&[
            log(1, 0, DataSource::Measured),
            log(2, 1, DataSource::Interpolated),
            log(3, 2, DataSource::Measured),
        ]);
        assert_eq!(f.history_fetch_start(now), at(2) + TimeDelta::seconds(1));

        f.add_history(&[log(4, 3, DataSource::Measured)]);
        assert_eq!(f.history_fetch_start(now), at(3) + TimeDelta::seconds(1));
    }

    #[test]
    fn test_run_pending_until_attempted() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        assert!(!f.run_pending());
        let status = |requested, attempted| SuiteStatus {
            state: MlState {
                retrain_requested_at: Some(at(requested)),
                last_attempt_at: Some(at(attempted)),
                last_error: None,
            },
            ..SuiteStatus::default()
        };
        f.set_status(status(10, 5));
        assert!(f.run_pending());
        f.set_status(status(10, 11));
        assert!(!f.run_pending());
    }

    #[test]
    fn test_refresh_keeps_future_points_and_caps_the_chart() {
        let mut f = Forecasting::new(1, TimeDelta::hours(1));
        f.set_published(vec![
            published(0, 10.0),
            published(30, 20.0),
            published(60, 30.0),
            published(90, 40.0),
        ]);
        f.refresh(&GymSchedule::default(), at(15));
        let charted: Vec<f64> = f.forecasts().iter().map(|p| p.predicted_value).collect();
        assert_eq!(charted, [20.0, 30.0], "within one hour of now");
        assert_eq!(f.forecasts()[0].confidence_low, 16.0);
        assert_eq!(f.forecasts()[0].confidence_high, 20.0, "no upper bound");
        assert_eq!(f.forecast_source(), Some("Persistence"));
        assert!(f.quiet_window().is_some());
    }

    #[test]
    fn test_nothing_published_means_no_forecast() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        f.refresh(&GymSchedule::default(), at(0));
        assert_eq!(f.forecasts().len(), 0);
        assert_eq!(f.forecast_source(), None);
    }

    #[test]
    fn test_reading_near_finds_close_measurement_only() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        f.add_history(&[
            log(1, 0, DataSource::Measured),
            log(2, 30, DataSource::Measured),
        ]);
        assert_eq!(f.reading_near(at(2)), Some(30.0));
        assert_eq!(f.reading_near(at(15)), None);
    }
}
