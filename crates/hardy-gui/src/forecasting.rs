//! The GUI's forecasting state: recent history from the database, the newest
//! model the daemon trained, and the forecasts derived from them.

use std::sync::Arc;

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::{
    GymSchedule,
    db::{MlState, ModelInfo, OccupancyLog},
};
use hardy_ml::{History, ModelArtifact, PredictionWithConfidence, SlotProfile, baseline_forecast};

use crate::quiet_window::{QuietWindow, next_quiet_window};

/// History kept for the baseline, which needs the most (model features
/// reach back one week).
pub(crate) const HISTORY_DAYS: i64 = hardy_ml::BASELINE_HISTORY_DAYS;

/// What the model view shows about the active model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSummary {
    pub algorithm: String,
    pub trained_at: DateTime<Utc>,
    pub training_samples: usize,
    pub holdout_mae: f64,
    pub baseline_mae: f64,
    pub tuned: bool,
    /// Holdout MAE per horizon (index `h - 1`).
    pub mae_by_horizon: Vec<f64>,
}

impl ModelSummary {
    /// Relative improvement over the slot-average baseline, e.g. 0.25 = 25%
    /// lower error.
    pub fn improvement(&self) -> f64 {
        if self.baseline_mae > 0.0 {
            1.0 - self.holdout_mae / self.baseline_mae
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
struct LoadedModel {
    id: i64,
    artifact: Arc<ModelArtifact>,
    summary: ModelSummary,
}

/// Forecasting state for the GUI.
#[derive(Debug)]
pub(crate) struct Forecasting {
    history: History,
    model: Option<LoadedModel>,
    state: MlState,
    max_hours_ahead: u32,
    /// No quiet window is suggested this soon after opening.
    opening_grace: TimeDelta,
    forecasts: Vec<PredictionWithConfidence>,
    quiet_window: Option<QuietWindow>,
}

impl Forecasting {
    pub(crate) fn new(max_hours_ahead: u32, opening_grace: TimeDelta) -> Self {
        Self {
            history: History::default(),
            model: None,
            state: MlState::default(),
            max_hours_ahead,
            opening_grace,
            forecasts: Vec::new(),
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

    /// Adds newly fetched rows (only measured, only newer ones).
    pub(crate) fn add_history(&mut self, logs: &[OccupancyLog]) {
        if self.history.is_empty() {
            self.history = History::from_logs(logs);
        } else {
            self.history
                .extend(History::from_logs(logs).view().iter().collect::<Vec<_>>());
        }
    }

    /// Whether `latest` is a different model from the one loaded.
    pub(crate) fn is_new_model(&self, latest: Option<&ModelInfo>) -> bool {
        latest.is_some_and(|info| self.model.as_ref().is_none_or(|m| m.id != info.id))
    }

    pub(crate) fn set_model(&mut self, info: &ModelInfo, artifact: Arc<ModelArtifact>) {
        let summary = ModelSummary {
            algorithm: info.algorithm.clone(),
            trained_at: info.trained_at,
            training_samples: info.training_samples,
            holdout_mae: info.holdout_mae,
            baseline_mae: info.baseline_mae,
            tuned: info.tuned,
            mae_by_horizon: artifact.metrics.mae_by_horizon.clone(),
        };
        self.model = Some(LoadedModel {
            id: info.id,
            artifact,
            summary,
        });
    }

    pub(crate) fn set_state(&mut self, state: MlState) {
        self.state = state;
    }

    /// Recomputes forecasts (from the model if one is loaded, else from
    /// slot averages) and the next quiet window.
    pub(crate) fn refresh(&mut self, schedule: &GymSchedule, now: DateTime<Utc>) {
        self.forecasts = match &self.model {
            Some(model) => model.artifact.forecast(&self.history, schedule, now),
            None => baseline_forecast(&self.history, schedule, now, self.max_hours_ahead),
        };
        let profile = SlotProfile::from_history(self.history.view(), schedule.timezone());
        self.quiet_window =
            next_quiet_window(now, schedule, &self.forecasts, &profile, self.opening_grace);
    }

    pub(crate) fn forecasts(&self) -> &[PredictionWithConfidence] {
        &self.forecasts
    }

    pub(crate) fn quiet_window(&self) -> Option<&QuietWindow> {
        self.quiet_window.as_ref()
    }

    /// The measured reading closest to `t`, within five minutes.
    pub(crate) fn reading_near(&self, t: DateTime<Utc>) -> Option<f64> {
        self.history.view().value_near(t, TimeDelta::minutes(5))
    }

    pub(crate) fn has_model(&self) -> bool {
        self.model.is_some()
    }

    pub(crate) fn summary(&self) -> Option<&ModelSummary> {
        self.model.as_ref().map(|m| &m.summary)
    }

    /// A retrain was requested and the daemon has not started it yet.
    pub(crate) fn retrain_pending(&self) -> bool {
        self.state
            .retrain_requested_at
            .is_some_and(|requested| self.state.last_attempt_at.is_none_or(|a| a < requested))
    }

    /// Why the last training produced no model, if it failed.
    pub(crate) fn last_error(&self) -> Option<&str> {
        self.state.last_error.as_deref()
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

    fn info(id: i64) -> ModelInfo {
        ModelInfo {
            id,
            trained_at: at(0),
            feature_version: 2,
            algorithm: "Random Forest".to_string(),
            training_samples: 100,
            holdout_mae: 4.0,
            baseline_mae: 5.0,
            tuned: false,
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
    fn test_is_new_model() {
        let f = Forecasting::new(6, TimeDelta::hours(1));
        assert!(!f.is_new_model(None));
        assert!(f.is_new_model(Some(&info(1))));
    }

    #[test]
    fn test_retrain_pending_until_attempted() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        assert!(!f.retrain_pending());
        f.set_state(MlState {
            retrain_requested_at: Some(at(10)),
            last_attempt_at: Some(at(5)),
            last_error: None,
        });
        assert!(f.retrain_pending());
        f.set_state(MlState {
            retrain_requested_at: Some(at(10)),
            last_attempt_at: Some(at(11)),
            last_error: Some("not enough data".to_string()),
        });
        assert!(!f.retrain_pending());
        assert_eq!(f.last_error(), Some("not enough data"));
    }

    #[test]
    fn test_summary_improvement() {
        let s = ModelSummary {
            algorithm: "Random Forest".to_string(),
            trained_at: at(0),
            training_samples: 10,
            holdout_mae: 3.0,
            baseline_mae: 4.0,
            tuned: true,
            mae_by_horizon: vec![],
        };
        assert!((s.improvement() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_refresh_without_model_uses_baseline() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        f.add_history(&[log(1, 0, DataSource::Measured)]);
        f.refresh(&GymSchedule::default(), at(0));
        assert!(
            f.forecasts()
                .iter()
                .all(|p| p.method == hardy_ml::PredictionMethod::HistoricalAverage)
        );
        assert!(!f.has_model());
        assert!(f.quiet_window().is_some());
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
