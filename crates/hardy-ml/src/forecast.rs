//! Live forecasts for the next hours from a trained model (see
//! [`crate::baseline`] for the forecast without one).

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::GymSchedule;

use crate::{
    confidence::{PredictionMethod, PredictionWithConfidence},
    features,
    history::History,
    model::Model,
    profile::SlotProfile,
    training::ModelArtifact,
};

/// Interval width (percentage points) that maps to zero confidence.
const ZERO_CONFIDENCE_WIDTH: f64 = 60.0;

impl ModelArtifact {
    /// Forecasts each open hour `now + 1h ..= now + max_hours_ahead`.
    ///
    /// Without a recent reading (gym just opened, daemon down) the model has
    /// no current state, so slot averages are returned instead.
    pub fn forecast(
        &self,
        history: &History,
        schedule: &GymSchedule,
        now: DateTime<Utc>,
    ) -> Vec<PredictionWithConfidence> {
        let mut forecasts = Vec::with_capacity(self.max_hours_ahead as usize);
        for hours_ahead in 1..=self.max_hours_ahead {
            let target = now + TimeDelta::hours(i64::from(hours_ahead));
            if !schedule.is_open(&target) {
                continue;
            }
            let row = features::extract(history, &self.profile, schedule, now, hours_ahead);
            let predicted = row.and_then(|r| self.model.predict(&r).ok());
            let forecast = match predicted {
                Some(value) => {
                    let (low, high) = self.intervals.offsets(hours_ahead);
                    let score = (1.0 - (high - low) / ZERO_CONFIDENCE_WIDTH).clamp(0.0, 1.0);
                    let method = match &self.model {
                        Model::RandomForest { params, .. } => PredictionMethod::RandomForest {
                            confidence: score,
                            n_trees: usize::from(params.n_trees()),
                        },
                        Model::Linear { .. } => {
                            PredictionMethod::MachineLearning { confidence: score }
                        }
                    };
                    PredictionWithConfidence::new(
                        target,
                        value,
                        value + low,
                        value + high,
                        score,
                        method,
                    )
                }
                None => baseline_point(&self.profile, schedule, target),
            };
            forecasts.push(forecast);
        }
        forecasts
    }
}

/// The stored profile's slot average for `target`: the model's fallback
/// without a current reading, when no correction could apply anyway.
fn baseline_point(
    profile: &SlotProfile,
    schedule: &GymSchedule,
    target: DateTime<Utc>,
) -> PredictionWithConfidence {
    let tz = schedule.timezone();
    let (mean, spread) = profile
        .stat(target, tz)
        .map_or((profile.overall_mean(), 15.0), |s| (s.mean, s.std_dev));
    PredictionWithConfidence::new(
        target,
        mean,
        mean - spread,
        mean + spread,
        0.5,
        PredictionMethod::HistoricalAverage,
    )
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};

    use super::*;
    use crate::{
        model::RfParams,
        training::{
            Tuning,
            tests::{learnable_history, local, options},
            train,
        },
    };

    #[test]
    fn test_forecast_uses_model_for_open_hours() -> Result<()> {
        let history = learnable_history(28);
        let schedule = GymSchedule::default();
        let artifact = train(
            &history,
            &schedule,
            &options(Tuning::Fixed(RfParams::new(30, 8, 10)?)),
            local(28, 0, 0),
        )?;

        // Friday 2024-06-28, 19:00 local: 20–23 are open, 00:00 and 01:00
        // are not.
        let now = local(25, 19, 0);
        let forecasts = artifact.forecast(&history, &schedule, now);
        assert_eq!(forecasts.len(), 4);
        for f in &forecasts {
            assert!(f.method.is_ml(), "{:?}", f.method);
            assert!(f.confidence_low <= f.predicted_value);
            assert!(f.predicted_value <= f.confidence_high);
        }
        let first = forecasts.first().context("forecast")?;
        assert_eq!(first.timestamp, now + TimeDelta::hours(1));
        Ok(())
    }

    #[test]
    fn test_forecast_falls_back_without_recent_reading() -> Result<()> {
        let history = learnable_history(28);
        let schedule = GymSchedule::default();
        let artifact = train(
            &history,
            &schedule,
            &options(Tuning::Fixed(RfParams::new(30, 8, 10)?)),
            local(28, 0, 0),
        )?;
        // Two days after the last reading: no current state.
        let forecasts = artifact.forecast(&history, &schedule, local(29, 12, 0));
        assert_ne!(forecasts.len(), 0);
        assert!(
            forecasts
                .iter()
                .all(|f| f.method == PredictionMethod::HistoricalAverage)
        );
        Ok(())
    }
}
