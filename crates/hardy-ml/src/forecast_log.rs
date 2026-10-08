//! Forecasts in the form the daemon logs them for accuracy tracking.

use chrono::{DateTime, Utc};
use hardy_core::{GymSchedule, db::ForecastLogEntry};

use crate::{forecast::baseline_forecast, history::History, training::ModelArtifact};

/// Entries for every open target `1..=max_hours_ahead` hours after `now`:
/// the model's forecast where it produced one (else the plain averages,
/// with no model id) alongside the plain averages.
pub fn forecast_log_entries(
    model: Option<(i64, &ModelArtifact)>,
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    max_hours_ahead: u32,
) -> Vec<ForecastLogEntry> {
    let (model_id, model_points) = model.map_or((None, Vec::new()), |(id, artifact)| {
        (Some(id), artifact.forecast(history, schedule, now))
    });
    baseline_forecast(history, schedule, now, max_hours_ahead)
        .into_iter()
        .filter_map(|base| {
            let hours = u32::try_from((base.timestamp - now).num_hours()).ok()?;
            let from_model = model_points
                .iter()
                .find(|p| p.timestamp == base.timestamp && p.method.is_ml());
            let (model_id, shown) = match from_model {
                Some(point) => (model_id, point),
                None => (None, &base),
            };
            Some(ForecastLogEntry {
                made_at: now,
                target: base.timestamp,
                horizon_hours: hours,
                model_id,
                predicted: shown.predicted_value,
                low: shown.confidence_low,
                high: shown.confidence_high,
                baseline: base.predicted_value,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;
    use chrono::TimeDelta;

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
    fn test_forecast_log_without_model_logs_averages() {
        let history = learnable_history(14);
        let schedule = GymSchedule::default();
        let now = local(14, 12, 0);
        let entries = forecast_log_entries(None, &history, &schedule, now, 6);
        let baseline = baseline_forecast(&history, &schedule, now, 6);

        assert_eq!(entries.len(), baseline.len());
        assert_ne!(entries.len(), 0);
        for (e, b) in entries.iter().zip(&baseline) {
            assert_eq!(e.made_at, now);
            assert_eq!(e.target, b.timestamp);
            assert_eq!(
                e.target - e.made_at,
                TimeDelta::hours(i64::from(e.horizon_hours))
            );
            assert_eq!(e.model_id, None);
            assert_relative_eq!(e.predicted, e.baseline);
            assert_relative_eq!(e.baseline, b.predicted_value);
        }
    }

    #[test]
    fn test_forecast_log_marks_model_forecasts() -> Result<()> {
        let history = learnable_history(28);
        let schedule = GymSchedule::default();
        let artifact = train(
            &history,
            &schedule,
            &options(Tuning::Fixed(RfParams::new(30, 8, 10)?)),
            local(28, 0, 0),
        )?;
        // Friday 19:00: 20–23 are open targets.
        let now = local(25, 19, 0);
        let entries = forecast_log_entries(Some((42, &artifact)), &history, &schedule, now, 6);

        assert_eq!(entries.len(), 4);
        assert!(entries.iter().all(|e| e.model_id == Some(42)));
        assert!(
            entries
                .iter()
                .all(|e| e.low <= e.predicted && e.predicted <= e.high)
        );
        assert_eq!(
            entries.iter().map(|e| e.horizon_hours).collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );

        // Without a recent reading the model falls back to averages.
        let stale = forecast_log_entries(
            Some((42, &artifact)),
            &history,
            &schedule,
            local(29, 12, 0),
            6,
        );
        assert!(stale.iter().all(|e| e.model_id.is_none()));
        Ok(())
    }
}
