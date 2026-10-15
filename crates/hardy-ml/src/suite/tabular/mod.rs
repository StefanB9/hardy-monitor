//! Regression and machine-learning models on tabular features: each learns
//! the deviation from the weekly profile at a target slot from what is known
//! at the origin, with one model per horizon bucket.

mod features;
mod learners;

use chrono::TimeDelta;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use self::{
    features::{forecast_rows, training_rows},
    learners::Learner,
};
use crate::{
    benchmark::Bucket,
    error::MlError,
    grid::HORIZON_STEPS,
    suite::{Forecaster, Observed, profiles::WeeklyProfile},
};

/// Deviations are capped: the profile plus this already spans 0–100.
const MAX_DEVIATION: f64 = 100.0;

/// A learner per horizon bucket on the weekly profile's deviation.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "L: Serialize", deserialize = "L: DeserializeOwned"))]
pub struct Tabular<L> {
    fitted: Option<Fitted<L>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "L: Serialize", deserialize = "L: DeserializeOwned"))]
struct Fitted<L> {
    profile: WeeklyProfile,
    /// One per bucket of [`Bucket::HORIZONS`].
    learners: Vec<L>,
}

impl<L> Default for Tabular<L> {
    fn default() -> Self {
        Self { fitted: None }
    }
}

impl<L: Learner> Forecaster for Tabular<L> {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(L::MIN_DAYS)
    }

    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
        let profile = WeeklyProfile::fit(data);
        let learners = training_rows(data, &profile)
            .iter()
            .map(L::fit)
            .collect::<Result<Vec<L>, MlError>>()?;
        self.fitted = Some(Fitted { profile, learners });
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| MlError::Fit("model is not fitted".to_string()))?;
        let grid = data.grid();
        let mut values = vec![0.0; HORIZON_STEPS];
        for (step, row) in forecast_rows(data, &fitted.profile) {
            let bucket = Bucket::of_step(step);
            let Some(learner) = Bucket::HORIZONS
                .iter()
                .position(|&b| b == bucket)
                .and_then(|i| fitted.learners.get(i))
            else {
                continue;
            };
            let deviation = learner.predict(&row).clamp(-MAX_DEVIATION, MAX_DEVIATION);
            values[step - 1] = fitted.profile.at(grid.step_end(step)) + deviation;
        }
        Ok(values)
    }
}

/// Ridge regression on the tabular features.
pub type Ridge = Tabular<learners::Ridge>;
/// Elastic-net regression on the tabular features.
pub type ElasticNet = Tabular<learners::ElasticNet>;
/// Additive P-spline regression on the tabular features.
pub type SplineGam = Tabular<learners::SplineGam>;
/// Median regression on the tabular features.
pub type MedianRegression = Tabular<learners::MedianRegression>;
/// Random forest on the tabular features.
pub type RandomForest = Tabular<learners::RandomForest>;
/// Gradient-boosted trees on the tabular features.
pub type BoostedTrees = Tabular<learners::BoostedTrees>;
/// Support-vector regression on the tabular features.
pub type SupportVectors = Tabular<learners::SupportVectors>;

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use chrono::{DateTime, Datelike, Timelike, Utc};
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{
        grid::{Grid, SLOT, is_open_slot, slot_end_of},
        history::History,
        suite::forecast,
        test_support::{local, minutely},
    };

    /// The weekly curve, plus 8 on day 9 from 09:00.
    fn value(t: DateTime<Utc>) -> f64 {
        let local = (slot_end_of(t) - TimeDelta::minutes(1))
            .with_timezone(&GymSchedule::default().timezone());
        let base = 20.0
            + f64::from(local.weekday().num_days_from_monday())
            + 10.0
                * (f64::from(local.hour()) / 24.0 * std::f64::consts::TAU)
                    .sin()
                    .abs();
        base + if t >= local_time(9, 9) { 8.0 } else { 0.0 }
    }

    fn local_time(day: i64, hour: u32) -> DateTime<Utc> {
        local(day, hour, 0)
    }

    fn forecast_at(model: &mut dyn Forecaster, history: &History) -> Result<Vec<f64>> {
        let schedule = GymSchedule::default();
        let grid = Grid::build(history, &schedule, local(0, 0, 0), local(10, 0, 0));
        let fit_history = history.until(local(9, 0, 0));
        model.fit(&Observed::new(
            &grid,
            &fit_history,
            &schedule,
            local(9, 0, 0),
        ))?;
        let origin = local(9, 12, 0);
        let seen = history.until(origin);
        Ok(forecast(
            model,
            &Observed::new(&grid, &seen, &schedule, origin),
        )?)
    }

    #[test]
    fn test_every_learner_follows_the_profile_and_todays_offset() -> Result<()> {
        let history = minutely(10, value);
        let models: [Box<dyn Forecaster>; 7] = [
            Box::new(Ridge::default()),
            Box::new(ElasticNet::default()),
            Box::new(SplineGam::default()),
            Box::new(MedianRegression::default()),
            Box::new(RandomForest::default()),
            Box::new(BoostedTrees::default()),
            Box::new(SupportVectors::default()),
        ];
        let schedule = GymSchedule::default();
        let origin = local(9, 12, 0);
        for mut model in models {
            let values = forecast_at(model.as_mut(), &history)?;
            assert_eq!(values.len(), HORIZON_STEPS);
            // Nothing in training showed an offset, so the models fall
            // back towards the profile; the next slot stays near it.
            let end = origin + SLOT;
            let expected = value(end - TimeDelta::minutes(1)) - 8.0;
            assert!(
                (values[0] - expected).abs() < 9.0,
                "{} vs {expected}",
                values[0]
            );
            for (k, v) in values.iter().enumerate() {
                let end = origin + SLOT * i32::try_from(k + 1)?;
                if !is_open_slot(&schedule, end) {
                    assert_eq!(*v, 0.0);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_ridge_follows_the_current_level() -> Result<()> {
        // Each day's level is set by the morning and holds all day, so the
        // current deviation predicts the afternoon.
        let history = minutely(29, |t| {
            let day = (t - local(0, 0, 0)).num_days();
            #[allow(clippy::cast_precision_loss)]
            let level = 20.0 + ((day * 37) % 5) as f64 * 10.0;
            level
        });
        let schedule = GymSchedule::default();
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(29, 0, 0));
        let fit_history = history.until(local(28, 0, 0));
        let mut ridge = Ridge::default();
        ridge.fit(&Observed::new(
            &grid,
            &fit_history,
            &schedule,
            local(28, 0, 0),
        ))?;
        let origin = local(28, 12, 0);
        let seen = history.until(origin);
        let values = ridge.predict(&Observed::new(&grid, &seen, &schedule, origin))?;
        let today = seen.view().latest().map_or(0.0, |(_, v)| v);
        // Two hours ahead the day is still at its level.
        assert!((values[7] - today).abs() < 5.0, "{} vs {today}", values[7]);
        Ok(())
    }

    #[test]
    fn test_predict_before_fit_fails_and_minimum_history() {
        let history = minutely(2, |_| 30.0);
        let schedule = GymSchedule::default();
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(2, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(1, 12, 0));
        assert!(Ridge::default().predict(&data).is_err());
        assert_eq!(Ridge::default().min_history(), TimeDelta::days(1));
        assert_eq!(SplineGam::default().min_history(), TimeDelta::days(2));
    }
}
