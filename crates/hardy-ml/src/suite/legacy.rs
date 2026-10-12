//! The models from before the suite: the profile baseline with a decaying
//! "busier than usual" offset, and the ridge / random-forest regressions on
//! lag and calendar features (one model for every whole hour ahead).

use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};

use super::{Forecaster, Observed};
use crate::{
    baseline::Baseline,
    error::MlError,
    features,
    grid::{HORIZON_STEPS, SLOT},
    model::{Algorithm, Model, RfParams},
    profile::SlotProfile,
    samples::{SampleWindow, build_samples},
};

/// Whole hours the regressions forecast; steps in between are interpolated.
const HOURS_AHEAD: u32 = 24;
/// Steps per hour.
const STEPS_PER_HOUR: usize = 4;

fn not_fitted() -> MlError {
    MlError::Fit("model is not fitted".to_string())
}

/// Middle of the slot ending at `end`: what profile lookups use.
fn middle(end: DateTime<Utc>) -> DateTime<Utc> {
    end - SLOT / 2
}

/// Typical weekday × quarter-hour profile, recency-weighted, plus the
/// current deviation from it carried forward with a fitted per-hour decay.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ProfileDecayingOffset {
    fitted: Option<Baseline>,
}

impl Forecaster for ProfileDecayingOffset {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(1)
    }

    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
        self.fitted = Some(Baseline::fit(data.history(), data.schedule(), HOURS_AHEAD));
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let baseline = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let grid = data.grid();
        Ok((1..=HORIZON_STEPS)
            .map(|k| baseline.predict_at(data.history(), data.origin(), middle(grid.step_end(k))))
            .collect())
    }
}

/// A fitted regression with the profile its features were built from.
#[derive(Debug, Serialize, Deserialize)]
struct Fitted {
    profile: SlotProfile,
    model: Model,
}

fn fit_tabular(data: &Observed<'_>, algorithm: Algorithm) -> Result<Fitted, MlError> {
    let history = data.history();
    let (Some(first), Some(last)) = (history.first_time(), history.last_time()) else {
        return Err(MlError::InsufficientData {
            found: 0,
            required: 2,
        });
    };
    let profile = SlotProfile::from_history(history.view(), data.schedule().timezone());
    let after_last = last + TimeDelta::seconds(1);
    let samples = build_samples(
        history,
        &profile,
        data.schedule(),
        SampleWindow {
            anchors_from: first,
            anchors_until: after_last,
            targets_until: after_last,
        },
        HOURS_AHEAD,
    );
    let rows: Vec<_> = samples.iter().map(|s| s.features).collect();
    let targets: Vec<f64> = samples.iter().map(|s| s.target).collect();
    let model = Model::fit(algorithm, RfParams::DEFAULT, &rows, &targets)?;
    Ok(Fitted { profile, model })
}

/// Forecasts every whole hour `0..=HOURS_AHEAD` and interpolates the slots
/// in between. Without a current reading there is nothing to regress from,
/// so the profile is used throughout (as before the suite).
fn predict_tabular(fitted: &Fitted, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
    let origin = data.origin();
    let tz = data.schedule().timezone();
    let hour = |h: u32| origin + TimeDelta::hours(i64::from(h));
    let rows: Option<Vec<_>> = (1..=HOURS_AHEAD)
        .map(|h| features::extract(data.history(), &fitted.profile, data.schedule(), origin, h))
        .collect();
    let hourly: Vec<f64> = match (rows, data.grid().last_known()) {
        (Some(rows), Some((_, current))) => std::iter::once(Ok(current))
            .chain(fitted.model.predict_batch(&rows)?.into_iter().map(Ok))
            .collect::<Result<_, MlError>>()?,
        _ => (0..=HOURS_AHEAD)
            .map(|h| fitted.profile.mean_at(middle(hour(h)), tz))
            .collect(),
    };
    Ok((1..=HORIZON_STEPS)
        .map(|k| {
            let (h, within) = (k / STEPS_PER_HOUR, k % STEPS_PER_HOUR);
            let next = hourly.get(h + 1).copied().unwrap_or(hourly[h]);
            #[allow(clippy::cast_precision_loss)]
            let fraction = within as f64 / STEPS_PER_HOUR as f64;
            hourly[h] * (1.0 - fraction) + next * fraction
        })
        .collect())
}

/// Ridge-regularised linear regression on lag and calendar features.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Ridge {
    fitted: Option<Fitted>,
}

impl Forecaster for Ridge {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(1)
    }

    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
        self.fitted = Some(fit_tabular(data, Algorithm::Linear)?);
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        predict_tabular(self.fitted.as_ref().ok_or_else(not_fitted)?, data)
    }
}

/// Random forest on lag and calendar features.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RandomForest {
    fitted: Option<Fitted>,
}

impl Forecaster for RandomForest {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(1)
    }

    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
        self.fitted = Some(fit_tabular(data, Algorithm::RandomForest)?);
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        predict_tabular(self.fitted.as_ref().ok_or_else(not_fitted)?, data)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{
        grid::Grid,
        history::History,
        test_support::{local, minutely},
    };

    /// Each day's level is set by the morning: the afternoon stays at the
    /// level the day started with, so the current reading predicts it.
    fn level_days(days: i64) -> History {
        minutely(days, |t| {
            let day = (t - local(0, 0, 0)).num_days();
            #[allow(clippy::cast_precision_loss)]
            let level = 20.0 + ((day * 37) % 5) as f64 * 10.0;
            level
        })
    }

    fn observed<'a>(
        grid: &'a Grid,
        history: &'a History,
        schedule: &'a GymSchedule,
        at: DateTime<Utc>,
    ) -> Observed<'a> {
        Observed::new(grid, history, schedule, at)
    }

    #[test]
    fn test_unfitted_models_refuse_to_predict() {
        let schedule = GymSchedule::default();
        let history = level_days(2);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(2, 0, 0));
        let data = observed(&grid, &history, &schedule, local(1, 12, 0));
        assert!(ProfileDecayingOffset::default().predict(&data).is_err());
        assert!(Ridge::default().predict(&data).is_err());
    }

    #[test]
    fn test_regressions_follow_the_current_level() -> Result<()> {
        let schedule = GymSchedule::default();
        // Four weeks, so a weekday's typical level mixes several days and
        // the current reading is the better guide.
        let history = level_days(29);
        let fit_cut = local(28, 0, 0);
        let fit_history = history.until(fit_cut);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(29, 0, 0));

        let mut ridge = Ridge::default();
        ridge.fit(&observed(&grid, &fit_history, &schedule, fit_cut))?;

        let origin = local(28, 12, 0);
        let seen = history.until(origin);
        let forecast = ridge.predict(&observed(&grid, &seen, &schedule, origin))?;
        let today = seen.view().latest().map_or(0.0, |(_, v)| v);
        // Two hours ahead the day is still at its level.
        assert!(
            (forecast[7] - today).abs() < 5.0,
            "{} vs {today}",
            forecast[7]
        );
        Ok(())
    }

    #[test]
    fn test_steps_between_hours_are_interpolated() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = level_days(4);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(4, 0, 0));
        let fit_history = history.until(local(3, 0, 0));
        let mut ridge = Ridge::default();
        ridge.fit(&observed(&grid, &fit_history, &schedule, local(3, 0, 0)))?;
        let origin = local(3, 12, 0);
        let seen = history.until(origin);
        let f = ridge.predict(&observed(&grid, &seen, &schedule, origin))?;
        // Step 6 (1.5 h) lies halfway between steps 4 (1 h) and 8 (2 h).
        assert!((f[5] - f64::midpoint(f[3], f[7])).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn test_profile_offset_carries_today_s_deviation() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = level_days(10);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(10, 0, 0));
        let fit_history = history.until(local(9, 0, 0));
        let mut model = ProfileDecayingOffset::default();
        model.fit(&observed(&grid, &fit_history, &schedule, local(9, 0, 0)))?;

        let origin = local(9, 12, 0);
        let seen = history.until(origin);
        let forecast = model.predict(&observed(&grid, &seen, &schedule, origin))?;
        let today = seen.view().latest().map_or(0.0, |(_, v)| v);
        assert!(
            (forecast[0] - today).abs() < 1.0,
            "{} vs {today}",
            forecast[0]
        );
        Ok(())
    }
}
