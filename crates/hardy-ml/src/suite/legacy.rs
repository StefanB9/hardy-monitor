//! The model from before the suite: the profile baseline with a decaying
//! "busier than usual" offset.

use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};

use super::{Forecaster, Observed};
use crate::{
    baseline::Baseline,
    error::MlError,
    grid::{HORIZON_STEPS, SLOT},
};

/// Whole hours the regressions forecast; steps in between are interpolated.
const HOURS_AHEAD: u32 = 24;
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
