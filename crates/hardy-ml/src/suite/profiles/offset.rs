//! Weekly profile + current offset: the typical week shifted by how far the
//! last hour was from it, for the rest of the current opening period.

use chrono::TimeDelta;
use serde::{Deserialize, Serialize};

use super::{WeeklyProfile, open_values};
use crate::{
    error::MlError,
    grid::HORIZON_STEPS,
    suite::{Forecaster, Observed},
};

/// The offset is the mean deviation over this window before the origin.
const OFFSET_WINDOW: TimeDelta = TimeDelta::hours(1);

/// See the module docs.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ProfileCurrentOffset {
    profile: Option<WeeklyProfile>,
}

impl Forecaster for ProfileCurrentOffset {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(7)
    }

    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
        self.profile = Some(WeeklyProfile::fit(data));
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let profile = self
            .profile
            .as_ref()
            .ok_or_else(|| MlError::Fit("model is not fitted".to_string()))?;
        let origin = data.origin();
        let (sum, n) = open_values(
            data,
            data.grid(),
            origin - OFFSET_WINDOW + TimeDelta::seconds(1),
        )
        .fold((0.0, 0.0), |(sum, n), (end, value)| {
            (sum + value - profile.at(end), n + 1.0)
        });
        let offset = if n > 0.0 { sum / n } else { 0.0 };
        // A busy morning says little about tomorrow: the offset holds until
        // the gym next closes.
        let closing = data.schedule().next_closing_after(origin);
        let grid = data.grid();
        Ok((1..=HORIZON_STEPS)
            .map(|k| {
                let end = grid.step_end(k);
                profile.at(end) + if end <= closing { offset } else { 0.0 }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{
        grid::Grid,
        test_support::{local, minutely},
    };

    #[test]
    fn test_offset_shifts_the_profile_until_closing() -> Result<()> {
        let schedule = GymSchedule::default();
        // A flat week at 30; day 7 runs 10 points busier from 9:00.
        let history = minutely(8, |t| if t >= local(7, 9, 0) { 40.0 } else { 30.0 });
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(8, 0, 0));
        let fit_history = history.until(local(7, 0, 0));
        let mut model = ProfileCurrentOffset::default();
        model.fit(&Observed::new(
            &grid,
            &fit_history,
            &schedule,
            local(7, 0, 0),
        ))?;

        let origin = local(7, 12, 0);
        let seen = history.until(origin);
        let forecast = model.predict(&Observed::new(&grid, &seen, &schedule, origin))?;
        assert_relative_eq!(forecast[0], 40.0, epsilon = 1e-9);
        // 22:45–23:00 is the last open slot (step 44); tomorrow is typical.
        assert_relative_eq!(forecast[43], 40.0, epsilon = 1e-9);
        assert_relative_eq!(forecast[80], 30.0, epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn test_without_recent_readings_the_profile_is_used() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = minutely(7, |_| 30.0);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(8, 0, 0));
        let mut model = ProfileCurrentOffset::default();
        model.fit(&Observed::new(&grid, &history, &schedule, local(7, 0, 0)))?;
        // Day 7 has no readings at all.
        let forecast =
            model.predict(&Observed::new(&grid, &history, &schedule, local(7, 12, 0)))?;
        assert_relative_eq!(forecast[0], 30.0, epsilon = 1e-9);
        assert!(
            ProfileCurrentOffset::default()
                .predict(&Observed::new(&grid, &history, &schedule, local(7, 12, 0)))
                .is_err()
        );
        Ok(())
    }
}
