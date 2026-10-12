//! Similar days (nearest neighbours): the earlier days of the same kind
//! (weekday or weekend) whose morning so far looked most like today's,
//! averaged at the times being forecast.

use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};

use super::{is_weekend_like, open_values, slot_date};
use crate::{
    error::MlError,
    grid::{HORIZON_STEPS, same_local_time_before},
    suite::{Forecaster, Observed},
};

/// Neighbours averaged.
const NEIGHBOURS: usize = 3;
/// Days searched back for neighbours.
const LOOKBACK_DAYS: u32 = 56;

/// See the module docs. Nothing is learned in advance: the neighbours are
/// chosen for each forecast from the data it sees.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimilarDays;

/// Days back (with their distance to today, smaller is closer) of the
/// `NEIGHBOURS` most similar earlier days.
fn neighbours(data: &Observed<'_>) -> Vec<(u32, f64)> {
    let origin = data.origin();
    let tz = data.schedule().timezone();
    let today = slot_date(origin, tz);
    let grid = data.grid();
    let observed_today: Vec<(DateTime<Utc>, f64)> =
        open_values(data, grid, origin - TimeDelta::days(1))
            .filter(|(end, _)| slot_date(*end, tz) == today)
            .collect();
    let mut candidates: Vec<(u32, f64)> = (1..=LOOKBACK_DAYS)
        .filter(|&back| {
            today
                .checked_sub_days(chrono::Days::new(u64::from(back)))
                .is_some_and(|day| is_weekend_like(day) == is_weekend_like(today))
        })
        .filter_map(|back| {
            let (sum, n) = observed_today
                .iter()
                .fold(
                    (0.0, 0_u32),
                    |(sum, n), (end, value)| match same_local_time_before(*end, back, tz)
                        .and_then(|t| grid.value_at(t))
                    {
                        Some(earlier) => (sum + (value - earlier).powi(2), n + 1),
                        None => (sum, n),
                    },
                );
            let day_seen = same_local_time_before(origin, back, tz)
                .is_some_and(|t| data.history().first_time().is_some_and(|first| first <= t));
            if n > 0 {
                Some((back, (sum / f64::from(n)).sqrt()))
            } else if day_seen && observed_today.is_empty() {
                // Nothing observed today yet: recent days count as closest.
                Some((back, f64::from(back)))
            } else {
                None
            }
        })
        .collect();
    candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    candidates.truncate(NEIGHBOURS);
    candidates
}

impl Forecaster for SimilarDays {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(2)
    }

    fn fit(&mut self, _: &Observed<'_>) -> Result<(), MlError> {
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let near = neighbours(data);
        if near.is_empty() {
            return Err(MlError::InsufficientData {
                found: 0,
                required: 1,
            });
        }
        let tz = data.schedule().timezone();
        let grid = data.grid();
        let mut previous = grid.last_known().map_or(0.0, |(_, v)| v);
        Ok((1..=HORIZON_STEPS)
            .map(|k| {
                let end = grid.step_end(k);
                let (sum, n) = near.iter().fold((0.0, 0.0), |(sum, n), (back, _)| {
                    // A neighbour's next day may lie after the origin.
                    match same_local_time_before(end, *back, tz).and_then(|t| grid.value_at(t)) {
                        Some(v) => (sum + v, n + 1.0),
                        None => (sum, n),
                    }
                });
                if n > 0.0 {
                    previous = sum / n;
                }
                previous
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

    /// Weekdays alternate between a quiet (20) and a busy (50) pattern;
    /// day 9 is busy.
    fn alternating() -> crate::history::History {
        minutely(10, |t| {
            let day = (t - local(0, 0, 0)).num_days();
            if day % 2 == 1 { 50.0 } else { 20.0 }
        })
    }

    #[test]
    fn test_neighbours_are_the_days_that_looked_alike() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = alternating();
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(10, 0, 0));
        // Day 8 is a quiet Tuesday.
        let origin = local(8, 10, 0);
        let seen = history.until(origin);
        let data = Observed::new(&grid, &seen, &schedule, origin);
        let near = neighbours(&data);
        // The quiet weekdays: 4, 6 and 8 days back (days 4, 2 and 0); the
        // weekend (2 and 3 back) is never a candidate.
        assert!(
            near.iter().all(|(back, distance)| {
                *distance < 1e-9 && (8 - i64::from(*back)).rem_euclid(2) == 0
            }),
            "{near:?}"
        );
        let forecast = SimilarDays.predict(&data)?;
        assert_relative_eq!(forecast[4], 20.0, epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn test_before_opening_the_most_recent_days_are_used() {
        let schedule = GymSchedule::default();
        let history = alternating();
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(10, 0, 0));
        let origin = local(8, 5, 0);
        let seen = history.until(origin);
        let near = neighbours(&Observed::new(&grid, &seen, &schedule, origin));
        let backs: Vec<u32> = near.iter().map(|(b, _)| *b).collect();
        // Day 8 is a Tuesday: the nearest earlier weekdays are 1, 4 and 5
        // days back (day 6 and 5 are the weekend).
        assert_eq!(backs, vec![1, 4, 5]);
    }

    #[test]
    fn test_without_earlier_days_there_is_no_forecast() {
        let schedule = GymSchedule::default();
        let history = minutely(1, |_| 30.0);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(1, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(0, 12, 0));
        assert!(SimilarDays.predict(&data).is_err());
    }
}
