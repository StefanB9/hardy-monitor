//! Baselines: simple forecasts every other model has to beat. None of them
//! learns parameters; each forecasts from the slots before the origin only.

use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};

use super::{Forecaster, Observed};
use crate::{
    error::MlError,
    grid::{GridView, HORIZON_STEPS, same_local_time_before},
};

/// Slots the recent average and trend look back over (one hour).
const RECENT_SLOTS: usize = 4;
/// Per-slot damping of the local trend: its effect levels off after a few
/// hours instead of running away.
const TREND_DAMPING: f64 = 0.9;
/// Weeks averaged by the weekly mean and median.
const SEASONAL_WEEKS: u32 = 4;
/// Weeks weighted by the recency-weighted mean.
const RECENCY_WEEKS: u32 = 8;
/// A week this many weeks old counts half as much as last week.
const RECENCY_HALF_LIFE_WEEKS: f64 = 2.0;
/// Days searched back for a same-time value before giving up.
const MAX_LOOKBACK_DAYS: u32 = 28;

fn no_data() -> MlError {
    MlError::InsufficientData {
        found: 0,
        required: 1,
    }
}

/// The newest known slot value.
fn last_value(grid: &GridView<'_>) -> Result<f64, MlError> {
    grid.last_known().map(|(_, v)| v).ok_or_else(no_data)
}

/// Known values of the slot ending at `end` on earlier days: `every` days
/// before it, then twice that, … (`count` candidates), skipping slots
/// without a reading.
fn seasonal_values(
    data: &Observed<'_>,
    end: DateTime<Utc>,
    every: u32,
    count: u32,
) -> impl Iterator<Item = (u32, f64)> {
    let grid = data.grid();
    let tz = data.schedule().timezone();
    (1..=count).filter_map(move |n| {
        let earlier = same_local_time_before(end, every * n, tz)?;
        grid.value_at(earlier).map(|v| (n, v))
    })
}

/// Same time on the most recent earlier day with a value; else the newest
/// value.
fn same_time_recent_day(data: &Observed<'_>, end: DateTime<Utc>) -> Result<f64, MlError> {
    match seasonal_values(data, end, 1, MAX_LOOKBACK_DAYS).next() {
        Some((_, v)) => Ok(v),
        None => last_value(&data.grid()),
    }
}

/// Builds a forecast from a per-step function of the step's end time.
fn per_step(
    data: &Observed<'_>,
    mut f: impl FnMut(DateTime<Utc>) -> Result<f64, MlError>,
) -> Result<Vec<f64>, MlError> {
    let grid = data.grid();
    (1..=HORIZON_STEPS).map(|k| f(grid.step_end(k))).collect()
}

/// Known values among the last `n` slots, newest last.
fn recent(grid: &GridView<'_>, n: usize) -> Vec<f64> {
    let values = grid.values();
    values[values.len().saturating_sub(n)..]
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .collect()
}

macro_rules! stateless {
    ($( $(#[$doc:meta])* $name:ident; )*) => {
        $(
            $(#[$doc])*
            #[derive(Debug, Clone, Default, Serialize, Deserialize)]
            pub struct $name;
        )*
    };
}

stateless! {
    /// The newest slot value for every step.
    Naive;
    /// Mean of the last hour for every step.
    MovingAverage;
    /// The newest value plus the last hour's slope, damped per step.
    DampedTrend;
    /// The same time yesterday (or the most recent earlier day with a value).
    DailySeasonalNaive;
    /// The same time a week ago.
    WeeklySeasonalNaive;
    /// Mean of the same time over the last four weeks.
    WeeklySeasonalMean;
    /// Median of the same time over the last four weeks.
    WeeklySeasonalMedian;
    /// Mean of the same time over the last eight weeks, recent weeks
    /// counting more.
    WeeklyRecencyMean;
}

impl Forecaster for Naive {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::zero()
    }

    fn fit(&mut self, _: &Observed<'_>) -> Result<(), MlError> {
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        Ok(vec![last_value(&data.grid())?; HORIZON_STEPS])
    }
}

impl Forecaster for MovingAverage {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::zero()
    }

    fn fit(&mut self, _: &Observed<'_>) -> Result<(), MlError> {
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let grid = data.grid();
        let window = recent(&grid, RECENT_SLOTS);
        let level = if window.is_empty() {
            last_value(&grid)?
        } else {
            #[allow(clippy::cast_precision_loss)]
            let n = window.len() as f64;
            window.iter().sum::<f64>() / n
        };
        Ok(vec![level; HORIZON_STEPS])
    }
}

/// Least-squares slope per slot of `values` (equally spaced); `None` with
/// fewer than two.
fn slope(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = values.len() as f64;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = values.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, y) in values.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let dx = i as f64 - mean_x;
        sxy += dx * (y - mean_y);
        sxx += dx * dx;
    }
    Some(sxy / sxx)
}

impl Forecaster for DampedTrend {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::zero()
    }

    fn fit(&mut self, _: &Observed<'_>) -> Result<(), MlError> {
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let grid = data.grid();
        let level = last_value(&grid)?;
        let trend = slope(&recent(&grid, RECENT_SLOTS)).unwrap_or(0.0);
        let mut damping = 1.0;
        let mut accumulated = 0.0;
        Ok((0..HORIZON_STEPS)
            .map(|_| {
                damping *= TREND_DAMPING;
                accumulated += damping;
                level + trend * accumulated
            })
            .collect())
    }
}

impl Forecaster for DailySeasonalNaive {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(1)
    }

    fn fit(&mut self, _: &Observed<'_>) -> Result<(), MlError> {
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        per_step(data, |end| same_time_recent_day(data, end))
    }
}

impl Forecaster for WeeklySeasonalNaive {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(7)
    }

    fn fit(&mut self, _: &Observed<'_>) -> Result<(), MlError> {
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        per_step(data, |end| {
            match seasonal_values(data, end, 7, MAX_LOOKBACK_DAYS / 7).next() {
                Some((_, v)) => Ok(v),
                None => same_time_recent_day(data, end),
            }
        })
    }
}

impl Forecaster for WeeklySeasonalMean {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(7)
    }

    fn fit(&mut self, _: &Observed<'_>) -> Result<(), MlError> {
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        per_step(data, |end| {
            let (sum, n) = seasonal_values(data, end, 7, SEASONAL_WEEKS)
                .fold((0.0, 0.0), |(sum, n), (_, v)| (sum + v, n + 1.0));
            if n > 0.0 {
                Ok(sum / n)
            } else {
                same_time_recent_day(data, end)
            }
        })
    }
}

/// Median of `values` (sorted in place); `None` when empty.
fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        f64::midpoint(values[mid - 1], values[mid])
    } else {
        values[mid]
    })
}

impl Forecaster for WeeklySeasonalMedian {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(7)
    }

    fn fit(&mut self, _: &Observed<'_>) -> Result<(), MlError> {
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let mut buffer = Vec::with_capacity(SEASONAL_WEEKS as usize);
        per_step(data, |end| {
            buffer.clear();
            buffer.extend(seasonal_values(data, end, 7, SEASONAL_WEEKS).map(|(_, v)| v));
            match median(&mut buffer) {
                Some(v) => Ok(v),
                None => same_time_recent_day(data, end),
            }
        })
    }
}

impl Forecaster for WeeklyRecencyMean {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(7)
    }

    fn fit(&mut self, _: &Observed<'_>) -> Result<(), MlError> {
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        per_step(data, |end| {
            let (sum, weight) = seasonal_values(data, end, 7, RECENCY_WEEKS).fold(
                (0.0, 0.0),
                |(sum, weight), (week, v)| {
                    let w = 0.5_f64.powf(f64::from(week - 1) / RECENCY_HALF_LIFE_WEEKS);
                    (sum + w * v, weight + w)
                },
            );
            if weight > 0.0 {
                Ok(sum / weight)
            } else {
                same_time_recent_day(data, end)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;
    use chrono::{Datelike, Timelike};
    use hardy_core::GymSchedule;
    use proptest::prelude::*;

    use super::*;
    use crate::{
        grid::Grid,
        history::History,
        test_support::{local, minutely},
    };

    fn run(model: &dyn Forecaster, history: &History, origin: DateTime<Utc>) -> Result<Vec<f64>> {
        let schedule = GymSchedule::default();
        let history = history.until(origin);
        let grid = Grid::build(&history, &schedule, local(-1, 0, 0), origin);
        let data = Observed::new(&grid, &history, &schedule, origin);
        Ok(model.predict(&data)?)
    }

    /// A value that encodes the day and the gym-local hour of the slot a
    /// reading at `t` falls into (constant within each slot).
    fn day_hour(t: DateTime<Utc>) -> f64 {
        let slot_start = crate::grid::slot_end_of(t) - TimeDelta::minutes(1);
        let local = slot_start.with_timezone(&GymSchedule::default().timezone());
        f64::from(local.ordinal() % 7) * 10.0 + f64::from(local.hour())
    }

    #[test]
    fn test_naive_repeats_the_newest_slot() -> Result<()> {
        let history = minutely(1, |t| if t < local(0, 10, 0) { 20.0 } else { 35.0 });
        let forecast = run(&Naive, &history, local(0, 10, 30))?;
        assert_eq!(forecast.len(), HORIZON_STEPS);
        assert!(forecast.iter().all(|&v| (v - 35.0).abs() < 1e-9));
        Ok(())
    }

    #[test]
    fn test_moving_average_uses_the_last_hour() -> Result<()> {
        // Slots ending 9:15, 9:30 at 20; 9:45, 10:00 at 40 (readings in
        // (t − 15, t]).
        let history = minutely(1, |t| if t <= local(0, 9, 30) { 20.0 } else { 40.0 });
        let forecast = run(&MovingAverage, &history, local(0, 10, 0))?;
        assert_relative_eq!(forecast[0], 30.0, epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn test_damped_trend_levels_off() -> Result<()> {
        // +1 point per minute: +15 per slot.
        let start = local(0, 9, 0);
        let history = minutely(1, |t| {
            #[allow(clippy::cast_precision_loss)]
            let minutes = (t - start).num_minutes().max(0) as f64;
            (10.0 + minutes / 4.0).min(90.0)
        });
        let forecast = run(&DampedTrend, &history, local(0, 10, 0))?;
        let level = run(&Naive, &history, local(0, 10, 0))?[0];
        // Slope 3.75 per slot; step 1 adds 0.9 × slope.
        assert_relative_eq!(forecast[0], level + 0.9 * 3.75, epsilon = 1e-9);
        // The sum of 0.9^i converges to 9: never more than 9 slopes away.
        assert!(forecast.iter().all(|&v| v <= level + 9.0 * 3.75 + 1e-9));
        assert!(forecast[95] > forecast[0]);
        Ok(())
    }

    #[test]
    fn test_daily_seasonal_naive_repeats_yesterday() -> Result<()> {
        let history = minutely(3, day_hour);
        let forecast = run(&DailySeasonalNaive, &history, local(2, 10, 0))?;
        // Step 4 ends at 11:00 on day 2: value from (10:45, 11:00] on day 1.
        assert_relative_eq!(forecast[3], day_hour(local(1, 10, 50)), epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn test_weekly_seasonal_naive_repeats_last_week() -> Result<()> {
        let history = minutely(15, day_hour);
        let forecast = run(&WeeklySeasonalNaive, &history, local(14, 10, 0))?;
        assert_relative_eq!(forecast[3], day_hour(local(7, 10, 50)), epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn test_weekly_mean_median_and_recency_weighting() -> Result<()> {
        // Same weekday-hour value every week, plus 4 points per week elapsed.
        let history = minutely(29, |t| {
            let week = (t - local(0, 0, 0)).num_days() / 7;
            #[allow(clippy::cast_precision_loss)]
            let drift = week as f64 * 4.0;
            day_hour(t) + drift
        });
        let origin = local(28, 10, 0);
        let typical = day_hour(local(28, 10, 50));
        // Weeks 3, 2, 1, 0 back from week 4: +12, +8, +4, +0.
        let mean = run(&WeeklySeasonalMean, &history, origin)?[3];
        assert_relative_eq!(mean, typical + 6.0, epsilon = 1e-9);
        let median = run(&WeeklySeasonalMedian, &history, origin)?[3];
        assert_relative_eq!(median, typical + 6.0, epsilon = 1e-9);
        // Only four weeks exist; weights 1, 2^-0.5, 2^-1, 2^-1.5.
        let w = [1.0, 0.5_f64.sqrt(), 0.5, 0.5_f64.powf(1.5)];
        let expected = (w[0] * 12.0 + w[1] * 8.0 + w[2] * 4.0) / w.iter().sum::<f64>();
        let recency = run(&WeeklyRecencyMean, &history, origin)?[3];
        assert_relative_eq!(recency, typical + expected, epsilon = 1e-9);
        assert!(recency > mean, "recent weeks count more");
        Ok(())
    }

    #[test]
    fn test_seasonal_models_fall_back_to_recent_days() -> Result<()> {
        // Only three days of data: no week-old value exists yet.
        let history = minutely(3, day_hour);
        let forecast = run(&WeeklySeasonalMean, &history, local(2, 10, 0))?;
        assert_relative_eq!(forecast[3], day_hour(local(1, 10, 50)), epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn test_baselines_need_some_data() {
        let history = History::default();
        assert!(run(&Naive, &history, local(0, 10, 0)).is_ok_and(|v| v[0] == 0.0));
        assert!(median(&mut []).is_none());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// The median lies within the values and splits them evenly.
        #[test]
        fn test_median_is_central(mut values in prop::collection::vec(-100.0_f64..100.0, 1..20)) {
            let m = median(&mut values).unwrap_or(f64::NAN);
            let below = values.iter().filter(|&&v| v < m).count();
            let above = values.iter().filter(|&&v| v > m).count();
            prop_assert!(below <= values.len() / 2 && above <= values.len() / 2);
        }
    }
}
