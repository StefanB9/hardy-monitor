//! Additive Holt–Winters with a linear trend, on occupancy itself, with a
//! daily (m = 96) or weekly (m = 672) season. A slot's season is its time of
//! day or week; level and trend move per open slot only.

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::Tz;
use serde::{Deserialize, Serialize};

use super::{
    optimize::{bounded, minimize, unbounded},
    series::{Slot, slots_after},
};
use crate::{
    error::MlError,
    grid::{HORIZON_STEPS, SLOT, SLOTS_PER_DAY, SLOTS_PER_WEEK, is_open_slot},
    suite::{
        Forecaster, Observed,
        profiles::{slot_of_day, slot_of_week},
    },
};

/// Nelder–Mead iterations of a fit.
const FIT_ITERATIONS: u64 = 200;
/// Range of the smoothing weights α, β (relative to α) and γ.
const WEIGHT_RANGE: (f64, f64) = (0.001, 0.999);
/// Range of β: a fast-moving trend extrapolates wildly over a day.
const TREND_RANGE: (f64, f64) = (0.0001, 0.3);

/// Weights and state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct State {
    alpha: f64,
    beta: f64,
    gamma: f64,
    level: f64,
    trend: f64,
    season: Vec<f64>,
}

impl State {
    /// Takes in one open slot of season `j` (`None` if it has no reading);
    /// returns the one-step error of an observed value.
    fn step(&mut self, j: usize, value: Option<f64>) -> Option<f64> {
        let seasonal = self.season.get(j).copied().unwrap_or(0.0);
        let error = value.map(|v| v - (self.level + self.trend + seasonal));
        let e = error.unwrap_or(0.0);
        self.level += self.trend + self.alpha * e;
        self.trend += self.alpha * self.beta * e;
        if let Some(s) = self.season.get_mut(j) {
            *s += self.gamma * (1.0 - self.alpha) * e;
        }
        error
    }

    /// The forecast `h` open slots ahead, in season `j`.
    fn ahead(&self, h: usize, j: usize) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let h = h as f64;
        self.level + h * self.trend + self.season.get(j).copied().unwrap_or(0.0)
    }
}

/// Initial level, trend and season from the first two seasons of
/// `series`: consecutive slots as (season index, value), `None` where closed
/// or missing. `None` without a value in each season.
fn initial(series: &[(usize, Option<f64>)], period: usize) -> Option<(f64, f64, Vec<f64>)> {
    let first = series.get(..period)?;
    let second = series.get(period..(2 * period).min(series.len()))?;
    let mean = |slots: &[(usize, Option<f64>)]| {
        let (sum, n) = slots
            .iter()
            .filter_map(|(_, v)| *v)
            .fold((0.0, 0_u32), |(s, n), v| (s + v, n + 1));
        (n > 0).then(|| (sum / f64::from(n), n))
    };
    let (mean1, n1) = mean(first)?;
    let (mean2, _) = mean(second)?;
    let mut sums = vec![(0.0, 0_u32); period];
    for (slots, m) in [(first, mean1), (second, mean2)] {
        for &(j, value) in slots {
            if let (Some(v), Some(sum)) = (value, sums.get_mut(j)) {
                sum.0 += v - m;
                sum.1 += 1;
            }
        }
    }
    let season = sums
        .into_iter()
        .map(|(sum, n)| if n > 0 { sum / f64::from(n) } else { 0.0 })
        .collect();
    Some((mean1, (mean2 - mean1) / f64::from(n1), season))
}

/// A fitted model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Fitted {
    period: usize,
    tz: Tz,
    state: State,
    end: DateTime<Utc>,
}

/// Season index of the slot ending at `end`.
fn season_of(period: usize, end: DateTime<Utc>, tz: Tz) -> usize {
    if period == SLOTS_PER_DAY {
        slot_of_day(end, tz)
    } else {
        slot_of_week(end, tz)
    }
}

fn fit(period: usize, data: &Observed<'_>) -> Result<Fitted, MlError> {
    let tz = data.schedule().timezone();
    let from = data.history().first_time().unwrap_or(data.origin()) - SLOT;
    let slots: Vec<(usize, Slot)> = slots_after(data, from)
        .map(|(end, slot)| (season_of(period, end, tz), slot))
        .collect();
    let start_slots: Vec<(usize, Option<f64>)> = slots
        .iter()
        .take(2 * period)
        .map(|&(j, slot)| match slot {
            Slot::Value(v) => (j, Some(v)),
            _ => (j, None),
        })
        .collect();
    let (level, trend, season) =
        initial(&start_slots, period).ok_or(MlError::InsufficientData {
            found: 0,
            required: 2 * period,
        })?;
    let open: Vec<(usize, Option<f64>)> = slots
        .iter()
        .filter_map(|&(j, slot)| match slot {
            Slot::Closed => None,
            Slot::Missing => Some((j, None)),
            Slot::Value(v) => Some((j, Some(v))),
        })
        .collect();
    let state_for = |p: &[f64]| State {
        alpha: bounded(p[0], WEIGHT_RANGE.0, WEIGHT_RANGE.1),
        beta: bounded(p[1], TREND_RANGE.0, TREND_RANGE.1),
        gamma: bounded(p[2], WEIGHT_RANGE.0, WEIGHT_RANGE.1),
        level,
        trend,
        season: season.clone(),
    };
    let cost = |p: &[f64]| {
        let mut state = state_for(p);
        let (sum, n) = open
            .iter()
            .filter_map(|&(j, v)| state.step(j, v))
            .fold((0.0, 0_u32), |(s, n), e| (s + e * e, n + 1));
        if n == 0 { 0.0 } else { sum / f64::from(n) }
    };
    let start = [
        unbounded(0.2, WEIGHT_RANGE.0, WEIGHT_RANGE.1),
        unbounded(0.01, TREND_RANGE.0, TREND_RANGE.1),
        unbounded(0.1, WEIGHT_RANGE.0, WEIGHT_RANGE.1),
    ];
    let params = minimize(cost, &start, 1.0, FIT_ITERATIONS);
    let mut state = state_for(&params);
    for &(j, value) in &open {
        state.step(j, value);
    }
    Ok(Fitted {
        period,
        tz,
        state,
        end: data.origin(),
    })
}

fn predict(fitted: &Fitted, data: &Observed<'_>) -> Vec<f64> {
    let mut state = fitted.state.clone();
    for (end, slot) in slots_after(data, fitted.end) {
        let j = season_of(fitted.period, end, fitted.tz);
        match slot {
            Slot::Closed => {}
            Slot::Missing => {
                state.step(j, None);
            }
            Slot::Value(v) => {
                state.step(j, Some(v));
            }
        }
    }
    let grid = data.grid();
    let mut h = 0;
    (1..=HORIZON_STEPS)
        .map(|k| {
            let end = grid.step_end(k);
            if !is_open_slot(data.schedule(), end) {
                return 0.0;
            }
            h += 1;
            state.ahead(h, season_of(fitted.period, end, fitted.tz))
        })
        .collect()
}

macro_rules! holt_winters_model {
    ($(#[$doc:meta])* $name:ident, $period:expr) => {
        $(#[$doc])*
        #[derive(Debug, Default, Serialize, Deserialize)]
        pub struct $name {
            fitted: Option<Fitted>,
        }

        impl Forecaster for $name {
            fn min_history(&self) -> TimeDelta {
                SLOT * i32::try_from(2 * $period).unwrap_or(i32::MAX)
            }

            fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
                self.fitted = Some(fit($period, data)?);
                Ok(())
            }

            fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
                let fitted = self
                    .fitted
                    .as_ref()
                    .ok_or_else(|| MlError::Fit("model is not fitted".to_string()))?;
                Ok(predict(fitted, data))
            }
        }
    };
}

holt_winters_model!(
    /// Holt–Winters with a daily season.
    HoltWintersDaily,
    SLOTS_PER_DAY
);
holt_winters_model!(
    /// Holt–Winters with a weekly season.
    HoltWintersWeekly,
    SLOTS_PER_WEEK
);

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;
    use chrono::{Datelike, Timelike};
    use hardy_core::GymSchedule;
    use proptest::prelude::*;

    use super::*;
    use crate::{
        grid::{Grid, slot_end_of},
        history::History,
        suite::forecast,
        test_support::{local, minutely},
    };

    fn state(level: f64, trend: f64, season: Vec<f64>) -> State {
        State {
            alpha: 0.5,
            beta: 0.5,
            gamma: 0.5,
            level,
            trend,
            season,
        }
    }

    #[test]
    fn test_step_reference_values() {
        // ŷ = 10 + 0 + 1 = 11, e = 2: l = 10 + 0.5·2, b = 0.5·0.5·2,
        // s₀ = 1 + 0.5·(1 − 0.5)·2.
        let mut s = state(10.0, 0.0, vec![1.0, -1.0]);
        assert_eq!(s.step(0, Some(13.0)), Some(2.0));
        assert_relative_eq!(s.level, 11.0);
        assert_relative_eq!(s.trend, 0.5);
        assert_relative_eq!(s.season[0], 1.5);
        assert_relative_eq!(s.ahead(1, 1), 10.5);
        assert_relative_eq!(s.ahead(3, 0), 14.0);
    }

    #[test]
    fn test_missing_value_moves_level_by_the_trend() {
        let mut s = state(10.0, 1.0, vec![0.0, 0.0]);
        assert_eq!(s.step(1, None), None);
        assert_relative_eq!(s.level, 11.0);
        assert_eq!(s.season, vec![0.0, 0.0]);
    }

    #[test]
    fn test_initial_state_from_two_seasons() {
        // Season of 2: values 10, 20 then 12, 22.
        let series = [
            (0, Some(10.0)),
            (1, Some(20.0)),
            (0, Some(12.0)),
            (1, Some(22.0)),
        ];
        let (level, trend, season) = initial(&series, 2).unwrap_or_default();
        assert_relative_eq!(level, 15.0);
        // Means 15 then 17, two slots apart.
        assert_relative_eq!(trend, 1.0);
        // Each slot's mean deviation from its season's mean: −5, +5.
        assert_relative_eq!(season[0], -5.0);
        assert_relative_eq!(season[1], 5.0);
        assert_eq!(initial(&[(0, None)], 2), None);
    }

    fn local_hour(t: DateTime<Utc>) -> (u32, u32) {
        let local = (slot_end_of(t) - TimeDelta::minutes(1))
            .with_timezone(&GymSchedule::default().timezone());
        (local.weekday().num_days_from_monday(), local.hour())
    }

    fn fitted_forecast(
        model: &mut dyn Forecaster,
        history: &History,
        fit_at: DateTime<Utc>,
        origin: DateTime<Utc>,
    ) -> Result<Vec<f64>> {
        let schedule = GymSchedule::default();
        let grid = Grid::build(history, &schedule, local(0, 0, 0), origin);
        let fit_history = history.until(fit_at);
        model.fit(&Observed::new(&grid, &fit_history, &schedule, fit_at))?;
        let seen = history.until(origin);
        Ok(forecast(
            model,
            &Observed::new(&grid, &seen, &schedule, origin),
        )?)
    }

    #[test]
    fn test_daily_season_reproduces_a_repeating_weekday() -> Result<()> {
        // The same curve every weekday (Mon 0 … Thu 3 used).
        let history = minutely(4, |t| 15.0 + f64::from(local_hour(t).1));
        let origin = local(3, 12, 0);
        let values = fitted_forecast(
            &mut HoltWintersDaily::default(),
            &history,
            local(3, 0, 0),
            origin,
        )?;
        for k in [1_usize, 4, 20] {
            let end = origin + SLOT * i32::try_from(k)?;
            let expected = 15.0 + f64::from(local_hour(end - TimeDelta::minutes(1)).1);
            assert_relative_eq!(values[k - 1], expected, epsilon = 0.5);
        }
        Ok(())
    }

    #[test]
    fn test_weekly_season_reproduces_a_repeating_week() -> Result<()> {
        let history = minutely(22, |t| {
            let (day, hour) = local_hour(t);
            20.0 + f64::from(day) * 4.0 + f64::from(hour)
        });
        let origin = local(21, 12, 0);
        let values = fitted_forecast(
            &mut HoltWintersWeekly::default(),
            &history,
            local(21, 0, 0),
            origin,
        )?;
        for k in [1_usize, 8, 40] {
            let end = origin + SLOT * i32::try_from(k)?;
            let (day, hour) = local_hour(end - TimeDelta::minutes(1));
            let expected = 20.0 + f64::from(day) * 4.0 + f64::from(hour);
            assert_relative_eq!(values[k - 1], expected, epsilon = 0.5);
        }
        Ok(())
    }

    #[test]
    fn test_minimum_history_is_two_seasons() {
        assert_eq!(
            HoltWintersDaily::default().min_history(),
            TimeDelta::days(2)
        );
        assert_eq!(
            HoltWintersWeekly::default().min_history(),
            TimeDelta::days(14)
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Any weights in range keep the state finite on any data.
        #[test]
        fn test_state_stays_finite(
            alpha in 0.001_f64..0.999,
            beta in 0.0001_f64..0.3,
            gamma in 0.001_f64..0.999,
            values in prop::collection::vec((0_usize..4, prop::option::of(0.0_f64..100.0)), 1..200),
        ) {
            let mut s = State { alpha, beta, gamma, level: 30.0, trend: 0.0, season: vec![0.0; 4] };
            for (j, v) in values {
                s.step(j, v);
            }
            prop_assert!(s.level.is_finite() && s.trend.is_finite());
            prop_assert!(s.ahead(68, 2).is_finite());
        }
    }
}
