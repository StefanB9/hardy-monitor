//! Exponential smoothing of the deviation from the weekly profile: simple,
//! Holt (linear trend), damped Holt, and Theta (simple smoothing plus half
//! the series' linear drift, Hyndman & Billah 2003).

use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};

use super::{
    ols::least_squares,
    optimize::{bounded, minimize, unbounded},
    series::{Slot, deviations_after, open_deviations},
};
use crate::{
    error::MlError,
    grid::{HORIZON_STEPS, is_open_slot},
    suite::{Forecaster, Observed, profiles::WeeklyProfile},
};

/// Deviations are capped: the profile plus this already spans 0–100.
const MAX_DEVIATION: f64 = 100.0;
/// Nelder–Mead iterations of a fit.
const FIT_ITERATIONS: u64 = 200;
/// Range of the smoothing weights α and β.
const WEIGHT_RANGE: (f64, f64) = (0.001, 0.999);
/// Range of the damping factor φ.
const DAMPING_RANGE: (f64, f64) = (0.8, 0.98);

/// Which smoothing model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Kind {
    Simple,
    Holt,
    Damped,
    Theta,
}

/// Smoothing weights and state of one model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Smoother {
    alpha: f64,
    beta: f64,
    phi: f64,
    level: f64,
    trend: f64,
}

impl Smoother {
    /// A smoother with the weights in `params` (unconstrained, as many as
    /// `kind` uses), starting at `level` without trend.
    fn new(kind: Kind, params: &[f64], level: f64) -> Self {
        let weight = |i: usize| {
            params
                .get(i)
                .map_or(0.0, |&p| bounded(p, WEIGHT_RANGE.0, WEIGHT_RANGE.1))
        };
        let (beta, phi) = match kind {
            Kind::Simple | Kind::Theta => (0.0, 1.0),
            Kind::Holt => (weight(1), 1.0),
            Kind::Damped => (
                weight(1),
                params.get(2).map_or(DAMPING_RANGE.1, |&p| {
                    bounded(p, DAMPING_RANGE.0, DAMPING_RANGE.1)
                }),
            ),
        };
        Self {
            alpha: weight(0),
            beta,
            phi,
            level,
            trend: 0.0,
        }
    }

    /// The one-step forecast.
    fn expected(&self) -> f64 {
        self.level + self.phi * self.trend
    }

    /// Takes in one open slot (`None` if it has no reading); returns the
    /// one-step error of an observed value.
    fn step(&mut self, value: Option<f64>) -> Option<f64> {
        let expected = self.expected();
        let error = value.map(|v| v - expected);
        let e = error.unwrap_or(0.0);
        self.level = expected + self.alpha * e;
        self.trend = self.phi * self.trend + self.alpha * self.beta * e;
        error
    }

    /// The forecast `h` open slots ahead (h ≥ 1).
    fn ahead(&self, h: usize) -> f64 {
        let h = i32::try_from(h).unwrap_or(i32::MAX);
        #[allow(clippy::float_cmp)]
        let damped_steps = if self.phi == 1.0 {
            f64::from(h)
        } else {
            self.phi * (1.0 - self.phi.powi(h)) / (1.0 - self.phi)
        };
        (self.level + damped_steps * self.trend).clamp(-MAX_DEVIATION, MAX_DEVIATION)
    }
}

/// Theta's drift term `h` open slots ahead (Hyndman & Billah 2003): half the
/// linear slope, phased in as simple smoothing would.
fn theta_drift(drift: f64, alpha: f64, observed: u64, h: usize) -> f64 {
    let n = i32::try_from(observed).unwrap_or(i32::MAX);
    #[allow(clippy::cast_precision_loss)]
    let h = h as f64;
    0.5 * drift * (h - 1.0 + 1.0 / alpha - (1.0 - alpha).powi(n) / alpha)
}

/// A fitted smoothing model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Fitted {
    profile: WeeklyProfile,
    smoother: Smoother,
    /// Theta only: the slope of the deviation per open slot.
    drift: f64,
    /// Values taken in so far.
    observed: u64,
    /// End of the last slot taken in.
    end: DateTime<Utc>,
}

/// Mean squared one-step error of a smoother with `params` over `series`.
fn one_step_cost(kind: Kind, params: &[f64], first: f64, series: &[Option<f64>]) -> f64 {
    let mut smoother = Smoother::new(kind, params, first);
    let (sum, count) = series
        .iter()
        .filter_map(|&v| smoother.step(v))
        .fold((0.0, 0_u32), |(s, n), e| (s + e * e, n + 1));
    if count == 0 {
        0.0
    } else {
        sum / f64::from(count)
    }
}

/// Slope of the observed values against their position.
fn linear_slope(series: &[Option<f64>]) -> f64 {
    let (x, y): (Vec<[f64; 2]>, Vec<f64>) = series
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f64;
            v.map(|v| ([1.0, t], v))
        })
        .unzip();
    let x: Vec<f64> = x.into_iter().flatten().collect();
    least_squares(&x, &y, 2).map_or(0.0, |b| b[1])
}

fn fit(kind: Kind, data: &Observed<'_>) -> Result<Fitted, MlError> {
    let profile = WeeklyProfile::fit(data);
    let series = open_deviations(data, &profile);
    let Some(first) = series.iter().position(Option::is_some) else {
        return Err(MlError::InsufficientData {
            found: 0,
            required: 1,
        });
    };
    let level = series[first].unwrap_or(0.0);
    let rest = &series[first + 1..];
    let start: Vec<f64> = match kind {
        Kind::Simple | Kind::Theta => vec![unbounded(0.3, WEIGHT_RANGE.0, WEIGHT_RANGE.1)],
        Kind::Holt => vec![
            unbounded(0.3, WEIGHT_RANGE.0, WEIGHT_RANGE.1),
            unbounded(0.1, WEIGHT_RANGE.0, WEIGHT_RANGE.1),
        ],
        Kind::Damped => vec![
            unbounded(0.3, WEIGHT_RANGE.0, WEIGHT_RANGE.1),
            unbounded(0.1, WEIGHT_RANGE.0, WEIGHT_RANGE.1),
            unbounded(0.9, DAMPING_RANGE.0, DAMPING_RANGE.1),
        ],
    };
    let params = minimize(
        |p| one_step_cost(kind, p, level, rest),
        &start,
        1.0,
        FIT_ITERATIONS,
    );
    let mut smoother = Smoother::new(kind, &params, level);
    for &value in rest {
        smoother.step(value);
    }
    let observed = series.iter().filter(|v| v.is_some()).count();
    Ok(Fitted {
        profile,
        smoother,
        drift: if kind == Kind::Theta {
            linear_slope(&series)
        } else {
            0.0
        },
        observed: u64::try_from(observed).unwrap_or(u64::MAX),
        end: data.origin(),
    })
}

fn predict(fitted: &Fitted, kind: Kind, data: &Observed<'_>) -> Vec<f64> {
    let mut smoother = fitted.smoother;
    let mut observed = fitted.observed;
    for (_, slot) in deviations_after(data, &fitted.profile, fitted.end) {
        match slot {
            Slot::Closed => {}
            Slot::Missing => {
                smoother.step(None);
            }
            Slot::Value(v) => {
                smoother.step(Some(v));
                observed += 1;
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
            let mut deviation = smoother.ahead(h);
            if kind == Kind::Theta {
                deviation += theta_drift(fitted.drift, smoother.alpha, observed, h);
            }
            fitted.profile.at(end) + deviation.clamp(-MAX_DEVIATION, MAX_DEVIATION)
        })
        .collect()
}

macro_rules! smoothing_model {
    ($(#[$doc:meta])* $name:ident, $kind:expr, $min_days:expr) => {
        $(#[$doc])*
        #[derive(Debug, Default, Serialize, Deserialize)]
        pub struct $name {
            fitted: Option<Fitted>,
        }

        impl Forecaster for $name {
            fn min_history(&self) -> TimeDelta {
                TimeDelta::days($min_days)
            }

            fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
                self.fitted = Some(fit($kind, data)?);
                Ok(())
            }

            fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
                let fitted = self
                    .fitted
                    .as_ref()
                    .ok_or_else(|| MlError::Fit("model is not fitted".to_string()))?;
                Ok(predict(fitted, $kind, data))
            }
        }
    };
}

smoothing_model!(
    /// Simple exponential smoothing of the deviation from the weekly profile.
    SimpleSmoothing,
    Kind::Simple,
    1
);
smoothing_model!(
    /// Holt's linear trend on the deviation from the weekly profile.
    HoltSmoothing,
    Kind::Holt,
    1
);
smoothing_model!(
    /// Damped Holt on the deviation from the weekly profile.
    DampedHolt,
    Kind::Damped,
    1
);
smoothing_model!(
    /// The Theta method on the deviation from the weekly profile.
    Theta,
    Kind::Theta,
    2
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
        grid::{Grid, SLOT, slot_end_of},
        history::History,
        suite::forecast,
        test_support::{local, minutely},
    };

    fn smoother(alpha: f64, beta: f64, phi: f64, level: f64, trend: f64) -> Smoother {
        Smoother {
            alpha,
            beta,
            phi,
            level,
            trend,
        }
    }

    #[test]
    fn test_simple_smoothing_reference_values() {
        let mut s = smoother(0.5, 0.0, 1.0, 0.0, 0.0);
        assert_eq!(s.step(Some(2.0)), Some(2.0));
        assert_relative_eq!(s.level, 1.0);
        assert_eq!(s.step(Some(4.0)), Some(3.0));
        assert_relative_eq!(s.level, 2.5);
        assert_relative_eq!(s.ahead(3), 2.5);
    }

    #[test]
    fn test_holt_reference_values() {
        // ŷ = 10, e = 2: l = 10 + 0.5·2, b = 0 + 0.5·0.5·2.
        let mut s = smoother(0.5, 0.5, 1.0, 10.0, 0.0);
        s.step(Some(12.0));
        assert_relative_eq!(s.level, 11.0);
        assert_relative_eq!(s.trend, 0.5);
        assert_relative_eq!(s.ahead(2), 12.0);
        assert_relative_eq!(s.expected(), 11.5);
    }

    #[test]
    fn test_damped_trend_fades() {
        let s = smoother(0.5, 0.5, 0.5, 10.0, 1.0);
        assert_relative_eq!(s.ahead(1), 10.5);
        assert_relative_eq!(s.ahead(2), 10.75);
        assert!(s.ahead(200) <= 11.0);
    }

    #[test]
    fn test_a_missing_value_follows_the_forecast() {
        let mut s = smoother(0.5, 0.5, 1.0, 10.0, 1.0);
        assert_eq!(s.step(None), None);
        assert_relative_eq!(s.level, 11.0);
        assert_relative_eq!(s.trend, 1.0);
    }

    #[test]
    fn test_theta_drift_reference_values() {
        // ½·2·(h − 1 + 1/α − (1 − α)ⁿ/α) with α = 0.5, n large, h = 1: 2.
        assert_relative_eq!(theta_drift(2.0, 0.5, 1_000, 1), 2.0, epsilon = 1e-12);
        assert_relative_eq!(theta_drift(2.0, 0.5, 1_000, 3), 4.0, epsilon = 1e-12);
        // With one observation (1 − α)¹/α = 1: h − 1 + 2 − 1.
        assert_relative_eq!(theta_drift(2.0, 0.5, 1, 1), 1.0, epsilon = 1e-12);
    }

    /// The same value at each time of the week.
    fn weekly(t: DateTime<Utc>) -> f64 {
        let local = (slot_end_of(t) - TimeDelta::minutes(1))
            .with_timezone(&GymSchedule::default().timezone());
        20.0 + f64::from(local.weekday().num_days_from_monday()) * 4.0 + f64::from(local.hour())
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
    fn test_every_kind_reproduces_an_exactly_repeating_week() -> Result<()> {
        let history = minutely(16, weekly);
        let origin = local(15, 12, 0);
        let schedule = GymSchedule::default();
        let models: [Box<dyn Forecaster>; 4] = [
            Box::new(SimpleSmoothing::default()),
            Box::new(HoltSmoothing::default()),
            Box::new(DampedHolt::default()),
            Box::new(Theta::default()),
        ];
        for mut model in models {
            let values = fitted_forecast(model.as_mut(), &history, local(15, 0, 0), origin)?;
            for (k, value) in values.iter().enumerate() {
                let end = origin + SLOT * i32::try_from(k + 1)?;
                if is_open_slot(&schedule, end) {
                    assert_relative_eq!(
                        *value,
                        weekly(end - TimeDelta::minutes(1)),
                        epsilon = 1e-6
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_todays_offset_carries_into_the_forecast() -> Result<()> {
        // Today runs 10 points above the usual week from opening on.
        let history = minutely(16, |t| {
            weekly(t) + if t >= local(15, 0, 0) { 10.0 } else { 0.0 }
        });
        let origin = local(15, 12, 0);
        let values = fitted_forecast(
            &mut SimpleSmoothing::default(),
            &history,
            local(15, 0, 0),
            origin,
        )?;
        let usual = weekly(origin);
        assert!(values[0] - usual > 8.0, "{} vs {usual}", values[0]);
        Ok(())
    }

    #[test]
    fn test_predict_before_fit_fails() {
        let history = minutely(2, |_| 30.0);
        let schedule = GymSchedule::default();
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(2, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(1, 12, 0));
        let result = Theta::default().predict(&data);
        assert!(result.is_err());
        let model = SimpleSmoothing::default();
        assert_eq!(model.min_history(), TimeDelta::days(1));
        assert_eq!(Theta::default().min_history(), TimeDelta::days(2));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Whatever the weights and data, forecasts stay finite and within
        /// the deviation cap.
        #[test]
        fn test_forecasts_stay_bounded(
            alpha in 0.001_f64..0.999,
            beta in 0.001_f64..0.999,
            phi in 0.8_f64..1.0,
            values in prop::collection::vec(prop::option::of(-100.0_f64..100.0), 1..60),
            h in 1_usize..100,
        ) {
            let mut s = smoother(alpha, alpha * beta, phi, 0.0, 0.0);
            for v in values {
                s.step(v);
            }
            let ahead = s.ahead(h);
            prop_assert!(ahead.is_finite());
            prop_assert!(ahead.abs() <= MAX_DEVIATION);
            let drift = theta_drift(5.0, alpha, 10, h);
            prop_assert!(drift.is_finite());
        }
    }
}
