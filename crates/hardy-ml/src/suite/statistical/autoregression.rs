//! Autoregression on the deviation from the weekly profile: AR(p) with an
//! intercept, the order chosen by AIC, fitted by least squares (faer).
//! Closed slots are skipped, so the first slots of a day follow on from the
//! last ones of the day before.

use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};

use super::{
    ols::least_squares,
    series::{Slot, deviations_after, open_deviations},
};
use crate::{
    error::MlError,
    grid::{HORIZON_STEPS, is_open_slot},
    suite::{Forecaster, Observed, profiles::WeeklyProfile},
};

/// Highest order tried (three hours of slots).
const MAX_ORDER: usize = 12;
/// Largest Σ|φ| kept, so the fitted process is stationary.
const MAX_PERSISTENCE: f64 = 0.98;
/// Deviations are capped: the profile plus this already spans 0–100.
const MAX_DEVIATION: f64 = 100.0;

/// Intercept and lag coefficients (lag 1 first).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Process {
    intercept: f64,
    coefficients: Vec<f64>,
}

impl Process {
    /// The next value after `recent` (newest last).
    fn next(&self, recent: &[f64]) -> f64 {
        self.intercept
            + self
                .coefficients
                .iter()
                .zip(recent.iter().rev())
                .map(|(c, x)| c * x)
                .sum::<f64>()
    }
}

/// The AR process of `series` with the lowest AIC, every order fitted on
/// the same steps (those with all [`MAX_ORDER`] lags observed); shrunk to
/// stationary. No lags without enough data.
fn fit_process(series: &[Option<f64>]) -> Process {
    // Steps whose value and every lag up to the highest order are known.
    let steps: Vec<usize> = (MAX_ORDER..series.len())
        .filter(|&t| series[t - MAX_ORDER..=t].iter().all(Option::is_some))
        .collect();
    let value = |t: usize| series[t].unwrap_or(0.0);
    let y: Vec<f64> = steps.iter().map(|&t| value(t)).collect();
    #[allow(clippy::cast_precision_loss)]
    let n = y.len() as f64;
    let mut best: Option<(f64, Process)> = None;
    for order in 1..=MAX_ORDER {
        if steps.len() <= 2 * (order + 1) {
            break;
        }
        let x: Vec<f64> = steps
            .iter()
            .flat_map(|&t| std::iter::once(1.0).chain((1..=order).map(move |lag| value(t - lag))))
            .collect();
        let Some(b) = least_squares(&x, &y, order + 1) else {
            continue;
        };
        let sse: f64 = (0..steps.len())
            .map(|row| {
                let fitted: f64 = (0..=order).map(|k| x[row * (order + 1) + k] * b[k]).sum();
                (y[row] - fitted).powi(2)
            })
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let aic = n * (sse.max(1e-12) / n).ln() + 2.0 * (order + 1) as f64;
        if best.as_ref().is_none_or(|(a, _)| aic < *a) {
            best = Some((
                aic,
                Process {
                    intercept: b[0],
                    coefficients: b[1..].to_vec(),
                },
            ));
        }
    }
    let mut process = best.map_or(
        Process {
            intercept: 0.0,
            coefficients: Vec::new(),
        },
        |(_, p)| p,
    );
    // Σ|φ| < 1 is sufficient for stationarity: forecasts then settle
    // instead of running away.
    let total: f64 = process.coefficients.iter().map(|c| c.abs()).sum();
    if total >= MAX_PERSISTENCE {
        for c in &mut process.coefficients {
            *c *= MAX_PERSISTENCE / total;
        }
    }
    process
}

/// See the module docs.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Autoregression {
    fitted: Option<Fitted>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Fitted {
    profile: WeeklyProfile,
    process: Process,
    /// The last [`MAX_ORDER`] deviations, newest last (missing ones filled
    /// in with the process's own forecast).
    recent: Vec<f64>,
    /// End of the last slot taken in.
    end: DateTime<Utc>,
}

/// Appends `value` (or the process's forecast when missing) to `recent`,
/// keeping the newest [`MAX_ORDER`].
fn push(recent: &mut Vec<f64>, process: &Process, value: Option<f64>) {
    let value = value.unwrap_or_else(|| process.next(recent));
    recent.push(value);
    if recent.len() > MAX_ORDER {
        recent.remove(0);
    }
}

impl Forecaster for Autoregression {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(1)
    }

    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
        let profile = WeeklyProfile::fit(data);
        let series = open_deviations(data, &profile);
        let process = fit_process(&series);
        let mut recent = Vec::with_capacity(MAX_ORDER + 1);
        for &value in &series {
            push(&mut recent, &process, value);
        }
        self.fitted = Some(Fitted {
            profile,
            process,
            recent,
            end: data.origin(),
        });
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| MlError::Fit("model is not fitted".to_string()))?;
        let mut recent = fitted.recent.clone();
        for (_, slot) in deviations_after(data, &fitted.profile, fitted.end) {
            match slot {
                Slot::Closed => {}
                Slot::Missing => push(&mut recent, &fitted.process, None),
                Slot::Value(v) => push(&mut recent, &fitted.process, Some(v)),
            }
        }
        let grid = data.grid();
        Ok((1..=HORIZON_STEPS)
            .map(|k| {
                let end = grid.step_end(k);
                if !is_open_slot(data.schedule(), end) {
                    return 0.0;
                }
                let deviation = fitted
                    .process
                    .next(&recent)
                    .clamp(-MAX_DEVIATION, MAX_DEVIATION);
                push(&mut recent, &fitted.process, Some(deviation));
                fitted.profile.at(end) + deviation
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;
    use hardy_core::GymSchedule;
    use proptest::prelude::*;

    use super::*;
    use crate::{
        grid::Grid,
        suite::forecast,
        test_support::{local, minutely},
    };

    /// Deterministic noise in −1…1.
    fn noise(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            unit * 2.0 - 1.0
        }
    }

    #[test]
    fn test_process_next_reference_value() {
        let process = Process {
            intercept: 1.0,
            coefficients: vec![0.5, -0.25],
        };
        // 1 + 0.5·4 − 0.25·8 (lag 1 is the newest).
        assert_relative_eq!(process.next(&[8.0, 4.0]), 1.0);
        // Too short a history: missing lags count as 0.
        assert_relative_eq!(process.next(&[4.0]), 3.0);
    }

    #[test]
    fn test_fit_recovers_an_ar2_process() {
        let mut e = noise(7);
        let mut series = vec![0.0, 0.0];
        for _ in 0..4_000 {
            let n = series.len();
            series.push(0.6 * series[n - 1] - 0.3 * series[n - 2] + e());
        }
        let series: Vec<Option<f64>> = series.into_iter().map(Some).collect();
        let process = fit_process(&series);
        assert!(process.coefficients.len() >= 2, "{process:?}");
        assert_relative_eq!(process.coefficients[0], 0.6, epsilon = 0.05);
        assert_relative_eq!(process.coefficients[1], -0.3, epsilon = 0.05);
        assert!(process.coefficients[2..].iter().all(|c| c.abs() < 0.06));
        assert!(process.intercept.abs() < 0.05);
    }

    #[test]
    fn test_explosive_fits_are_shrunk_to_stationary() {
        let series: Vec<Option<f64>> = (0..40).map(|i| Some(1.2_f64.powi(i))).collect();
        let process = fit_process(&series);
        let total: f64 = process.coefficients.iter().map(|c| c.abs()).sum();
        assert!(total < 1.0, "{process:?}");
    }

    #[test]
    fn test_too_little_data_gives_no_lags() {
        let process = fit_process(&[Some(1.0), Some(2.0), None, Some(3.0)]);
        assert_eq!(process.coefficients, Vec::<f64>::new());
    }

    #[test]
    fn test_reproduces_an_exactly_repeating_day() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = minutely(3, |_| 30.0);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(3, 0, 0));
        let mut model = Autoregression::default();
        let fit_history = history.until(local(2, 0, 0));
        model.fit(&Observed::new(
            &grid,
            &fit_history,
            &schedule,
            local(2, 0, 0),
        ))?;
        let seen = history.until(local(2, 12, 0));
        let values = forecast(
            &model,
            &Observed::new(&grid, &seen, &schedule, local(2, 12, 0)),
        )?;
        assert!(
            values
                .iter()
                .filter(|v| **v > 0.0)
                .all(|v| (v - 30.0).abs() < 1e-6)
        );
        assert_eq!(model.min_history(), TimeDelta::days(1));
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Whatever the data, the process is stationary and iterating it
        /// stays bounded.
        #[test]
        fn test_fitted_processes_are_stationary(
            values in prop::collection::vec(prop::option::of(-60.0_f64..60.0), 0..200),
        ) {
            let process = fit_process(&values);
            let total: f64 = process.coefficients.iter().map(|c| c.abs()).sum();
            prop_assert!(total < 1.0);
            prop_assert!(process.coefficients.len() <= MAX_ORDER);
            let mut recent = vec![50.0; MAX_ORDER];
            for _ in 0..500 {
                let next = process.next(&recent);
                prop_assert!(next.is_finite());
                push(&mut recent, &process, Some(next));
            }
        }
    }
}
