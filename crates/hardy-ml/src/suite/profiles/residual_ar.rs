//! Weekly profile + residual autoregression: the deviation from the typical
//! week follows an AR(p) process within each opening period, fitted by
//! least squares and iterated forward from the last known deviations.

use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};

use super::{WeeklyProfile, open_values};
use crate::{
    error::MlError,
    grid::{HORIZON_STEPS, SLOT, is_open_slot},
    suite::{Forecaster, Observed, linalg::solve},
};

/// Lags of the autoregression (one hour of slots).
const ORDER: usize = 4;
/// Keeps the normal equations solvable when the residuals barely vary.
const RIDGE: f64 = 1e-6;
/// Deviations are capped: the profile plus this already spans 0–100.
const MAX_RESIDUAL: f64 = 100.0;
/// Largest Σ|φ| kept, so the fitted process is stationary.
const MAX_PERSISTENCE: f64 = 0.98;

/// See the module docs.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ProfileResidualAr {
    fitted: Option<(WeeklyProfile, [f64; ORDER])>,
}

/// Deviations from the profile of consecutive open slots, split where a
/// slot is closed or missing.
fn runs(data: &Observed<'_>, profile: &WeeklyProfile, from: DateTime<Utc>) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    let mut previous: Option<DateTime<Utc>> = None;
    for (end, value) in open_values(data, data.grid(), from) {
        if previous != Some(end - SLOT) {
            out.push(Vec::new());
        }
        if let Some(run) = out.last_mut() {
            run.push(value - profile.at(end));
        }
        previous = Some(end);
    }
    out
}

/// Least-squares AR coefficients (lag 1 first) of the runs; zero without
/// enough data.
fn fit_coefficients(runs: &[Vec<f64>]) -> [f64; ORDER] {
    let mut xtx = [0.0; ORDER * ORDER];
    let mut xty = [0.0; ORDER];
    let mut rows = 0;
    for run in runs {
        for t in ORDER..run.len() {
            let lags: [f64; ORDER] = std::array::from_fn(|i| run[t - 1 - i]);
            for i in 0..ORDER {
                xty[i] += lags[i] * run[t];
                for j in 0..ORDER {
                    xtx[i * ORDER + j] += lags[i] * lags[j];
                }
            }
            rows += 1;
        }
    }
    if rows <= ORDER {
        return [0.0; ORDER];
    }
    for i in 0..ORDER {
        xtx[i * ORDER + i] += RIDGE;
    }
    let mut coefficients = solve(&xtx, &xty)
        .ok()
        .and_then(|c| <[f64; ORDER]>::try_from(c).ok())
        .unwrap_or([0.0; ORDER]);
    // Σ|φ| < 1 is sufficient for stationarity: forecasts then decay back
    // to the profile instead of running away.
    let total: f64 = coefficients.iter().map(|c| c.abs()).sum();
    if total >= MAX_PERSISTENCE {
        for c in &mut coefficients {
            *c *= MAX_PERSISTENCE / total;
        }
    }
    coefficients
}

impl Forecaster for ProfileResidualAr {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(7)
    }

    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
        let profile = WeeklyProfile::fit(data);
        let from = data.history().first_time().unwrap_or(data.origin());
        let coefficients = fit_coefficients(&runs(data, &profile, from));
        self.fitted = Some((profile, coefficients));
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let (profile, coefficients) = self
            .fitted
            .as_ref()
            .ok_or_else(|| MlError::Fit("model is not fitted".to_string()))?;
        let origin = data.origin();
        // The current run of open slots up to the origin, newest last.
        let mut recent: Vec<f64> = runs(data, profile, origin - TimeDelta::hours(2))
            .pop()
            .filter(|_| {
                data.grid()
                    .last_known()
                    .is_some_and(|(end, _)| end == origin)
            })
            .unwrap_or_default();
        let grid = data.grid();
        Ok((1..=HORIZON_STEPS)
            .map(|k| {
                let end = grid.step_end(k);
                if !is_open_slot(data.schedule(), end) {
                    // A new opening period starts from the profile.
                    recent.clear();
                    return 0.0;
                }
                let residual = if recent.len() >= ORDER {
                    coefficients
                        .iter()
                        .enumerate()
                        .map(|(i, c)| c * recent[recent.len() - 1 - i])
                        .sum::<f64>()
                        .clamp(-MAX_RESIDUAL, MAX_RESIDUAL)
                } else {
                    recent.last().copied().unwrap_or(0.0)
                };
                recent.push(residual);
                profile.at(end) + residual
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
        test_support::{local, minutely},
    };

    #[test]
    fn test_coefficients_of_a_known_ar1_process() {
        // r_t = 0.5 r_{t-1} + e_t with deterministic pseudo-random e_t.
        let mut state: u64 = 42;
        let mut noise = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            unit * 2.0 - 1.0
        };
        let runs: Vec<Vec<f64>> = (0..40)
            .map(|_| {
                let mut r = vec![0.0];
                for _ in 0..60 {
                    let next = r[r.len() - 1] * 0.5 + noise();
                    r.push(next);
                }
                r
            })
            .collect();
        let c = fit_coefficients(&runs);
        assert_relative_eq!(c[0], 0.5, epsilon = 0.05);
        assert!(c[1..].iter().all(|x| x.abs() < 0.08), "{c:?}");
    }

    #[test]
    fn test_explosive_fits_are_shrunk_to_stationary() {
        // r_t = 1.5 r_{t-1}: grows without bound.
        let run: Vec<f64> = (0..20).map(|i| 1.5_f64.powi(i)).collect();
        let c = fit_coefficients(&[run]);
        let total: f64 = c.iter().map(|x| x.abs()).sum();
        assert!(total < 1.0, "{c:?}");
    }

    #[test]
    fn test_too_little_data_gives_no_autoregression() {
        assert_eq!(fit_coefficients(&[vec![1.0, 2.0, 3.0]]), [0.0; ORDER]);
    }

    #[test]
    fn test_deviation_decays_back_to_the_profile() -> Result<()> {
        let schedule = GymSchedule::default();
        // Each day: typical 30, but from 12:00 the deviation halves every
        // slot from +16 (after a spike).
        let history = minutely(9, |t| {
            let day = (t - local(0, 0, 0)).num_days();
            let spike = local(day, 12, 0);
            if t > spike {
                let slots = (t - spike).num_minutes().div_euclid(15);
                30.0 + 16.0 * 0.5_f64.powi(i32::try_from(slots).unwrap_or(0))
            } else {
                30.0
            }
        });
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(9, 0, 0));
        let fit = history.until(local(8, 0, 0));
        let mut model = ProfileResidualAr::default();
        model.fit(&Observed::new(&grid, &fit, &schedule, local(8, 0, 0)))?;

        let origin = local(8, 12, 30);
        let seen = history.until(origin);
        let forecast = model.predict(&Observed::new(&grid, &seen, &schedule, origin))?;
        // Far ahead the forecast returns to the typical level.
        assert!((forecast[20] - profile_level(&model, &grid)).abs() < 1.0);
        assert!(forecast[0] > forecast[3], "the deviation decays");
        Ok(())
    }

    fn profile_level(model: &ProfileResidualAr, _: &Grid) -> f64 {
        model
            .fitted
            .as_ref()
            .map_or(0.0, |(p, _)| p.at(local(8, 18, 0)))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Whatever the data, forecasts stay finite and within the cap.
        #[test]
        fn test_forecasts_stay_bounded(values in prop::collection::vec(0.0_f64..100.0, 64)) {
            let schedule = GymSchedule::default();
            let history = minutely(8, |t| {
                let i = usize::try_from(t.timestamp().rem_euclid(64)).unwrap_or(0);
                values[i]
            });
            let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(8, 0, 0));
            let mut model = ProfileResidualAr::default();
            let data = Observed::new(&grid, &history, &schedule, local(7, 20, 0));
            prop_assert!(model.fit(&data).is_ok());
            let forecast = model.predict(&data).unwrap_or_default();
            prop_assert_eq!(forecast.len(), HORIZON_STEPS);
            prop_assert!(forecast.iter().all(|v| v.is_finite() && v.abs() <= 200.0));
        }
    }
}
