//! Structural state-space model (Durbin & Koopman 2012), filtered with a
//! Kalman filter: a local linear trend plus daily and weekly seasonality as
//! Fourier terms whose coefficients drift as random walks. The slot's time
//! of day and week only enters the observation row, so closed slots can be
//! skipped without losing the seasonal phase. The noise variances (relative
//! to the observation noise, which is concentrated out of the likelihood)
//! are estimated by maximum likelihood with Nelder–Mead.

use std::f64::consts::TAU;

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

/// Daily harmonics (periods of 24 h, 12 h, …).
const DAILY_HARMONICS: usize = 8;
/// Weekly harmonics.
const WEEKLY_HARMONICS: usize = 3;
/// State size: level, slope and a cosine and sine coefficient per harmonic.
const DIM: usize = 2 + 2 * (DAILY_HARMONICS + WEEKLY_HARMONICS);
/// Initial state variance relative to the observation noise: vague, so the
/// first days set the state.
const DIFFUSE: f64 = 1e4;
/// Range of the log₁₀ variance ratios (level, slope, seasonal).
const LOG_RATIO_RANGE: (f64, f64) = (-8.0, 1.0);
/// Nelder–Mead iterations of a fit.
const FIT_ITERATIONS: u64 = 120;

/// Observation row of the slot at `day` (0–95) and `week` (0–671).
fn design(day: usize, week: usize) -> [f64; DIM] {
    let mut z = [0.0; DIM];
    z[0] = 1.0;
    #[allow(clippy::cast_precision_loss)]
    let harmonics = (1..=DAILY_HARMONICS)
        .map(|k| k as f64 * day as f64 / SLOTS_PER_DAY as f64)
        .chain((1..=WEEKLY_HARMONICS).map(|k| k as f64 * week as f64 / SLOTS_PER_WEEK as f64));
    for (i, cycles) in harmonics.enumerate() {
        z[2 + 2 * i] = (TAU * cycles).cos();
        z[3 + 2 * i] = (TAU * cycles).sin();
    }
    z
}

/// State estimate and its covariance (row-major).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Filter {
    state: Vec<f64>,
    covariance: Vec<f64>,
}

impl Filter {
    fn dim(&self) -> usize {
        self.state.len()
    }

    /// Moves one open slot ahead: the level grows by the slope, every state
    /// takes on its noise (`noise[0]` level, `[1]` slope, `[2]` the rest).
    fn advance(&mut self, noise: &[f64; 3]) {
        let n = self.dim();
        let p = &mut self.covariance;
        if n >= 2 {
            self.state[0] += self.state[1];
            // P ← T P Tᵀ with T = I + e₀e₁ᵀ: row 0 += row 1, then
            // column 0 += column 1.
            for j in 0..n {
                p[j] += p[n + j];
            }
            for i in 0..n {
                p[i * n] += p[i * n + 1];
            }
        }
        for i in 0..n {
            p[i * n + i] += noise[i.min(2)];
        }
    }

    /// Takes in observation `y` with row `z` (observation noise 1); returns
    /// the innovation and its variance.
    fn update(&mut self, z: &[f64], y: f64) -> (f64, f64) {
        let n = self.dim();
        let pz: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| self.covariance[i * n + j] * z[j]).sum())
            .collect();
        let variance = z.iter().zip(&pz).map(|(a, b)| a * b).sum::<f64>() + 1.0;
        let innovation = y - z.iter().zip(&self.state).map(|(a, b)| a * b).sum::<f64>();
        for i in 0..n {
            self.state[i] += pz[i] * innovation / variance;
            for j in 0..n {
                self.covariance[i * n + j] -= pz[i] * pz[j] / variance;
            }
            // Rounding must not leave a negative variance.
            let d = &mut self.covariance[i * n + i];
            *d = d.max(0.0);
        }
        (innovation, variance)
    }

    /// The expected observation for row `z`, `h` open slots ahead.
    fn ahead(&self, z: &[f64], h: usize) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let h = h as f64;
        let slope = self.state.get(1).copied().unwrap_or(0.0);
        z.iter().zip(&self.state).map(|(a, b)| a * b).sum::<f64>() + z[0] * h * slope
    }
}

/// Variance ratios (level, slope, seasonal) from unconstrained parameters.
fn noise_of(params: &[f64]) -> [f64; 3] {
    std::array::from_fn(|i| {
        let log = params.get(i).map_or(LOG_RATIO_RANGE.0, |&p| {
            bounded(p, LOG_RATIO_RANGE.0, LOG_RATIO_RANGE.1)
        });
        10_f64.powf(log)
    })
}

/// A fitted model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Fitted {
    tz: Tz,
    noise: [f64; 3],
    filter: Filter,
    end: DateTime<Utc>,
}

/// See the module docs.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StructuralModel {
    fitted: Option<Fitted>,
}

impl Forecaster for StructuralModel {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(2)
    }

    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
        let tz = data.schedule().timezone();
        let from = data.history().first_time().unwrap_or(data.origin()) - SLOT;
        let open: Vec<([f64; DIM], Option<f64>)> = slots_after(data, from)
            .filter_map(|(end, slot)| {
                let z = design(slot_of_day(end, tz), slot_of_week(end, tz));
                match slot {
                    Slot::Closed => None,
                    Slot::Missing => Some((z, None)),
                    Slot::Value(v) => Some((z, Some(v))),
                }
            })
            .collect();
        let first = open
            .iter()
            .find_map(|(_, v)| *v)
            .ok_or(MlError::InsufficientData {
                found: 0,
                required: 1,
            })?;
        let start_filter = || {
            let mut state = vec![0.0; DIM];
            state[0] = first;
            Filter {
                state,
                covariance: (0..DIM * DIM)
                    .map(|i| if i % (DIM + 1) == 0 { DIFFUSE } else { 0.0 })
                    .collect(),
            }
        };
        let cost = |p: &[f64]| {
            let noise = noise_of(p);
            let mut filter = start_filter();
            let (mut log_variances, mut scaled, mut n, mut seen) = (0.0, 0.0, 0_u32, 0);
            for (z, value) in &open {
                filter.advance(&noise);
                if let Some(y) = value {
                    let (v, f) = filter.update(z, *y);
                    seen += 1;
                    // The first observations only pin down the vague start.
                    if seen > DIM {
                        log_variances += f.ln();
                        scaled += v * v / f;
                        n += 1;
                    }
                }
            }
            if n == 0 {
                0.0
            } else {
                // −2 × the log-likelihood with the observation variance
                // concentrated out, constants dropped.
                log_variances + f64::from(n) * (scaled / f64::from(n)).ln()
            }
        };
        let start = [-2.0, -6.0, -4.0].map(|r| unbounded(r, LOG_RATIO_RANGE.0, LOG_RATIO_RANGE.1));
        let params = minimize(cost, &start, 1.0, FIT_ITERATIONS);
        let noise = noise_of(&params);
        let mut filter = start_filter();
        for (z, value) in &open {
            filter.advance(&noise);
            if let Some(y) = value {
                filter.update(z, *y);
            }
        }
        self.fitted = Some(Fitted {
            tz,
            noise,
            filter,
            end: data.origin(),
        });
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| MlError::Fit("model is not fitted".to_string()))?;
        let tz = fitted.tz;
        let mut filter = fitted.filter.clone();
        for (end, slot) in slots_after(data, fitted.end) {
            match slot {
                Slot::Closed => {}
                Slot::Missing => filter.advance(&fitted.noise),
                Slot::Value(y) => {
                    filter.advance(&fitted.noise);
                    filter.update(&design(slot_of_day(end, tz), slot_of_week(end, tz)), y);
                }
            }
        }
        let grid = data.grid();
        let mut h = 0;
        Ok((1..=HORIZON_STEPS)
            .map(|k| {
                let end = grid.step_end(k);
                if !is_open_slot(data.schedule(), end) {
                    return 0.0;
                }
                h += 1;
                filter.ahead(&design(slot_of_day(end, tz), slot_of_week(end, tz)), h)
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

    #[test]
    fn test_update_reference_values() {
        // Prior N(0, 1), observation 2 with unit noise: posterior N(1, ½).
        let mut f = Filter {
            state: vec![0.0],
            covariance: vec![1.0],
        };
        let (v, var) = f.update(&[1.0], 2.0);
        assert_relative_eq!(v, 2.0);
        assert_relative_eq!(var, 2.0);
        assert_relative_eq!(f.state[0], 1.0);
        assert_relative_eq!(f.covariance[0], 0.5);
    }

    #[test]
    fn test_advance_reference_values() {
        // T = [[1, 1], [0, 1]]: T·I·Tᵀ = [[2, 1], [1, 1]], plus noise.
        let mut f = Filter {
            state: vec![1.0, 2.0],
            covariance: vec![1.0, 0.0, 0.0, 1.0],
        };
        f.advance(&[0.1, 0.2, 0.3]);
        assert_eq!(f.state, vec![3.0, 2.0]);
        assert_relative_eq!(f.covariance[0], 2.1);
        assert_relative_eq!(f.covariance[1], 1.0);
        assert_relative_eq!(f.covariance[2], 1.0);
        assert_relative_eq!(f.covariance[3], 1.2);
        // h ahead: level + h·slope.
        assert_relative_eq!(f.ahead(&[1.0, 0.0], 2), 7.0);
    }

    #[test]
    fn test_design_row() {
        let z = design(0, 0);
        assert_eq!(z[0], 1.0);
        assert_eq!(z[1], 0.0);
        // At position 0 every cosine is 1 and every sine 0.
        for k in 0..DAILY_HARMONICS + WEEKLY_HARMONICS {
            assert_relative_eq!(z[2 + 2 * k], 1.0);
            assert_relative_eq!(z[3 + 2 * k], 0.0);
        }
        // A quarter day: the first daily cosine is 0, its sine 1.
        let z = design(24, 24);
        assert_relative_eq!(z[2], 0.0, epsilon = 1e-12);
        assert_relative_eq!(z[3], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_learns_a_smooth_daily_curve() -> Result<()> {
        let schedule = GymSchedule::default();
        let tz = schedule.timezone();
        let curve = |t: DateTime<Utc>| {
            #[allow(clippy::cast_precision_loss)]
            let d = slot_of_day(crate::grid::slot_end_of(t), tz) as f64 / 96.0;
            30.0 + 10.0 * (TAU * d).cos() + 5.0 * (2.0 * TAU * d).sin()
        };
        let history = minutely(4, curve);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(4, 0, 0));
        let mut model = StructuralModel::default();
        let fit_history = history.until(local(3, 0, 0));
        model.fit(&Observed::new(
            &grid,
            &fit_history,
            &schedule,
            local(3, 0, 0),
        ))?;
        let origin = local(3, 12, 0);
        let seen = history.until(origin);
        let values = forecast(&model, &Observed::new(&grid, &seen, &schedule, origin))?;
        for k in [1_usize, 4, 24] {
            let end = origin + SLOT * i32::try_from(k)?;
            assert_relative_eq!(
                values[k - 1],
                curve(end - TimeDelta::minutes(1)),
                epsilon = 1.5
            );
        }
        assert_eq!(model.min_history(), TimeDelta::days(2));
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// On any data the filter stays finite with non-negative variances.
        #[test]
        fn test_filter_stays_finite(
            noise in prop::array::uniform3(1e-8_f64..10.0),
            values in prop::collection::vec((0_usize..96, prop::option::of(0.0_f64..100.0)), 1..60),
        ) {
            let mut f = Filter {
                state: vec![0.0; DIM],
                covariance: (0..DIM * DIM).map(|i| if i % (DIM + 1) == 0 { DIFFUSE } else { 0.0 }).collect(),
            };
            for (day, value) in values {
                f.advance(&noise);
                if let Some(y) = value {
                    let (_, var) = f.update(&design(day, day * 3), y);
                    prop_assert!(var > 0.0);
                }
            }
            prop_assert!(f.state.iter().all(|x| x.is_finite()));
            prop_assert!((0..DIM).all(|i| f.covariance[i * DIM + i] >= 0.0));
            prop_assert!(f.ahead(&design(5, 5), 68).is_finite());
        }
    }
}
