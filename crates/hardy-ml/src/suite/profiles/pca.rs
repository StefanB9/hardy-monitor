//! Daily-profile PCA + regression: past days as 96-slot vectors, reduced to
//! their main shapes (principal components). A day's component scores are
//! predicted from its weekday, then corrected by projecting what today has
//! shown so far onto the components.

use chrono::{Datelike, NaiveDate, TimeDelta};
use serde::{Deserialize, Serialize};

use super::{open_values, slot_date, slot_of_day};
use crate::{
    error::MlError,
    grid::{HORIZON_STEPS, SLOTS_PER_DAY},
    suite::{
        Forecaster, Observed,
        linalg::{dot, solve, top_eigenvectors},
    },
};

/// Components kept (fewer with few days).
const COMPONENTS: usize = 3;
/// A day with less of its open slots observed is left out of the fit.
const MIN_DAY_COVERAGE: f64 = 0.8;
/// Assumed variance of a slot's value around its day's shape (percentage
/// points²). Today's scores are the posterior given the weekday prior with
/// each component's variance (its eigenvalue): a component that varies a
/// lot between days follows the observations readily.
const NOISE_VARIANCE: f64 = 9.0;

/// See the module docs.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DailyPca {
    fitted: Option<Fitted>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Fitted {
    mean: Vec<f64>,
    /// Unit-length components, each `SLOTS_PER_DAY` long.
    components: Vec<Vec<f64>>,
    /// Variance of the days' scores along each component.
    eigenvalues: Vec<f64>,
    /// Mean scores per weekday (Monday first); `None` for weekdays not seen.
    weekday_scores: Vec<Option<Vec<f64>>>,
}

impl Fitted {
    /// Scores expected for `date` before anything of it is observed.
    fn prior(&self, date: NaiveDate) -> Vec<f64> {
        let weekday = date.weekday().num_days_from_monday() as usize;
        self.weekday_scores[weekday]
            .clone()
            .unwrap_or_else(|| vec![0.0; self.components.len()])
    }

    /// Scores closest to the observed `(slot, value)` pairs, pulled
    /// towards `prior`.
    fn fit_scores(&self, observed: &[(usize, f64)], prior: &[f64]) -> Vec<f64> {
        let k = self.components.len();
        let mut a = vec![0.0; k * k];
        let mut b = vec![0.0; k];
        for i in 0..k {
            let weight = NOISE_VARIANCE / self.eigenvalues[i].max(1e-9);
            a[i * k + i] = weight;
            b[i] = weight * prior[i];
        }
        for &(slot, value) in observed {
            let residual = value - self.mean[slot];
            for i in 0..k {
                b[i] += self.components[i][slot] * residual;
                for j in 0..k {
                    a[i * k + j] += self.components[i][slot] * self.components[j][slot];
                }
            }
        }
        solve(&a, &b).unwrap_or_else(|_| prior.to_vec())
    }

    fn reconstruct(&self, scores: &[f64], slot: usize) -> f64 {
        self.mean[slot]
            + self
                .components
                .iter()
                .zip(scores)
                .map(|(c, s)| c[slot] * s)
                .sum::<f64>()
    }
}

/// The complete days in `data` as (date, 96 slot values); closed slots 0.
fn day_vectors(data: &Observed<'_>) -> Vec<(NaiveDate, Vec<f64>)> {
    let tz = data.schedule().timezone();
    let origin = data.origin();
    let from = data.history().first_time().unwrap_or(origin);
    let today = slot_date(origin, tz);
    let mut days: Vec<(NaiveDate, Vec<f64>, usize)> = Vec::new();
    for (end, value) in open_values(data, data.grid(), from) {
        let day = slot_date(end, tz);
        if day >= today {
            continue;
        }
        if days.last().is_none_or(|(d, _, _)| *d != day) {
            days.push((day, vec![f64::NAN; SLOTS_PER_DAY], 0));
        }
        if let Some((_, values, seen)) = days.last_mut() {
            values[slot_of_day(end, tz)] = value;
            *seen += 1;
        }
    }
    days.into_iter()
        .filter_map(|(date, mut values, seen)| {
            let open = open_slots_on(data, date);
            #[allow(clippy::cast_precision_loss)]
            let coverage = seen as f64 / open.len().max(1) as f64;
            if coverage < MIN_DAY_COVERAGE {
                return None;
            }
            // Fill missing open slots from their neighbours in the day.
            let mut last = None;
            for (slot, v) in values.iter_mut().enumerate() {
                if v.is_nan() {
                    *v = if open.contains(&slot) {
                        last.unwrap_or(0.0)
                    } else {
                        0.0
                    };
                } else {
                    last = Some(*v);
                }
            }
            Some((date, values))
        })
        .collect()
}

/// Slots of `date` (by index) that are open.
fn open_slots_on(data: &Observed<'_>, day: NaiveDate) -> Vec<usize> {
    let tz = data.schedule().timezone();
    let start = hardy_core::analytics::midnight_local_as_utc(day, tz);
    (1..=SLOTS_PER_DAY)
        .map(|k| start + crate::grid::SLOT * i32::try_from(k).unwrap_or(0))
        .filter(|end| crate::grid::is_open_slot(data.schedule(), *end))
        .map(|end| slot_of_day(end, tz))
        .collect()
}

impl Forecaster for DailyPca {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(3)
    }

    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
        let days = day_vectors(data);
        if days.len() < 2 {
            return Err(MlError::InsufficientData {
                found: days.len(),
                required: 2,
            });
        }
        #[allow(clippy::cast_precision_loss)]
        let n = days.len() as f64;
        let mean: Vec<f64> = (0..SLOTS_PER_DAY)
            .map(|s| days.iter().map(|(_, v)| v[s]).sum::<f64>() / n)
            .collect();
        let mut covariance = vec![0.0; SLOTS_PER_DAY * SLOTS_PER_DAY];
        for (_, values) in &days {
            let centred: Vec<f64> = values.iter().zip(&mean).map(|(v, m)| v - m).collect();
            for i in 0..SLOTS_PER_DAY {
                for j in 0..SLOTS_PER_DAY {
                    covariance[i * SLOTS_PER_DAY + j] += centred[i] * centred[j] / n;
                }
            }
        }
        let k = COMPONENTS.min(days.len() - 1);
        let (eigenvalues, components): (Vec<f64>, Vec<Vec<f64>>) =
            top_eigenvectors(&covariance, SLOTS_PER_DAY, k)
                .into_iter()
                .unzip();

        let mut sums = vec![(vec![0.0; components.len()], 0.0); 7];
        for (date, values) in &days {
            let centred: Vec<f64> = values.iter().zip(&mean).map(|(v, m)| v - m).collect();
            let entry = &mut sums[date.weekday().num_days_from_monday() as usize];
            for (score, c) in entry.0.iter_mut().zip(&components) {
                *score += dot(&centred, c);
            }
            entry.1 += 1.0;
        }
        let weekday_scores = sums
            .into_iter()
            .map(|(total, count)| (count > 0.0).then(|| total.iter().map(|s| s / count).collect()))
            .collect();
        self.fitted = Some(Fitted {
            mean,
            components,
            eigenvalues,
            weekday_scores,
        });
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| MlError::Fit("model is not fitted".to_string()))?;
        let tz = data.schedule().timezone();
        let origin = data.origin();
        let today = slot_date(origin, tz);
        let observed: Vec<(usize, f64)> =
            open_values(data, data.grid(), origin - TimeDelta::days(1))
                .filter(|(end, _)| slot_date(*end, tz) == today)
                .map(|(end, v)| (slot_of_day(end, tz), v))
                .collect();
        let today_scores = fitted.fit_scores(&observed, &fitted.prior(today));
        let grid = data.grid();
        Ok((1..=HORIZON_STEPS)
            .map(|k| {
                let end = grid.step_end(k);
                let day = slot_date(end, tz);
                let scores = if day == today {
                    today_scores.clone()
                } else {
                    fitted.prior(day)
                };
                fitted.reconstruct(&scores, slot_of_day(end, tz))
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::Timelike;
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{
        grid::Grid,
        test_support::{local, minutely},
    };

    /// A day's level times a fixed daily shape: one component explains it.
    fn shaped(level: impl Fn(i64) -> f64) -> crate::history::History {
        let tz = GymSchedule::default().timezone();
        minutely(15, move |t| {
            let day = (t - local(0, 0, 0)).num_days();
            let hour = f64::from(
                (crate::grid::slot_end_of(t) - TimeDelta::minutes(1))
                    .with_timezone(&tz)
                    .hour(),
            );
            level(day) * (1.0 + (hour - 6.0) / 17.0)
        })
    }

    #[test]
    fn test_today_s_scale_is_read_from_the_morning() -> Result<()> {
        let schedule = GymSchedule::default();
        // Levels vary from day to day; day 14 runs at 30.
        let history = shaped(|d| {
            if d == 14 {
                30.0
            } else {
                10.0 + 4.0 * f64::from((d % 5) as i32)
            }
        });
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(15, 0, 0));
        let fit = history.until(local(14, 0, 0));
        let mut model = DailyPca::default();
        model.fit(&Observed::new(&grid, &fit, &schedule, local(14, 0, 0)))?;

        let origin = local(14, 12, 0);
        let seen = history.until(origin);
        let forecast = model.predict(&Observed::new(&grid, &seen, &schedule, origin))?;
        // 16:00 on day 14: 30 × (1 + 10/17). Mondays so far ran at 10 and
        // 18 (a prior near 22): the morning moves the forecast most of the
        // way, short of a level no training day reached.
        let expected = 30.0 * (1.0 + 10.0 / 17.0);
        assert_relative_eq!(forecast[15], expected, epsilon = 3.0);
        Ok(())
    }

    #[test]
    fn test_scores_without_observations_are_the_weekday_prior() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = shaped(|d| 10.0 + 2.0 * f64::from((d % 3) as i32));
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(15, 0, 0));
        let mut model = DailyPca::default();
        model.fit(&Observed::new(&grid, &history, &schedule, local(15, 0, 0)))?;
        let fitted = model.fitted.as_ref().context("fitted")?;
        let date = slot_date(local(15, 12, 0), schedule.timezone());
        let prior = fitted.prior(date);
        for (score, expected) in fitted.fit_scores(&[], &prior).iter().zip(&prior) {
            assert_relative_eq!(*score, *expected, epsilon = 1e-9);
        }
        assert!(fitted.components.len() <= COMPONENTS);
        Ok(())
    }

    #[test]
    fn test_one_complete_day_is_not_enough() {
        let schedule = GymSchedule::default();
        let history = minutely(2, |_| 30.0);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(2, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(1, 12, 0));
        assert!(DailyPca::default().fit(&data).is_err());
    }
}
