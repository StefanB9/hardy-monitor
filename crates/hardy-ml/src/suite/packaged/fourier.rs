//! Fourier regression: occupancy on daily and weekly harmonics of the slot's
//! time, fitted by least squares (faer). The residual ARIMA lives in
//! [`super::models`].

use std::f64::consts::TAU;

use chrono::{DateTime, Utc};
use hardy_core::Tz;

use crate::{
    grid::{SLOTS_PER_DAY, SLOTS_PER_WEEK},
    suite::{
        profiles::{slot_of_day, slot_of_week},
        statistical::ols::least_squares,
    },
};

/// Daily harmonics (periods of 24 h, 12 h, …).
const DAILY_HARMONICS: usize = 10;
/// Weekly harmonics.
const WEEKLY_HARMONICS: usize = 4;
/// Columns: intercept plus a cosine and sine per harmonic.
const COLUMNS: usize = 1 + 2 * (DAILY_HARMONICS + WEEKLY_HARMONICS);

/// Regressors of the slot ending at `end`.
fn features(end: DateTime<Utc>, tz: Tz) -> [f64; COLUMNS] {
    #[allow(clippy::cast_precision_loss)]
    let (day, week) = (
        slot_of_day(end, tz) as f64 / SLOTS_PER_DAY as f64,
        slot_of_week(end, tz) as f64 / SLOTS_PER_WEEK as f64,
    );
    let mut x = [0.0; COLUMNS];
    x[0] = 1.0;
    #[allow(clippy::cast_precision_loss)]
    let cycles = (1..=DAILY_HARMONICS)
        .map(|k| k as f64 * day)
        .chain((1..=WEEKLY_HARMONICS).map(|k| k as f64 * week));
    for (i, c) in cycles.enumerate() {
        x[1 + 2 * i] = (TAU * c).cos();
        x[2 + 2 * i] = (TAU * c).sin();
    }
    x
}

/// A fitted Fourier regression.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Fourier {
    tz: Tz,
    coefficients: Vec<f64>,
}

impl Fourier {
    /// Fits the regression on `(end, value)` points; `None` if the points
    /// cannot pin every coefficient down.
    pub(super) fn fit(points: &[(DateTime<Utc>, f64)], tz: Tz) -> Option<Self> {
        let x: Vec<f64> = points
            .iter()
            .flat_map(|(end, _)| features(*end, tz))
            .collect();
        let y: Vec<f64> = points.iter().map(|(_, v)| *v).collect();
        least_squares(&x, &y, COLUMNS).map(|coefficients| Self { tz, coefficients })
    }

    /// The fitted value for the slot ending at `end`.
    pub(super) fn at(&self, end: DateTime<Utc>) -> f64 {
        features(end, self.tz)
            .iter()
            .zip(&self.coefficients)
            .map(|(x, b)| x * b)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::TimeDelta;
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{grid::SLOT, test_support::local};

    #[test]
    fn test_features_at_the_start_of_the_week() {
        let tz = GymSchedule::default().timezone();
        // Monday 00:15 ends slot 0 of the day and of the week.
        let x = features(local(0, 0, 15), tz);
        assert_eq!(x[0], 1.0);
        for k in 0..DAILY_HARMONICS + WEEKLY_HARMONICS {
            assert_relative_eq!(x[1 + 2 * k], 1.0);
            assert_relative_eq!(x[2 + 2 * k], 0.0);
        }
        // Six hours in: a quarter of the first daily cycle.
        let x = features(local(0, 6, 15), tz);
        assert_relative_eq!(x[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(x[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_fit_recovers_daily_harmonics() -> Result<()> {
        let tz = GymSchedule::default().timezone();
        let truth = |end: DateTime<Utc>| {
            #[allow(clippy::cast_precision_loss)]
            let d = slot_of_day(end, tz) as f64 / 96.0;
            30.0 + 8.0 * (TAU * d).cos() - 4.0 * (2.0 * TAU * d).sin()
        };
        let points: Vec<(DateTime<Utc>, f64)> = (1..=14 * 96)
            .map(|i| local(0, 0, 0) + SLOT * i)
            .map(|end| (end, truth(end)))
            .collect();
        let fit = Fourier::fit(&points, tz).context("fitted")?;
        for hours in [3, 11, 19] {
            let end = local(15, 0, 0) + TimeDelta::hours(hours);
            assert_relative_eq!(fit.at(end), truth(end), epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_fit_needs_enough_points() {
        let tz = GymSchedule::default().timezone();
        let points = vec![(local(0, 12, 0), 30.0); 5];
        assert_eq!(Fourier::fit(&points, tz), None);
    }
}
