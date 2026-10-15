//! Error measures for the benchmark and the conformal intervals built from
//! past errors.

use serde::{Deserialize, Serialize};

/// Share of errors the interval leaves below it (and above it): an 80 %
/// interval.
pub const INTERVAL_TAIL: f64 = 0.1;
/// Fewer past errors than this give no interval.
pub const MIN_INTERVAL_ERRORS: usize = 20;

/// Offsets added to a point forecast to get its interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub low: f64,
    pub high: f64,
}

impl Interval {
    /// The interval of past `errors` (actual − forecast): their 10 % and
    /// 90 % quantiles; `None` with fewer than [`MIN_INTERVAL_ERRORS`].
    pub fn from_errors(errors: &[f64]) -> Option<Self> {
        if errors.len() < MIN_INTERVAL_ERRORS {
            return None;
        }
        let mut sorted = errors.to_vec();
        sorted.sort_by(f64::total_cmp);
        Some(Self {
            low: quantile(&sorted, INTERVAL_TAIL),
            high: quantile(&sorted, 1.0 - INTERVAL_TAIL),
        })
    }

    /// The bounds around `forecast`, within 0–100.
    pub fn around(self, forecast: f64) -> (f64, f64) {
        (
            (forecast + self.low).clamp(0.0, 100.0),
            (forecast + self.high).clamp(0.0, 100.0),
        )
    }
}

/// Quantile `q` of sorted values, interpolating between neighbours.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let position = q * (sorted.len() - 1) as f64;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let below = position.floor() as usize;
    let above = (below + 1).min(sorted.len() - 1);
    let fraction = position - position.floor();
    sorted[below] * (1.0 - fraction) + sorted[above] * fraction
}

/// Pinball (quantile) loss of `bound` as the `q` quantile of `actual`.
pub fn pinball(q: f64, actual: f64, bound: f64) -> f64 {
    let diff = actual - bound;
    (q * diff).max((q - 1.0) * diff)
}

/// Scores of one model in one horizon bucket.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub mae: f64,
    pub rmse: f64,
    /// MAE relative to the seasonal naive one-step error on the history.
    pub mase: Option<f64>,
    /// Share of actual values inside the interval.
    pub coverage: Option<f64>,
    /// Mean pinball loss of the interval's bounds.
    pub pinball: Option<f64>,
    /// Number of scored values.
    pub errors: usize,
}

/// Running sums for a [`Score`].
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Accumulator {
    abs: f64,
    squared: f64,
    count: usize,
    covered: usize,
    pinball: f64,
    with_interval: usize,
}

impl Accumulator {
    /// Adds one forecast against its actual value, with the interval the
    /// model had at the time (if any).
    pub(super) fn add(&mut self, actual: f64, forecast: f64, interval: Option<Interval>) {
        let error = actual - forecast;
        self.abs += error.abs();
        self.squared += error * error;
        self.count += 1;
        if let Some(interval) = interval {
            let (low, high) = interval.around(forecast);
            self.with_interval += 1;
            if (low..=high).contains(&actual) {
                self.covered += 1;
            }
            self.pinball += f64::midpoint(
                pinball(INTERVAL_TAIL, actual, low),
                pinball(1.0 - INTERVAL_TAIL, actual, high),
            );
        }
    }

    /// The score; `None` without values. `scale` turns MAE into MASE.
    pub(super) fn score(&self, scale: Option<f64>) -> Option<Score> {
        if self.count == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let n = self.count as f64;
        #[allow(clippy::cast_precision_loss)]
        let m = self.with_interval as f64;
        let mae = self.abs / n;
        let with_interval = self.with_interval > 0;
        Some(Score {
            mae,
            rmse: (self.squared / n).sqrt(),
            mase: scale.filter(|s| *s > 0.0).map(|s| mae / s),
            #[allow(clippy::cast_precision_loss)]
            coverage: with_interval.then(|| self.covered as f64 / m),
            pinball: with_interval.then(|| self.pinball / m),
            errors: self.count,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_interval_needs_enough_errors() {
        assert!(Interval::from_errors(&[1.0; MIN_INTERVAL_ERRORS - 1]).is_none());
        assert!(Interval::from_errors(&[1.0; MIN_INTERVAL_ERRORS]).is_some());
    }

    #[test]
    fn test_interval_quantiles_of_a_uniform_ramp() -> Result<()> {
        // -10, -9, …, 10: the 10 % / 90 % quantiles are -8 and 8.
        let errors: Vec<f64> = (-10..=10).map(f64::from).collect();
        let interval = Interval::from_errors(&errors).context("enough errors")?;
        assert_relative_eq!(interval.low, -8.0, epsilon = 1e-9);
        assert_relative_eq!(interval.high, 8.0, epsilon = 1e-9);
        assert_eq!(interval.around(95.0), (87.0, 100.0));
        Ok(())
    }

    #[test]
    fn test_pinball_penalises_the_wrong_side_more() {
        // A 90 % bound below the actual value costs 0.9 per point.
        assert_relative_eq!(pinball(0.9, 10.0, 8.0), 1.8);
        assert_relative_eq!(pinball(0.9, 10.0, 12.0), 0.2);
        assert_relative_eq!(pinball(0.1, 10.0, 8.0), 0.2);
    }

    #[test]
    fn test_accumulator_scores() -> Result<()> {
        let mut acc = Accumulator::default();
        assert!(acc.score(None).is_none());
        let interval = Interval {
            low: -1.0,
            high: 1.0,
        };
        acc.add(10.0, 7.0, Some(interval)); // error 3, outside
        acc.add(10.0, 10.0, Some(interval)); // error 0, inside
        acc.add(10.0, 14.0, None); // error -4, no interval
        let score = acc.score(Some(2.0)).context("values added")?;
        assert_relative_eq!(score.mae, 7.0 / 3.0);
        assert_relative_eq!(score.rmse, (25.0_f64 / 3.0).sqrt());
        assert_relative_eq!(score.mase.unwrap_or(0.0), 7.0 / 6.0);
        assert_relative_eq!(score.coverage.unwrap_or(0.0), 0.5);
        assert_eq!(score.errors, 3);
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// The interval of past errors covers roughly 80 % of them.
        #[test]
        fn test_interval_covers_its_own_errors(errors in prop::collection::vec(-30.0_f64..30.0, 20..200)) {
            let interval = Interval::from_errors(&errors).unwrap_or(Interval { low: 0.0, high: 0.0 });
            prop_assert!(interval.low <= interval.high);
            let inside = errors.iter().filter(|e| (interval.low..=interval.high).contains(*e)).count();
            #[allow(clippy::cast_precision_loss)]
            let share = inside as f64 / errors.len() as f64;
            prop_assert!(share >= 0.7, "{share}");
        }

        /// Pinball loss is never negative and zero only on the bound.
        #[test]
        fn test_pinball_is_non_negative(q in 0.01_f64..0.99, actual in 0.0_f64..100.0, bound in 0.0_f64..100.0) {
            let loss = pinball(q, actual, bound);
            prop_assert!(loss >= 0.0);
            prop_assert!(loss > 0.0 || (actual - bound).abs() < 1e-12);
        }
    }
}
