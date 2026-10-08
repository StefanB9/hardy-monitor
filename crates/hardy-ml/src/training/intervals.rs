//! Per-horizon prediction intervals from holdout residuals.

use serde::{Deserialize, Serialize};

use crate::samples::Sample;

/// 10th / 90th percentile of out-of-sample residuals (actual − predicted),
/// per horizon; index `h - 1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HorizonIntervals {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl HorizonIntervals {
    /// `(lower, upper)` residual offsets for a horizon; `(0, 0)` if unknown.
    pub fn offsets(&self, hours_ahead: u32) -> (f64, f64) {
        let i = (hours_ahead as usize).saturating_sub(1);
        (
            self.lower.get(i).copied().unwrap_or(0.0),
            self.upper.get(i).copied().unwrap_or(0.0),
        )
    }
}

pub(super) fn horizon_statistics(
    holdout: &[Sample],
    predictions: &[f64],
    max_hours_ahead: u32,
) -> (HorizonIntervals, Vec<f64>) {
    let horizons = max_hours_ahead as usize;
    let mut residuals: Vec<Vec<f64>> = vec![Vec::new(); horizons];
    for (sample, predicted) in holdout.iter().zip(predictions) {
        if let Some(bucket) = residuals.get_mut((sample.hours_ahead as usize).saturating_sub(1)) {
            bucket.push(sample.target - predicted);
        }
    }
    let mut lower = Vec::with_capacity(horizons);
    let mut upper = Vec::with_capacity(horizons);
    let mut mae_by_horizon = Vec::with_capacity(horizons);
    for mut bucket in residuals {
        bucket.sort_by(f64::total_cmp);
        lower.push(quantile(&bucket, 0.10));
        upper.push(quantile(&bucket, 0.90));
        #[allow(clippy::cast_precision_loss)]
        let mae = if bucket.is_empty() {
            f64::NAN
        } else {
            bucket.iter().map(|r| r.abs()).sum::<f64>() / bucket.len() as f64
        };
        mae_by_horizon.push(mae);
    }
    (HorizonIntervals { lower, upper }, mae_by_horizon)
}

/// Linear-interpolated quantile of sorted data; 0 for empty input.
pub(super) fn quantile(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => 0.0,
        1 => sorted[0],
        n => {
            #[allow(clippy::cast_precision_loss)]
            let position = q.clamp(0.0, 1.0) * (n - 1) as f64;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let below = position.floor() as usize;
            let above = (below + 1).min(n - 1);
            #[allow(clippy::cast_precision_loss)]
            let weight = position - below as f64;
            sorted[below] * (1.0 - weight) + sorted[above] * weight
        }
    }
}

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_quantile() {
        assert_relative_eq!(quantile(&[], 0.5), 0.0);
        assert_relative_eq!(quantile(&[3.0], 0.9), 3.0);
        assert_relative_eq!(quantile(&[0.0, 10.0], 0.1), 1.0);
        assert_relative_eq!(quantile(&[0.0, 5.0, 10.0], 0.5), 5.0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn quantile_is_bounded_and_monotonic(
            mut values in prop::collection::vec(-50.0f64..50.0, 1..100),
            q1 in 0.0f64..1.0,
            q2 in 0.0f64..1.0,
        ) {
            values.sort_by(f64::total_cmp);
            let (lo, hi) = (q1.min(q2), q1.max(q2));
            let a = quantile(&values, lo);
            let b = quantile(&values, hi);
            prop_assert!(a <= b + 1e-12);
            prop_assert!(a >= values[0] - 1e-12);
            prop_assert!(b <= values[values.len() - 1] + 1e-12);
        }
    }
}
