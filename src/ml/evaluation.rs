/// Mean Squared Error.
///
/// Returns `None` if inputs are empty or have mismatched lengths.
pub fn mse(predictions: &[f64], targets: &[f64]) -> Option<f64> {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return None;
    }

    let sum_sq: f64 = predictions
        .iter()
        .zip(targets)
        .map(|(p, t)| (p - t).powi(2))
        .sum();

    Some(sum_sq / predictions.len() as f64)
}

/// Root Mean Squared Error.
///
/// Returns `None` if inputs are empty or have mismatched lengths.
pub fn rmse(predictions: &[f64], targets: &[f64]) -> Option<f64> {
    mse(predictions, targets).map(f64::sqrt)
}

/// Mean Absolute Error.
///
/// Returns `None` if inputs are empty or have mismatched lengths.
pub fn mae(predictions: &[f64], targets: &[f64]) -> Option<f64> {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return None;
    }

    let sum_abs: f64 = predictions
        .iter()
        .zip(targets)
        .map(|(p, t)| (p - t).abs())
        .sum();

    Some(sum_abs / predictions.len() as f64)
}

/// Mean Absolute Percentage Error.
///
/// Skips samples where `|target| < 1e-10`. Returns `None` if no valid samples
/// remain or inputs are empty/mismatched.
pub fn mape(predictions: &[f64], targets: &[f64]) -> Option<f64> {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return None;
    }

    let (sum, count) = predictions
        .iter()
        .zip(targets)
        .filter(|(_, t)| t.abs() >= 1e-10)
        .fold((0.0_f64, 0_usize), |(sum, count), (p, t)| {
            (sum + ((p - t) / t).abs(), count + 1)
        });

    if count == 0 {
        return None;
    }

    Some(sum / count as f64 * 100.0)
}

/// Coefficient of determination (R²).
///
/// Returns `None` if SS_tot ≈ 0 (all targets identical), inputs are empty, or
/// lengths mismatch.
pub fn r_squared(predictions: &[f64], targets: &[f64]) -> Option<f64> {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return None;
    }

    let mean_target = targets.iter().sum::<f64>() / targets.len() as f64;

    let ss_tot: f64 = targets.iter().map(|t| (t - mean_target).powi(2)).sum();

    if ss_tot < 1e-10 {
        return None;
    }

    let ss_res: f64 = predictions
        .iter()
        .zip(targets)
        .map(|(p, t)| (t - p).powi(2))
        .sum();

    Some(1.0 - ss_res / ss_tot)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    // ── MSE ──────────────────────────────────────────────────────────

    #[test]
    fn test_mse_simple() {
        let predictions = [10.0, 20.0, 30.0];
        let targets = [12.0, 18.0, 32.0];

        let result = mse(&predictions, &targets);

        assert_relative_eq!(result.unwrap_or(f64::NAN), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_perfect() {
        let values = [10.0, 20.0, 30.0];

        let result = mse(&values, &values);

        assert_relative_eq!(result.unwrap_or(f64::NAN), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_empty() {
        assert!(mse(&[], &[]).is_none());
    }

    #[test]
    fn test_mse_mismatched_lengths() {
        assert!(mse(&[1.0, 2.0], &[1.0]).is_none());
    }

    // ── RMSE ─────────────────────────────────────────────────────────

    #[test]
    fn test_rmse_simple() {
        let predictions = [10.0, 20.0, 30.0];
        let targets = [12.0, 18.0, 32.0];

        let result = rmse(&predictions, &targets);

        assert_relative_eq!(result.unwrap_or(f64::NAN), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rmse_perfect() {
        let values = [10.0, 20.0, 30.0];

        let result = rmse(&values, &values);

        assert_relative_eq!(result.unwrap_or(f64::NAN), 0.0, epsilon = 1e-10);
    }

    // ── MAE ──────────────────────────────────────────────────────────

    #[test]
    fn test_mae_simple() {
        let predictions = [10.0, 20.0, 30.0];
        let targets = [12.0, 18.0, 32.0];

        let result = mae(&predictions, &targets);

        assert_relative_eq!(result.unwrap_or(f64::NAN), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mae_perfect() {
        let values = [10.0, 20.0, 30.0];

        let result = mae(&values, &values);

        assert_relative_eq!(result.unwrap_or(f64::NAN), 0.0, epsilon = 1e-10);
    }

    // ── MAPE ─────────────────────────────────────────────────────────

    #[test]
    fn test_mape_simple() {
        // |10-12|/12 + |20-18|/20 + |30-32|/30 = 2/12 + 2/20 + 2/30
        // = 0.1667 + 0.1 + 0.0667 = 0.3333 → /3 * 100 = 11.11%
        let predictions = [10.0, 20.0, 30.0];
        let targets = [12.0, 20.0, 30.0];

        let result = mape(&predictions, &targets);
        // |10-12|/12 = 0.1667, |20-20|/20 = 0, |30-30|/30 = 0
        // mean = 0.1667/3 * 100 = 5.556%
        assert_relative_eq!(
            result.unwrap_or(f64::NAN),
            5.555_555_555_555_555,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_mape_skips_zero_targets() {
        let predictions = [10.0, 20.0, 30.0];
        let targets = [0.0, 20.0, 30.0];

        let result = mape(&predictions, &targets);

        // Skips first pair (target=0), computes on remaining 2
        assert!(result.is_some());
    }

    #[test]
    fn test_mape_all_zero_targets() {
        let predictions = [10.0, 20.0];
        let targets = [0.0, 0.0];

        assert!(mape(&predictions, &targets).is_none());
    }

    // ── R² ───────────────────────────────────────────────────────────

    #[test]
    fn test_r_squared_perfect() {
        let values = [10.0, 20.0, 30.0, 40.0];

        let result = r_squared(&values, &values);

        assert_relative_eq!(result.unwrap_or(f64::NAN), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_r_squared_mean_prediction() {
        let targets = [10.0, 20.0, 30.0, 40.0];
        let mean = 25.0;
        let predictions = [mean; 4];

        let result = r_squared(&predictions, &targets);

        assert_relative_eq!(result.unwrap_or(f64::NAN), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_r_squared_constant_targets() {
        let targets = [50.0; 4];
        let predictions = [50.0; 4];

        assert!(r_squared(&predictions, &targets).is_none());
    }

    #[test]
    fn test_r_squared_negative() {
        // Predictions worse than predicting the mean
        let targets = [10.0, 20.0, 30.0];
        let predictions = [30.0, 10.0, 20.0]; // badly shuffled

        let result = r_squared(&predictions, &targets);

        assert!(result.is_some());
        assert!(result.unwrap_or(0.0) < 0.0);
    }

    // ── Property-based tests ─────────────────────────────────────────

    fn finite_vec(size: usize) -> impl Strategy<Value = Vec<f64>> {
        proptest::collection::vec(-1000.0_f64..1000.0, size)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_mse_non_negative(
            values in finite_vec(10),
            offsets in finite_vec(10),
        ) {
            let predictions: Vec<f64> = values.iter()
                .zip(&offsets)
                .map(|(v, o)| v + o)
                .collect();
            if let Some(result) = mse(&predictions, &values) {
                prop_assert!(result >= 0.0, "MSE was negative: {result}");
            }
        }

        #[test]
        fn prop_rmse_non_negative(
            values in finite_vec(10),
            offsets in finite_vec(10),
        ) {
            let predictions: Vec<f64> = values.iter()
                .zip(&offsets)
                .map(|(v, o)| v + o)
                .collect();
            if let Some(result) = rmse(&predictions, &values) {
                prop_assert!(result >= 0.0, "RMSE was negative: {result}");
            }
        }

        #[test]
        fn prop_mae_non_negative(
            values in finite_vec(10),
            offsets in finite_vec(10),
        ) {
            let predictions: Vec<f64> = values.iter()
                .zip(&offsets)
                .map(|(v, o)| v + o)
                .collect();
            if let Some(result) = mae(&predictions, &values) {
                prop_assert!(result >= 0.0, "MAE was negative: {result}");
            }
        }

        #[test]
        fn prop_mape_non_negative(
            values in proptest::collection::vec(1.0_f64..1000.0, 10),
            offsets in finite_vec(10),
        ) {
            let predictions: Vec<f64> = values.iter()
                .zip(&offsets)
                .map(|(v, o)| v + o)
                .collect();
            if let Some(result) = mape(&predictions, &values) {
                prop_assert!(result >= 0.0, "MAPE was negative: {result}");
            }
        }

        #[test]
        fn prop_r_squared_perfect_self(
            values in proptest::collection::vec(-1000.0_f64..1000.0, 10),
        ) {
            // Only test when targets aren't constant
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max - min > 1e-6 {
                let result = r_squared(&values, &values);
                if let Some(r2) = result {
                    prop_assert!(
                        (r2 - 1.0).abs() < 1e-6,
                        "R² for self-prediction should be 1.0, got {r2}"
                    );
                }
            }
        }

        #[test]
        fn prop_metrics_zero_for_self(
            values in finite_vec(10),
        ) {
            if let Some(result) = mse(&values, &values) {
                prop_assert!(
                    result.abs() < 1e-10,
                    "MSE(x, x) should be 0.0, got {result}"
                );
            }
            if let Some(result) = mae(&values, &values) {
                prop_assert!(
                    result.abs() < 1e-10,
                    "MAE(x, x) should be 0.0, got {result}"
                );
            }
        }
    }
}
