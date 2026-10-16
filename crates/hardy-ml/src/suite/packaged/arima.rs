//! ARIMA fits with a stability check: a fit whose AR part is not stationary
//! or whose MA part is not invertible is refused rather than forecast from,
//! as its forecasts drift or explode.
//!
//! `AutoARIMA` checks its own choice: since anofox-forecast 0.16.1 it fits
//! the way R's `auto.arima` does and keeps that fit, refusing any candidate
//! with a root within 1.01 of the unit circle. Its forecasts come straight
//! from that model; refitting the chosen order with the crate's own
//! estimator would give different, possibly unstable coefficients.

use anofox_forecast::models::{
    Forecaster as _,
    arima::{AutoARIMA, SARIMA},
};

use super::models::{ahead, failed, time_series};
use crate::{error::MlError, grid::SLOTS_PER_DAY};

/// `AutoARIMA`'s choice for `values`, forecast `horizon` steps ahead.
pub(super) fn auto_arima(values: &[f64], horizon: usize) -> Result<Vec<f64>, MlError> {
    let series = time_series(values)?;
    let mut search = AutoARIMA::new();
    search.fit(&series).map_err(failed)?;
    ahead(&search, horizon)
}

/// `SARIMA(1,0,1)(1,1,1)` with a daily season on `values`, checked and
/// forecast `horizon` steps ahead.
pub(super) fn sarima(values: &[f64], horizon: usize) -> Result<Vec<f64>, MlError> {
    let mut model = SARIMA::new(1, 0, 1, 1, 1, 1, SLOTS_PER_DAY);
    model.fit(&time_series(values)?).map_err(failed)?;
    check_sarima(&model)?;
    ahead(&model, horizon)
}

fn check_sarima(model: &SARIMA) -> Result<(), MlError> {
    check(
        model.ar_coefficients(),
        model.ma_coefficients(),
        model.seasonal_ar_coefficients(),
        model.seasonal_ma_coefficients(),
    )
}

/// Whether every root of `1 − a₁z − … − aₚzᵖ` lies outside the unit circle
/// (Schur–Cohn step-down: the partial autocorrelations it implies must
/// all lie strictly inside (−1, 1)).
pub(super) fn is_stable(coefficients: &[f64]) -> bool {
    let mut a = coefficients.to_vec();
    while let Some(&last) = a.last() {
        // NaN fails too.
        if last.is_nan() || last.abs() >= 1.0 {
            return false;
        }
        let k = a.len();
        let scale = 1.0 - last * last;
        let reduced: Vec<f64> = (0..k - 1)
            .map(|j| (a[j] + last * a[k - 2 - j]) / scale)
            .collect();
        a = reduced;
    }
    true
}

/// Refuses a fit whose AR or seasonal AR part (`1 − φ(B)`) is not
/// stationary or whose MA or seasonal MA part (`1 + θ(B)`) is not
/// invertible.
pub(super) fn check(
    ar: &[f64],
    ma: &[f64],
    seasonal_ar: &[f64],
    seasonal_ma: &[f64],
) -> Result<(), MlError> {
    let negated = |c: &[f64]| c.iter().map(|v| -v).collect::<Vec<f64>>();
    let parts = [
        ("AR part is not stationary", ar.to_vec()),
        ("MA part is not invertible", negated(ma)),
        ("seasonal AR part is not stationary", seasonal_ar.to_vec()),
        ("seasonal MA part is not invertible", negated(seasonal_ma)),
    ];
    match parts.iter().find(|(_, c)| !is_stable(c)) {
        Some((problem, _)) => Err(MlError::Fit(format!("ARIMA fit refused: {problem}"))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_stability_of_known_processes() {
        assert!(is_stable(&[]));
        assert!(is_stable(&[0.9]));
        assert!(is_stable(&[-0.9]));
        assert!(!is_stable(&[1.0]));
        assert!(!is_stable(&[-1.2]));
        // AR(2): stationary iff φ₁ + φ₂ < 1, φ₂ − φ₁ < 1 and |φ₂| < 1.
        assert!(is_stable(&[0.5, 0.3]));
        assert!(!is_stable(&[0.9, 0.2]));
        assert!(!is_stable(&[-0.9, 0.2]));
        assert!(is_stable(&[1.2, -0.5]));
        assert!(!is_stable(&[0.0, 1.1]));
        assert!(!is_stable(&[f64::NAN]));
    }

    #[test]
    fn test_check_names_the_failing_part() {
        assert!(check(&[0.5], &[0.4], &[0.3], &[-0.6]).is_ok());
        let message = |r: Result<(), MlError>| r.err().map_or_else(String::new, |e| e.to_string());
        assert!(message(check(&[1.1], &[], &[], &[])).contains("AR part"));
        // 1 + θz with θ = −1.5 has its root inside the unit circle.
        assert!(message(check(&[], &[-1.5], &[], &[])).contains("MA part"));
        assert!(message(check(&[], &[], &[1.0], &[])).contains("seasonal AR"));
        assert!(message(check(&[], &[], &[], &[2.0])).contains("seasonal MA"));
    }

    /// Coefficients of `(1 − z/r₁)(1 − z/r₂)…` written as `1 − a₁z − …`.
    fn from_roots(roots: &[f64]) -> Vec<f64> {
        let mut poly = vec![1.0];
        for r in roots {
            let mut next = vec![0.0; poly.len() + 1];
            for (i, c) in poly.iter().enumerate() {
                next[i] += c;
                next[i + 1] -= c / r;
            }
            poly = next;
        }
        poly[1..].iter().map(|c| -c).collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Real roots all outside the unit circle: stable; one inside: not.
        #[test]
        fn test_stability_matches_the_roots(
            outside in proptest::collection::vec((1.05_f64..5.0, any::<bool>()), 0..5),
            inside in 0.2_f64..0.95,
            negative: bool,
        ) {
            let mut roots: Vec<f64> = outside
                .iter()
                .map(|&(r, neg)| if neg { -r } else { r })
                .collect();
            prop_assert!(is_stable(&from_roots(&roots)));
            roots.push(if negative { -inside } else { inside });
            prop_assert!(!is_stable(&from_roots(&roots)));
        }
    }
}
