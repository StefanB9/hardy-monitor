use std::f64::consts::PI;

/// Encode a value as a point on the unit circle for a given period.
///
/// Returns `(sin, cos)` where both values are in `[-1.0, 1.0]`. This preserves
/// the circular continuity of periodic features (e.g., hour 23 is close to hour
/// 0).
pub(super) fn cyclical_encode(value: f64, period: f64) -> (f64, f64) {
    let angle = 2.0 * PI * value / period;
    (angle.sin(), angle.cos())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_cyclical_encoding_continuity() {
        let (sin_23, cos_23) = cyclical_encode(23.0, 24.0);
        let (sin_0, cos_0) = cyclical_encode(0.0, 24.0);

        let distance = ((sin_23 - sin_0).powi(2) + (cos_23 - cos_0).powi(2)).sqrt();

        assert!(distance < 0.5, "Distance was {distance}");
    }

    #[test]
    fn test_cyclical_encoding_opposite() {
        let (_sin_0, cos_0) = cyclical_encode(0.0, 24.0);
        let (_sin_12, cos_12) = cyclical_encode(12.0, 24.0);

        assert_relative_eq!(cos_0, -cos_12, epsilon = 1e-10);
    }

    #[test]
    fn test_cyclical_encoding_quarter() {
        let (sin_6, cos_6) = cyclical_encode(6.0, 24.0);

        assert_relative_eq!(sin_6, 1.0, epsilon = 1e-10);
        assert_relative_eq!(cos_6, 0.0, epsilon = 1e-10);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_cyclical_sin_cos_unit_circle(
            value in -1000.0_f64..1000.0,
            period in 0.1_f64..1000.0,
        ) {
            let (sin, cos) = cyclical_encode(value, period);
            let magnitude = sin.powi(2) + cos.powi(2);
            prop_assert!(
                (magnitude - 1.0).abs() < 1e-10,
                "sin²+cos² should be 1.0, got {magnitude}"
            );
        }

        #[test]
        fn prop_cyclical_range(
            value in -1000.0_f64..1000.0,
            period in 0.1_f64..1000.0,
        ) {
            let (sin, cos) = cyclical_encode(value, period);
            prop_assert!(sin >= -1.0 && sin <= 1.0, "sin out of range: {sin}");
            prop_assert!(cos >= -1.0 && cos <= 1.0, "cos out of range: {cos}");
        }

        #[test]
        fn prop_cyclical_period_invariance(
            value in -1000.0_f64..1000.0,
            period in 0.1_f64..1000.0,
        ) {
            let (sin1, cos1) = cyclical_encode(value, period);
            let (sin2, cos2) = cyclical_encode(value + period, period);
            prop_assert!(
                (sin1 - sin2).abs() < 1e-8,
                "sin not period-invariant: {sin1} vs {sin2}"
            );
            prop_assert!(
                (cos1 - cos2).abs() < 1e-8,
                "cos not period-invariant: {cos1} vs {cos2}"
            );
        }

        #[test]
        fn prop_cyclical_output_finite(
            value in -1000.0_f64..1000.0,
            period in 0.1_f64..1000.0,
        ) {
            let (sin, cos) = cyclical_encode(value, period);
            prop_assert!(sin.is_finite(), "sin not finite: {sin}");
            prop_assert!(cos.is_finite(), "cos not finite: {cos}");
        }
    }
}
