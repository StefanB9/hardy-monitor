//! Parameter search for the smoothing and state-space models: Nelder–Mead
//! (argmin) on unconstrained coordinates, mapped into each parameter's
//! range.

use argmin::{
    core::{CostFunction, Executor, State},
    solver::neldermead::NelderMead,
};

/// `x` mapped smoothly into the open interval `(low, high)`.
pub(super) fn bounded(x: f64, low: f64, high: f64) -> f64 {
    low + (high - low) / (1.0 + (-x).exp())
}

/// The unconstrained coordinate that [`bounded`] maps to `value`.
pub(super) fn unbounded(value: f64, low: f64, high: f64) -> f64 {
    let share = ((value - low) / (high - low)).clamp(1e-12, 1.0 - 1e-12);
    (share / (1.0 - share)).ln()
}

/// The point near `start` (unconstrained coordinates) where `cost` is
/// lowest, found by Nelder–Mead from a simplex of side `step`. Falls back to
/// `start` if the search fails; non-finite costs count as infinitely bad.
pub(super) fn minimize(
    cost: impl Fn(&[f64]) -> f64,
    start: &[f64],
    step: f64,
    max_iters: u64,
) -> Vec<f64> {
    if start.is_empty() {
        return Vec::new();
    }
    let mut simplex = vec![start.to_vec()];
    for i in 0..start.len() {
        let mut vertex = start.to_vec();
        vertex[i] += step;
        simplex.push(vertex);
    }
    let Ok(solver) = NelderMead::new(simplex).with_sd_tolerance(1e-10) else {
        return start.to_vec();
    };
    Executor::new(Cost(cost), solver)
        .configure(|state| state.max_iters(max_iters))
        .run()
        .ok()
        .and_then(|result| result.state().get_best_param().cloned())
        .unwrap_or_else(|| start.to_vec())
}

/// A cost closure as an argmin problem.
struct Cost<F>(F);

impl<F: Fn(&[f64]) -> f64> CostFunction for Cost<F> {
    type Output = f64;
    type Param = Vec<f64>;

    fn cost(&self, param: &Vec<f64>) -> Result<f64, argmin::core::Error> {
        let value = (self.0)(param);
        Ok(if value.is_finite() { value } else { f64::MAX })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_minimize_finds_the_bottom_of_a_bowl() {
        let cost = |p: &[f64]| (p[0] - 1.5).powi(2) + 3.0 * (p[1] + 0.5).powi(2);
        let best = minimize(cost, &[0.0, 0.0], 0.5, 500);
        assert_relative_eq!(best[0], 1.5, epsilon = 1e-4);
        assert_relative_eq!(best[1], -0.5, epsilon = 1e-4);
    }

    #[test]
    fn test_minimize_survives_non_finite_costs() {
        // NaN left of zero: the search must stay on the right.
        let cost = |p: &[f64]| {
            if p[0] < 0.0 {
                f64::NAN
            } else {
                (p[0] - 2.0).powi(2)
            }
        };
        let best = minimize(cost, &[1.0], 0.5, 500);
        assert_relative_eq!(best[0], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_minimize_without_parameters_returns_nothing() {
        assert_eq!(minimize(|_| 0.0, &[], 0.5, 10), Vec::<f64>::new());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Every coordinate lands strictly inside the range, and the
        /// mapping inverts.
        #[test]
        fn test_bounded_stays_inside_and_inverts(
            x in -30.0_f64..30.0,
            low in -5.0_f64..5.0,
            width in 0.01_f64..10.0,
        ) {
            let high = low + width;
            let value = bounded(x, low, high);
            prop_assert!(value >= low && value <= high);
            if (-15.0..15.0).contains(&x) {
                prop_assert!((unbounded(value, low, high) - x).abs() < 1e-6);
            }
        }
    }
}
