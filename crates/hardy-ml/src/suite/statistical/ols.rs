//! Ordinary least squares through faer's QR decomposition.

use faer::{Mat, prelude::SolveLstsq};

/// Coefficients minimising `‖X b − y‖²`, with `x` row-major (`y.len()` rows
/// of `cols` columns). `None` with fewer rows than columns or a
/// rank-deficient `X`.
pub(super) fn least_squares(x: &[f64], y: &[f64], cols: usize) -> Option<Vec<f64>> {
    let rows = y.len();
    if cols == 0 || rows < cols || x.len() != rows * cols {
        return None;
    }
    let design = Mat::from_fn(rows, cols, |row, col| x[row * cols + col]);
    let target = Mat::from_fn(rows, 1, |row, _| y[row]);
    let decomposition = design.col_piv_qr();
    // A rank-deficient X shows as a vanishing diagonal of R.
    let upper = decomposition.R();
    let diagonal = |k: usize| upper[(k, k)].abs();
    let largest = (0..cols).map(diagonal).fold(0.0, f64::max);
    if (0..cols).any(|k| diagonal(k) <= largest * RANK_TOLERANCE) {
        return None;
    }
    let solution = decomposition.solve_lstsq(&target);
    let coefficients: Vec<f64> = (0..cols).map(|k| solution[(k, 0)]).collect();
    coefficients
        .iter()
        .all(|c| c.is_finite())
        .then_some(coefficients)
}

/// Relative size below which a diagonal entry of R counts as zero.
const RANK_TOLERANCE: f64 = 1e-10;

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_least_squares_fits_a_line_exactly() -> Result<()> {
        // y = 2 + 3x.
        let x: Vec<f64> = (0..10).flat_map(|i| [1.0, f64::from(i)]).collect();
        let y: Vec<f64> = (0..10).map(|i| 2.0 + 3.0 * f64::from(i)).collect();
        let b = least_squares(&x, &y, 2).context("solved")?;
        assert_relative_eq!(b[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(b[1], 3.0, epsilon = 1e-10);
        Ok(())
    }

    #[test]
    fn test_least_squares_refuses_degenerate_systems() {
        // Fewer rows than columns.
        assert_eq!(least_squares(&[1.0, 2.0], &[1.0], 2), None);
        // Two identical columns.
        let x: Vec<f64> = (0..6).flat_map(|i| [f64::from(i), f64::from(i)]).collect();
        let y: Vec<f64> = (0..6).map(f64::from).collect();
        assert_eq!(least_squares(&x, &y, 2), None);
        // Misshaped input.
        assert_eq!(least_squares(&[1.0, 2.0, 3.0], &[1.0, 2.0], 2), None);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// The residual is orthogonal to every column (normal equations).
        #[test]
        fn test_least_squares_residual_is_orthogonal(
            rows in prop::collection::vec(prop::array::uniform3(-10.0_f64..10.0), 6..30),
            y in prop::collection::vec(-50.0_f64..50.0, 30),
        ) {
            let x: Vec<f64> = rows.iter().flat_map(|r| [1.0, r[0], r[1] + 0.5 * r[2]]).collect();
            let y = &y[..rows.len()];
            if let Some(b) = least_squares(&x, y, 3) {
                for c in 0..3 {
                    let dot: f64 = rows
                        .iter()
                        .enumerate()
                        .map(|(i, _)| {
                            let fitted: f64 = (0..3).map(|k| x[i * 3 + k] * b[k]).sum();
                            x[i * 3 + c] * (y[i] - fitted)
                        })
                        .sum();
                    prop_assert!(dot.abs() < 1e-6 * (1.0 + y.iter().map(|v| v.abs()).sum::<f64>()));
                }
            }
        }
    }
}
