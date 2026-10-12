//! Small dense linear algebra for model fits: a linear solve and the top
//! eigenvectors of a symmetric matrix. Matrices are row-major `Vec<f64>`;
//! the sizes involved (a few dozen) do not justify a library.

use crate::error::MlError;

/// Solves `a · x = b` for square `a` (`n × n`, row-major) by Gaussian
/// elimination with partial pivoting.
pub fn solve(a: &[f64], b: &[f64]) -> Result<Vec<f64>, MlError> {
    let n = b.len();
    if a.len() != n * n {
        return Err(MlError::Fit(format!(
            "{} matrix values for size {n}",
            a.len()
        )));
    }
    let mut work = a.to_vec();
    let mut solution = b.to_vec();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| work[i * n + col].abs().total_cmp(&work[j * n + col].abs()))
            .unwrap_or(col);
        if work[pivot * n + col].abs() < 1e-12 {
            return Err(MlError::Fit("singular system".to_string()));
        }
        if pivot != col {
            for k in 0..n {
                work.swap(col * n + k, pivot * n + k);
            }
            solution.swap(col, pivot);
        }
        for row in col + 1..n {
            let factor = work[row * n + col] / work[col * n + col];
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                work[row * n + k] -= factor * work[col * n + k];
            }
            solution[row] -= factor * solution[col];
        }
    }
    for col in (0..n).rev() {
        let tail: f64 = (col + 1..n).map(|k| work[col * n + k] * solution[k]).sum();
        solution[col] = (solution[col] - tail) / work[col * n + col];
    }
    Ok(solution)
}

/// Iterations of the power method per eigenvector.
const POWER_ITERATIONS: usize = 300;

/// The `k` eigenvectors of symmetric `a` (`n × n`) with the largest
/// eigenvalues, by power iteration with deflation; each unit length, with
/// its eigenvalue. Stops early when the remaining eigenvalues vanish.
pub fn top_eigenvectors(a: &[f64], n: usize, k: usize) -> Vec<(f64, Vec<f64>)> {
    let mut work = a.to_vec();
    let mut out = Vec::with_capacity(k);
    for index in 0..k {
        // A start vector not orthogonal to the leading eigenvector.
        #[allow(clippy::cast_precision_loss)]
        let mut v: Vec<f64> = (0..n)
            .map(|i| 1.0 + ((i + index) % 7) as f64 * 0.1)
            .collect();
        normalize(&mut v);
        let mut value = 0.0;
        for _ in 0..POWER_ITERATIONS {
            let mut next = multiply(&work, &v, n);
            value = dot(&v, &next);
            if normalize(&mut next) < 1e-12 {
                return out;
            }
            v = next;
        }
        if value <= 1e-9 {
            break;
        }
        for i in 0..n {
            for j in 0..n {
                work[i * n + j] -= value * v[i] * v[j];
            }
        }
        out.push((value, v));
    }
    out
}

fn multiply(a: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            a[i * n..(i + 1) * n]
                .iter()
                .zip(v)
                .map(|(x, y)| x * y)
                .sum()
        })
        .collect()
}

/// Dot product.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Scales `v` to unit length; returns its former length.
fn normalize(v: &mut [f64]) -> f64 {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_solve_a_small_system() -> Result<()> {
        // 2x + y = 5, x + 3y = 10 → x = 1, y = 3.
        let x = solve(&[2.0, 1.0, 1.0, 3.0], &[5.0, 10.0])?;
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(x[1], 3.0, epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn test_solve_needs_a_pivot_swap() -> Result<()> {
        let x = solve(&[0.0, 1.0, 1.0, 0.0], &[2.0, 3.0])?;
        assert_eq!(x, vec![3.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_solve_rejects_singular_and_misshaped_systems() {
        assert!(solve(&[1.0, 2.0, 2.0, 4.0], &[1.0, 2.0]).is_err());
        assert!(solve(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_top_eigenvectors_of_a_diagonal_matrix() {
        let a = [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let vectors = top_eigenvectors(&a, 3, 2);
        assert_eq!(vectors.len(), 2);
        assert_relative_eq!(vectors[0].0, 3.0, epsilon = 1e-6);
        assert_relative_eq!(vectors[0].1[0].abs(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(vectors[1].0, 2.0, epsilon = 1e-6);
        assert_relative_eq!(vectors[1].1[2].abs(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_top_eigenvectors_stop_at_rank() {
        // Rank one: v vᵀ with v = (1, 1).
        let vectors = top_eigenvectors(&[1.0, 1.0, 1.0, 1.0], 2, 2);
        assert_eq!(vectors.len(), 1);
        assert_relative_eq!(vectors[0].0, 2.0, epsilon = 1e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// A diagonally dominant system is solved to within rounding.
        #[test]
        fn test_solve_reproduces_the_right_hand_side(
            off in prop::collection::vec(-1.0_f64..1.0, 9),
            b in prop::collection::vec(-50.0_f64..50.0, 3),
        ) {
            let mut a = off;
            for i in 0..3 {
                a[i * 3 + i] += 5.0;
            }
            let x = solve(&a, &b).unwrap_or_default();
            for i in 0..3 {
                let row: f64 = (0..3).map(|j| a[i * 3 + j] * x[j]).sum();
                prop_assert!((row - b[i]).abs() < 1e-9);
            }
        }
    }
}
