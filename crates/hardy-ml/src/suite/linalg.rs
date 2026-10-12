//! Small dense linear algebra for model fits: a linear solve, the top
//! eigenvectors of a symmetric matrix and least squares on the simplex.
//! Matrices are row-major `Vec<f64>`; the sizes involved (a few dozen) do not
//! justify a library.

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

/// Weights `w ≥ 0` with `Σ w = 1` minimising `‖X w − y‖²`, given the Gram
/// matrix `gram = XᵀX` (`n × n`, row-major) and `cross = Xᵀy`.
///
/// Accelerated projected gradient (FISTA): the problem is convex, so it
/// converges to the optimum; members' forecasts are highly correlated, which
/// slows plain gradient descent but not the answer.
pub fn simplex_least_squares(gram: &[f64], cross: &[f64]) -> Vec<f64> {
    const MAX_ITERATIONS: usize = 20_000;
    const TOLERANCE: f64 = 1e-13;
    let n = cross.len();
    if n == 0 || gram.len() != n * n {
        return Vec::new();
    }
    // Lipschitz constant of the gradient 2(Gw − c): twice the largest
    // eigenvalue; the trace bounds it if power iteration finds none.
    let largest = top_eigenvectors(gram, n, 1).first().map_or_else(
        || (0..n).map(|i| gram[i * n + i]).sum(),
        |(value, _)| *value,
    );
    #[allow(clippy::cast_precision_loss)]
    let uniform = 1.0 / n as f64;
    let mut w = vec![uniform; n];
    if largest <= 0.0 {
        return w;
    }
    let step = 1.0 / (2.0 * largest);
    let mut momentum_point = w.clone();
    let mut t = 1.0_f64;
    let mut next = vec![0.0; n];
    for _ in 0..MAX_ITERATIONS {
        let product = multiply(gram, &momentum_point, n);
        for i in 0..n {
            next[i] = momentum_point[i] - step * 2.0 * (product[i] - cross[i]);
        }
        project_onto_simplex(&mut next);
        let t_next = f64::midpoint(1.0, (1.0 + 4.0 * t * t).sqrt());
        let blend = (t - 1.0) / t_next;
        let mut change = 0.0_f64;
        for i in 0..n {
            change = change.max((next[i] - w[i]).abs());
            momentum_point[i] = next[i] + blend * (next[i] - w[i]);
        }
        w.copy_from_slice(&next);
        t = t_next;
        if change < TOLERANCE {
            break;
        }
    }
    w
}

/// Replaces `v` by its Euclidean projection onto the simplex
/// `{w ≥ 0, Σ w = 1}` (sort-based, Duchi et al. 2008).
fn project_onto_simplex(v: &mut [f64]) {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mut sum = 0.0;
    let mut shift = 0.0;
    for (k, &x) in sorted.iter().enumerate() {
        sum += x;
        #[allow(clippy::cast_precision_loss)]
        let candidate = (sum - 1.0) / (k + 1) as f64;
        if x - candidate > 0.0 {
            shift = candidate;
        }
    }
    for x in v.iter_mut() {
        *x = (*x - shift).max(0.0);
    }
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

    #[test]
    fn test_simplex_least_squares_recovers_a_convex_mix() {
        // y = 0.3 a + 0.7 b exactly, with a third column that does not help.
        let (gram, cross) = normal_equations(&mix_columns(), &[0.3, 0.7, 0.0]);
        let w = simplex_least_squares(&gram, &cross);
        assert_relative_eq!(w[0], 0.3, epsilon = 1e-6);
        assert_relative_eq!(w[1], 0.7, epsilon = 1e-6);
        assert_relative_eq!(w[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_simplex_least_squares_puts_all_weight_on_an_exact_column() {
        // y is twice column b: the best convex mix is b alone.
        let columns = mix_columns();
        let y: Vec<f64> = columns.iter().map(|row| 2.0 * row[1]).collect();
        let (gram, cross) = normal_equations_for(&columns, &y);
        let w = simplex_least_squares(&gram, &cross);
        assert_relative_eq!(w[1], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_simplex_least_squares_without_columns_is_empty() {
        assert_eq!(simplex_least_squares(&[], &[]), Vec::<f64>::new());
    }

    /// Rows of three member forecasts.
    fn mix_columns() -> Vec<[f64; 3]> {
        (0..40)
            .map(|i| {
                let t = f64::from(i);
                [10.0 + t, 50.0 - t * 0.5, 30.0 + (t * 0.7).sin() * 5.0]
            })
            .collect()
    }

    fn normal_equations(columns: &[[f64; 3]], weights: &[f64; 3]) -> (Vec<f64>, Vec<f64>) {
        let y: Vec<f64> = columns.iter().map(|row| dot(row, weights)).collect();
        normal_equations_for(columns, &y)
    }

    fn normal_equations_for(columns: &[[f64; 3]], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut gram = vec![0.0; 9];
        let mut cross = vec![0.0; 3];
        for (row, &target) in columns.iter().zip(y) {
            for i in 0..3 {
                cross[i] += row[i] * target;
                for j in 0..3 {
                    gram[i * 3 + j] += row[i] * row[j];
                }
            }
        }
        (gram, cross)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// The weights always form a convex mix and are no worse than any
        /// single column.
        #[test]
        fn test_simplex_least_squares_is_convex_and_beats_every_column(
            rows in prop::collection::vec(prop::array::uniform3(0.0_f64..100.0), 3..30),
            y in prop::collection::vec(0.0_f64..100.0, 30),
        ) {
            let y = &y[..rows.len()];
            let (gram, cross) = normal_equations_for(&rows, y);
            let w = simplex_least_squares(&gram, &cross);
            prop_assert_eq!(w.len(), 3);
            prop_assert!(w.iter().all(|&x| x >= 0.0));
            prop_assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            let loss = |weights: &[f64]| -> f64 {
                rows.iter()
                    .zip(y)
                    .map(|(row, t)| (dot(row, weights) - t).powi(2))
                    .sum()
            };
            let mixed = loss(&w);
            for k in 0..3 {
                let mut single = [0.0; 3];
                single[k] = 1.0;
                let alone = loss(&single);
                prop_assert!(mixed <= alone * (1.0 + 1e-6) + 1e-6, "{mixed} > {alone}");
            }
        }
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
