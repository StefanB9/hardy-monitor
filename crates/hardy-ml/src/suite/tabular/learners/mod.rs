//! The learners behind the tabular models: each maps a feature row to the
//! deviation from the weekly profile.

mod linear;
mod spline;
mod svr;
mod trees;

use faer::{Mat, Side, prelude::Solve};
pub(super) use linear::{ElasticNet, MedianRegression, Ridge};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
pub(super) use spline::SplineGam;
pub(super) use svr::SupportVectors;
pub(super) use trees::{BoostedTrees, RandomForest};

use super::features::{FEATURES, Row, Rows};
use crate::error::MlError;

/// A regression from feature rows to deviations.
pub trait Learner: Sized + Serialize + DeserializeOwned + std::fmt::Debug {
    /// History needed before the learner can be fitted.
    const MIN_DAYS: i64 = 1;

    /// Fits the learner on `rows`.
    fn fit(rows: &Rows) -> Result<Self, MlError>;

    /// The fitted deviation for `row`.
    fn predict(&self, row: &Row) -> f64;
}

/// Centres and scales each feature to unit variance (constant features are
/// only centred).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Standardizer {
    mean: Vec<f64>,
    scale: Vec<f64>,
}

impl Standardizer {
    /// The standardizer of `rows`' features.
    pub(super) fn fit(rows: &[Row]) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let n = rows.len().max(1) as f64;
        let mean: Vec<f64> = (0..FEATURES)
            .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / n)
            .collect();
        let scale = (0..FEATURES)
            .map(|j| {
                let var = rows.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>() / n;
                if var > 1e-12 { var.sqrt() } else { 1.0 }
            })
            .collect();
        Self { mean, scale }
    }

    /// `row` standardized.
    pub(super) fn apply(&self, row: &Row) -> Row {
        std::array::from_fn(|j| (row[j] - self.mean[j]) / self.scale[j])
    }
}

/// Solves `a · x = b` for symmetric positive definite `a` (`n × n`,
/// row-major) by Cholesky (faer); `None` if `a` is not positive definite.
pub(super) fn solve_spd(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n * n {
        return None;
    }
    let matrix = Mat::from_fn(n, n, |i, j| a[i * n + j]);
    let rhs = Mat::from_fn(n, 1, |i, _| b[i]);
    let solution = matrix.llt(Side::Lower).ok()?.solve(&rhs);
    let x: Vec<f64> = (0..n).map(|i| solution[(i, 0)]).collect();
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// The normal equations of standardized rows against centred targets:
/// (`ZᵀWZ`, `ZᵀWy`), weighted by `weights` (all 1 when `None`).
pub(super) fn normal_equations(
    z: &[Row],
    y: &[f64],
    weights: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let mut gram = vec![0.0; FEATURES * FEATURES];
    let mut cross = vec![0.0; FEATURES];
    for (i, (row, target)) in z.iter().zip(y).enumerate() {
        let w = weights.map_or(1.0, |w| w[i]);
        for a in 0..FEATURES {
            cross[a] += w * row[a] * target;
            for b in a..FEATURES {
                gram[a * FEATURES + b] += w * row[a] * row[b];
            }
        }
    }
    for a in 0..FEATURES {
        for b in 0..a {
            gram[a * FEATURES + b] = gram[b * FEATURES + a];
        }
    }
    (gram, cross)
}

/// Checks that `rows` can be fitted: rows present and as many targets.
pub(super) fn check(rows: &Rows) -> Result<(), MlError> {
    if rows.x.is_empty() || rows.x.len() != rows.y.len() {
        return Err(MlError::InsufficientData {
            found: rows.x.len().min(rows.y.len()),
            required: 1,
        });
    }
    Ok(())
}

#[cfg(test)]
pub(super) mod test_rows {
    use super::*;

    /// Deterministic values in −1…1.
    pub(crate) fn noise(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            unit * 2.0 - 1.0
        }
    }

    /// `n` rows of random features (scaled by column) with target `f(row)`.
    pub(crate) fn rows(n: usize, mut target: impl FnMut(&Row) -> f64) -> Rows {
        let mut random = noise(11);
        let x: Vec<Row> = (0..n)
            .map(|_| {
                std::array::from_fn(|j| {
                    #[allow(clippy::cast_precision_loss)]
                    let scale = 1.0 + j as f64;
                    random() * scale
                })
            })
            .collect();
        let y = x.iter().map(&mut target).collect();
        Rows { x, y }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{test_rows::rows, *};

    #[test]
    fn test_standardizer_centres_and_scales() {
        let data = rows(500, |_| 0.0);
        let mut constant = data.x.clone();
        for row in &mut constant {
            row[3] = 7.0;
        }
        let s = Standardizer::fit(&constant);
        let standardized: Vec<Row> = constant.iter().map(|r| s.apply(r)).collect();
        for j in [0, 5, 15] {
            #[allow(clippy::cast_precision_loss)]
            let n = standardized.len() as f64;
            let mean = standardized.iter().map(|r| r[j]).sum::<f64>() / n;
            let var = standardized
                .iter()
                .map(|r| (r[j] - mean).powi(2))
                .sum::<f64>()
                / n;
            assert_relative_eq!(mean, 0.0, epsilon = 1e-9);
            assert_relative_eq!(var, 1.0, epsilon = 1e-9);
        }
        assert!(
            standardized.iter().all(|r| r[3] == 0.0),
            "constant column centred"
        );
    }

    #[test]
    fn test_solve_spd() {
        // [[4, 1], [1, 3]] x = [1, 2] → x = [1/11, 7/11].
        let x = solve_spd(&[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0]).unwrap_or_default();
        assert_relative_eq!(x[0], 1.0 / 11.0, epsilon = 1e-12);
        assert_relative_eq!(x[1], 7.0 / 11.0, epsilon = 1e-12);
        assert_eq!(solve_spd(&[1.0, 2.0, 2.0, 1.0], &[1.0, 1.0]), None);
        assert_eq!(solve_spd(&[1.0, 2.0, 3.0], &[1.0, 1.0]), None);
    }

    #[test]
    fn test_check_refuses_empty_or_misshaped_rows() {
        assert!(check(&Rows::default()).is_err());
        let mut data = rows(3, |_| 0.0);
        data.y.pop();
        assert!(check(&data).is_err());
        assert!(check(&rows(3, |_| 0.0)).is_ok());
    }
}
