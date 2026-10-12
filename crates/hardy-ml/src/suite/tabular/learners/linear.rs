//! Linear learners on standardized features: ridge (λ by GCV), elastic-net
//! (coordinate descent) and median regression (IRLS).

use serde::{Deserialize, Serialize};

use super::{Learner, Standardizer, check, normal_equations, solve_spd};
use crate::{
    error::MlError,
    suite::tabular::features::{FEATURES, Row, Rows},
};

/// A linear model on standardized features.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Linear {
    standardizer: Standardizer,
    intercept: f64,
    coefficients: Vec<f64>,
}

impl Linear {
    fn predict(&self, row: &Row) -> f64 {
        let z = self.standardizer.apply(row);
        self.intercept
            + z.iter()
                .zip(&self.coefficients)
                .map(|(x, b)| x * b)
                .sum::<f64>()
    }
}

/// Standardized rows, the standardizer and the targets' mean.
fn prepare(rows: &Rows) -> (Standardizer, Vec<Row>, f64) {
    let standardizer = Standardizer::fit(&rows.x);
    let z = rows.x.iter().map(|r| standardizer.apply(r)).collect();
    #[allow(clippy::cast_precision_loss)]
    let mean = rows.y.iter().sum::<f64>() / rows.y.len().max(1) as f64;
    (standardizer, z, mean)
}

/// `gram` with `penalty` added to its diagonal.
fn penalized(gram: &[f64], penalty: f64) -> Vec<f64> {
    let mut a = gram.to_vec();
    for j in 0..FEATURES {
        a[j * FEATURES + j] += penalty;
    }
    a
}

/// Penalties tried by ridge, relative to the number of rows.
const RIDGE_PENALTIES: [f64; 8] = [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0];

/// Share of lasso in the elastic-net penalty.
const EN_MIX: f64 = 0.5;
/// Elastic-net penalty as a share of the smallest one that zeroes every
/// coefficient.
const EN_LAMBDA_SHARE: f64 = 0.01;
/// Coordinate-descent sweeps at most.
const EN_MAX_SWEEPS: usize = 1_000;
/// Reweighting rounds of median regression.
const IRLS_ITERATIONS: usize = 50;
/// Smallest residual a weight is computed from (keeps weights finite).
const IRLS_FLOOR: f64 = 1e-4;
/// Tiny ridge that keeps the weighted system solvable.
const IRLS_RIDGE: f64 = 1e-8;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    value.signum() * (value.abs() - threshold).max(0.0)
}

/// Ridge regression, the penalty chosen by generalised cross-validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ridge(Linear);

impl Learner for Ridge {
    fn fit(rows: &Rows) -> Result<Self, MlError> {
        check(rows)?;
        let (standardizer, z, mean) = prepare(rows);
        let centred: Vec<f64> = rows.y.iter().map(|y| y - mean).collect();
        let (gram, cross) = normal_equations(&z, &centred, None);
        let yy: f64 = centred.iter().map(|y| y * y).sum();
        #[allow(clippy::cast_precision_loss)]
        let n = z.len() as f64;
        let mut best: Option<(f64, Vec<f64>)> = None;
        for relative in RIDGE_PENALTIES {
            let a = penalized(&gram, relative * n);
            let Some(beta) = solve_spd(&a, &cross) else {
                continue;
            };
            // Residuals from the normal equations: yᵀy − 2βᵀc + βᵀGβ.
            let g_beta: Vec<f64> = (0..FEATURES)
                .map(|i| {
                    (0..FEATURES)
                        .map(|j| gram[i * FEATURES + j] * beta[j])
                        .sum()
                })
                .collect();
            let rss = (yy - 2.0 * dot(&beta, &cross) + dot(&beta, &g_beta)).max(0.0);
            // Effective parameters: trace((G + λI)⁻¹ G), column by column.
            let trace: f64 = (0..FEATURES)
                .filter_map(|j| {
                    let column: Vec<f64> = (0..FEATURES).map(|i| gram[i * FEATURES + j]).collect();
                    solve_spd(&a, &column).map(|x| x[j])
                })
                .sum();
            let gcv = n * rss / (n - trace).max(1.0).powi(2);
            if best.as_ref().is_none_or(|(g, _)| gcv < *g) {
                best = Some((gcv, beta));
            }
        }
        let (_, coefficients) =
            best.ok_or_else(|| MlError::Fit("ridge system is singular".to_string()))?;
        Ok(Self(Linear {
            standardizer,
            intercept: mean,
            coefficients,
        }))
    }

    fn predict(&self, row: &Row) -> f64 {
        self.0.predict(row)
    }
}

/// Elastic-net regression (half lasso, half ridge) by coordinate descent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElasticNet(Linear);

impl Learner for ElasticNet {
    fn fit(rows: &Rows) -> Result<Self, MlError> {
        check(rows)?;
        let (standardizer, z, mean) = prepare(rows);
        let centred: Vec<f64> = rows.y.iter().map(|y| y - mean).collect();
        let (gram, cross) = normal_equations(&z, &centred, None);
        #[allow(clippy::cast_precision_loss)]
        let n = z.len() as f64;
        let lambda_max = cross.iter().map(|c| c.abs()).fold(0.0, f64::max) / (n * EN_MIX);
        let lambda = EN_LAMBDA_SHARE * lambda_max;
        let mut beta = vec![0.0; FEATURES];
        for _ in 0..EN_MAX_SWEEPS {
            let mut largest_change = 0.0_f64;
            for j in 0..FEATURES {
                let diagonal = gram[j * FEATURES + j];
                if diagonal <= 0.0 {
                    continue;
                }
                let partial = cross[j]
                    - (0..FEATURES)
                        .filter(|&k| k != j)
                        .map(|k| gram[j * FEATURES + k] * beta[k])
                        .sum::<f64>();
                let next = soft_threshold(partial, n * lambda * EN_MIX)
                    / (diagonal + n * lambda * (1.0 - EN_MIX));
                largest_change = largest_change.max((next - beta[j]).abs());
                beta[j] = next;
            }
            if largest_change < 1e-9 {
                break;
            }
        }
        Ok(Self(Linear {
            standardizer,
            intercept: mean,
            coefficients: beta,
        }))
    }

    fn predict(&self, row: &Row) -> f64 {
        self.0.predict(row)
    }
}

/// Median (least absolute deviation) regression by iteratively reweighted
/// least squares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MedianRegression(Linear);

impl Learner for MedianRegression {
    fn fit(rows: &Rows) -> Result<Self, MlError> {
        check(rows)?;
        let (standardizer, z, _) = prepare(rows);
        // The intercept rides along as a constant first column.
        let extended: Vec<Vec<f64>> = z
            .iter()
            .map(|r| std::iter::once(1.0).chain(r.iter().copied()).collect())
            .collect();
        let p = FEATURES + 1;
        let mut weights = vec![1.0; extended.len()];
        let mut beta = vec![0.0; p];
        for _ in 0..IRLS_ITERATIONS {
            let mut gram = vec![0.0; p * p];
            let mut cross = vec![0.0; p];
            for ((row, y), w) in extended.iter().zip(&rows.y).zip(&weights) {
                for a in 0..p {
                    cross[a] += w * row[a] * y;
                    for b in 0..p {
                        gram[a * p + b] += w * row[a] * row[b];
                    }
                }
            }
            for a in 0..p {
                gram[a * p + a] += IRLS_RIDGE;
            }
            let Some(next) = solve_spd(&gram, &cross) else {
                break;
            };
            beta = next;
            for ((row, y), w) in extended.iter().zip(&rows.y).zip(weights.iter_mut()) {
                let residual = y - dot(row, &beta);
                *w = 1.0 / residual.abs().max(IRLS_FLOOR);
            }
        }
        Ok(Self(Linear {
            standardizer,
            intercept: beta[0],
            coefficients: beta[1..].to_vec(),
        }))
    }

    fn predict(&self, row: &Row) -> f64 {
        self.0.predict(row)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;

    use super::*;
    use crate::suite::tabular::learners::test_rows::{noise, rows};

    fn truth(row: &Row) -> f64 {
        1.0 + 2.0 * row[0] - 0.5 * row[3]
    }

    #[test]
    fn test_ridge_recovers_a_linear_relation() -> Result<()> {
        let data = rows(2_000, truth);
        let model = Ridge::fit(&data)?;
        for row in data.x.iter().take(20) {
            assert_relative_eq!(model.predict(row), truth(row), epsilon = 0.05);
        }
        Ok(())
    }

    #[test]
    fn test_elastic_net_drops_irrelevant_features() -> Result<()> {
        let mut e = noise(3);
        let data = rows(2_000, |r| truth(r) + 0.1 * e());
        let model = ElasticNet::fit(&data)?;
        let Linear { coefficients, .. } = &model.0;
        // Features other than 0 and 3 do not matter.
        for (j, c) in coefficients.iter().enumerate() {
            if j != 0 && j != 3 {
                assert!(c.abs() < 0.05, "feature {j}: {c}");
            }
        }
        assert!(coefficients[0] > 1.0, "{coefficients:?}");
        for row in data.x.iter().take(20) {
            assert_relative_eq!(model.predict(row), truth(row), epsilon = 0.5);
        }
        Ok(())
    }

    #[test]
    fn test_median_regression_ignores_outliers() -> Result<()> {
        // Every tenth target is 50 too high: least squares would shift up.
        let data = rows(2_000, truth);
        let mut skewed = data.clone();
        for (i, y) in skewed.y.iter_mut().enumerate() {
            if i % 10 == 0 {
                *y += 50.0;
            }
        }
        let model = MedianRegression::fit(&skewed)?;
        for row in data.x.iter().skip(1).step_by(10).take(20) {
            assert_relative_eq!(model.predict(row), truth(row), epsilon = 0.3);
        }
        Ok(())
    }

    #[test]
    fn test_learners_refuse_empty_rows() {
        assert!(Ridge::fit(&Rows::default()).is_err());
        assert!(ElasticNet::fit(&Rows::default()).is_err());
        assert!(MedianRegression::fit(&Rows::default()).is_err());
    }
}
