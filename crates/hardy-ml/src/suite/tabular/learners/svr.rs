//! Support-vector regression with an RBF kernel, fitted by smartcore on the
//! newest rows (a kernel fit grows with the square of the rows).
//!
//! smartcore's fitted SVR borrows its parameters and cannot predict once
//! deserialized, so the support vectors, their weights and the bias are
//! read out of it after the fit and evaluated here.

use serde::{Deserialize, Serialize};
use smartcore::{
    linalg::basic::matrix::DenseMatrix,
    svm::{
        Kernels,
        svr::{SVR, SVRParameters},
    },
};

use super::{Learner, Standardizer, check};
use crate::{
    error::MlError,
    suite::tabular::features::{FEATURES, Row, Rows},
};

/// Newest rows the SVR is fitted on.
pub(super) const MAX_ROWS: usize = 4_000;
/// Penalty on errors outside the tube.
const COST: f64 = 10.0;
/// Half-width of the tube errors are free in (points).
const TUBE: f64 = 0.5;

/// See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportVectors {
    standardizer: Standardizer,
    gamma: f64,
    vectors: Vec<Vec<f64>>,
    weights: Vec<f64>,
    bias: f64,
}

impl Learner for SupportVectors {
    fn fit(rows: &Rows) -> Result<Self, MlError> {
        check(rows)?;
        let start = rows.x.len().saturating_sub(MAX_ROWS);
        let (x, y) = (&rows.x[start..], &rows.y[start..]);
        let standardizer = Standardizer::fit(x);
        let inputs: Vec<Vec<f64>> = x.iter().map(|r| standardizer.apply(r).to_vec()).collect();
        let failed = |e: smartcore::error::Failed| MlError::Fit(format!("SVR: {e}"));
        let matrix = DenseMatrix::from_2d_vec(&inputs).map_err(failed)?;
        #[allow(clippy::cast_precision_loss)]
        let gamma = 1.0 / FEATURES as f64;
        let parameters = SVRParameters::default()
            .with_c(COST)
            .with_eps(TUBE)
            .with_kernel(Kernels::rbf().with_gamma(gamma));
        let targets = y.to_vec();
        let fitted = SVR::fit(&matrix, &targets, &parameters).map_err(failed)?;
        let fields = serde_json::to_value(&fitted)
            .map_err(|e| MlError::Serialization(format!("SVR: {e}")))?;
        let read = |name: &str| {
            fields
                .get(name)
                .cloned()
                .ok_or_else(|| MlError::Serialization(format!("SVR has no field {name}")))
        };
        let parse = |e: serde_json::Error| MlError::Serialization(format!("SVR: {e}"));
        Ok(Self {
            standardizer,
            gamma,
            vectors: serde_json::from_value(read("instances")?).map_err(parse)?,
            weights: serde_json::from_value(read("w")?).map_err(parse)?,
            bias: serde_json::from_value(read("b")?).map_err(parse)?,
        })
    }

    fn predict(&self, row: &Row) -> f64 {
        let z = self.standardizer.apply(row);
        self.bias
            + self
                .vectors
                .iter()
                .zip(&self.weights)
                .map(|(v, w)| {
                    let distance: f64 = v.iter().zip(&z).map(|(a, b)| (a - b).powi(2)).sum();
                    w * (-self.gamma * distance).exp()
                })
                .sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;

    use super::*;
    use crate::suite::{decode, encode, tabular::learners::test_rows::rows};

    fn truth(row: &Row) -> f64 {
        3.0 * (row[0]).sin() + 0.2 * row[4]
    }

    #[test]
    fn test_prediction_matches_smartcore() -> Result<()> {
        let data = rows(600, truth);
        let model = SupportVectors::fit(&data)?;
        // smartcore's own prediction on the same standardized rows.
        let standardized: Vec<Vec<f64>> = data
            .x
            .iter()
            .map(|r| model.standardizer.apply(r).to_vec())
            .collect();
        let x = DenseMatrix::from_2d_vec(&standardized)?;
        let parameters = SVRParameters::default()
            .with_c(COST)
            .with_eps(TUBE)
            .with_kernel(Kernels::rbf().with_gamma(model.gamma));
        let reference = SVR::fit(&x, &data.y, &parameters)?.predict(&x)?;
        for (row, expected) in data.x.iter().zip(&reference).take(50) {
            assert_relative_eq!(model.predict(row), *expected, epsilon = 1e-9);
        }
        Ok(())
    }

    #[test]
    fn test_fits_the_newest_rows_and_round_trips() -> Result<()> {
        let mut data = rows(MAX_ROWS + 500, truth);
        // The oldest rows are garbage; only the newest count.
        for y in data.y.iter_mut().take(500) {
            *y = 99.0;
        }
        let model = SupportVectors::fit(&data)?;
        let row = data.x.last().context("a row")?;
        assert_relative_eq!(model.predict(row), truth(row), epsilon = 1.0);
        let restored: SupportVectors = decode(&encode(&model)?)?;
        assert_eq!(restored, model);
        Ok(())
    }

    #[test]
    fn test_svr_refuses_empty_rows() {
        assert!(SupportVectors::fit(&Rows::default()).is_err());
    }
}
