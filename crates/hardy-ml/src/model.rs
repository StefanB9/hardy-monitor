//! Regression backends: random forest (smartcore) and ridge-regularised
//! linear regression (linfa).

use std::sync::Arc;

use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use smartcore::{
    ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters},
    linalg::basic::matrix::DenseMatrix,
};

use crate::{
    error::MlError,
    features::{FeatureRow, NUM_FEATURES},
};

/// Largest forest accepted; keeps stored models small (see plan).
pub const MAX_TREES: u16 = 100;
/// Deepest tree accepted.
pub const MAX_DEPTH: u16 = 12;
/// Smallest leaf accepted.
pub const MIN_LEAF: u16 = 10;

/// Fixed seed so training is reproducible.
const RF_SEED: u64 = 0x4A52_4459;
/// L2 penalty for the linear model; stabilises correlated features.
const RIDGE_LAMBDA: f64 = 1e-3;

pub type Forest = RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>;

/// Which backend to train.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algorithm {
    RandomForest,
    Linear,
}

impl Algorithm {
    pub fn name(self) -> &'static str {
        match self {
            Algorithm::RandomForest => "Random Forest",
            Algorithm::Linear => "Linear Regression",
        }
    }
}

impl From<&hardy_core::MlAlgorithm> for Algorithm {
    fn from(algorithm: &hardy_core::MlAlgorithm) -> Self {
        match algorithm {
            hardy_core::MlAlgorithm::RandomForest => Algorithm::RandomForest,
            hardy_core::MlAlgorithm::LinearRegression => Algorithm::Linear,
        }
    }
}

/// Random forest hyperparameters, validated against the size caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RfParams {
    n_trees: u16,
    max_depth: u16,
    min_samples_leaf: u16,
}

impl RfParams {
    /// Used nightly until a weekly search picks something better.
    pub const DEFAULT: RfParams = RfParams {
        n_trees: 100,
        max_depth: 12,
        min_samples_leaf: 10,
    };

    pub fn new(n_trees: u16, max_depth: u16, min_samples_leaf: u16) -> Result<Self, MlError> {
        if !(1..=MAX_TREES).contains(&n_trees) {
            return Err(MlError::InvalidParams(format!(
                "n_trees must be 1–{MAX_TREES}, got {n_trees}"
            )));
        }
        if !(1..=MAX_DEPTH).contains(&max_depth) {
            return Err(MlError::InvalidParams(format!(
                "max_depth must be 1–{MAX_DEPTH}, got {max_depth}"
            )));
        }
        if min_samples_leaf < MIN_LEAF {
            return Err(MlError::InvalidParams(format!(
                "min_samples_leaf must be >= {MIN_LEAF}, got {min_samples_leaf}"
            )));
        }
        Ok(Self {
            n_trees,
            max_depth,
            min_samples_leaf,
        })
    }

    /// The weekly search space: 8 configurations.
    pub fn weekly_grid() -> Vec<RfParams> {
        let mut grid = Vec::with_capacity(8);
        for n_trees in [50, 100] {
            for max_depth in [8, 12] {
                for min_samples_leaf in [10, 20] {
                    grid.push(RfParams {
                        n_trees,
                        max_depth,
                        min_samples_leaf,
                    });
                }
            }
        }
        grid
    }

    pub fn n_trees(self) -> u16 {
        self.n_trees
    }

    pub fn max_depth(self) -> u16 {
        self.max_depth
    }

    pub fn min_samples_leaf(self) -> u16 {
        self.min_samples_leaf
    }
}

/// A fitted model.
#[derive(Debug, Clone)]
pub enum Model {
    Linear {
        coefficients: Vec<f64>,
        intercept: f64,
    },
    RandomForest {
        forest: Arc<Forest>,
        params: RfParams,
    },
}

impl Model {
    /// Fits `algorithm` to the rows; `params` only apply to the forest.
    pub fn fit(
        algorithm: Algorithm,
        params: RfParams,
        rows: &[FeatureRow],
        targets: &[f64],
    ) -> Result<Self, MlError> {
        if rows.len() != targets.len() {
            return Err(MlError::Fit(format!(
                "{} rows but {} targets",
                rows.len(),
                targets.len()
            )));
        }
        if rows.len() < 2 {
            return Err(MlError::InsufficientData {
                found: rows.len(),
                required: 2,
            });
        }
        match algorithm {
            Algorithm::RandomForest => fit_forest(params, rows, targets),
            Algorithm::Linear => fit_linear(rows, targets),
        }
    }

    pub fn algorithm(&self) -> Algorithm {
        match self {
            Model::Linear { .. } => Algorithm::Linear,
            Model::RandomForest { .. } => Algorithm::RandomForest,
        }
    }

    /// Forest hyperparameters; `None` for the linear model.
    pub fn rf_params(&self) -> Option<RfParams> {
        match self {
            Model::RandomForest { params, .. } => Some(*params),
            Model::Linear { .. } => None,
        }
    }

    /// Predictions for many rows.
    pub fn predict_batch(&self, rows: &[FeatureRow]) -> Result<Vec<f64>, MlError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            Model::Linear {
                coefficients,
                intercept,
            } => Ok(rows
                .iter()
                .map(|row| {
                    row.as_slice()
                        .iter()
                        .zip(coefficients)
                        .map(|(x, c)| x * c)
                        .sum::<f64>()
                        + intercept
                })
                .collect()),
            Model::RandomForest { forest, .. } => forest
                .predict(&to_matrix(rows)?)
                .map_err(|e| MlError::Fit(format!("forest prediction failed: {e}"))),
        }
    }

    /// Prediction for one row.
    pub fn predict(&self, row: &FeatureRow) -> Result<f64, MlError> {
        self.predict_batch(std::slice::from_ref(row))?
            .first()
            .copied()
            .ok_or_else(|| MlError::Fit("empty prediction".to_string()))
    }
}

fn fit_forest(params: RfParams, rows: &[FeatureRow], targets: &[f64]) -> Result<Model, MlError> {
    let rf_params = RandomForestRegressorParameters::default()
        .with_n_trees(usize::from(params.n_trees))
        .with_max_depth(params.max_depth)
        .with_min_samples_leaf(usize::from(params.min_samples_leaf))
        .with_min_samples_split(usize::from(params.min_samples_leaf) * 2)
        .with_seed(RF_SEED);
    let forest = RandomForestRegressor::fit(&to_matrix(rows)?, &targets.to_vec(), rf_params)
        .map_err(|e| MlError::Fit(format!("random forest: {e}")))?;
    Ok(Model::RandomForest {
        forest: Arc::new(forest),
        params,
    })
}

fn fit_linear(rows: &[FeatureRow], targets: &[f64]) -> Result<Model, MlError> {
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.0).collect();
    let x = Array2::from_shape_vec((rows.len(), NUM_FEATURES), flat)
        .map_err(|e| MlError::Fit(e.to_string()))?;
    let y = Array1::from_vec(targets.to_vec());

    // Ridge via data augmentation: append sqrt(lambda) * I with zero targets.
    let penalty = Array2::<f64>::eye(NUM_FEATURES) * RIDGE_LAMBDA.sqrt();
    let x_aug = ndarray::concatenate(Axis(0), &[x.view(), penalty.view()])
        .map_err(|e| MlError::Fit(e.to_string()))?;
    let y_aug = ndarray::concatenate(Axis(0), &[y.view(), Array1::zeros(NUM_FEATURES).view()])
        .map_err(|e| MlError::Fit(e.to_string()))?;

    let fitted = LinearRegression::default()
        .fit(&Dataset::new(x_aug, y_aug))
        .map_err(|e| MlError::Fit(format!("linear regression: {e}")))?;
    Ok(Model::Linear {
        coefficients: fitted.params().to_vec(),
        intercept: fitted.intercept(),
    })
}

fn to_matrix(rows: &[FeatureRow]) -> Result<DenseMatrix<f64>, MlError> {
    let nested: Vec<Vec<f64>> = rows.iter().map(|r| r.0.to_vec()).collect();
    DenseMatrix::from_2d_vec(&nested).map_err(|e| MlError::Fit(format!("feature matrix: {e}")))
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;

    use super::*;

    /// Target = 2 * feature 10 + 5, other features noise-free constants.
    fn linear_data(n: usize) -> (Vec<FeatureRow>, Vec<f64>) {
        (0..n)
            .map(|i| {
                let mut row = [1.0; NUM_FEATURES];
                #[allow(clippy::cast_precision_loss)]
                let x = (i % 50) as f64;
                row[10] = x;
                (FeatureRow(row), 2.0 * x + 5.0)
            })
            .unzip()
    }

    #[test]
    fn test_rf_params_validation() {
        assert!(RfParams::new(0, 8, 10).is_err());
        assert!(RfParams::new(MAX_TREES + 1, 8, 10).is_err());
        assert!(RfParams::new(50, MAX_DEPTH + 1, 10).is_err());
        assert!(RfParams::new(50, 8, MIN_LEAF - 1).is_err());
        assert!(RfParams::new(50, 8, 10).is_ok());
    }

    #[test]
    fn test_weekly_grid_within_caps() {
        let grid = RfParams::weekly_grid();
        assert_eq!(grid.len(), 8);
        for p in grid {
            assert!(RfParams::new(p.n_trees(), p.max_depth(), p.min_samples_leaf()).is_ok());
        }
        assert!(RfParams::new(100, 12, 10).is_ok_and(|p| p == RfParams::DEFAULT));
    }

    #[test]
    fn test_linear_model_recovers_relationship() -> Result<()> {
        let (rows, targets) = linear_data(200);
        let model = Model::fit(Algorithm::Linear, RfParams::DEFAULT, &rows, &targets)?;
        let mut probe = [1.0; NUM_FEATURES];
        probe[10] = 30.0;
        assert_relative_eq!(model.predict(&FeatureRow(probe))?, 65.0, epsilon = 0.1);
        assert_eq!(model.algorithm(), Algorithm::Linear);
        Ok(())
    }

    #[test]
    fn test_random_forest_fits_and_is_deterministic() -> Result<()> {
        let (rows, targets) = linear_data(300);
        let params = RfParams::new(20, 8, 10)?;
        let a = Model::fit(Algorithm::RandomForest, params, &rows, &targets)?;
        let b = Model::fit(Algorithm::RandomForest, params, &rows, &targets)?;
        let mut probe = [1.0; NUM_FEATURES];
        probe[10] = 25.0;
        let pa = a.predict(&FeatureRow(probe))?;
        assert_relative_eq!(pa, b.predict(&FeatureRow(probe))?);
        assert!(
            (pa - 55.0).abs() < 10.0,
            "forest prediction {pa} far from 55"
        );
        assert_eq!(a.rf_params(), Some(params));
        Ok(())
    }

    #[test]
    fn test_fit_rejects_mismatched_and_tiny_inputs() {
        let (rows, targets) = linear_data(10);
        assert!(Model::fit(Algorithm::Linear, RfParams::DEFAULT, &rows, &targets[..5]).is_err());
        assert!(matches!(
            Model::fit(
                Algorithm::Linear,
                RfParams::DEFAULT,
                &rows[..1],
                &targets[..1]
            ),
            Err(MlError::InsufficientData { .. })
        ));
    }
}
