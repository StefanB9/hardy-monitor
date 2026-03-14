use std::sync::Arc;

use smartcore::{
    ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters},
    linalg::basic::matrix::DenseMatrix,
};

use super::TrainingError;
use crate::ml::features::PredictionFeatures;

/// Random Forest regression model backend using smartcore.
///
/// The smartcore `RandomForestRegressor` does not implement `Clone`, so we
/// wrap it in `Arc` to allow `TrainedModel` (and thus iced `Message` enums)
/// to remain `Clone`.
#[derive(Debug, Clone)]
pub(crate) struct RandomForestModel {
    model: Arc<RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    n_trees: usize,
}

/// Hyperparameters for Random Forest training.
#[derive(Debug, Clone)]
pub(crate) struct RfHyperparameters {
    pub n_trees: usize,
    pub max_depth: Option<u16>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
}

impl Default for RfHyperparameters {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
        }
    }
}

impl RandomForestModel {
    /// Train a Random Forest regressor from a `DenseMatrix` of features and
    /// a target vector.
    pub fn train(
        features_matrix: &DenseMatrix<f64>,
        targets: &[f64],
        params: &RfHyperparameters,
    ) -> Result<Self, TrainingError> {
        let target_vec = targets.to_vec();

        let mut rf_params = RandomForestRegressorParameters::default()
            .with_n_trees(params.n_trees)
            .with_min_samples_split(params.min_samples_split)
            .with_min_samples_leaf(params.min_samples_leaf);

        if let Some(depth) = params.max_depth {
            rf_params = rf_params.with_max_depth(depth);
        }

        if let Some(max_feat) = params.max_features {
            rf_params = rf_params.with_m(max_feat);
        }

        let model = RandomForestRegressor::fit(features_matrix, &target_vec, rf_params)
            .map_err(|e| TrainingError::FitError(format!("Random Forest training failed: {e}")))?;

        Ok(Self {
            model: Arc::new(model),
            n_trees: params.n_trees,
        })
    }

    /// Predict a single sample from a flat feature vector.
    pub fn predict(&self, features: &[f64]) -> Option<f64> {
        let matrix = single_to_dense_matrix(features).ok()?;
        let predictions = self.model.predict(&matrix).ok()?;
        predictions.first().copied()
    }

    /// Predict a batch of samples from a `DenseMatrix`.
    pub fn predict_batch(&self, features_matrix: &DenseMatrix<f64>) -> Vec<f64> {
        self.model.predict(features_matrix).unwrap_or_default()
    }

    /// Returns the number of trees in the forest.
    pub fn n_trees(&self) -> usize {
        self.n_trees
    }

    /// Feature importance (stub — smartcore v0.4 does not expose this).
    #[allow(clippy::unused_self)]
    pub fn feature_importance(&self) -> Option<Vec<f64>> {
        None
    }

    /// Per-tree predictions (stub — smartcore v0.4 does not expose this).
    #[allow(clippy::unused_self)]
    pub fn per_tree_predictions(&self) -> Option<Vec<f64>> {
        None
    }
}

/// Convert a slice of `PredictionFeatures` into a `DenseMatrix`.
pub(crate) fn features_to_dense_matrix(
    features: &[PredictionFeatures],
) -> Result<DenseMatrix<f64>, TrainingError> {
    if features.is_empty() {
        return Err(TrainingError::InsufficientData(0));
    }

    let rows: Vec<Vec<f64>> = features.iter().map(PredictionFeatures::to_vec).collect();

    DenseMatrix::from_2d_vec(&rows)
        .map_err(|e| TrainingError::ArrayError(format!("DenseMatrix conversion failed: {e}")))
}

/// Convert a single flat feature vector into a 1-row `DenseMatrix`.
fn single_to_dense_matrix(features: &[f64]) -> Result<DenseMatrix<f64>, TrainingError> {
    if features.is_empty() {
        return Err(TrainingError::ArrayError(
            "Empty feature vector".to_string(),
        ));
    }

    let row = vec![features.to_vec()];
    DenseMatrix::from_2d_vec(&row)
        .map_err(|e| TrainingError::ArrayError(format!("DenseMatrix conversion failed: {e}")))
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use proptest::prelude::*;
    use smartcore::linalg::basic::arrays::Array;

    use super::*;

    #[allow(clippy::cast_precision_loss)]
    fn create_test_features(n: usize) -> Vec<PredictionFeatures> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                let noise1 = (t * 0.1).sin() * 0.01;
                let noise2 = (t * 0.17).cos() * 0.01;

                PredictionFeatures {
                    hour_sin: (t * 0.3).sin() + noise1,
                    hour_cos: (t * 0.31).cos() + noise2,
                    weekday_sin: (t * 0.07).sin() + noise1,
                    weekday_cos: (t * 0.071).cos() + noise2,
                    historical_avg: 30.0 + (t % 40.0) + noise1 * 100.0,
                    historical_std: 5.0 + (t % 15.0),
                    recent_avg_1h: 35.0 + (t % 35.0),
                    recent_avg_3h: 40.0 + ((t * 1.3) % 30.0),
                    recent_trend: -10.0 + (t % 20.0),
                    day_avg_so_far: 30.0 + (t % 45.0),
                    prev_day_avg: 45.0 + ((t * 0.7) % 25.0),
                    is_weekend: if (i % 7) >= 5 { 1.0 } else { 0.0 },
                    is_holiday: if i % 30 == 0 { 1.0 } else { 0.0 },
                    week_of_year_sin: (t * 0.02).sin() + noise1,
                    week_of_year_cos: (t * 0.021).cos() + noise2,
                    hours_ahead: 1.0 + (t % 6.0),
                    raw_hour: t % 24.0,
                    raw_weekday: t % 7.0,
                    time_to_close: 5.0 + (t % 12.0),
                    occupancy_volatility: 2.0 + (t % 10.0),
                    recent_avg_6h: 42.0 + ((t * 0.9) % 28.0),
                    prev_week_same_slot: 38.0 + (t % 35.0),
                }
            })
            .collect()
    }

    #[test]
    fn test_rf_train_success() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let matrix = features_to_dense_matrix(&features)?;
        let params = RfHyperparameters {
            n_trees: 10, // fewer trees for test speed
            max_depth: Some(5),
            ..Default::default()
        };

        let model = RandomForestModel::train(&matrix, &targets, &params)?;
        let prediction = model.predict(&features[0].to_vec());
        assert!(prediction.is_some());

        Ok(())
    }

    #[test]
    fn test_rf_predict_single() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let matrix = features_to_dense_matrix(&features)?;
        let params = RfHyperparameters {
            n_trees: 10,
            max_depth: Some(5),
            ..Default::default()
        };

        let model = RandomForestModel::train(&matrix, &targets, &params)?;
        let prediction = model.predict(&features[50].to_vec());

        assert!(prediction.is_some());
        assert!(prediction.unwrap_or(f64::NAN).is_finite());

        Ok(())
    }

    #[test]
    fn test_rf_predict_batch() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let matrix = features_to_dense_matrix(&features)?;
        let params = RfHyperparameters {
            n_trees: 10,
            max_depth: Some(5),
            ..Default::default()
        };

        let model = RandomForestModel::train(&matrix, &targets, &params)?;

        let batch_features = &features[0..10];
        let batch_matrix = features_to_dense_matrix(batch_features)?;
        let predictions = model.predict_batch(&batch_matrix);

        assert_eq!(predictions.len(), 10);

        Ok(())
    }

    #[test]
    fn test_rf_feature_importance_returns_none() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let matrix = features_to_dense_matrix(&features)?;
        let params = RfHyperparameters {
            n_trees: 10,
            max_depth: Some(5),
            ..Default::default()
        };

        let model = RandomForestModel::train(&matrix, &targets, &params)?;
        assert!(model.feature_importance().is_none());

        Ok(())
    }

    #[test]
    fn test_rf_per_tree_predictions_returns_none() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let matrix = features_to_dense_matrix(&features)?;
        let params = RfHyperparameters {
            n_trees: 10,
            max_depth: Some(5),
            ..Default::default()
        };

        let model = RandomForestModel::train(&matrix, &targets, &params)?;
        assert!(model.per_tree_predictions().is_none());

        Ok(())
    }

    #[test]
    fn test_dense_matrix_conversion() -> Result<()> {
        let features = create_test_features(50);
        let matrix = features_to_dense_matrix(&features)?;

        let (rows, cols) = matrix.shape();
        assert_eq!(rows, 50);
        assert_eq!(cols, PredictionFeatures::NUM_FEATURES);

        Ok(())
    }

    #[test]
    fn test_rf_train_insufficient_data() {
        let features = create_test_features(0);
        let result = features_to_dense_matrix(&features);

        assert!(matches!(result, Err(TrainingError::InsufficientData(0))));
    }

    #[test]
    fn test_dense_matrix_empty_fails() {
        let result = features_to_dense_matrix(&[]);
        assert!(matches!(result, Err(TrainingError::InsufficientData(0))));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_rf_predictions_finite(idx in 0_usize..50) {
            let features = create_test_features(100);
            let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

            if let Ok(matrix) = features_to_dense_matrix(&features) {
                let params = RfHyperparameters {
                    n_trees: 5,
                    max_depth: Some(3),
                    ..Default::default()
                };

                if let Ok(model) = RandomForestModel::train(&matrix, &targets, &params) {
                    let prediction = model.predict(&features[idx].to_vec());
                    if let Some(p) = prediction {
                        prop_assert!(p.is_finite(), "RF prediction at index {idx} not finite: {p}");
                    }
                }
            }
        }

        #[test]
        fn prop_dense_matrix_dimensions(n in 10_usize..50) {
            let features = create_test_features(n);

            if let Ok(matrix) = features_to_dense_matrix(&features) {
                let (rows, cols) = matrix.shape();
                prop_assert_eq!(rows, n);
                prop_assert_eq!(cols, PredictionFeatures::NUM_FEATURES);
            }
        }
    }
}
