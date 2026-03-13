use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2, Axis};

use super::TrainingError;
use crate::ml::{evaluation, features::PredictionFeatures};

/// Linear regression model backend using linfa.
#[derive(Debug, Clone)]
pub(crate) struct LinearRegressionModel {
    model: linfa_linear::FittedLinearRegression<f64>,
}

impl LinearRegressionModel {
    /// Train a new linear regression model from features and targets.
    ///
    /// When `ridge_lambda > 0`, augments the data matrix with a scaled identity
    /// matrix for L2 regularization, stabilizing near-singular Gram matrices.
    pub fn train(
        features: &[PredictionFeatures],
        targets: &[f64],
        fit_intercept: bool,
        ridge_lambda: f64,
    ) -> Result<Self, TrainingError> {
        if features.is_empty() || targets.is_empty() {
            return Err(TrainingError::InsufficientData(0));
        }

        if features.len() != targets.len() {
            return Err(TrainingError::MismatchedLengths {
                features: features.len(),
                targets: targets.len(),
            });
        }

        let n_samples = features.len();
        let n_features = PredictionFeatures::NUM_FEATURES;

        let flat_features: Vec<f64> = features
            .iter()
            .flat_map(PredictionFeatures::to_vec)
            .collect();
        let x = Array2::from_shape_vec((n_samples, n_features), flat_features)
            .map_err(|e| TrainingError::ArrayError(e.to_string()))?;
        let y = Array1::from_vec(targets.to_vec());

        let fit_dataset = if ridge_lambda > 0.0 {
            let scale = ridge_lambda.sqrt();
            let identity = Array2::<f64>::eye(n_features) * scale;
            let x_aug = ndarray::concatenate(Axis(0), &[x.view(), identity.view()])
                .map_err(|e| TrainingError::ArrayError(e.to_string()))?;
            let zeros = Array1::<f64>::zeros(n_features);
            let y_aug = ndarray::concatenate(Axis(0), &[y.view(), zeros.view()])
                .map_err(|e| TrainingError::ArrayError(e.to_string()))?;
            Dataset::new(x_aug, y_aug)
        } else {
            Dataset::new(x, y)
        };

        let model = LinearRegression::default()
            .with_intercept(fit_intercept)
            .fit(&fit_dataset)
            .map_err(|e: linfa_linear::LinearError<f64>| TrainingError::FitError(e.to_string()))?;

        Ok(Self { model })
    }

    /// Predict a single sample from a flat feature vector.
    pub fn predict(&self, features: &[f64]) -> Option<f64> {
        let array = Array2::from_shape_vec((1, features.len()), features.to_vec()).ok()?;
        let predictions = self.model.predict(&array);
        predictions.first().copied()
    }

    /// Predict a batch of samples from a 2D feature matrix.
    pub fn predict_batch(&self, feature_matrix: &Array2<f64>) -> Vec<f64> {
        self.model.predict(feature_matrix).to_vec()
    }

    /// Return the fitted coefficients.
    pub fn coefficients(&self) -> &Array1<f64> {
        self.model.params()
    }

    /// Return the fitted intercept.
    pub fn intercept(&self) -> f64 {
        self.model.intercept()
    }

    /// Compute training MSE given features and original targets.
    pub fn compute_training_mse(&self, x: &Array2<f64>, targets: &[f64]) -> f64 {
        let predictions = self.model.predict(x);
        evaluation::mse(&predictions.to_vec(), targets).unwrap_or(f64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

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
                }
            })
            .collect()
    }

    #[test]
    fn test_lr_train_success() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let model = LinearRegressionModel::train(&features, &targets, true, 0.0)?;

        assert_eq!(model.coefficients().len(), PredictionFeatures::NUM_FEATURES);
        Ok(())
    }

    #[test]
    fn test_lr_predict_single() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let model = LinearRegressionModel::train(&features, &targets, true, 0.0)?;
        let prediction = model.predict(&features[0].to_vec());

        assert!(prediction.is_some());
        assert!(prediction.unwrap_or(f64::NAN).is_finite());
        Ok(())
    }

    #[test]
    fn test_lr_predict_batch() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let model = LinearRegressionModel::train(&features, &targets, true, 0.0)?;

        let flat: Vec<f64> = features[0..5]
            .iter()
            .flat_map(PredictionFeatures::to_vec)
            .collect();
        let matrix = Array2::from_shape_vec((5, PredictionFeatures::NUM_FEATURES), flat)
            .map_err(|e| anyhow::anyhow!("Failed to create array from shape: {e}"))?;

        let predictions = model.predict_batch(&matrix);
        assert_eq!(predictions.len(), 5);
        Ok(())
    }

    #[test]
    fn test_lr_coefficients_length() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let model = LinearRegressionModel::train(&features, &targets, true, 0.0)?;

        assert_eq!(model.coefficients().len(), PredictionFeatures::NUM_FEATURES);
        Ok(())
    }

    #[test]
    fn test_lr_ridge_handles_singular() -> Result<()> {
        let features: Vec<PredictionFeatures> = (0..200)
            .map(|i| {
                let t = f64::from(i);
                PredictionFeatures {
                    hour_sin: (t * 0.3).sin(),
                    hour_cos: (t * 0.31).cos(),
                    weekday_sin: (t * 0.07).sin(),
                    weekday_cos: (t * 0.071).cos(),
                    historical_avg: 30.0 + (t % 40.0),
                    historical_std: 5.0 + (t % 15.0),
                    recent_avg_1h: 35.0 + (t % 35.0),
                    recent_avg_3h: 40.0 + ((t * 1.3) % 30.0),
                    recent_trend: -10.0 + (t % 20.0),
                    day_avg_so_far: 30.0 + (t % 45.0),
                    prev_day_avg: 45.0 + ((t * 0.7) % 25.0),
                    is_weekend: if (i % 7) >= 5 { 1.0 } else { 0.0 },
                    is_holiday: 0.0,
                    week_of_year_sin: (t * 0.02).sin(),
                    week_of_year_cos: (t * 0.021).cos(),
                    hours_ahead: 0.0, // constant column → singular
                }
            })
            .collect();
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        // Without ridge: should fail
        assert!(
            LinearRegressionModel::train(&features, &targets, true, 0.0).is_err(),
            "expected fit failure with constant hours_ahead and no regularisation"
        );

        // With ridge: should recover
        let model = LinearRegressionModel::train(&features, &targets, true, 1e-3)?;
        assert_eq!(model.coefficients().len(), PredictionFeatures::NUM_FEATURES);

        Ok(())
    }

    #[test]
    fn test_lr_compute_training_mse() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let model = LinearRegressionModel::train(&features, &targets, true, 0.0)?;

        let flat: Vec<f64> = features
            .iter()
            .flat_map(PredictionFeatures::to_vec)
            .collect();
        let x = Array2::from_shape_vec((100, PredictionFeatures::NUM_FEATURES), flat)
            .map_err(|e| anyhow::anyhow!("Failed to create array: {e}"))?;

        let mse = model.compute_training_mse(&x, &targets);
        assert!(mse >= 0.0);
        assert!(mse.is_finite());

        Ok(())
    }

    #[test]
    fn test_lr_intercept() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let model = LinearRegressionModel::train(&features, &targets, true, 0.0)?;
        let intercept = model.intercept();
        assert!(intercept.is_finite());

        // When fit_intercept is false, intercept should be 0
        let model_no_intercept = LinearRegressionModel::train(&features, &targets, false, 0.0)?;
        assert_relative_eq!(model_no_intercept.intercept(), 0.0, epsilon = 1e-10);

        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_lr_predictions_finite(idx in 0_usize..50) {
            let features = create_test_features(100);
            let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

            // Use ridge to ensure training always succeeds
            if let Ok(model) = LinearRegressionModel::train(&features, &targets, true, 1e-3) {
                let prediction = model.predict(&features[idx].to_vec());
                if let Some(p) = prediction {
                    prop_assert!(p.is_finite(), "Prediction at index {idx} is not finite: {p}");
                }
            }
        }
    }
}
