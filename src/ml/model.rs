use chrono::{DateTime, Utc};
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2, Axis};

use super::{evaluation, features::PredictionFeatures};

#[derive(Debug, Clone)]
pub struct TrainedModel {
    model: linfa_linear::FittedLinearRegression<f64>,
    pub training_mse: f64,
    pub validation_mse: Option<f64>,
    pub training_samples: usize,
    pub created_at: DateTime<Utc>,
}

impl TrainedModel {
    pub fn new(
        model: linfa_linear::FittedLinearRegression<f64>,
        training_mse: f64,
        validation_mse: Option<f64>,
        training_samples: usize,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            model,
            training_mse,
            validation_mse,
            training_samples,
            created_at,
        }
    }

    pub fn predict(&self, features: &PredictionFeatures) -> Option<f64> {
        let feature_vec = features.to_vec();
        let array = Array2::from_shape_vec((1, feature_vec.len()), feature_vec).ok()?;

        let predictions = self.model.predict(&array);
        predictions.first().copied()
    }

    pub fn predict_batch(&self, features: &[PredictionFeatures]) -> Vec<f64> {
        if features.is_empty() {
            return Vec::new();
        }

        let n_samples = features.len();
        let n_features = PredictionFeatures::NUM_FEATURES;

        let flat: Vec<f64> = features.iter().flat_map(|f| f.to_vec()).collect();

        match Array2::from_shape_vec((n_samples, n_features), flat) {
            Ok(array) => self.model.predict(&array).to_vec(),
            Err(_) => Vec::new(),
        }
    }

    pub fn info(&self) -> String {
        format!(
            "TrainedModel(samples={}, train_mse={:.2}, val_mse={}, created={})",
            self.training_samples,
            self.training_mse,
            self.validation_mse
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "N/A".to_string()),
            self.created_at.format("%Y-%m-%d %H:%M")
        )
    }

    pub fn coefficients(&self) -> &Array1<f64> {
        self.model.params()
    }

    pub fn intercept(&self) -> f64 {
        self.model.intercept()
    }
}

pub struct ModelBuilder {
    fit_intercept: bool,
    ridge_lambda: f64,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            ridge_lambda: 0.0,
        }
    }
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit_intercept(mut self, fit: bool) -> Self {
        self.fit_intercept = fit;
        self
    }

    pub fn ridge_lambda(mut self, lambda: f64) -> Self {
        self.ridge_lambda = lambda;
        self
    }

    pub fn max_depth(self, _depth: usize) -> Self {
        self
    }

    pub fn min_samples_split(self, _samples: usize) -> Self {
        self
    }

    pub fn min_samples_leaf(self, _samples: usize) -> Self {
        self
    }

    pub fn train(
        &self,
        features: &[PredictionFeatures],
        targets: &[f64],
    ) -> Result<TrainedModel, TrainingError> {
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

        let flat_features: Vec<f64> = features.iter().flat_map(|f| f.to_vec()).collect();
        let x = Array2::from_shape_vec((n_samples, n_features), flat_features)
            .map_err(|e| TrainingError::ArrayError(e.to_string()))?;
        let y = Array1::from_vec(targets.to_vec());

        let fit_dataset = if self.ridge_lambda > 0.0 {
            let scale = self.ridge_lambda.sqrt();
            let identity = Array2::<f64>::eye(n_features) * scale;
            let x_aug = ndarray::concatenate(Axis(0), &[x.view(), identity.view()])
                .map_err(|e| TrainingError::ArrayError(e.to_string()))?;
            let zeros = Array1::<f64>::zeros(n_features);
            let y_aug = ndarray::concatenate(Axis(0), &[y.view(), zeros.view()])
                .map_err(|e| TrainingError::ArrayError(e.to_string()))?;
            Dataset::new(x_aug, y_aug)
        } else {
            Dataset::new(x.clone(), y)
        };

        let model = LinearRegression::default()
            .with_intercept(self.fit_intercept)
            .fit(&fit_dataset)
            .map_err(|e: linfa_linear::LinearError<f64>| TrainingError::FitError(e.to_string()))?;

        let predictions = model.predict(&x);
        let mse = evaluation::mse(&predictions.to_vec(), targets).unwrap_or(f64::MAX);

        Ok(TrainedModel::new(model, mse, None, n_samples, Utc::now()))
    }

    pub fn train_with_validation(
        &self,
        features: &[PredictionFeatures],
        targets: &[f64],
        validation_split: f64,
    ) -> Result<TrainedModel, TrainingError> {
        if features.len() < 10 {
            return Err(TrainingError::InsufficientData(features.len()));
        }

        let split_idx = ((1.0 - validation_split) * features.len() as f64) as usize;

        let train_features = &features[..split_idx];
        let train_targets = &targets[..split_idx];
        let val_features = &features[split_idx..];
        let val_targets = &targets[split_idx..];

        let mut model = self.train(train_features, train_targets)?;

        let val_predictions = model.predict_batch(val_features);
        let val_mse = evaluation::mse(&val_predictions, val_targets).unwrap_or(f64::MAX);
        model.validation_mse = Some(val_mse);

        Ok(model)
    }
}

#[derive(Debug, Clone)]
pub enum TrainingError {
    InsufficientData(usize),
    MismatchedLengths { features: usize, targets: usize },
    ArrayError(String),
    FitError(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingError::InsufficientData(n) => {
                write!(f, "Insufficient data for training: {n} samples")
            }
            TrainingError::MismatchedLengths { features, targets } => {
                write!(
                    f,
                    "Feature and target lengths mismatch: {features} vs {targets}",
                )
            }
            TrainingError::ArrayError(e) => write!(f, "Array error: {e}"),
            TrainingError::FitError(e) => write!(f, "Model fitting error: {e}"),
        }
    }
}

impl std::error::Error for TrainingError {}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;

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
    fn test_model_builder_default() {
        let builder = ModelBuilder::default();
        assert!(builder.fit_intercept);
        assert_relative_eq!(builder.ridge_lambda, 0.0);
    }

    #[test]
    fn test_model_builder_customization() {
        let builder = ModelBuilder::new().fit_intercept(false).ridge_lambda(1e-3);

        assert!(!builder.fit_intercept);
        assert!((builder.ridge_lambda - 1e-3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ridge_handles_singular_data() -> Result<()> {
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
                    hours_ahead: 0.0,
                }
            })
            .collect();
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let no_ridge = ModelBuilder::new();
        assert!(
            no_ridge.train(&features, &targets).is_err(),
            "expected fit failure with constant hours_ahead column and no regularisation"
        );

        let with_ridge = ModelBuilder::new().ridge_lambda(1e-3);
        let result = with_ridge.train(&features, &targets);
        assert!(
            result.is_ok(),
            "ridge must recover from singular Gram matrix: {:?}",
            result.err()
        );
        let model = result?;
        assert_eq!(model.training_samples, 200);
        assert!(model.training_mse >= 0.0);

        Ok(())
    }

    #[test]
    fn test_train_empty_data() {
        let builder = ModelBuilder::new();
        let result = builder.train(&[], &[]);

        assert!(matches!(result, Err(TrainingError::InsufficientData(0))));
    }

    #[test]
    fn test_train_mismatched_lengths() {
        let features = create_test_features(10);
        let targets = vec![50.0; 5];

        let builder = ModelBuilder::new();
        let result = builder.train(&features, &targets);

        assert!(matches!(
            result,
            Err(TrainingError::MismatchedLengths { .. })
        ));
    }

    #[test]
    fn test_train_success() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let result = builder.train(&features, &targets);

        assert!(result.is_ok());
        let model = result?;
        assert_eq!(model.training_samples, 100);
        assert!(model.training_mse >= 0.0);

        Ok(())
    }

    #[test]
    fn test_train_with_validation() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let result = builder.train_with_validation(&features, &targets, 0.2);

        assert!(result.is_ok());
        let model = result?;
        assert!(model.validation_mse.is_some());

        Ok(())
    }

    #[test]
    fn test_model_predict_single() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let model = builder.train(&features, &targets)?;

        let test_feature = &features[0];
        let prediction = model.predict(test_feature);

        assert!(prediction.is_some());

        Ok(())
    }

    #[test]
    fn test_model_predict_batch() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let model = builder.train(&features, &targets)?;

        let test_features = &features[0..5];
        let predictions = model.predict_batch(test_features);

        assert_eq!(predictions.len(), 5);

        Ok(())
    }

    #[test]
    fn test_model_info() -> Result<()> {
        let features = create_test_features(50);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let model = builder.train(&features, &targets)?;

        let info = model.info();
        assert!(info.contains("samples=50"));
        assert!(info.contains("train_mse="));

        Ok(())
    }

    #[test]
    fn test_model_coefficients() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let model = builder.train(&features, &targets)?;

        let coeffs = model.coefficients();
        assert_eq!(coeffs.len(), PredictionFeatures::NUM_FEATURES);

        Ok(())
    }
}
