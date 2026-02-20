//! ML model wrapper for Linear Regression

use chrono::{DateTime, Utc};
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2, Axis};

use super::features::PredictionFeatures;

/// A trained ML model for occupancy prediction
#[derive(Debug, Clone)]
pub struct TrainedModel {
    /// The underlying linear regression model
    model: linfa_linear::FittedLinearRegression<f64>,
    /// Training mean squared error
    pub training_mse: f64,
    /// Validation mean squared error (if available)
    pub validation_mse: Option<f64>,
    /// Number of samples used for training
    pub training_samples: usize,
    /// Timestamp when model was created
    pub created_at: DateTime<Utc>,
}

impl TrainedModel {
    /// Create a new trained model
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

    /// Predict occupancy for a single feature vector
    pub fn predict(&self, features: &PredictionFeatures) -> Option<f64> {
        let feature_vec = features.to_vec();
        let array = Array2::from_shape_vec((1, feature_vec.len()), feature_vec).ok()?;

        let predictions = self.model.predict(&array);
        predictions.first().copied()
    }

    /// Predict occupancy for multiple feature vectors
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

    /// Get model information as a string
    pub fn info(&self) -> String {
        format!(
            "TrainedModel(samples={}, train_mse={:.2}, val_mse={}, created={})",
            self.training_samples,
            self.training_mse,
            self.validation_mse
                .map(|v| format!("{:.2}", v))
                .unwrap_or_else(|| "N/A".to_string()),
            self.created_at.format("%Y-%m-%d %H:%M")
        )
    }

    /// Get the model coefficients
    pub fn coefficients(&self) -> &Array1<f64> {
        self.model.params()
    }

    /// Get the model intercept
    pub fn intercept(&self) -> f64 {
        self.model.intercept()
    }
}

/// Builder for training a model
pub struct ModelBuilder {
    /// Whether to fit intercept
    fit_intercept: bool,
    /// L2 regularization strength (λ).  0.0 means no regularization.
    /// When > 0, training uses data augmentation equivalent to ridge regression:
    /// appending sqrt(λ)*I rows to X and zeros to y, so the Gram matrix becomes
    /// X^T X + λI — positive-definite and invertible even for zero-variance columns.
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
    /// Create a new model builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit: bool) -> Self {
        self.fit_intercept = fit;
        self
    }

    /// Set L2 regularization strength (λ ≥ 0).
    ///
    /// Ridge regression via data augmentation: appending `sqrt(λ)*I` rows to X
    /// and zeros to y makes the Gram matrix `X^T X + λI`, which is positive-definite
    /// and invertible even when X has zero-variance columns (e.g. `hours_ahead = 0`
    /// constant across all training samples).  A value of `1e-3` works well in practice.
    pub fn ridge_lambda(mut self, lambda: f64) -> Self {
        self.ridge_lambda = lambda;
        self
    }

    /// Provided for API compatibility - ignored for linear regression
    pub fn max_depth(self, _depth: usize) -> Self {
        self
    }

    /// Provided for API compatibility - ignored for linear regression
    pub fn min_samples_split(self, _samples: usize) -> Self {
        self
    }

    /// Provided for API compatibility - ignored for linear regression
    pub fn min_samples_leaf(self, _samples: usize) -> Self {
        self
    }

    /// Train a model on the provided data
    ///
    /// Time complexity : O(n_samples × n_features) to build X, O(n_features³) for OLS solve
    /// Allocations     : 1 Array2 (n_samples × n_features), 1 Array1 (n_samples)
    ///                   + O(n_features²) extra when ridge_lambda > 0
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

        // Optionally apply ridge regularisation via data augmentation.
        //
        // Appending sqrt(λ)·I rows to X and n_features zeros to y is algebraically
        // equivalent to minimising ‖Xβ − y‖² + λ‖β‖²  (Tikhonov / L2 ridge), i.e.
        // solving (X^T X + λI)β = X^T y.  Adding λI to the Gram matrix makes it
        // positive-definite, preventing "Matrix is non-invertible" when X has
        // zero-variance columns (e.g. hours_ahead = 0 for every training row).
        //
        // Allocation: O(n_features²) extra rows — negligible relative to n_samples.
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
            // No regularisation: clone x so the original remains available for MSE.
            // Training runs at most once per day; this allocation is not on a hot path.
            Dataset::new(x.clone(), y)
        };

        let model = LinearRegression::default()
            .with_intercept(self.fit_intercept)
            .fit(&fit_dataset)
            .map_err(|e: linfa_linear::LinearError<f64>| TrainingError::FitError(e.to_string()))?;

        // MSE is computed on the original n_samples only — augmented rows are
        // not real observations and must not inflate the reported error.
        let predictions = model.predict(&x);
        let mse = calculate_mse(&predictions.to_vec(), targets);

        Ok(TrainedModel::new(model, mse, None, n_samples, Utc::now()))
    }

    /// Train with validation split
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
        let val_mse = calculate_mse(&val_predictions, val_targets);
        model.validation_mse = Some(val_mse);

        Ok(model)
    }
}

/// Calculate mean squared error
fn calculate_mse(predictions: &[f64], targets: &[f64]) -> f64 {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return f64::MAX;
    }

    let sum_sq_error: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();

    sum_sq_error / predictions.len() as f64
}

/// Errors that can occur during model training
#[derive(Debug, Clone)]
pub enum TrainingError {
    /// Not enough data to train
    InsufficientData(usize),
    /// Feature and target arrays have different lengths
    MismatchedLengths { features: usize, targets: usize },
    /// Error creating array
    ArrayError(String),
    /// Error fitting model
    FitError(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingError::InsufficientData(n) => {
                write!(f, "Insufficient data for training: {} samples", n)
            }
            TrainingError::MismatchedLengths { features, targets } => {
                write!(
                    f,
                    "Feature and target lengths mismatch: {} vs {}",
                    features, targets
                )
            }
            TrainingError::ArrayError(e) => write!(f, "Array error: {}", e),
            TrainingError::FitError(e) => write!(f, "Model fitting error: {}", e),
        }
    }
}

impl std::error::Error for TrainingError {}

#[cfg(test)]
mod tests {
    use super::*;

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
                    hours_ahead: 1.0 + (i % 6) as f64,
                }
            })
            .collect()
    }

    #[test]
    fn test_model_builder_default() {
        let builder = ModelBuilder::default();
        assert!(builder.fit_intercept);
        assert_eq!(builder.ridge_lambda, 0.0);
    }

    #[test]
    fn test_model_builder_customization() {
        let builder = ModelBuilder::new().fit_intercept(false).ridge_lambda(1e-3);

        assert!(!builder.fit_intercept);
        assert!((builder.ridge_lambda - 1e-3).abs() < f64::EPSILON);
    }

    /// Regression guard: `hours_ahead = 0.0` for every row creates a zero-variance
    /// column that makes the Gram matrix singular.  Without ridge the fit errors;
    /// with ridge (λ = 1e-3) it must succeed.
    #[test]
    fn test_ridge_handles_singular_data() {
        let features: Vec<PredictionFeatures> = (0..200)
            .map(|i| {
                let t = i as f64;
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
                    is_holiday: 0.0,  // constant — second degenerate column
                    week_of_year_sin: (t * 0.02).sin(),
                    week_of_year_cos: (t * 0.021).cos(),
                    hours_ahead: 0.0, // constant — makes Gram matrix singular
                }
            })
            .collect();
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        // Without ridge: singular Gram matrix → fit error.
        let no_ridge = ModelBuilder::new();
        assert!(
            no_ridge.train(&features, &targets).is_err(),
            "expected fit failure with constant hours_ahead column and no regularisation"
        );

        // With ridge: positive-definite Gram matrix → fit succeeds.
        let with_ridge = ModelBuilder::new().ridge_lambda(1e-3);
        let result = with_ridge.train(&features, &targets);
        assert!(result.is_ok(), "ridge must recover from singular Gram matrix: {:?}", result.err());
        let model = result.unwrap();
        assert_eq!(model.training_samples, 200);
        assert!(model.training_mse >= 0.0);
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
    fn test_train_success() {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let result = builder.train(&features, &targets);

        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.training_samples, 100);
        assert!(model.training_mse >= 0.0);
    }

    #[test]
    fn test_train_with_validation() {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let result = builder.train_with_validation(&features, &targets, 0.2);

        assert!(result.is_ok());
        let model = result.unwrap();
        assert!(model.validation_mse.is_some());
    }

    #[test]
    fn test_model_predict_single() {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let model = builder.train(&features, &targets).unwrap();

        let test_feature = &features[0];
        let prediction = model.predict(test_feature);

        assert!(prediction.is_some());
    }

    #[test]
    fn test_model_predict_batch() {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let model = builder.train(&features, &targets).unwrap();

        let test_features = &features[0..5];
        let predictions = model.predict_batch(test_features);

        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_calculate_mse() {
        let predictions = vec![10.0, 20.0, 30.0];
        let targets = vec![12.0, 18.0, 32.0];

        let mse = calculate_mse(&predictions, &targets);

        assert!((mse - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_model_info() {
        let features = create_test_features(50);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let model = builder.train(&features, &targets).unwrap();

        let info = model.info();
        assert!(info.contains("samples=50"));
        assert!(info.contains("train_mse="));
    }

    #[test]
    fn test_model_coefficients() {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let model = builder.train(&features, &targets).unwrap();

        let coeffs = model.coefficients();
        assert_eq!(coeffs.len(), PredictionFeatures::NUM_FEATURES);
    }
}
