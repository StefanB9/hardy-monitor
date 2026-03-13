pub(crate) mod linear;
pub(crate) mod random_forest;

use chrono::{DateTime, Utc};
use ndarray::Array2;
use random_forest::features_to_dense_matrix;

use super::{evaluation, features::PredictionFeatures};

/// Internal enum dispatching to the concrete model backend.
#[derive(Debug, Clone)]
pub(crate) enum ModelBackend {
    LinearRegression(linear::LinearRegressionModel),
    RandomForest(random_forest::RandomForestModel),
}

/// A trained occupancy prediction model.
///
/// Wraps an internal `ModelBackend` (LR or RF) and exposes a stable public API.
/// Callers interact only with `TrainedModel`; the backend is an implementation
/// detail.
#[derive(Debug, Clone)]
pub struct TrainedModel {
    backend: ModelBackend,
    pub training_mse: f64,
    pub validation_mse: Option<f64>,
    pub training_samples: usize,
    pub created_at: DateTime<Utc>,
}

impl TrainedModel {
    /// Create a `TrainedModel` wrapping the given backend.
    pub(crate) fn new(
        backend: ModelBackend,
        training_mse: f64,
        validation_mse: Option<f64>,
        training_samples: usize,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            backend,
            training_mse,
            validation_mse,
            training_samples,
            created_at,
        }
    }

    /// Predict occupancy for a single feature set.
    pub fn predict(&self, features: &PredictionFeatures) -> Option<f64> {
        let vec = features.to_vec();
        match &self.backend {
            ModelBackend::LinearRegression(lr) => lr.predict(&vec),
            ModelBackend::RandomForest(rf) => rf.predict(&vec),
        }
    }

    /// Predict occupancy for a batch of feature sets.
    pub fn predict_batch(&self, features: &[PredictionFeatures]) -> Vec<f64> {
        if features.is_empty() {
            return Vec::new();
        }

        match &self.backend {
            ModelBackend::LinearRegression(lr) => {
                let n_samples = features.len();
                let n_features = PredictionFeatures::NUM_FEATURES;

                let flat: Vec<f64> = features
                    .iter()
                    .flat_map(PredictionFeatures::to_vec)
                    .collect();

                match Array2::from_shape_vec((n_samples, n_features), flat) {
                    Ok(array) => lr.predict_batch(&array),
                    Err(_) => Vec::new(),
                }
            }
            ModelBackend::RandomForest(rf) => match features_to_dense_matrix(features) {
                Ok(matrix) => rf.predict_batch(&matrix),
                Err(_) => Vec::new(),
            },
        }
    }

    /// Human-readable model information.
    pub fn info(&self) -> String {
        format!(
            "TrainedModel(type={}, samples={}, train_mse={:.2}, val_mse={}, created={})",
            self.model_type(),
            self.training_samples,
            self.training_mse,
            self.validation_mse
                .map_or_else(|| "N/A".to_string(), |v| format!("{v:.2}")),
            self.created_at.format("%Y-%m-%d %H:%M")
        )
    }

    /// Returns the model type name.
    pub fn model_type(&self) -> &'static str {
        match &self.backend {
            ModelBackend::LinearRegression(_) => "LinearRegression",
            ModelBackend::RandomForest(_) => "RandomForest",
        }
    }

    /// Feature importance, if available. Currently only RF could provide this
    /// (returns `None` in v0.4 stub); LR always returns `None`.
    pub fn feature_importance(&self) -> Option<Vec<f64>> {
        match &self.backend {
            ModelBackend::LinearRegression(_) => None,
            ModelBackend::RandomForest(rf) => rf.feature_importance(),
        }
    }
}

/// Builder for constructing and training models.
pub struct ModelBuilder {
    fit_intercept: bool,
    ridge_lambda: f64,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    n_trees: usize,
    max_features: Option<usize>,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            ridge_lambda: 0.0,
            max_depth: 0,
            min_samples_split: 2,
            min_samples_leaf: 1,
            n_trees: 100,
            max_features: None,
        }
    }
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn fit_intercept(mut self, fit: bool) -> Self {
        self.fit_intercept = fit;
        self
    }

    #[must_use]
    pub fn ridge_lambda(mut self, lambda: f64) -> Self {
        self.ridge_lambda = lambda;
        self
    }

    #[must_use]
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    #[must_use]
    pub fn min_samples_split(mut self, samples: usize) -> Self {
        self.min_samples_split = samples;
        self
    }

    #[must_use]
    pub fn min_samples_leaf(mut self, samples: usize) -> Self {
        self.min_samples_leaf = samples;
        self
    }

    #[must_use]
    pub fn n_trees(mut self, n: usize) -> Self {
        self.n_trees = n;
        self
    }

    #[must_use]
    pub fn max_features(mut self, max: Option<usize>) -> Self {
        self.max_features = max;
        self
    }

    /// Train a Linear Regression model.
    pub fn train(
        &self,
        features: &[PredictionFeatures],
        targets: &[f64],
    ) -> Result<TrainedModel, TrainingError> {
        let lr = linear::LinearRegressionModel::train(
            features,
            targets,
            self.fit_intercept,
            self.ridge_lambda,
        )?;

        let n_samples = features.len();
        let n_features = PredictionFeatures::NUM_FEATURES;

        let flat: Vec<f64> = features
            .iter()
            .flat_map(PredictionFeatures::to_vec)
            .collect();
        let x = Array2::from_shape_vec((n_samples, n_features), flat)
            .map_err(|e| TrainingError::ArrayError(e.to_string()))?;

        let mse = lr.compute_training_mse(&x, targets);

        Ok(TrainedModel::new(
            ModelBackend::LinearRegression(lr),
            mse,
            None,
            n_samples,
            Utc::now(),
        ))
    }

    /// Train a Linear Regression model with a validation split.
    pub fn train_with_validation(
        &self,
        features: &[PredictionFeatures],
        targets: &[f64],
        validation_split: f64,
    ) -> Result<TrainedModel, TrainingError> {
        if features.len() < 10 {
            return Err(TrainingError::InsufficientData(features.len()));
        }

        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
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

    /// Train a Random Forest model.
    pub fn train_rf(
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

        let matrix = features_to_dense_matrix(features)?;

        let max_depth = if self.max_depth == 0 {
            None
        } else {
            Some(u16::try_from(self.max_depth).unwrap_or(u16::MAX))
        };

        let params = random_forest::RfHyperparameters {
            n_trees: self.n_trees,
            max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
        };

        let rf = random_forest::RandomForestModel::train(&matrix, targets, &params)?;

        // Compute training MSE
        let predictions = rf.predict_batch(&matrix);
        let mse = evaluation::mse(&predictions, targets).unwrap_or(f64::MAX);

        Ok(TrainedModel::new(
            ModelBackend::RandomForest(rf),
            mse,
            None,
            features.len(),
            Utc::now(),
        ))
    }

    /// Train a Random Forest model with a validation split.
    pub fn train_rf_with_validation(
        &self,
        features: &[PredictionFeatures],
        targets: &[f64],
        validation_split: f64,
    ) -> Result<TrainedModel, TrainingError> {
        if features.len() < 10 {
            return Err(TrainingError::InsufficientData(features.len()));
        }

        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let split_idx = ((1.0 - validation_split) * features.len() as f64) as usize;

        let train_features = &features[..split_idx];
        let train_targets = &targets[..split_idx];
        let val_features = &features[split_idx..];
        let val_targets = &targets[split_idx..];

        let mut model = self.train_rf(train_features, train_targets)?;

        let val_predictions = model.predict_batch(val_features);
        let val_mse = evaluation::mse(&val_predictions, val_targets).unwrap_or(f64::MAX);
        model.validation_mse = Some(val_mse);

        Ok(model)
    }
}

/// Errors that can occur during model training.
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

    // ── ModelBuilder defaults ────────────────────────────────────────

    #[test]
    fn test_model_builder_default() {
        let builder = ModelBuilder::default();
        assert!(builder.fit_intercept);
        assert_relative_eq!(builder.ridge_lambda, 0.0);
        assert_eq!(builder.n_trees, 100);
        assert_eq!(builder.max_depth, 0);
        assert_eq!(builder.min_samples_split, 2);
        assert_eq!(builder.min_samples_leaf, 1);
        assert!(builder.max_features.is_none());
    }

    #[test]
    fn test_model_builder_max_features() {
        let builder = ModelBuilder::new().max_features(Some(8));
        assert_eq!(builder.max_features, Some(8));

        let builder_none = ModelBuilder::new().max_features(None);
        assert!(builder_none.max_features.is_none());
    }

    #[test]
    fn test_train_rf_with_max_features() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new()
            .n_trees(10)
            .max_depth(5)
            .max_features(Some(4));

        let model = builder.train_rf(&features, &targets)?;
        assert_eq!(model.training_samples, 200);
        assert!(model.training_mse >= 0.0);
        assert_eq!(model.model_type(), "RandomForest");

        Ok(())
    }

    #[test]
    fn test_model_builder_customization() {
        let builder = ModelBuilder::new().fit_intercept(false).ridge_lambda(1e-3);

        assert!(!builder.fit_intercept);
        assert!((builder.ridge_lambda - 1e-3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_builder_n_trees() {
        let builder = ModelBuilder::new().n_trees(50);
        assert_eq!(builder.n_trees, 50);
    }

    // ── LR via ModelBuilder ──────────────────────────────────────────

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

    // ── TrainedModel predict dispatch ────────────────────────────────

    #[test]
    fn test_trained_model_lr_predict() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let model = builder.train(&features, &targets)?;

        let prediction = model.predict(&features[0]);
        assert!(prediction.is_some());

        Ok(())
    }

    #[test]
    fn test_trained_model_rf_predict() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new().n_trees(10).max_depth(5);
        let model = builder.train_rf(&features, &targets)?;

        let prediction = model.predict(&features[0]);
        assert!(prediction.is_some());

        Ok(())
    }

    // ── TrainedModel batch predict ───────────────────────────────────

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
    fn test_rf_model_predict_batch() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new().n_trees(10).max_depth(5);
        let model = builder.train_rf(&features, &targets)?;

        let predictions = model.predict_batch(&features[0..10]);
        assert_eq!(predictions.len(), 10);

        Ok(())
    }

    // ── TrainedModel info / model_type ───────────────────────────────

    #[test]
    fn test_trained_model_info_lr() -> Result<()> {
        let features = create_test_features(50);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new();
        let model = builder.train(&features, &targets)?;

        let info = model.info();
        assert!(info.contains("LinearRegression"));
        assert!(info.contains("samples=50"));
        assert!(info.contains("train_mse="));

        Ok(())
    }

    #[test]
    fn test_trained_model_info_rf() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new().n_trees(10).max_depth(5);
        let model = builder.train_rf(&features, &targets)?;

        let info = model.info();
        assert!(info.contains("RandomForest"));

        Ok(())
    }

    #[test]
    fn test_trained_model_model_type() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let lr_model = ModelBuilder::new().train(&features, &targets)?;
        assert_eq!(lr_model.model_type(), "LinearRegression");

        let rf_model = ModelBuilder::new()
            .n_trees(10)
            .max_depth(5)
            .train_rf(&features, &targets)?;
        assert_eq!(rf_model.model_type(), "RandomForest");

        Ok(())
    }

    // ── feature_importance ───────────────────────────────────────────

    #[test]
    fn test_trained_model_feature_importance_lr() -> Result<()> {
        let features = create_test_features(100);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let model = ModelBuilder::new().train(&features, &targets)?;
        assert!(model.feature_importance().is_none());

        Ok(())
    }

    #[test]
    fn test_trained_model_feature_importance_rf() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let model = ModelBuilder::new()
            .n_trees(10)
            .max_depth(5)
            .train_rf(&features, &targets)?;
        // Stub returns None in v0.4
        assert!(model.feature_importance().is_none());

        Ok(())
    }

    // ── RF via ModelBuilder ──────────────────────────────────────────

    #[test]
    fn test_train_rf_success() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new()
            .n_trees(10)
            .max_depth(5)
            .min_samples_split(5)
            .min_samples_leaf(2);

        let model = builder.train_rf(&features, &targets)?;
        assert_eq!(model.training_samples, 200);
        assert!(model.training_mse >= 0.0);
        assert_eq!(model.model_type(), "RandomForest");

        Ok(())
    }

    #[test]
    fn test_train_rf_with_validation() -> Result<()> {
        let features = create_test_features(200);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let builder = ModelBuilder::new().n_trees(10).max_depth(5);

        let model = builder.train_rf_with_validation(&features, &targets, 0.2)?;
        assert!(model.validation_mse.is_some());

        Ok(())
    }

    #[test]
    fn test_train_rf_insufficient_data() {
        let builder = ModelBuilder::new().n_trees(10);
        let result = builder.train_rf(&[], &[]);

        assert!(matches!(result, Err(TrainingError::InsufficientData(0))));
    }

    #[test]
    fn test_train_rf_mismatched_lengths() {
        let features = create_test_features(10);
        let targets = vec![50.0; 5];

        let builder = ModelBuilder::new().n_trees(10);
        let result = builder.train_rf(&features, &targets);

        assert!(matches!(
            result,
            Err(TrainingError::MismatchedLengths { .. })
        ));
    }
}
