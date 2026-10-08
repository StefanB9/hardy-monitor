//! Forecasting model settings.

use serde::Deserialize;

/// Which ML algorithm to use for occupancy prediction.
#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
pub enum MlAlgorithm {
    #[default]
    RandomForest,
    LinearRegression,
}

/// Configuration for forecasting. Training runs in the daemon.
#[derive(Debug, Clone, Deserialize)]
pub struct MlConfig {
    pub enabled: bool,
    /// Days of history used for training.
    pub training_window_days: i64,
    /// Furthest forecast, in hours.
    pub prediction_horizon_hours: i64,
    /// Minimum training samples (15-minute anchors × horizons).
    pub min_samples_for_training: usize,
    #[serde(default)]
    pub algorithm: MlAlgorithm,
    /// Most recent days held out to check a new model against the
    /// slot-average baseline before it is accepted.
    #[serde(default = "default_holdout_days")]
    pub holdout_days: i64,
}

fn default_holdout_days() -> i64 {
    7
}

impl Default for MlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            training_window_days: 56,
            prediction_horizon_hours: 6,
            min_samples_for_training: 500,
            algorithm: MlAlgorithm::default(),
            holdout_days: default_holdout_days(),
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

    #[test]
    fn test_ml_config_defaults() {
        let config = MlConfig::default();

        assert!(config.enabled);
        assert_eq!(config.training_window_days, 56);
        assert_eq!(config.prediction_horizon_hours, 6);
        assert_eq!(config.min_samples_for_training, 500);
        assert_eq!(config.algorithm, MlAlgorithm::RandomForest);
        assert_eq!(config.holdout_days, 7);
    }

    #[test]
    fn test_ml_algorithm_default() {
        let algo = MlAlgorithm::default();
        assert_eq!(algo, MlAlgorithm::RandomForest);
    }

    #[test]
    fn test_ml_algorithm_deserialize() -> Result<()> {
        #[derive(Deserialize)]
        struct Wrapper {
            algorithm: MlAlgorithm,
        }

        let rf: Wrapper = toml::from_str("algorithm = \"RandomForest\"")?;
        assert_eq!(rf.algorithm, MlAlgorithm::RandomForest);

        let lr: Wrapper = toml::from_str("algorithm = \"LinearRegression\"")?;
        assert_eq!(lr.algorithm, MlAlgorithm::LinearRegression);

        Ok(())
    }

    #[test]
    fn test_ml_config_ignores_removed_fields() -> Result<()> {
        // Keys from older config files must not break loading.
        let toml_str = r#"
            enabled = true
            training_window_days = 56
            retrain_interval_hours = 24
            prediction_horizon_hours = 6
            min_samples_for_training = 500
            model_path = "C:/old/model.bin"
            fallback_on_error = true
            tune_hyperparameters = false
        "#;

        let config: MlConfig = toml::from_str(toml_str)?;
        assert_eq!(config.algorithm, MlAlgorithm::RandomForest);
        assert_eq!(config.holdout_days, 7);

        Ok(())
    }
}
