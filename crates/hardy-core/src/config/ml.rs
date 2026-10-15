//! Forecasting settings.

use serde::Deserialize;

/// Configuration for forecasting. Models are fitted and benchmarked in the
/// daemon.
#[derive(Debug, Clone, Deserialize)]
pub struct MlConfig {
    pub enabled: bool,
    /// Days of history each model is fitted on.
    pub training_window_days: i64,
    /// Furthest forecast shown on the chart and logged for accuracy, in
    /// hours (models always forecast 24 hours).
    pub prediction_horizon_hours: i64,
    /// Most recent complete days the nightly benchmark backtests.
    #[serde(default = "default_benchmark_days")]
    pub benchmark_days: i64,
}

fn default_benchmark_days() -> i64 {
    28
}

impl Default for MlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            training_window_days: 56,
            prediction_horizon_hours: 6,
            benchmark_days: default_benchmark_days(),
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
        assert_eq!(config.benchmark_days, 28);
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
            algorithm = "RandomForest"
            holdout_days = 7
            model_path = "C:/old/model.bin"
            fallback_on_error = true
            tune_hyperparameters = false
        "#;

        let config: MlConfig = toml::from_str(toml_str)?;
        assert_eq!(config.benchmark_days, 28);

        Ok(())
    }
}
