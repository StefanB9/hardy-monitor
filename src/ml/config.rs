use std::path::PathBuf;

use serde::Deserialize;

/// Configuration for the ML prediction pipeline.
#[derive(Debug, Clone, Deserialize)]
pub struct MlConfig {
    pub enabled: bool,
    pub training_window_days: i64,
    pub retrain_interval_hours: i64,
    pub prediction_horizon_hours: i64,
    pub min_samples_for_training: usize,
    pub model_path: Option<PathBuf>,
    pub fallback_on_error: bool,
}

impl Default for MlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            training_window_days: 56,
            retrain_interval_hours: 24,
            prediction_horizon_hours: 6,
            min_samples_for_training: 500,
            model_path: None,
            fallback_on_error: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = MlConfig::default();

        assert!(config.enabled);
        assert_eq!(config.training_window_days, 56);
        assert_eq!(config.retrain_interval_hours, 24);
        assert_eq!(config.prediction_horizon_hours, 6);
        assert_eq!(config.min_samples_for_training, 500);
        assert!(config.fallback_on_error);
    }
}
