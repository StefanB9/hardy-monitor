use std::path::PathBuf;

use serde::Deserialize;

/// Which ML algorithm to use for occupancy prediction.
#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
pub enum MlAlgorithm {
    #[default]
    RandomForest,
    LinearRegression,
}

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
    #[serde(default)]
    pub algorithm: MlAlgorithm,
    #[serde(default = "default_cv_folds")]
    pub cv_folds: usize,
    #[serde(default = "default_cv_gap_hours")]
    pub cv_gap_hours: i64,
    #[serde(default = "default_tune_hyperparameters")]
    pub tune_hyperparameters: bool,
}

fn default_cv_folds() -> usize {
    4
}

fn default_cv_gap_hours() -> i64 {
    24
}

fn default_tune_hyperparameters() -> bool {
    true
}

impl MlConfig {
    /// Resolve the model persistence path.
    ///
    /// Uses `model_path` if set, otherwise derives a default using
    /// `dirs::data_dir()` (e.g., `AppData/Roaming/hardy-monitor/model.bin`).
    /// Returns `None` only if `dirs::data_dir()` is unavailable.
    pub fn resolve_model_path(&self) -> Option<PathBuf> {
        if let Some(ref path) = self.model_path {
            return Some(path.clone());
        }
        dirs::data_dir().map(|d| d.join("hardy-monitor").join("model.bin"))
    }
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
            algorithm: MlAlgorithm::default(),
            cv_folds: default_cv_folds(),
            cv_gap_hours: default_cv_gap_hours(),
            tune_hyperparameters: default_tune_hyperparameters(),
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

    #[test]
    fn test_ml_algorithm_default() {
        let algo = MlAlgorithm::default();
        assert_eq!(algo, MlAlgorithm::RandomForest);
    }

    #[test]
    fn test_ml_config_new_fields_default() {
        let config = MlConfig::default();
        assert_eq!(config.algorithm, MlAlgorithm::RandomForest);
        assert_eq!(config.cv_folds, 4);
        assert_eq!(config.cv_gap_hours, 24);
        assert!(config.tune_hyperparameters);
    }

    #[test]
    fn test_ml_algorithm_deserialize() -> anyhow::Result<()> {
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
    fn test_resolve_model_path_explicit() {
        let config = MlConfig {
            model_path: Some(PathBuf::from("/custom/path/model.bin")),
            ..MlConfig::default()
        };

        let resolved = config.resolve_model_path();
        assert_eq!(resolved, Some(PathBuf::from("/custom/path/model.bin")));
    }

    #[test]
    fn test_resolve_model_path_default() {
        let config = MlConfig::default();
        assert!(config.model_path.is_none());

        let resolved = config.resolve_model_path();
        assert!(resolved.is_some());

        let path = resolved.unwrap_or_else(|| unreachable!());
        let path_str = path.to_string_lossy();
        assert!(
            path_str.contains("hardy-monitor"),
            "default path should contain 'hardy-monitor': {path_str}"
        );
        assert!(
            path_str.ends_with("model.bin"),
            "default path should end with 'model.bin': {path_str}"
        );
    }

    #[test]
    fn test_resolve_model_path_default_not_none() {
        // dirs::data_dir() should return Some on Windows/macOS/Linux
        let data_dir = dirs::data_dir();
        assert!(
            data_dir.is_some(),
            "dirs::data_dir() should be available on this platform"
        );
    }

    #[test]
    fn test_ml_config_deserialize_without_new_fields() -> anyhow::Result<()> {
        let toml_str = r"
            enabled = true
            training_window_days = 56
            retrain_interval_hours = 24
            prediction_horizon_hours = 6
            min_samples_for_training = 500
            fallback_on_error = true
        ";

        let config: MlConfig = toml::from_str(toml_str)?;
        assert_eq!(config.algorithm, MlAlgorithm::RandomForest);
        assert_eq!(config.cv_folds, 4);
        assert_eq!(config.cv_gap_hours, 24);
        assert!(config.tune_hyperparameters);

        Ok(())
    }
}
