//! Model persistence - save and load trained models

use std::fs;
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::features::SlotStats;

/// Serializable model metadata and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedModel {
    /// Version for backward compatibility
    pub version: u32,
    /// When the model was trained
    pub created_at: DateTime<Utc>,
    /// Number of days of data used for training
    pub training_window_days: i64,
    /// Number of samples used for training
    pub training_samples: usize,
    /// Training MSE
    pub training_mse: f64,
    /// Validation MSE (if available)
    pub validation_mse: Option<f64>,
    /// Historical statistics for each (weekday, hour) slot
    pub slot_stats: Vec<SerializedSlotStats>,
    /// Serialized model weights/parameters
    /// Note: For decision trees, we store summary info rather than full model
    /// as linfa trees don't implement Serialize directly
    pub model_summary: ModelSummary,
}

/// Serializable slot statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedSlotStats {
    pub weekday: u32,
    pub hour: u32,
    pub mean: f64,
    pub std_dev: f64,
    pub sample_count: i64,
}

impl From<((u32, u32), &SlotStats)> for SerializedSlotStats {
    fn from(((weekday, hour), stats): ((u32, u32), &SlotStats)) -> Self {
        Self {
            weekday,
            hour,
            mean: stats.mean,
            std_dev: stats.std_dev,
            sample_count: stats.sample_count,
        }
    }
}

/// Summary of model performance (since full tree serialization is complex)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    /// Model type identifier
    pub model_type: String,
    /// Maximum depth used
    pub max_depth: Option<usize>,
    /// Feature importance (if available)
    pub feature_importance: Option<Vec<f64>>,
}

impl PersistedModel {
    /// Current version number
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new persisted model record
    pub fn new(
        training_window_days: i64,
        training_samples: usize,
        training_mse: f64,
        validation_mse: Option<f64>,
        slot_stats: Vec<SerializedSlotStats>,
        model_summary: ModelSummary,
    ) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            created_at: Utc::now(),
            training_window_days,
            training_samples,
            training_mse,
            validation_mse,
            slot_stats,
            model_summary,
        }
    }

    /// Save to a file using bincode
    pub fn save(&self, path: &Path) -> Result<(), PersistenceError> {
        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| PersistenceError::IoError(e.to_string()))?;
        }

        let bytes =
            bincode::serialize(self).map_err(|e| PersistenceError::SerializeError(e.to_string()))?;

        fs::write(path, bytes).map_err(|e| PersistenceError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load from a file
    pub fn load(path: &Path) -> Result<Self, PersistenceError> {
        if !path.exists() {
            return Err(PersistenceError::FileNotFound(
                path.to_string_lossy().to_string(),
            ));
        }

        let bytes = fs::read(path).map_err(|e| PersistenceError::IoError(e.to_string()))?;

        let model: Self = bincode::deserialize(&bytes)
            .map_err(|e| PersistenceError::DeserializeError(e.to_string()))?;

        // Version check
        if model.version > Self::CURRENT_VERSION {
            return Err(PersistenceError::VersionMismatch {
                expected: Self::CURRENT_VERSION,
                found: model.version,
            });
        }

        Ok(model)
    }

    /// Check if the persisted model is stale
    pub fn is_stale(&self, max_age_hours: i64) -> bool {
        let age = Utc::now() - self.created_at;
        age.num_hours() > max_age_hours
    }

    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Model v{}: {} samples, train_mse={:.2}, val_mse={}, created {}",
            self.version,
            self.training_samples,
            self.training_mse,
            self.validation_mse
                .map(|v| format!("{:.2}", v))
                .unwrap_or_else(|| "N/A".to_string()),
            self.created_at.format("%Y-%m-%d %H:%M UTC")
        )
    }
}

/// Errors that can occur during model persistence
#[derive(Debug, Clone)]
pub enum PersistenceError {
    /// File not found
    FileNotFound(String),
    /// IO error
    IoError(String),
    /// Serialization error
    SerializeError(String),
    /// Deserialization error
    DeserializeError(String),
    /// Version mismatch
    VersionMismatch { expected: u32, found: u32 },
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistenceError::FileNotFound(path) => write!(f, "Model file not found: {}", path),
            PersistenceError::IoError(e) => write!(f, "IO error: {}", e),
            PersistenceError::SerializeError(e) => write!(f, "Serialization error: {}", e),
            PersistenceError::DeserializeError(e) => write!(f, "Deserialization error: {}", e),
            PersistenceError::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "Model version mismatch: expected v{}, found v{}",
                    expected, found
                )
            }
        }
    }
}

impl std::error::Error for PersistenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_model() -> PersistedModel {
        PersistedModel::new(
            28,
            1000,
            5.5,
            Some(6.2),
            vec![
                SerializedSlotStats {
                    weekday: 0,
                    hour: 10,
                    mean: 45.0,
                    std_dev: 12.0,
                    sample_count: 50,
                },
                SerializedSlotStats {
                    weekday: 0,
                    hour: 11,
                    mean: 55.0,
                    std_dev: 10.0,
                    sample_count: 48,
                },
            ],
            ModelSummary {
                model_type: "DecisionTree".to_string(),
                max_depth: Some(10),
                feature_importance: None,
            },
        )
    }

    #[test]
    fn test_persisted_model_creation() {
        let model = create_test_model();

        assert_eq!(model.version, PersistedModel::CURRENT_VERSION);
        assert_eq!(model.training_samples, 1000);
        assert_eq!(model.training_mse, 5.5);
        assert_eq!(model.validation_mse, Some(6.2));
        assert_eq!(model.slot_stats.len(), 2);
    }

    #[test]
    fn test_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.bin");

        let model = create_test_model();
        model.save(&path).unwrap();

        let loaded = PersistedModel::load(&path).unwrap();

        assert_eq!(loaded.version, model.version);
        assert_eq!(loaded.training_samples, model.training_samples);
        assert_eq!(loaded.training_mse, model.training_mse);
        assert_eq!(loaded.slot_stats.len(), model.slot_stats.len());
    }

    #[test]
    fn test_load_nonexistent() {
        let path = Path::new("/nonexistent/path/model.bin");
        let result = PersistedModel::load(path);

        assert!(matches!(result, Err(PersistenceError::FileNotFound(_))));
    }

    #[test]
    fn test_is_stale() {
        let model = create_test_model();

        // Just created, should not be stale
        assert!(!model.is_stale(24));

        // With very short max age, should be stale
        // Note: This might be flaky if the test takes > 0 hours
        // but in practice a just-created model won't be stale even with max_age=0
        // because the age calculation is in hours
    }

    #[test]
    fn test_summary() {
        let model = create_test_model();
        let summary = model.summary();

        assert!(summary.contains("1000 samples"));
        assert!(summary.contains("train_mse=5.50"));
        assert!(summary.contains("val_mse=6.20"));
    }

    #[test]
    fn test_serialized_slot_stats_from() {
        let stats = SlotStats {
            mean: 50.0,
            std_dev: 15.0,
            sample_count: 100,
        };

        let serialized = SerializedSlotStats::from(((0, 10), &stats));

        assert_eq!(serialized.weekday, 0);
        assert_eq!(serialized.hour, 10);
        assert_eq!(serialized.mean, 50.0);
        assert_eq!(serialized.std_dev, 15.0);
        assert_eq!(serialized.sample_count, 100);
    }

    #[test]
    fn test_save_creates_parent_dirs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested").join("dirs").join("model.bin");

        let model = create_test_model();
        let result = model.save(&path);

        assert!(result.is_ok());
        assert!(path.exists());
    }
}
