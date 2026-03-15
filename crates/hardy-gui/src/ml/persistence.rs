use std::{collections::HashMap, fs, path::Path};

use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use super::{
    features::SlotStats,
    model::{self, SerializedModelWeights},
    residuals::{ResidualQuantiles, SlotQuantiles},
};

/// zstd magic bytes for detecting compressed files.
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// Maximum decompressed size (1 MB) — safety limit for
/// `zstd::bulk::decompress`.
const MAX_DECOMPRESSED_SIZE: usize = 1_048_576;

/// zstd compression level (3 = good ratio, fast).
const ZSTD_COMPRESSION_LEVEL: i32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedModel {
    pub version: u32,
    pub created_at: DateTime<Utc>,
    pub training_window_days: i64,
    pub training_samples: usize,
    pub training_mse: f64,
    pub validation_mse: Option<f64>,
    pub slot_stats: Vec<SerializedSlotStats>,
    pub model_summary: ModelSummary,
    /// Per-slot residual quantiles for calibrated confidence intervals (v3+).
    pub residual_quantiles: Option<Vec<SerializedSlotQuantiles>>,
    /// Global residual quantiles as (`q_low`, `q_high`, count) (v3+).
    pub global_quantiles: Option<(f64, f64, usize)>,
    /// Best hyperparameters from grid search (v4+).
    pub best_hyperparameters: Option<SerializedHyperparameters>,
    /// Cross-validation scores (v4+).
    pub cv_scores: Option<SerializedCvScores>,
    /// Serialized model weights for full model reconstruction (v5+).
    pub model_weights: Option<SerializedModelWeights>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedSlotStats {
    pub weekday: u32,
    pub hour: u32,
    pub mean: f64,
    pub std_dev: f64,
    pub sample_count: i64,
}

/// Serialized per-slot residual quantiles for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedSlotQuantiles {
    pub weekday: u32,
    pub hour: u32,
    pub q_low: f64,
    pub q_high: f64,
    pub count: usize,
}

/// Serialized hyperparameters for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedHyperparameters {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
}

/// Serialized cross-validation scores for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedCvScores {
    pub rmse_mean: f64,
    pub rmse_std: f64,
    pub mae_mean: f64,
    pub mae_std: f64,
    pub r_squared_mean: f64,
    pub r_squared_std: f64,
    pub mse_mean: f64,
    pub mse_std: f64,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    pub model_type: String,
    pub max_depth: Option<usize>,
    pub feature_importance: Option<Vec<f64>>,
}

impl PersistedModel {
    pub const CURRENT_VERSION: u32 = 5;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        training_window_days: i64,
        training_samples: usize,
        training_mse: f64,
        validation_mse: Option<f64>,
        slot_stats: Vec<SerializedSlotStats>,
        model_summary: ModelSummary,
        quantiles: Option<&ResidualQuantiles>,
        best_hyperparameters: Option<SerializedHyperparameters>,
        cv_scores: Option<SerializedCvScores>,
        model_weights: Option<SerializedModelWeights>,
    ) -> Self {
        let (residual_quantiles, global_quantiles) = quantiles.map_or((None, None), |q| {
            let slots: Vec<SerializedSlotQuantiles> = q
                .slot_quantiles_map()
                .iter()
                .map(|(&(weekday, hour), sq)| SerializedSlotQuantiles {
                    weekday,
                    hour,
                    q_low: sq.q_low,
                    q_high: sq.q_high,
                    count: sq.count,
                })
                .collect();
            let g = q.global_quantiles();
            (Some(slots), Some((g.q_low, g.q_high, g.count)))
        });

        Self {
            version: Self::CURRENT_VERSION,
            created_at: Utc::now(),
            training_window_days,
            training_samples,
            training_mse,
            validation_mse,
            slot_stats,
            model_summary,
            residual_quantiles,
            global_quantiles,
            best_hyperparameters,
            cv_scores,
            model_weights,
        }
    }

    /// Reconstruct `ResidualQuantiles` from the persisted data.
    pub fn to_residual_quantiles(&self) -> Option<ResidualQuantiles> {
        let serialized_slots = self.residual_quantiles.as_ref()?;
        let (g_low, g_high, g_count) = self.global_quantiles?;

        let slot_map: HashMap<(u32, u32), SlotQuantiles> = serialized_slots
            .iter()
            .map(|s| {
                (
                    (s.weekday, s.hour),
                    SlotQuantiles {
                        q_low: s.q_low,
                        q_high: s.q_high,
                        count: s.count,
                    },
                )
            })
            .collect();

        let global = SlotQuantiles {
            q_low: g_low,
            q_high: g_high,
            count: g_count,
        };

        Some(ResidualQuantiles::from_persisted(slot_map, global))
    }

    /// Save the model to disk with zstd compression.
    pub fn save(&self, path: &Path) -> Result<(), PersistenceError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| PersistenceError::IoError(e.to_string()))?;
        }

        let bytes = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| PersistenceError::SerializeError(e.to_string()))?;

        let compressed = zstd::bulk::compress(&bytes, ZSTD_COMPRESSION_LEVEL)
            .map_err(|e| PersistenceError::SerializeError(e.to_string()))?;

        fs::write(path, compressed).map_err(|e| PersistenceError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load a model from disk, supporting both zstd-compressed and raw bincode
    /// files.
    pub fn load(path: &Path) -> Result<Self, PersistenceError> {
        if !path.exists() {
            return Err(PersistenceError::FileNotFound(
                path.to_string_lossy().to_string(),
            ));
        }

        let raw = fs::read(path).map_err(|e| PersistenceError::IoError(e.to_string()))?;
        let bytes = decompress_or_raw(&raw)?;

        let (model, _): (Self, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .map_err(|e| PersistenceError::DeserializeError(e.to_string()))?;

        if model.version > Self::CURRENT_VERSION {
            return Err(PersistenceError::VersionMismatch {
                expected: Self::CURRENT_VERSION,
                found: model.version,
            });
        }

        Ok(model)
    }

    pub fn is_stale(&self, max_age_hours: i64) -> bool {
        let age = Utc::now() - self.created_at;
        age.num_hours() > max_age_hours
    }

    pub fn summary(&self) -> String {
        format!(
            "Model v{}: {} samples, train_mse={:.2}, val_mse={}, created {}",
            self.version,
            self.training_samples,
            self.training_mse,
            self.validation_mse
                .map_or_else(|| "N/A".to_string(), |v| format!("{v:.2}")),
            self.created_at.format("%Y-%m-%d %H:%M UTC")
        )
    }

    /// Reconstruct a functional `TrainedModel` from the persisted weights.
    ///
    /// Returns an error if no weights are stored or if deserialization fails.
    pub fn to_trained_model(&self) -> Result<model::TrainedModel, PersistenceError> {
        let weights = self.model_weights.as_ref().ok_or_else(|| {
            PersistenceError::ModelWeightsInvalid("No model weights in persisted data".to_string())
        })?;

        let backend = match weights {
            SerializedModelWeights::LinearRegression {
                coefficients,
                intercept,
            } => {
                let lr = model::linear::LinearRegressionModel::from_coefficients(
                    Array1::from_vec(coefficients.clone()),
                    *intercept,
                );
                model::ModelBackend::LinearRegression(lr)
            }
            SerializedModelWeights::RandomForest(bytes) => {
                let n_trees = self
                    .best_hyperparameters
                    .as_ref()
                    .map_or(100, |hp| hp.n_trees);
                let rf = model::random_forest::RandomForestModel::from_serialized(bytes, n_trees)
                    .map_err(|e| PersistenceError::ModelWeightsInvalid(e.to_string()))?;
                model::ModelBackend::RandomForest(rf)
            }
        };

        Ok(model::TrainedModel::new(
            backend,
            self.training_mse,
            self.validation_mse,
            self.training_samples,
            self.created_at,
        ))
    }
}

/// Decompress zstd data or return raw bytes for backward compatibility.
fn decompress_or_raw(data: &[u8]) -> Result<Vec<u8>, PersistenceError> {
    if data.len() >= 4 && data[..4] == ZSTD_MAGIC {
        zstd::bulk::decompress(data, MAX_DECOMPRESSED_SIZE)
            .map_err(|e| PersistenceError::DeserializeError(e.to_string()))
    } else {
        Ok(data.to_vec())
    }
}

#[derive(Debug, Clone)]
pub enum PersistenceError {
    FileNotFound(String),
    IoError(String),
    SerializeError(String),
    DeserializeError(String),
    VersionMismatch { expected: u32, found: u32 },
    ModelWeightsInvalid(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistenceError::FileNotFound(path) => write!(f, "Model file not found: {path}"),
            PersistenceError::IoError(e) => write!(f, "IO error: {e}"),
            PersistenceError::SerializeError(e) => write!(f, "Serialization error: {e}"),
            PersistenceError::DeserializeError(e) => write!(f, "Deserialization error: {e}"),
            PersistenceError::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "Model version mismatch: expected v{expected}, found v{found}",
                )
            }
            PersistenceError::ModelWeightsInvalid(e) => {
                write!(f, "Invalid model weights: {e}")
            }
        }
    }
}

impl std::error::Error for PersistenceError {}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use approx::assert_relative_eq;
    use tempfile::tempdir;

    use super::*;

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
            None,
            None,
            None,
            None,
        )
    }

    fn create_test_hyperparameters() -> SerializedHyperparameters {
        SerializedHyperparameters {
            n_trees: 150,
            max_depth: 12,
            min_samples_leaf: 3,
            max_features: Some(8),
        }
    }

    fn create_test_cv_scores() -> SerializedCvScores {
        SerializedCvScores {
            rmse_mean: 4.21,
            rmse_std: 0.35,
            mae_mean: 3.12,
            mae_std: 0.28,
            r_squared_mean: 0.87,
            r_squared_std: 0.03,
            mse_mean: 17.72,
            mse_std: 2.95,
        }
    }

    #[test]
    fn test_persisted_model_creation() {
        let model = create_test_model();

        assert_eq!(model.version, PersistedModel::CURRENT_VERSION);
        assert_eq!(model.training_samples, 1000);
        assert_relative_eq!(model.training_mse, 5.5);
        assert_eq!(model.validation_mse, Some(6.2));
        assert_eq!(model.slot_stats.len(), 2);
    }

    #[test]
    fn test_save_and_load() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("model.bin");

        let model = create_test_model();
        model.save(&path)?;

        let loaded = PersistedModel::load(&path)?;

        assert_eq!(loaded.version, model.version);
        assert_eq!(loaded.training_samples, model.training_samples);
        assert_relative_eq!(loaded.training_mse, model.training_mse);
        assert_eq!(loaded.slot_stats.len(), model.slot_stats.len());

        Ok(())
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

        assert!(!model.is_stale(24));
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
        assert_relative_eq!(serialized.mean, 50.0);
        assert_relative_eq!(serialized.std_dev, 15.0);
        assert_eq!(serialized.sample_count, 100);
    }

    #[test]
    fn test_save_creates_parent_dirs() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("nested").join("dirs").join("model.bin");

        let model = create_test_model();
        let result = model.save(&path);

        assert!(result.is_ok());
        assert!(path.exists());

        Ok(())
    }

    // ── residual quantile persistence tests ─────────────────────────

    fn create_test_quantiles() -> ResidualQuantiles {
        let residuals: Vec<(u32, u32, f64)> = (0..20)
            .map(|i| (0, 10, -5.0 + f64::from(i) * 0.5))
            .chain((0..15).map(|i| (3, 15, -8.0 + f64::from(i))))
            .collect();

        ResidualQuantiles::from_residuals(&residuals).unwrap_or_else(|| unreachable!())
    }

    fn create_test_model_with_quantiles() -> PersistedModel {
        let quantiles = create_test_quantiles();
        PersistedModel::new(
            28,
            1000,
            5.5,
            Some(6.2),
            vec![SerializedSlotStats {
                weekday: 0,
                hour: 10,
                mean: 45.0,
                std_dev: 12.0,
                sample_count: 50,
            }],
            ModelSummary {
                model_type: "RandomForest".to_string(),
                max_depth: Some(10),
                feature_importance: None,
            },
            Some(&quantiles),
            None,
            None,
            None,
        )
    }

    #[test]
    fn test_persisted_model_version_is_5() {
        let version = PersistedModel::CURRENT_VERSION;
        assert_eq!(version, 5);
    }

    #[test]
    fn test_persisted_model_v3_roundtrip() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("model_v3.bin");

        let model = create_test_model_with_quantiles();
        assert!(model.residual_quantiles.is_some());
        assert!(model.global_quantiles.is_some());

        model.save(&path)?;
        let loaded = PersistedModel::load(&path)?;

        assert_eq!(loaded.version, PersistedModel::CURRENT_VERSION);
        assert!(loaded.residual_quantiles.is_some());
        assert!(loaded.global_quantiles.is_some());

        let slots = loaded.residual_quantiles.unwrap_or_else(|| unreachable!());
        assert_eq!(slots.len(), 2); // two distinct (weekday, hour) slots

        let (g_low, g_high, g_count) = loaded.global_quantiles.unwrap_or_else(|| unreachable!());
        assert!(g_low < g_high);
        assert_eq!(g_count, 35); // 20 + 15

        Ok(())
    }

    #[test]
    fn test_persisted_model_v3_without_quantiles() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("model_v3_no_q.bin");

        let model = create_test_model(); // None quantiles
        assert!(model.residual_quantiles.is_none());
        assert!(model.global_quantiles.is_none());

        model.save(&path)?;
        let loaded = PersistedModel::load(&path)?;

        assert_eq!(loaded.version, PersistedModel::CURRENT_VERSION);
        assert!(loaded.residual_quantiles.is_none());
        assert!(loaded.global_quantiles.is_none());

        Ok(())
    }

    #[test]
    fn test_to_residual_quantiles_roundtrip() {
        let original = create_test_quantiles();
        let model = create_test_model_with_quantiles();

        let restored = model.to_residual_quantiles();
        assert!(restored.is_some());

        let restored = restored.unwrap_or_else(|| unreachable!());

        // Verify the restored quantiles produce the same confidence interval
        let (orig_low, orig_high, orig_score) =
            original.compute_confidence_interval(50.0, 0, 10, 1);
        let (rest_low, rest_high, rest_score) =
            restored.compute_confidence_interval(50.0, 0, 10, 1);

        assert_relative_eq!(orig_low, rest_low, epsilon = 1e-10);
        assert_relative_eq!(orig_high, rest_high, epsilon = 1e-10);
        assert_relative_eq!(orig_score, rest_score, epsilon = 1e-10);
    }

    #[test]
    fn test_to_residual_quantiles_none_when_missing() {
        let model = create_test_model();
        assert!(model.to_residual_quantiles().is_none());
    }

    #[test]
    fn test_serialized_slot_quantiles() {
        let sq = SerializedSlotQuantiles {
            weekday: 2,
            hour: 14,
            q_low: -5.3,
            q_high: 8.7,
            count: 25,
        };

        assert_eq!(sq.weekday, 2);
        assert_eq!(sq.hour, 14);
        assert_relative_eq!(sq.q_low, -5.3);
        assert_relative_eq!(sq.q_high, 8.7);
        assert_eq!(sq.count, 25);
    }

    // ── v4: hyperparameters, CV scores, zstd compression ────────────

    #[test]
    fn test_serialized_hyperparameters() {
        let hp = create_test_hyperparameters();

        assert_eq!(hp.n_trees, 150);
        assert_eq!(hp.max_depth, 12);
        assert_eq!(hp.min_samples_leaf, 3);
        assert_eq!(hp.max_features, Some(8));
    }

    #[test]
    fn test_serialized_cv_scores() {
        let cv = create_test_cv_scores();

        assert_relative_eq!(cv.rmse_mean, 4.21);
        assert_relative_eq!(cv.rmse_std, 0.35);
        assert_relative_eq!(cv.mae_mean, 3.12);
        assert_relative_eq!(cv.mae_std, 0.28);
        assert_relative_eq!(cv.r_squared_mean, 0.87);
        assert_relative_eq!(cv.r_squared_std, 0.03);
        assert_relative_eq!(cv.mse_mean, 17.72);
        assert_relative_eq!(cv.mse_std, 2.95);
    }

    #[test]
    fn test_v4_roundtrip_with_all_fields() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("model_v4_full.bin");

        let quantiles = create_test_quantiles();
        let model = PersistedModel::new(
            56,
            2500,
            12.3,
            Some(14.1),
            vec![SerializedSlotStats {
                weekday: 1,
                hour: 14,
                mean: 60.0,
                std_dev: 8.0,
                sample_count: 100,
            }],
            ModelSummary {
                model_type: "RandomForest".to_string(),
                max_depth: Some(12),
                feature_importance: None,
            },
            Some(&quantiles),
            Some(create_test_hyperparameters()),
            Some(create_test_cv_scores()),
            None,
        );

        assert!(model.best_hyperparameters.is_some());
        assert!(model.cv_scores.is_some());

        model.save(&path)?;
        let loaded = PersistedModel::load(&path)?;

        assert_eq!(loaded.version, PersistedModel::CURRENT_VERSION);
        assert_eq!(loaded.training_samples, 2500);
        assert_relative_eq!(loaded.training_mse, 12.3);

        let hp = loaded
            .best_hyperparameters
            .unwrap_or_else(|| unreachable!());
        assert_eq!(hp.n_trees, 150);
        assert_eq!(hp.max_depth, 12);
        assert_eq!(hp.min_samples_leaf, 3);
        assert_eq!(hp.max_features, Some(8));

        let cv = loaded.cv_scores.unwrap_or_else(|| unreachable!());
        assert_relative_eq!(cv.rmse_mean, 4.21);
        assert_relative_eq!(cv.r_squared_mean, 0.87);

        assert!(loaded.residual_quantiles.is_some());
        assert!(loaded.global_quantiles.is_some());

        Ok(())
    }

    #[test]
    fn test_v4_roundtrip_without_optional_fields() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("model_v4_minimal.bin");

        let model = create_test_model(); // None for quantiles, hp, cv
        assert!(model.best_hyperparameters.is_none());
        assert!(model.cv_scores.is_none());

        model.save(&path)?;
        let loaded = PersistedModel::load(&path)?;

        assert_eq!(loaded.version, PersistedModel::CURRENT_VERSION);
        assert!(loaded.best_hyperparameters.is_none());
        assert!(loaded.cv_scores.is_none());
        assert!(loaded.residual_quantiles.is_none());
        assert!(loaded.global_quantiles.is_none());

        Ok(())
    }

    #[test]
    fn test_zstd_compressed_smaller() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("model_compressed.bin");

        let model = create_test_model_with_quantiles();

        // Get uncompressed size
        let uncompressed = bincode::serde::encode_to_vec(&model, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        model.save(&path)?;
        let compressed = fs::read(&path)?;

        assert!(
            compressed.len() < uncompressed.len(),
            "compressed ({}) should be smaller than uncompressed ({})",
            compressed.len(),
            uncompressed.len()
        );

        Ok(())
    }

    #[test]
    fn test_load_uncompressed_v3_compat() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("model_raw.bin");

        // Simulate a v3 file: write raw bincode without zstd compression
        let model = create_test_model();
        let raw_bytes = bincode::serde::encode_to_vec(&model, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Verify it does NOT start with zstd magic
        assert!(
            raw_bytes.len() < 4 || raw_bytes[..4] != ZSTD_MAGIC,
            "raw bincode should not start with zstd magic bytes"
        );

        fs::write(&path, &raw_bytes)?;

        // load() should handle uncompressed data via decompress_or_raw
        let loaded = PersistedModel::load(&path)?;
        assert_eq!(loaded.version, model.version);
        assert_eq!(loaded.training_samples, 1000);

        Ok(())
    }

    // ── v5: model weights persistence ───────────────────────────────

    #[test]
    fn test_v5_roundtrip_with_lr_weights() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("model_v5_lr.bin");

        let weights = SerializedModelWeights::LinearRegression {
            coefficients: vec![1.0, 2.0, 3.0],
            intercept: 0.5,
        };

        let model = PersistedModel::new(
            28,
            1000,
            5.5,
            Some(6.2),
            vec![],
            ModelSummary {
                model_type: "LinearRegression".to_string(),
                max_depth: None,
                feature_importance: None,
            },
            None,
            None,
            None,
            Some(weights),
        );

        model.save(&path)?;
        let loaded = PersistedModel::load(&path)?;

        assert_eq!(loaded.version, PersistedModel::CURRENT_VERSION);
        assert!(loaded.model_weights.is_some());

        if let Some(SerializedModelWeights::LinearRegression {
            coefficients,
            intercept,
        }) = loaded.model_weights
        {
            assert_eq!(coefficients.len(), 3);
            assert_relative_eq!(intercept, 0.5);
        } else {
            anyhow::bail!("Expected LinearRegression weights");
        }

        Ok(())
    }

    #[test]
    fn test_v5_roundtrip_with_rf_weights() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("model_v5_rf.bin");

        let weights = SerializedModelWeights::RandomForest(vec![42, 43, 44, 45]);

        let model = PersistedModel::new(
            28,
            2000,
            3.1,
            Some(4.0),
            vec![],
            ModelSummary {
                model_type: "RandomForest".to_string(),
                max_depth: Some(10),
                feature_importance: None,
            },
            None,
            None,
            None,
            Some(weights),
        );

        model.save(&path)?;
        let loaded = PersistedModel::load(&path)?;

        assert_eq!(loaded.version, PersistedModel::CURRENT_VERSION);
        if let Some(SerializedModelWeights::RandomForest(bytes)) = loaded.model_weights {
            assert_eq!(bytes, vec![42, 43, 44, 45]);
        } else {
            anyhow::bail!("Expected RandomForest weights");
        }

        Ok(())
    }

    #[test]
    fn test_v5_without_weights() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("model_v5_no_weights.bin");

        let model = PersistedModel::new(
            28,
            500,
            8.0,
            None,
            vec![],
            ModelSummary {
                model_type: "LinearRegression".to_string(),
                max_depth: None,
                feature_importance: None,
            },
            None,
            None,
            None,
            None,
        );

        model.save(&path)?;
        let loaded = PersistedModel::load(&path)?;

        assert_eq!(loaded.version, PersistedModel::CURRENT_VERSION);
        assert!(loaded.model_weights.is_none());

        Ok(())
    }

    #[test]
    fn test_to_trained_model_lr() -> Result<()> {
        use crate::ml::features::PredictionFeatures;

        let n_features = PredictionFeatures::NUM_FEATURES;
        let coefficients = vec![1.0; n_features];
        let intercept = 5.0;

        let weights = SerializedModelWeights::LinearRegression {
            coefficients,
            intercept,
        };

        let model = PersistedModel::new(
            28,
            1000,
            5.5,
            Some(6.2),
            vec![],
            ModelSummary {
                model_type: "LinearRegression".to_string(),
                max_depth: None,
                feature_importance: None,
            },
            None,
            None,
            None,
            Some(weights),
        );

        let trained = model.to_trained_model()?;
        assert_eq!(trained.model_type(), "LinearRegression");
        assert_eq!(trained.training_samples, 1000);

        Ok(())
    }

    #[test]
    fn test_to_trained_model_no_weights_returns_error() {
        let model = PersistedModel::new(
            28,
            500,
            8.0,
            None,
            vec![],
            ModelSummary {
                model_type: "LinearRegression".to_string(),
                max_depth: None,
                feature_importance: None,
            },
            None,
            None,
            None,
            None,
        );

        let result = model.to_trained_model();
        assert!(
            matches!(result, Err(PersistenceError::ModelWeightsInvalid(_))),
            "Should fail when no weights present"
        );
    }
}
