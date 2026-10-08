//! Model artifacts as compact bytes (bincode + zstd) for database storage.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    error::MlError,
    features::FEATURE_VERSION,
    model::{Forest, Model, RfParams},
    profile::SlotProfile,
    training::{HorizonIntervals, ModelArtifact, TrainingMetrics},
};

/// Layout version of the stored bytes (independent of the feature version).
const FORMAT_VERSION: u32 = 1;
/// Safety limit when decompressing; capped models are far smaller.
pub const MAX_DECOMPRESSED_BYTES: usize = 64 * 1024 * 1024;
const ZSTD_LEVEL: i32 = 3;

#[derive(Serialize)]
enum ModelRef<'a> {
    Linear {
        coefficients: &'a [f64],
        intercept: f64,
    },
    RandomForest {
        forest: &'a Forest,
        params: RfParams,
    },
}

#[derive(Deserialize)]
enum ModelOwned {
    Linear {
        coefficients: Vec<f64>,
        intercept: f64,
    },
    RandomForest {
        forest: Forest,
        params: RfParams,
    },
}

#[derive(Serialize)]
struct StoredRef<'a> {
    format_version: u32,
    feature_version: u32,
    trained_at: DateTime<Utc>,
    data_until: DateTime<Utc>,
    max_hours_ahead: u32,
    tuned: bool,
    model: ModelRef<'a>,
    profile: &'a SlotProfile,
    intervals: &'a HorizonIntervals,
    metrics: &'a TrainingMetrics,
}

#[derive(Deserialize)]
struct StoredOwned {
    format_version: u32,
    feature_version: u32,
    trained_at: DateTime<Utc>,
    data_until: DateTime<Utc>,
    max_hours_ahead: u32,
    tuned: bool,
    model: ModelOwned,
    profile: SlotProfile,
    intervals: HorizonIntervals,
    metrics: TrainingMetrics,
}

impl ModelArtifact {
    /// Compressed bytes for storage.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MlError> {
        let model = match &self.model {
            Model::Linear {
                coefficients,
                intercept,
            } => ModelRef::Linear {
                coefficients,
                intercept: *intercept,
            },
            Model::RandomForest { forest, params } => ModelRef::RandomForest {
                forest,
                params: *params,
            },
        };
        let stored = StoredRef {
            format_version: FORMAT_VERSION,
            feature_version: self.feature_version,
            trained_at: self.trained_at,
            data_until: self.data_until,
            max_hours_ahead: self.max_hours_ahead,
            tuned: self.tuned,
            model,
            profile: &self.profile,
            intervals: &self.intervals,
            metrics: &self.metrics,
        };
        let raw = bincode::serde::encode_to_vec(&stored, bincode::config::standard())
            .map_err(|e| MlError::Serialization(e.to_string()))?;
        zstd::bulk::compress(&raw, ZSTD_LEVEL).map_err(|e| MlError::Serialization(e.to_string()))
    }

    /// Restores an artifact; rejects other layouts and feature versions.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MlError> {
        let raw = zstd::bulk::decompress(bytes, MAX_DECOMPRESSED_BYTES)
            .map_err(|e| MlError::Serialization(e.to_string()))?;
        let (stored, _): (StoredOwned, usize) =
            bincode::serde::decode_from_slice(&raw, bincode::config::standard())
                .map_err(|e| MlError::Serialization(e.to_string()))?;
        if stored.format_version != FORMAT_VERSION {
            return Err(MlError::Serialization(format!(
                "unsupported format version {}",
                stored.format_version
            )));
        }
        if stored.feature_version != FEATURE_VERSION {
            return Err(MlError::IncompatibleVersion {
                found: stored.feature_version,
                expected: FEATURE_VERSION,
            });
        }
        let model = match stored.model {
            ModelOwned::Linear {
                coefficients,
                intercept,
            } => Model::Linear {
                coefficients,
                intercept,
            },
            ModelOwned::RandomForest { forest, params } => Model::RandomForest {
                forest: std::sync::Arc::new(forest),
                params,
            },
        };
        Ok(Self {
            feature_version: stored.feature_version,
            trained_at: stored.trained_at,
            data_until: stored.data_until,
            max_hours_ahead: stored.max_hours_ahead,
            tuned: stored.tuned,
            model,
            profile: stored.profile,
            intervals: stored.intervals,
            metrics: stored.metrics,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{
        model::{Algorithm, MAX_DEPTH, MAX_TREES, MIN_LEAF},
        training::{
            Tuning,
            tests::{learnable_history, local, options},
            train_ungated,
        },
    };

    /// An artifact to store and restore; its quality does not matter here.
    fn trained(algorithm: Algorithm, params: RfParams, days: i64) -> Result<ModelArtifact> {
        let mut opts = options(Tuning::Fixed(params));
        opts.algorithm = algorithm;
        Ok(train_ungated(
            &learnable_history(days),
            &GymSchedule::default(),
            &opts,
            local(days, 0, 0),
        )?)
    }

    #[test]
    fn test_artifact_round_trips_with_identical_forecasts() -> Result<()> {
        for algorithm in [Algorithm::RandomForest, Algorithm::Linear] {
            let artifact = trained(algorithm, RfParams::new(20, 8, 10)?, 21)?;
            let restored = ModelArtifact::from_bytes(&artifact.to_bytes()?)?;
            assert_eq!(restored.metrics, artifact.metrics);
            assert_eq!(restored.profile, artifact.profile);
            let history = learnable_history(21);
            let schedule = GymSchedule::default();
            let now = local(20, 12, 0);
            assert_eq!(
                restored.forecast(&history, &schedule, now),
                artifact.forecast(&history, &schedule, now)
            );
        }
        Ok(())
    }

    #[test]
    fn test_artifact_rejects_other_feature_version_and_garbage() -> Result<()> {
        let mut artifact = trained(Algorithm::Linear, RfParams::DEFAULT, 21)?;
        artifact.feature_version = FEATURE_VERSION + 1;
        assert!(matches!(
            ModelArtifact::from_bytes(&artifact.to_bytes()?),
            Err(MlError::IncompatibleVersion { .. })
        ));
        assert!(matches!(
            ModelArtifact::from_bytes(b"not a model"),
            Err(MlError::Serialization(_))
        ));
        Ok(())
    }

    #[test]
    fn test_largest_allowed_forest_stays_small() -> Result<()> {
        // Eight weeks of 5-minute data with the largest permitted forest.
        let params = RfParams::new(MAX_TREES, MAX_DEPTH, MIN_LEAF)?;
        let artifact = trained(Algorithm::RandomForest, params, 56)?;
        let bytes = artifact.to_bytes()?;
        assert!(
            bytes.len() < 8 * 1024 * 1024,
            "compressed model is {} bytes",
            bytes.len()
        );
        Ok(())
    }
}
