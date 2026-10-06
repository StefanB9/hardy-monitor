//! Errors from training, prediction and model persistence.

use thiserror::Error;

/// Why a model could not be trained, used or stored.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MlError {
    #[error("not enough data: {found} samples, need {required}")]
    InsufficientData { found: usize, required: usize },

    #[error("model fitting failed: {0}")]
    Fit(String),

    #[error(
        "model rejected: holdout MAE {model_mae:.2} is not better than the slot-average baseline \
         {baseline_mae:.2}"
    )]
    QualityGate { model_mae: f64, baseline_mae: f64 },

    #[error("invalid hyperparameters: {0}")]
    InvalidParams(String),

    #[error("model serialization failed: {0}")]
    Serialization(String),

    #[error("stored model has feature version {found}, expected {expected}")]
    IncompatibleVersion { found: u32, expected: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_error_display() {
        let cases = [
            (
                MlError::InsufficientData {
                    found: 3,
                    required: 500,
                },
                "not enough data: 3 samples, need 500",
            ),
            (
                MlError::Fit("singular".to_string()),
                "model fitting failed: singular",
            ),
            (
                MlError::QualityGate {
                    model_mae: 7.5,
                    baseline_mae: 6.25,
                },
                "model rejected: holdout MAE 7.50 is not better than the slot-average baseline \
                 6.25",
            ),
            (
                MlError::InvalidParams("too many trees".to_string()),
                "invalid hyperparameters: too many trees",
            ),
            (
                MlError::Serialization("eof".to_string()),
                "model serialization failed: eof",
            ),
            (
                MlError::IncompatibleVersion {
                    found: 1,
                    expected: 2,
                },
                "stored model has feature version 1, expected 2",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }
}
