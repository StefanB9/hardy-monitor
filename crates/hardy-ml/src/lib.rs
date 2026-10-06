//! Occupancy forecasting: feature extraction, model training, evaluation and
//! persistence. Shared by the daemon (nightly training) and the GUI.

pub mod confidence;
pub mod config;
pub mod error;
pub mod evaluation;
pub mod features;
pub mod forecast;
pub mod history;
pub mod maintenance;
pub mod model;
pub mod persistence;
pub mod profile;
pub mod retrain;
pub mod samples;
pub mod training;

pub use confidence::{PredictionMethod, PredictionWithConfidence};
pub use config::{MlAlgorithm, MlConfig};
pub use error::MlError;
pub use forecast::baseline_forecast;
pub use history::{History, HistoryView};
pub use model::{Algorithm, RfParams};
pub use profile::SlotProfile;
pub use training::{ModelArtifact, TrainOptions, TrainingMetrics, Tuning, train};
