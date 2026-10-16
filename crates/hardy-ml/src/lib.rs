//! Occupancy forecasting: the suite of models, their benchmark and the
//! service that fits them nightly and publishes live forecasts. Used by the
//! daemon; the GUI shares the history and profile types.

pub mod baseline;
pub mod benchmark;
pub mod confidence;
pub mod config;
pub mod error;
pub mod grid;
pub mod history;
pub mod live;
pub mod nightly;
pub mod profile;
pub mod progress;
pub mod retrain;
pub mod service;
pub mod suite;
#[cfg(test)]
mod test_support;

pub use baseline::Baseline;
pub use confidence::{PredictionMethod, PredictionWithConfidence};
pub use config::MlConfig;
pub use error::MlError;
pub use history::{History, HistoryView};
pub use profile::SlotProfile;
