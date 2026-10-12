//! Statistical models: exponential smoothing, Theta, autoregression,
//! Holt–Winters and a structural state-space model.
//!
//! The models without seasonality of their own run on the deviation from
//! the weekly profile; the seasonal ones on occupancy itself. Closed slots
//! are skipped: time moves on, but nothing is learnt from them. Each model
//! keeps its state at the end of the fit and filters forward from there to
//! a forecast's origin with fixed parameters.

mod autoregression;
mod holt_winters;
mod ols;
mod optimize;
mod series;
mod smoothing;
mod structural;

pub use autoregression::Autoregression;
pub use holt_winters::{HoltWintersDaily, HoltWintersWeekly};
pub use smoothing::{DampedHolt, HoltSmoothing, SimpleSmoothing, Theta};
pub use structural::StructuralModel;
