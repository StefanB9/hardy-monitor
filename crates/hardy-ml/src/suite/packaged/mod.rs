//! Models from packaged libraries: anofox-forecast (ETS, ARIMA, SARIMA,
//! TBATS, MFLES, Theta) and stlrs (STL / MSTL decompositions).
//!
//! anofox models cannot take in new data without refitting, so each one
//! runs its full search and fit on the last [`series::LOOKBACK`] up to every
//! origin; `fit` only prepares the weekly profile. Seasonal models see the
//! whole slot grid (closed slots 0, missing ones the profile's value); the
//! non-seasonal ARIMA runs on the open-slot deviation from the profile.

mod decompose;
mod fourier;
mod models;
mod series;

pub use models::{
    AutoArima, AutoEts, AutoTbats, AutoTheta, FourierArima, Mfles, MstlArima, MstlEts, Sarima,
    StlArima, StlEts,
};
