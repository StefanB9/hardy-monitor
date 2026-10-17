//! "Plan your visit": the expected occupancy of a day per quarter hour and
//! the quietest windows of a given length in it.
//!
//! Free of forecasting types: the caller lays forecast points over the
//! recent-weeks estimates with [`with_forecast`].

mod estimates;
mod windows;

use chrono::{DateTime, TimeDelta, Utc};
pub use estimates::{recent_weeks_estimates, with_forecast};
pub use windows::{DayPart, VisitConstraints, VisitPlan, plan_day};

/// Length of one estimate.
pub const QUARTER: TimeDelta = TimeDelta::minutes(15);

/// Where an estimate comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimateSource {
    /// The live forecast.
    Forecast,
    /// The same weekday over recent weeks.
    RecentWeeks,
    /// Recent Sundays, for a holiday.
    RecentSundays,
}

/// Expected occupancy in percent, with a plausible range around it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Expected {
    pub mean: f64,
    pub low: f64,
    pub high: f64,
}

/// The estimate for the quarter hour starting at `start`; `expected` is
/// `None` where there is not enough data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotEstimate {
    pub start: DateTime<Utc>,
    pub expected: Option<Expected>,
    pub source: EstimateSource,
}

/// A suggested visit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisitWindow {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    /// Mean of the window's quarter hours (means, lows and highs).
    pub expected: Expected,
    /// [`EstimateSource::Forecast`] if any quarter hour is forecast.
    pub source: EstimateSource,
}
