//! Historical occupancy analytics: period comparison and trend, summary
//! statistics, busiest and quietest slots, insights, visit planning, and
//! gym-local time helpers.

mod comparison;
mod insights;
mod slots;
mod stats;
mod time;
mod trend;
pub mod visit;

pub use comparison::{
    ComparisonMode, HourlyComparison, PeriodComparison, build_hourly_comparisons, compare_periods,
};
pub use insights::{Insight, InsightCategory, generate_insights};
pub use slots::{MIN_DAYS, TimePeriod, find_peak_hours, find_quiet_hours, find_quiet_windows};
pub use stats::{DayAnalysis, OccupancyStats, analyze_days, calculate_stats};
pub use time::{midnight_local_as_utc, weekday_name, weekday_short};
pub use trend::{TrendDirection, determine_trend, mean_point_change, trend_from_points};
