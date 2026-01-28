//! Hardy Monitor Library
//!
//! This module exposes the core components of the Hardy Monitor application
//! for testing and potential reuse.

pub mod analytics;
pub mod api;
pub mod config;
pub mod db;
pub mod repair;
pub mod schedule;
pub mod traits;

// GUI-only modules
#[cfg(feature = "gui")]
pub mod style;
#[cfg(feature = "gui")]
pub mod widgets;

// Re-export commonly used types
pub use analytics::{
    // Comparison types
    ComparisonMode,
    DayAnalysis,
    HourlyComparison,
    // Insights
    Insight,
    InsightCategory,
    // Statistical analysis
    OccupancyStats,
    PeriodComparison,
    // Peak and quiet time analysis
    TimePeriod,
    TrendDirection,
    analyze_days,
    // Comparison functions
    build_hourly_comparisons,
    // Core prediction functions
    calculate_predictions,
    calculate_predictions_with_clock,
    calculate_stats,
    compare_periods,
    determine_trend,
    find_best_time_today,
    find_best_time_today_with_clock,
    find_peak_hours,
    find_quiet_hours,
    find_quiet_windows,
    generate_insights,
    midnight_utc,
    // Utility functions
    weekday_name,
    weekday_short,
};
pub use api::{GymApiClient, GymResponse};
pub use config::AppConfig;
pub use db::{Database, HourlyAverage, OccupancyLog};
pub use repair::{DataRepairer, RepairProgress, RepairSummary};
pub use schedule::{GymSchedule, is_bavarian_holiday};
pub use traits::{Clock, MockClock, MockNotifier, Notifier, SystemClock};
#[cfg(feature = "gui")]
pub use traits::{CombinedNotifier, SystemNotifier};
