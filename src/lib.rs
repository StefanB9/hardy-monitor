pub mod analytics;
pub mod api;
pub mod config;
pub mod db;
pub mod error;
pub mod repair;
pub mod schedule;
pub mod traits;

#[cfg(feature = "gui")]
pub mod ml;
#[cfg(feature = "gui")]
pub mod style;
#[cfg(feature = "gui")]
pub mod widgets;

pub use analytics::{
    ComparisonMode, DayAnalysis, HourlyComparison, Insight, InsightCategory, OccupancyStats,
    PeriodComparison, TimePeriod, TrendDirection, analyze_days, build_hourly_comparisons,
    calculate_predictions, calculate_predictions_with_clock, calculate_stats, compare_periods,
    determine_trend, find_best_time_today, find_best_time_today_with_clock, find_peak_hours,
    find_quiet_hours, find_quiet_windows, generate_insights, midnight_utc, weekday_name,
    weekday_short,
};
pub use api::{GymApiClient, GymResponse};
pub use config::AppConfig;
pub use db::{Database, HourlyAverage, OccupancyLog};
pub use error::{AppError, DatabaseError, NetworkErrorKind};
#[cfg(feature = "gui")]
pub use ml::{
    MlConfig, OccupancyPredictor, PredictionMethod, PredictionWithConfidence, TrainingResult,
};
pub use repair::{DataRepairer, RepairProgress, RepairSummary};
pub use schedule::{GymSchedule, is_bavarian_holiday};
pub use traits::{Clock, MockClock, MockNotifier, Notifier, SystemClock};
#[cfg(feature = "gui")]
pub use traits::{CombinedNotifier, SystemNotifier};
