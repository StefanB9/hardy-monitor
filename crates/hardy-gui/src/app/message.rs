//! What the app reacts to, and the views it can show.

use chrono::{DateTime, Utc};
use hardy_core::{
    alert::{AlertDuration, AlertSettings},
    db::{
        ForecastHistoryEntry, HorizonAccuracy, HourlyAverage, LiveForecast, OccupancyLog,
        QuarterHourAverage, SchemaStatus,
    },
    error::AppError,
    repair::RepairSummary,
};
use hardy_ml::suite::Family;

use super::loads::RequestId;
use crate::{
    data_table::DataPeriod,
    forecasting::{ForecastChoice, ModelOption, SuiteStatus},
    leaderboard::SortKey,
    plan_settings::VisitLength,
    planner::{RangeEnd, RangeKind},
    time_range::{AnalyticsRange, ChartRange, Step},
    tray::TrayAction,
};

/// What the chart shows of one range.
#[derive(Debug, Clone, Default)]
pub struct ChartData {
    pub readings: Vec<OccupancyLog>,
    /// The hour-ahead forecast of past slots (week and month views only).
    pub past_forecasts: Vec<ForecastHistoryEntry>,
}

/// The top-level views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    #[default]
    Now,
    Week,
    /// Every stored reading of a day or week, as a table.
    Data,
    Insights,
    /// When to go on one of the next seven days.
    Plan,
    ModelData,
}

impl ViewMode {
    /// All views in sidebar order.
    pub const ALL: [ViewMode; 6] = [
        ViewMode::Now,
        ViewMode::Week,
        ViewMode::Data,
        ViewMode::Insights,
        ViewMode::Plan,
        ViewMode::ModelData,
    ];

    /// Page title shown in the header and sidebar.
    pub fn title(self) -> &'static str {
        match self {
            ViewMode::Now => "Now",
            ViewMode::Week => "Week",
            ViewMode::Data => "Data",
            ViewMode::Insights => "Insights",
            ViewMode::Plan => "Plan",
            ViewMode::ModelData => "Model & Data",
        }
    }

    /// Sidebar icon glyph.
    pub fn icon(self) -> &'static str {
        match self {
            ViewMode::Now => "◉",
            ViewMode::Week => "▦",
            ViewMode::Data => "≡",
            ViewMode::Insights => "✦",
            ViewMode::Plan => "◷",
            ViewMode::ModelData => "⚙",
        }
    }
}

/// Date presets for Data Repair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairPreset {
    Last7Days,
    Last30Days,
    AllData,
}

/// Everything the application reacts to.
#[derive(Debug, Clone)]
pub enum Message {
    Tick,
    FetchTick,
    FetchAlignmentComplete,
    RefreshNow,

    /// Newest stored reading: when it was taken and its value.
    FetchCompleted(Result<Option<(DateTime<Utc>, f64)>, AppError>),
    /// Chart data; applied only if `RequestId` is the newest chart load.
    HistoryLoaded(RequestId, Result<ChartData, AppError>),
    /// Hourly averages; applied only if `RequestId` is the newest one.
    AnalyticsLoaded(RequestId, Result<Vec<HourlyAverage>, AppError>),
    InsightsDataLoaded {
        current: Result<Vec<HourlyAverage>, AppError>,
        baseline: Result<Vec<HourlyAverage>, AppError>,
    },

    NotificationThresholdChanged(f64),
    NotificationThresholdReleased,
    NotificationToggled(bool),
    NotificationDurationSelected(AlertDuration),
    AlertSettingsLoaded(Result<AlertSettings, AppError>),
    NotificationSent,

    SwitchView(ViewMode),
    SwitchAnalyticsRange(AnalyticsRange),
    ChartRangeSelected(ChartRange),
    /// A leaderboard column header clicked: sort by it (again: reverse).
    LeaderboardSorted(SortKey),
    /// The leaderboard narrowed to a family (`None`: every model).
    LeaderboardFiltered(Option<Family>),
    CustomStartChanged(String),
    CustomEndChanged(String),
    ApplyCustomRange,
    /// Arrow button or key: show the previous or next period.
    ChartStep(Step),
    /// Oldest reading, the limit for stepping back.
    FirstReadingLoaded(Result<Option<DateTime<Utc>>, AppError>),

    /// The Data tab shows a day or a week.
    DataPeriodSelected(DataPeriod),
    /// The Data tab's table scrolled: offset from the top and visible
    /// height, in logical pixels.
    DataScrolled {
        offset: f32,
        height: f32,
    },
    /// Readings of the Data tab's period; applied only if `RequestId` is the
    /// newest load.
    DataLoaded(RequestId, Result<Vec<OccupancyLog>, AppError>),

    ExportCsv,
    ExportCompleted(Result<String, AppError>),
    ClearExportStatus,
    /// A tray click or menu entry.
    Tray(TrayAction),
    WindowCloseRequested,

    RepairStartDateChanged(String),
    RepairEndDateChanged(String),
    RepairPresetSelected(RepairPreset),
    StartRepairJob,
    RepairCompleted(Result<RepairSummary, AppError>),

    /// New rows for the forecasting history.
    ForecastHistoryLoaded(Result<Vec<OccupancyLog>, AppError>),
    /// The forecast models' states, benchmark and selection.
    SuiteStatusLoaded(Result<SuiteStatus, AppError>),
    /// Every forecast the daemon published.
    PublishedForecastLoaded(Result<Vec<LiveForecast>, AppError>),
    ForecastRunRequested,
    /// Which forecast the chart and the quiet hour use.
    ForecastChoiceSelected(ForecastChoice),
    /// A model added to or removed from the chart's comparison lines.
    OverlayAdded(ModelOption),
    OverlayRemoved(ModelOption),
    AccuracyLoaded(Result<Vec<HorizonAccuracy>, AppError>),

    /// Quarter-hour averages of the last 28 days, for the Plan tab.
    PlanAveragesLoaded(Result<Vec<QuarterHourAverage>, AppError>),
    /// A day chip clicked (0: today).
    PlanDaySelected(usize),
    PlanLengthSelected(VisitLength),
    /// One end of the workday or weekend time range moved to an hour.
    PlanRangeChanged(RangeKind, RangeEnd, u32),

    /// Result of comparing the database schema with this build.
    SchemaChecked(Result<SchemaStatus, AppError>),
    RetrySchemaCheck,
}

impl Message {
    /// Whether this delivers the result of a database load counted as
    /// pending. Every counted load produces exactly one such message.
    pub(crate) fn finishes_load(&self) -> bool {
        match self {
            Message::FetchCompleted(_)
            | Message::HistoryLoaded(..)
            | Message::AnalyticsLoaded(..)
            | Message::InsightsDataLoaded { .. }
            | Message::AlertSettingsLoaded(_)
            | Message::ExportCompleted(_)
            | Message::RepairCompleted(_)
            | Message::ForecastHistoryLoaded(_)
            | Message::SuiteStatusLoaded(_)
            | Message::PublishedForecastLoaded(_)
            | Message::AccuracyLoaded(_)
            | Message::FirstReadingLoaded(_)
            | Message::PlanAveragesLoaded(_)
            | Message::DataLoaded(..) => true,
            Message::Tick
            | Message::FetchTick
            | Message::FetchAlignmentComplete
            | Message::RefreshNow
            | Message::NotificationThresholdChanged(_)
            | Message::NotificationThresholdReleased
            | Message::NotificationToggled(_)
            | Message::NotificationDurationSelected(_)
            | Message::NotificationSent
            | Message::SwitchView(_)
            | Message::SwitchAnalyticsRange(_)
            | Message::ChartRangeSelected(_)
            | Message::LeaderboardSorted(_)
            | Message::LeaderboardFiltered(_)
            | Message::CustomStartChanged(_)
            | Message::CustomEndChanged(_)
            | Message::ApplyCustomRange
            | Message::ChartStep(_)
            | Message::DataPeriodSelected(_)
            | Message::DataScrolled { .. }
            | Message::ExportCsv
            | Message::ClearExportStatus
            | Message::Tray(_)
            | Message::WindowCloseRequested
            | Message::RepairStartDateChanged(_)
            | Message::RepairEndDateChanged(_)
            | Message::RepairPresetSelected(_)
            | Message::StartRepairJob
            | Message::ForecastRunRequested
            | Message::ForecastChoiceSelected(_)
            | Message::OverlayAdded(_)
            | Message::OverlayRemoved(_)
            | Message::PlanDaySelected(_)
            | Message::PlanLengthSelected(_)
            | Message::PlanRangeChanged(..)
            | Message::SchemaChecked(_)
            | Message::RetrySchemaCheck => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use hardy_core::error::AppError;

    use super::*;

    fn failed<T>() -> Result<T, AppError> {
        Err(AppError::validation("x"))
    }

    #[test]
    fn test_message_finishes_load_only_for_load_results() {
        let results = [
            Message::FetchCompleted(failed()),
            Message::HistoryLoaded(RequestId::default(), failed()),
            Message::AnalyticsLoaded(RequestId::default(), failed()),
            Message::InsightsDataLoaded {
                current: failed(),
                baseline: failed(),
            },
            Message::AlertSettingsLoaded(failed()),
            Message::ExportCompleted(failed()),
            Message::RepairCompleted(failed()),
            Message::ForecastHistoryLoaded(failed()),
            Message::SuiteStatusLoaded(failed()),
            Message::PublishedForecastLoaded(failed()),
            Message::AccuracyLoaded(failed()),
            Message::FirstReadingLoaded(failed()),
            Message::PlanAveragesLoaded(failed()),
            Message::DataLoaded(RequestId::default(), failed()),
        ];
        assert!(results.iter().all(Message::finishes_load));

        let others = [
            Message::Tick,
            Message::RefreshNow,
            Message::ApplyCustomRange,
            Message::ChartStep(Step::Back),
            Message::SchemaChecked(failed()),
            Message::NotificationSent,
            Message::LeaderboardSorted(SortKey::Name),
            Message::LeaderboardFiltered(None),
            Message::DataPeriodSelected(DataPeriod::Week),
            Message::DataScrolled {
                offset: 0.0,
                height: 0.0,
            },
            Message::PlanDaySelected(1),
            Message::PlanLengthSelected(VisitLength::OneHour),
            Message::PlanRangeChanged(RangeKind::Workday, RangeEnd::Earliest, 17),
        ];
        assert!(!others.iter().any(Message::finishes_load));
    }

    #[test]
    fn test_plan_sits_between_insights_and_model_data() {
        let titles: Vec<_> = ViewMode::ALL.iter().map(|m| m.title()).collect();
        assert_eq!(
            titles,
            vec!["Now", "Week", "Data", "Insights", "Plan", "Model & Data"]
        );
        assert_eq!(ViewMode::Plan.icon(), "◷");
    }
}
