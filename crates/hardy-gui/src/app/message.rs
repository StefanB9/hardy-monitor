//! What the app reacts to, and the views it can show.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use hardy_core::{
    alert::{AlertDuration, AlertSettings},
    db::{HorizonAccuracy, HourlyAverage, MlState, ModelInfo, OccupancyLog, SchemaStatus},
    error::AppError,
    repair::RepairSummary,
};
use hardy_ml::ModelArtifact;

use super::loads::RequestId;
use crate::time_range::{AnalyticsRange, ChartRange};

/// The four top-level views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    #[default]
    Now,
    Week,
    Insights,
    ModelData,
}

impl ViewMode {
    /// All views in sidebar order.
    pub const ALL: [ViewMode; 4] = [
        ViewMode::Now,
        ViewMode::Week,
        ViewMode::Insights,
        ViewMode::ModelData,
    ];

    /// Page title shown in the header and sidebar.
    pub fn title(self) -> &'static str {
        match self {
            ViewMode::Now => "Now",
            ViewMode::Week => "Week",
            ViewMode::Insights => "Insights",
            ViewMode::ModelData => "Model & Data",
        }
    }

    /// Sidebar icon glyph.
    pub fn icon(self) -> &'static str {
        match self {
            ViewMode::Now => "◉",
            ViewMode::Week => "▦",
            ViewMode::Insights => "✦",
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
    /// Chart readings; applied only if `RequestId` is the newest chart load.
    HistoryLoaded(RequestId, Result<Vec<OccupancyLog>, AppError>),
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
    CustomStartChanged(String),
    CustomEndChanged(String),
    ApplyCustomRange,

    ExportCsv,
    ExportCompleted(Result<String, AppError>),
    ClearExportStatus,
    TrayCheck,
    WindowCloseRequested,

    RepairStartDateChanged(String),
    RepairEndDateChanged(String),
    RepairPresetSelected(RepairPreset),
    StartRepairJob,
    RepairCompleted(Result<RepairSummary, AppError>),

    /// New rows for the forecasting history.
    ForecastHistoryLoaded(Result<Vec<OccupancyLog>, AppError>),
    /// Newest stored model and training state.
    ModelStatusLoaded(Result<(Option<ModelInfo>, MlState), AppError>),
    ModelLoaded(Result<(ModelInfo, Arc<ModelArtifact>), AppError>),
    TrainModelRequested,
    AccuracyLoaded(Result<Vec<HorizonAccuracy>, AppError>),

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
            | Message::ModelStatusLoaded(_)
            | Message::ModelLoaded(_)
            | Message::AccuracyLoaded(_) => true,
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
            | Message::CustomStartChanged(_)
            | Message::CustomEndChanged(_)
            | Message::ApplyCustomRange
            | Message::ExportCsv
            | Message::ClearExportStatus
            | Message::TrayCheck
            | Message::WindowCloseRequested
            | Message::RepairStartDateChanged(_)
            | Message::RepairEndDateChanged(_)
            | Message::RepairPresetSelected(_)
            | Message::StartRepairJob
            | Message::TrainModelRequested
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
            Message::ModelStatusLoaded(failed()),
            Message::ModelLoaded(failed()),
            Message::AccuracyLoaded(failed()),
        ];
        assert!(results.iter().all(Message::finishes_load));

        let others = [
            Message::Tick,
            Message::RefreshNow,
            Message::ApplyCustomRange,
            Message::SchemaChecked(failed()),
            Message::NotificationSent,
        ];
        assert!(!others.iter().any(Message::finishes_load));
    }
}
