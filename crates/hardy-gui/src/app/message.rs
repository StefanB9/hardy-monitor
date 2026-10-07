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

    pub fn title(self) -> &'static str {
        match self {
            ViewMode::Now => "Now",
            ViewMode::Week => "Week",
            ViewMode::Insights => "Insights",
            ViewMode::ModelData => "Model & Data",
        }
    }

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
    HistoryLoaded(Result<Vec<OccupancyLog>, AppError>),
    AnalyticsLoaded(Result<Vec<HourlyAverage>, AppError>),
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
