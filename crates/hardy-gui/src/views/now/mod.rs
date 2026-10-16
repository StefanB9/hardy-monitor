//! "Now": live occupancy, the next quiet hour, alerts and the chart.

mod alerts;
mod chart;
mod quiet;
mod status;

use chrono::{DateTime, Utc};
use hardy_core::{alert::AlertDuration, db::OccupancyLog, schedule::GymSchedule};
use hardy_ml::PredictionWithConfidence;
use iced::{
    Element, Length,
    widget::{canvas::Cache, column, combo_box, row},
};

use crate::{
    app::Message,
    forecasting::{ForecastChoice, ForecastLine, ModelOption},
    quiet_window::QuietWindow,
    style,
    time_range::ChartRange,
};

/// Everything the Now view shows.
pub struct NowProps<'a> {
    pub now: DateTime<Utc>,
    pub schedule: &'a GymSchedule,
    pub occupancy: Option<f64>,
    pub reading_15_min_ago: Option<f64>,
    pub last_update: Option<DateTime<Utc>>,
    /// Set when the newest reading is too old to be live.
    pub stale_warning: Option<String>,
    pub low_threshold: f64,
    pub high_threshold: f64,
    pub quiet_window: Option<&'a QuietWindow>,
    pub forecast: &'a [PredictionWithConfidence],
    /// Name of the model behind the forecast, if one was published.
    pub forecast_source: Option<&'static str>,
    /// Set when the forecast is old (the forecaster seems offline).
    pub forecast_notice: Option<String>,
    /// Which forecast is shown, and the models it can be switched to.
    pub forecast_choice: ForecastChoice,
    pub forecast_picker: &'a combo_box::State<ForecastChoice>,
    /// Models that can be added for comparison.
    pub compare_picker: &'a combo_box::State<ModelOption>,
    /// Comparison forecasts drawn on the chart, in overlay order.
    pub overlays: &'a [ForecastLine],
    /// Alerts armed and not expired.
    pub alert_active: bool,
    pub alert_threshold: f64,
    /// Duration used when arming.
    pub alert_duration: AlertDuration,
    /// e.g. "On below 25% until 21:00 · set from phone".
    pub alert_status: String,
    pub history: &'a [OccupancyLog],
    pub chart_range: ChartRange,
    /// Periods back from the current one (preset ranges).
    pub chart_offset: u32,
    /// Oldest reading; no stepping back past it.
    pub first_reading: Option<DateTime<Utc>>,
    pub custom_start: &'a str,
    pub custom_end: &'a str,
    pub chart_cache: &'a Cache,
    pub gauge_cache: &'a Cache,
}

/// The Now page.
pub fn view<'a>(props: &NowProps<'a>) -> Element<'a, Message> {
    let top = row![
        status::card(props).width(Length::FillPortion(4)),
        quiet::card(props).width(Length::FillPortion(5)),
        alerts::card(props).width(Length::FillPortion(4)),
    ]
    .spacing(style::SPACE_L)
    .height(Length::Fixed(300.0));

    column![top, chart::card(props)]
        .spacing(style::SPACE_L)
        .height(Length::Fill)
        .into()
}
