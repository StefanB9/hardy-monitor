use chrono::{DateTime, Duration as ChronoDuration, Local, Utc};
use hardy_monitor::{
    PredictionMethod, PredictionWithConfidence, style, widgets::history_chart::HistoryChart,
};
use iced::{
    Alignment, Element, Length,
    widget::{Canvas, Space, canvas, column, container, row, scrollable, text},
};

use crate::{app::Message, views::components::card_container};

#[derive(Clone, Copy)]
pub struct MLPredictionsProps<'a> {
    pub ml_predictions: &'a [PredictionWithConfidence],
    pub ml_predictions_simple: &'a [(DateTime<Utc>, f64)],
    pub ml_has_model: bool,
    pub ml_training_in_progress: bool,
    pub ml_last_trained: Option<DateTime<Utc>>,
    pub chart_cache: &'a canvas::Cache,
    pub now: DateTime<Utc>,
}

#[allow(clippy::too_many_lines)]
pub fn view(props: MLPredictionsProps<'_>) -> Element<'_, Message> {
    let (status_text, status_color) = if props.ml_training_in_progress {
        ("Training...", style::ACCENT_ORANGE)
    } else if props.ml_has_model {
        ("Active", style::ACCENT_GREEN)
    } else {
        ("Collecting data", style::TEXT_MUTED)
    };

    let trained_str = props.ml_last_trained.map_or_else(
        || "N/A".to_string(),
        |t| t.with_timezone(&Local).format("%Y-%m-%d %H:%M").to_string(),
    );

    let status_card = card_container(column![
        text("ML Prediction Model")
            .size(14)
            .color(style::TEXT_MUTED),
        Space::new().height(15),
        row![
            text("Status:").size(12).color(style::TEXT_MUTED),
            Space::new().width(8),
            text(status_text).size(12).color(status_color),
            Space::new().width(Length::Fill),
            text("Last trained:").size(12).color(style::TEXT_MUTED),
            Space::new().width(8),
            text(trained_str).size(12).color(style::TEXT_BRIGHT),
        ]
        .align_y(Alignment::Center),
    ])
    .width(Length::Fill);

    let (range_start, range_end) = if props.ml_predictions.is_empty() {
        let now = props.now;
        (now, now + ChronoDuration::hours(6))
    } else {
        let first_ts = props.ml_predictions[0].timestamp;
        let last_ts = props.ml_predictions[props.ml_predictions.len() - 1].timestamp;
        (
            first_ts - ChronoDuration::minutes(15),
            last_ts + ChronoDuration::minutes(30),
        )
    };

    let chart_inner: Element<'_, Message> = if props.ml_predictions.is_empty() {
        container(text("No predictions yet — waiting for data...").color(style::TEXT_MUTED))
            .height(Length::Fixed(200.0))
            .into()
    } else {
        Element::from(
            Canvas::new(HistoryChart {
                history: &[],
                predictions: props.ml_predictions_simple,
                confidence_band: props.ml_predictions,
                range_start,
                range_end,
                cache: props.chart_cache,
            })
            .width(Length::Fill)
            .height(Length::Fixed(220.0)),
        )
        .map(|_| Message::ChartInteraction)
    };

    let chart_card = card_container(column![
        text("Predicted Occupancy — Next Hours")
            .size(14)
            .color(style::TEXT_MUTED),
        Space::new().height(15),
        chart_inner,
    ])
    .width(Length::Fill);

    let col_time = Length::Fixed(60.0);
    let col_pred = Length::Fixed(80.0);
    let col_low = Length::Fixed(60.0);
    let col_high = Length::Fixed(60.0);
    let col_width = Length::Fixed(60.0);
    let col_conf = Length::Fixed(90.0);
    let col_method = Length::Fill;

    let header_row = row![
        text("Time")
            .size(12)
            .color(style::TEXT_MUTED)
            .width(col_time),
        text("Predicted")
            .size(12)
            .color(style::TEXT_MUTED)
            .width(col_pred),
        text("Low").size(12).color(style::TEXT_MUTED).width(col_low),
        text("High")
            .size(12)
            .color(style::TEXT_MUTED)
            .width(col_high),
        text("±Width")
            .size(12)
            .color(style::TEXT_MUTED)
            .width(col_width),
        text("Confidence")
            .size(12)
            .color(style::TEXT_MUTED)
            .width(col_conf),
        text("Method")
            .size(12)
            .color(style::TEXT_MUTED)
            .width(col_method),
    ]
    .spacing(0);

    let table_body: Element<'_, Message> = if props.ml_predictions.is_empty() {
        text("No predictions available — model needs more data.")
            .color(style::TEXT_MUTED)
            .size(13)
            .into()
    } else {
        let mut rows_col = column![].spacing(6);
        for p in props.ml_predictions {
            let conf_color = if p.confidence_score >= 0.7 {
                style::ACCENT_GREEN
            } else if p.confidence_score >= 0.4 {
                style::ACCENT_ORANGE
            } else {
                style::ACCENT_RED
            };
            let (method_label, method_color) = match p.method {
                PredictionMethod::MachineLearning { .. } => ("ML", style::ACCENT_CYAN),
                PredictionMethod::HistoricalAverage => ("Historical", style::TEXT_MUTED),
            };
            let data_row = row![
                text(
                    p.timestamp
                        .with_timezone(&Local)
                        .format("%H:%M")
                        .to_string()
                )
                .size(12)
                .color(style::TEXT_BRIGHT)
                .width(col_time),
                text(format!("{:.1}%", p.predicted_value))
                    .size(12)
                    .color(style::TEXT_BRIGHT)
                    .width(col_pred),
                text(format!("{:.1}%", p.confidence_low))
                    .size(12)
                    .color(style::TEXT_MUTED)
                    .width(col_low),
                text(format!("{:.1}%", p.confidence_high))
                    .size(12)
                    .color(style::TEXT_MUTED)
                    .width(col_high),
                text(format!("{:.1}%", p.interval_width()))
                    .size(12)
                    .color(style::TEXT_MUTED)
                    .width(col_width),
                text(format!("{:.0}%", p.confidence_score * 100.0))
                    .size(12)
                    .color(conf_color)
                    .width(col_conf),
                text(method_label)
                    .size(12)
                    .color(method_color)
                    .width(col_method),
            ]
            .spacing(0);
            rows_col = rows_col.push(data_row);
        }
        rows_col.into()
    };

    let table_card = card_container(column![
        text("Prediction Details").size(14).color(style::TEXT_MUTED),
        Space::new().height(15),
        header_row,
        Space::new().height(8),
        table_body,
    ])
    .width(Length::Fill);

    let content = column![
        status_card,
        Space::new().height(20),
        chart_card,
        Space::new().height(20),
        table_card,
    ]
    .padding(10);

    scrollable(content)
        .height(Length::Fill)
        .width(Length::Fill)
        .into()
}
