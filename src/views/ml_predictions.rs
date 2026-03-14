use chrono::{DateTime, Duration as ChronoDuration, Local, Utc};
use hardy_monitor::{
    PredictionMethod, PredictionWithConfidence, ml::TrainingInfo, style,
    widgets::history_chart::HistoryChart,
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
    pub training_info: Option<&'a TrainingInfo>,
}

#[allow(clippy::too_many_lines)]
pub fn view(props: MLPredictionsProps<'_>) -> Element<'_, Message> {
    // ── Status card ─────────────────────────────────────────────────
    let (status_text, status_color) = if props.ml_training_in_progress {
        ("Training...".to_string(), style::ACCENT_ORANGE)
    } else if props.ml_has_model {
        let algo = props.training_info.map_or_else(
            || "Active".to_string(),
            |ti| format!("Active ({})", ti.algorithm),
        );
        (algo, style::ACCENT_GREEN)
    } else {
        ("Collecting data".to_string(), style::TEXT_MUTED)
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

    // ── Model details card (only if training info available) ────────
    let details_card = props.training_info.map(|ti| build_details_card(ti));

    // ── Chart card ──────────────────────────────────────────────────
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

    // ── Prediction details table ────────────────────────────────────
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
                PredictionMethod::RandomForest { .. } => ("RF", style::ACCENT_CYAN),
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

    // ── Assemble layout ─────────────────────────────────────────────
    let mut content = column![status_card, Space::new().height(20),].padding(10);

    if let Some(card) = details_card {
        content = content.push(card);
        content = content.push(Space::new().height(20));
    }

    content = content.push(chart_card);
    content = content.push(Space::new().height(20));
    content = content.push(table_card);

    scrollable(content)
        .height(Length::Fill)
        .width(Length::Fill)
        .into()
}

/// Build the "Model Details" card showing training metrics.
#[allow(clippy::too_many_lines)]
fn build_details_card(ti: &TrainingInfo) -> Element<'_, Message> {
    let mut col = column![
        text("Model Details").size(14).color(style::TEXT_MUTED),
        Space::new().height(15),
        // Algorithm + samples row
        row![
            text("Algorithm:").size(12).color(style::TEXT_MUTED),
            Space::new().width(6),
            text(&ti.algorithm).size(12).color(style::TEXT_BRIGHT),
            Space::new().width(Length::Fill),
            text("Samples:").size(12).color(style::TEXT_MUTED),
            Space::new().width(6),
            text(format!("{}", ti.training_samples))
                .size(12)
                .color(style::TEXT_BRIGHT),
        ]
        .align_y(Alignment::Center),
        Space::new().height(4),
        // Training window + MSE row
        row![
            text("Window:").size(12).color(style::TEXT_MUTED),
            Space::new().width(6),
            text(format!("{} days", ti.training_window_days))
                .size(12)
                .color(style::TEXT_BRIGHT),
            Space::new().width(Length::Fill),
            text("Train MSE:").size(12).color(style::TEXT_MUTED),
            Space::new().width(6),
            text(format!("{:.2}", ti.training_mse))
                .size(12)
                .color(style::TEXT_BRIGHT),
        ]
        .align_y(Alignment::Center),
    ];

    // Hyperparameters section (RF only)
    if let Some(ref hp) = ti.best_hyperparameters {
        let features_str = hp
            .max_features
            .map_or_else(|| "auto".to_string(), |f| format!("{f}"));

        col = col
            .push(Space::new().height(10))
            .push(
                text("Best Hyperparameters")
                    .size(12)
                    .color(style::TEXT_MUTED),
            )
            .push(Space::new().height(4))
            .push(
                row![
                    text("Trees:").size(11).color(style::TEXT_MUTED),
                    Space::new().width(4),
                    text(format!("{}", hp.n_trees))
                        .size(11)
                        .color(style::TEXT_BRIGHT),
                    Space::new().width(12),
                    text("Depth:").size(11).color(style::TEXT_MUTED),
                    Space::new().width(4),
                    text(format!("{}", hp.max_depth))
                        .size(11)
                        .color(style::TEXT_BRIGHT),
                    Space::new().width(12),
                    text("Min Leaf:").size(11).color(style::TEXT_MUTED),
                    Space::new().width(4),
                    text(format!("{}", hp.min_samples_leaf))
                        .size(11)
                        .color(style::TEXT_BRIGHT),
                    Space::new().width(12),
                    text("Features:").size(11).color(style::TEXT_MUTED),
                    Space::new().width(4),
                    text(features_str).size(11).color(style::TEXT_BRIGHT),
                ]
                .align_y(Alignment::Center),
            );
    }

    // CV scores section
    if let Some(ref cv) = ti.cv_scores {
        let r2_color = if cv.r_squared_mean >= 0.8 {
            style::ACCENT_GREEN
        } else if cv.r_squared_mean >= 0.5 {
            style::ACCENT_ORANGE
        } else {
            style::ACCENT_RED
        };

        col = col
            .push(Space::new().height(10))
            .push(
                text("Cross-Validation Scores")
                    .size(12)
                    .color(style::TEXT_MUTED),
            )
            .push(Space::new().height(4))
            .push(
                row![
                    text("RMSE:").size(11).color(style::TEXT_MUTED),
                    Space::new().width(4),
                    text(format!("{:.2} \u{00b1} {:.2}", cv.rmse_mean, cv.rmse_std))
                        .size(11)
                        .color(style::TEXT_BRIGHT),
                    Space::new().width(16),
                    text("R\u{00b2}:").size(11).color(style::TEXT_MUTED),
                    Space::new().width(4),
                    text(format!(
                        "{:.3} \u{00b1} {:.3}",
                        cv.r_squared_mean, cv.r_squared_std
                    ))
                    .size(11)
                    .color(r2_color),
                ]
                .align_y(Alignment::Center),
            )
            .push(Space::new().height(4))
            .push(
                row![
                    text("MAE:").size(11).color(style::TEXT_MUTED),
                    Space::new().width(4),
                    text(format!("{:.2} \u{00b1} {:.2}", cv.mae_mean, cv.mae_std))
                        .size(11)
                        .color(style::TEXT_BRIGHT),
                ]
                .align_y(Alignment::Center),
            );
    }

    card_container(col).width(Length::Fill).into()
}
