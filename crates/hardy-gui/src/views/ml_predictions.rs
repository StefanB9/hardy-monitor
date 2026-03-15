use chrono::{DateTime, Duration as ChronoDuration, Local, Utc};
use hardy_core::db::OccupancyLog;
use iced::{
    Alignment, Element, Length,
    widget::{Canvas, Space, button, canvas, column, container, row, scrollable, text},
};

use crate::{
    app::Message,
    ml::{PredictionWithConfidence, TrainingInfo},
    style,
    views::components::card_container,
    widgets::history_chart::HistoryChart,
};

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
    pub history: &'a [OccupancyLog],
    pub show_model_details: bool,
    pub retrain_interval_hours: i64,
}

// ── Prediction Highlights extraction ──────────────────────────────────

struct PredictionHighlights {
    next_hour: Option<HighlightEntry>,
    peak: Option<HighlightEntry>,
    quietest: Option<HighlightEntry>,
    avg_confidence: f64,
    prediction_count: usize,
}

struct HighlightEntry {
    time: DateTime<Utc>,
    value: f64,
    confidence_low: f64,
    confidence_high: f64,
    confidence_score: f64,
}

impl HighlightEntry {
    fn from_prediction(p: &PredictionWithConfidence) -> Self {
        Self {
            time: p.timestamp,
            value: p.predicted_value,
            confidence_low: p.confidence_low,
            confidence_high: p.confidence_high,
            confidence_score: p.confidence_score,
        }
    }

    fn interval_width(&self) -> f64 {
        self.confidence_high - self.confidence_low
    }
}

fn extract_highlights(
    predictions: &[PredictionWithConfidence],
    now: DateTime<Utc>,
) -> Option<PredictionHighlights> {
    if predictions.is_empty() {
        return None;
    }

    let target = now + ChronoDuration::hours(1);
    let next_hour = predictions
        .iter()
        .min_by_key(|p| (p.timestamp - target).num_seconds().unsigned_abs())
        .map(HighlightEntry::from_prediction);

    let peak = predictions
        .iter()
        .max_by(|a, b| a.predicted_value.total_cmp(&b.predicted_value))
        .map(HighlightEntry::from_prediction);

    let quietest = predictions
        .iter()
        .min_by(|a, b| a.predicted_value.total_cmp(&b.predicted_value))
        .map(HighlightEntry::from_prediction);

    #[allow(clippy::cast_precision_loss)]
    let avg_confidence: f64 =
        predictions.iter().map(|p| p.confidence_score).sum::<f64>() / predictions.len() as f64;

    Some(PredictionHighlights {
        next_hour,
        peak,
        quietest,
        avg_confidence,
        prediction_count: predictions.len(),
    })
}

// ── View ──────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
pub fn view(props: MLPredictionsProps<'_>) -> Element<'_, Message> {
    // ── Status card (4-state) ─────────────────────────────────────────
    let status_card = build_status_card(&props);

    // ── Chart card ────────────────────────────────────────────────────
    let chart_card = build_chart_card(&props);

    // ── Prediction Highlights card ────────────────────────────────────
    let highlights_card = build_highlights_card(&props);

    // ── Assemble layout ───────────────────────────────────────────────
    let content = column![
        status_card,
        Space::new().height(20),
        chart_card,
        Space::new().height(20),
        highlights_card,
    ]
    .padding(10);

    scrollable(content)
        .height(Length::Fill)
        .width(Length::Fill)
        .into()
}

/// Small styled action button.
fn build_action_button(label: &str, message: Message) -> Element<'_, Message> {
    button(text(label).size(11).color(style::TEXT_BRIGHT))
        .on_press(message)
        .padding([4, 10])
        .style(|_theme, status| {
            let bg = match status {
                button::Status::Hovered => style::ACCENT_BLUE,
                _ => style::STROKE_DIM,
            };
            button::Style {
                background: Some(iced::Background::Color(bg)),
                border: iced::Border {
                    radius: 4.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        })
        .into()
}

/// Small cancel button with red-ish styling.
fn build_cancel_button() -> Element<'static, Message> {
    button(text("\u{2715}").size(11).color(style::TEXT_BRIGHT))
        .on_press(Message::CancelTrainingRequested)
        .padding([4, 8])
        .style(|_theme, status| {
            let bg = match status {
                button::Status::Hovered => style::ACCENT_RED,
                _ => style::STROKE_DIM,
            };
            button::Style {
                background: Some(iced::Background::Color(bg)),
                border: iced::Border {
                    radius: 4.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        })
        .into()
}

#[allow(clippy::too_many_lines)]
fn build_status_card<'a>(props: &MLPredictionsProps<'a>) -> Element<'a, Message> {
    let trained_str = props.ml_last_trained.map_or_else(
        || "N/A".to_string(),
        |t| t.with_timezone(&Local).format("%H:%M").to_string(),
    );

    let mut col = column![
        text("ML Prediction Model")
            .size(14)
            .color(style::TEXT_MUTED),
        Space::new().height(15),
    ];

    match (props.ml_has_model, props.ml_training_in_progress) {
        // State 1: no model, not training
        (false, false) => {
            col = col.push(
                row![
                    text("No model loaded").size(12).color(style::TEXT_MUTED),
                    Space::new().width(Length::Fill),
                    build_action_button("Load Model", Message::LoadModelRequested),
                    Space::new().width(8),
                    build_action_button("Train Model", Message::TrainModelRequested),
                ]
                .align_y(Alignment::Center),
            );
        }
        // State 2: no model, training
        (false, true) => {
            col = col.push(
                row![
                    text("Training initial model...")
                        .size(12)
                        .color(style::ACCENT_ORANGE),
                    Space::new().width(Length::Fill),
                    build_cancel_button(),
                ]
                .align_y(Alignment::Center),
            );
        }
        // State 3: has model, not training
        (true, false) => {
            let mut status_row = row![].align_y(Alignment::Center);

            let algo_text = props.training_info.map_or_else(
                || "Active".to_string(),
                |ti| format!("Active ({})", ti.algorithm),
            );
            status_row = status_row
                .push(text(algo_text).size(12).color(style::ACCENT_GREEN))
                .push(Space::new().width(12));

            // R² badge
            if let Some(cv) = props.training_info.and_then(|ti| ti.cv_scores.as_ref()) {
                let r2_color = r_squared_color(cv.r_squared_mean);
                status_row = status_row.push(
                    text(format!("R\u{00b2} {:.3}", cv.r_squared_mean))
                        .size(12)
                        .color(r2_color),
                );
            }

            status_row = status_row.push(Space::new().width(Length::Fill));

            // Trained age display
            status_row = status_row
                .push(text("Trained:").size(12).color(style::TEXT_MUTED))
                .push(Space::new().width(6))
                .push(text(trained_str.clone()).size(12).color(style::TEXT_BRIGHT))
                .push(Space::new().width(12))
                .push(build_action_button("Retrain", Message::TrainModelRequested));

            col = col.push(status_row);

            // Staleness hint
            if let Some(trained_at) = props.ml_last_trained {
                let age_hours = (props.now - trained_at).num_hours();
                if age_hours >= props.retrain_interval_hours {
                    col = col.push(Space::new().height(4));
                    col = col.push(
                        text(format!(
                            "Trained {age_hours}h ago \u{2014} consider retraining for improved \
                             accuracy"
                        ))
                        .size(11)
                        .color(style::ACCENT_ORANGE),
                    );
                }
            }
        }
        // State 4: has model, retraining
        (true, true) => {
            let mut status_row = row![].align_y(Alignment::Center);

            let algo_text = props.training_info.map_or_else(
                || "Active".to_string(),
                |ti| format!("Active ({})", ti.algorithm),
            );
            status_row = status_row
                .push(text(algo_text).size(12).color(style::ACCENT_GREEN))
                .push(Space::new().width(12));

            if let Some(cv) = props.training_info.and_then(|ti| ti.cv_scores.as_ref()) {
                let r2_color = r_squared_color(cv.r_squared_mean);
                status_row = status_row.push(
                    text(format!("R\u{00b2} {:.3}", cv.r_squared_mean))
                        .size(12)
                        .color(r2_color),
                );
            }

            status_row = status_row.push(Space::new().width(Length::Fill));

            status_row = status_row
                .push(text("Trained:").size(12).color(style::TEXT_MUTED))
                .push(Space::new().width(6))
                .push(text(trained_str).size(12).color(style::TEXT_BRIGHT));

            col = col.push(status_row);
            col = col.push(Space::new().height(6));
            col = col.push(
                row![
                    text("Retraining...").size(11).color(style::ACCENT_ORANGE),
                    Space::new().width(8),
                    text("Showing previous predictions")
                        .size(11)
                        .color(style::TEXT_MUTED),
                    Space::new().width(Length::Fill),
                    build_cancel_button(),
                ]
                .align_y(Alignment::Center),
            );
        }
    }

    // Model details toggle (only if training info available)
    if props.training_info.is_some() {
        let toggle_label = if props.show_model_details {
            "Hide model details \u{25b2}"
        } else {
            "Show model details \u{25bc}"
        };
        col = col.push(Space::new().height(10));
        col = col.push(
            button(text(toggle_label).size(11).color(style::ACCENT_BLUE))
                .on_press(Message::ModelDetailsToggled(!props.show_model_details))
                .padding(0)
                .style(|_theme, _status| button::Style {
                    background: None,
                    ..Default::default()
                }),
        );
    }

    // Expanded model details
    if props.show_model_details
        && let Some(ti) = props.training_info
    {
        col = col.push(Space::new().height(10));
        col = col.push(build_details_content(ti));
    }

    card_container(col).width(Length::Fill).into()
}

fn build_chart_card<'a>(props: &MLPredictionsProps<'a>) -> Element<'a, Message> {
    let now = props.now;
    let six_hours_ago = now - ChronoDuration::hours(6);

    let (range_start, range_end) = if props.ml_predictions.is_empty() && props.history.is_empty() {
        (six_hours_ago, now + ChronoDuration::hours(6))
    } else {
        let history_start = props
            .history
            .first()
            .map_or(six_hours_ago, |h| h.timestamp)
            .max(six_hours_ago);

        let pred_end = props
            .ml_predictions
            .last()
            .map_or(now + ChronoDuration::hours(6), |p| {
                p.timestamp + ChronoDuration::minutes(30)
            });

        (history_start, pred_end)
    };

    let chart_inner: Element<'_, Message> = if props.ml_predictions.is_empty()
        && props.history.is_empty()
    {
        container(text("No predictions yet \u{2014} waiting for data...").color(style::TEXT_MUTED))
            .height(Length::Fixed(280.0))
            .into()
    } else {
        Element::from(
            Canvas::new(HistoryChart {
                history: props.history,
                predictions: props.ml_predictions_simple,
                confidence_band: props.ml_predictions,
                range_start,
                range_end,
                cache: props.chart_cache,
            })
            .width(Length::Fill)
            .height(Length::Fixed(280.0)),
        )
        .map(|_| Message::ChartInteraction)
    };

    card_container(column![
        text("Occupancy \u{2014} Actual vs Predicted")
            .size(14)
            .color(style::TEXT_MUTED),
        Space::new().height(15),
        chart_inner,
    ])
    .width(Length::Fill)
    .into()
}

fn build_highlights_card<'a>(props: &MLPredictionsProps<'a>) -> Element<'a, Message> {
    let highlights = extract_highlights(props.ml_predictions, props.now);

    let body: Element<'_, Message> = match highlights {
        None => text("No predictions available")
            .color(style::TEXT_MUTED)
            .size(13)
            .into(),
        Some(h) => build_highlights_grid(h),
    };

    card_container(column![
        text("Prediction Highlights")
            .size(14)
            .color(style::TEXT_MUTED),
        Space::new().height(15),
        body,
    ])
    .width(Length::Fill)
    .into()
}

fn build_highlights_grid(h: PredictionHighlights) -> Element<'static, Message> {
    // Top row: Next Hour + Peak
    let next_hour_col = build_highlight_item("Next Hour", h.next_hour, HighlightFormat::WithRange);
    let peak_col = build_highlight_item("Peak Predicted", h.peak, HighlightFormat::WithTime);

    let top_row = row![
        next_hour_col.width(Length::FillPortion(1)),
        Space::new().width(20),
        peak_col.width(Length::FillPortion(1)),
    ];

    // Bottom row: Quietest + Avg Confidence
    let quietest_col =
        build_highlight_item("Quietest Predicted", h.quietest, HighlightFormat::WithTime);

    let conf_color = confidence_color(h.avg_confidence);
    let avg_conf_col = column![
        text("Avg Confidence").size(11).color(style::TEXT_MUTED),
        Space::new().height(4),
        text(format!("{:.0}%", h.avg_confidence * 100.0))
            .size(20)
            .color(conf_color),
        Space::new().height(2),
        text(format!("({} predictions)", h.prediction_count))
            .size(11)
            .color(style::TEXT_MUTED),
    ];

    let bottom_row = row![
        quietest_col.width(Length::FillPortion(1)),
        Space::new().width(20),
        avg_conf_col.width(Length::FillPortion(1)),
    ];

    column![top_row, Space::new().height(16), bottom_row].into()
}

#[derive(Clone, Copy)]
enum HighlightFormat {
    WithRange,
    WithTime,
}

fn build_highlight_item(
    label: &'static str,
    entry: Option<HighlightEntry>,
    format: HighlightFormat,
) -> iced::widget::Column<'static, Message> {
    let mut col = column![
        text(label).size(11).color(style::TEXT_MUTED),
        Space::new().height(4),
    ];

    if let Some(e) = entry {
        col = col.push(
            text(format!("{:.1}%", e.value))
                .size(20)
                .color(style::TEXT_BRIGHT),
        );
        col = col.push(Space::new().height(2));

        match format {
            HighlightFormat::WithRange => {
                let conf_color = confidence_color(e.confidence_score);
                let conf_label = if e.confidence_score >= 0.7 {
                    "High"
                } else if e.confidence_score >= 0.4 {
                    "Med"
                } else {
                    "Low"
                };
                col = col.push(
                    row![
                        text(format!("\u{00b1}{:.1}pp", e.interval_width() / 2.0))
                            .size(11)
                            .color(style::TEXT_MUTED),
                        Space::new().width(8),
                        text(conf_label).size(11).color(conf_color),
                    ]
                    .align_y(Alignment::Center),
                );
            }
            HighlightFormat::WithTime => {
                let time_str = e.time.with_timezone(&Local).format("%H:%M").to_string();
                col = col.push(
                    text(format!("at {time_str}"))
                        .size(11)
                        .color(style::TEXT_MUTED),
                );
            }
        }
    } else {
        col = col.push(text("--").size(20).color(style::TEXT_MUTED));
    }

    col
}

// ── Model details content (inline, not wrapped in card) ───────────────

#[allow(clippy::too_many_lines)]
fn build_details_content(ti: &TrainingInfo) -> Element<'_, Message> {
    let mut col = column![
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
        let r2_color = r_squared_color(cv.r_squared_mean);

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

    col.into()
}

// ── Color helpers ─────────────────────────────────────────────────────

fn r_squared_color(r2: f64) -> iced::Color {
    if r2 >= 0.8 {
        style::ACCENT_GREEN
    } else if r2 >= 0.5 {
        style::ACCENT_ORANGE
    } else {
        style::ACCENT_RED
    }
}

fn confidence_color(score: f64) -> iced::Color {
    if score >= 0.7 {
        style::ACCENT_GREEN
    } else if score >= 0.4 {
        style::ACCENT_ORANGE
    } else {
        style::ACCENT_RED
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use approx::assert_relative_eq;
    use chrono::TimeZone;

    use super::*;
    use crate::ml::{PredictionMethod, PredictionWithConfidence};

    fn make_prediction(ts: DateTime<Utc>, value: f64, confidence: f64) -> PredictionWithConfidence {
        PredictionWithConfidence {
            timestamp: ts,
            predicted_value: value,
            confidence_low: value - 5.0,
            confidence_high: value + 5.0,
            confidence_score: confidence,
            method: PredictionMethod::RandomForest {
                confidence,
                n_trees: 100,
            },
        }
    }

    #[test]
    fn test_extract_highlights_empty_returns_none() {
        let now = Utc::now();
        let result = extract_highlights(&[], now);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_highlights_single_prediction() -> anyhow::Result<()> {
        let now = Utc.with_ymd_and_hms(2026, 3, 14, 10, 0, 0).single();
        let now = now.context("failed to create timestamp")?;
        let pred = make_prediction(now + ChronoDuration::hours(1), 45.0, 0.8);

        let h = extract_highlights(&[pred], now).context("expected Some")?;

        assert_eq!(h.prediction_count, 1);

        let next = h.next_hour.context("expected next_hour")?;
        assert_relative_eq!(next.value, 45.0);

        let peak = h.peak.context("expected peak")?;
        assert_relative_eq!(peak.value, 45.0);

        let quietest = h.quietest.context("expected quietest")?;
        assert_relative_eq!(quietest.value, 45.0);

        Ok(())
    }

    #[test]
    fn test_extract_highlights_finds_peak_and_quietest() -> anyhow::Result<()> {
        let now = Utc.with_ymd_and_hms(2026, 3, 14, 10, 0, 0).single();
        let now = now.context("failed to create timestamp")?;

        let predictions = vec![
            make_prediction(now + ChronoDuration::hours(1), 30.0, 0.8),
            make_prediction(now + ChronoDuration::hours(2), 80.0, 0.7),
            make_prediction(now + ChronoDuration::hours(3), 15.0, 0.9),
            make_prediction(now + ChronoDuration::hours(4), 55.0, 0.6),
        ];

        let h = extract_highlights(&predictions, now).context("expected Some")?;

        let peak = h.peak.context("expected peak")?;
        assert_relative_eq!(peak.value, 80.0);

        let quietest = h.quietest.context("expected quietest")?;
        assert_relative_eq!(quietest.value, 15.0);

        Ok(())
    }

    #[test]
    fn test_extract_highlights_next_hour_selection() -> anyhow::Result<()> {
        let now = Utc.with_ymd_and_hms(2026, 3, 14, 10, 0, 0).single();
        let now = now.context("failed to create timestamp")?;

        let predictions = vec![
            make_prediction(now + ChronoDuration::minutes(30), 20.0, 0.8),
            make_prediction(now + ChronoDuration::minutes(55), 40.0, 0.7),
            make_prediction(now + ChronoDuration::hours(2), 60.0, 0.9),
            make_prediction(now + ChronoDuration::hours(3), 50.0, 0.6),
        ];

        let h = extract_highlights(&predictions, now).context("expected Some")?;

        // Closest to now + 1h (=10:55 is 5 min away, 11:00 would be
        // exact)
        let next = h.next_hour.context("expected next_hour")?;
        assert_relative_eq!(next.value, 40.0);

        Ok(())
    }

    #[test]
    fn test_extract_highlights_avg_confidence() -> anyhow::Result<()> {
        let now = Utc.with_ymd_and_hms(2026, 3, 14, 10, 0, 0).single();
        let now = now.context("failed to create timestamp")?;

        let predictions = vec![
            make_prediction(now + ChronoDuration::hours(1), 30.0, 0.6),
            make_prediction(now + ChronoDuration::hours(2), 50.0, 0.8),
            make_prediction(now + ChronoDuration::hours(3), 40.0, 1.0),
        ];

        let h = extract_highlights(&predictions, now).context("expected Some")?;

        // (0.6 + 0.8 + 1.0) / 3 = 0.8
        assert_relative_eq!(h.avg_confidence, 0.8, epsilon = 1e-10);
        assert_eq!(h.prediction_count, 3);

        Ok(())
    }
}
