use chrono::{DateTime, Duration as ChronoDuration, Utc};
use hardy_core::db::OccupancyLog;
use hardy_ml::PredictionWithConfidence;
use iced::{
    Alignment, Element, Length,
    widget::{Canvas, Space, button, canvas, column, container, row, scrollable, text},
};

use crate::{
    app::Message, forecasting::ModelSummary, style, views::components::card_container,
    widgets::history_chart::HistoryChart,
};

#[derive(Clone, Copy)]
pub struct MLPredictionsProps<'a> {
    /// Gym timezone used for every displayed wall-clock time.
    pub timezone: hardy_core::Tz,
    pub ml_predictions: &'a [PredictionWithConfidence],
    pub ml_predictions_simple: &'a [(DateTime<Utc>, f64)],
    pub ml_has_model: bool,
    /// A retrain was requested and the daemon has not started it yet.
    pub retrain_pending: bool,
    /// Why the daemon's last training produced no model.
    pub last_training_error: Option<&'a str>,
    pub chart_cache: &'a canvas::Cache,
    pub now: DateTime<Utc>,
    pub model: Option<&'a ModelSummary>,
    pub history: &'a [OccupancyLog],
    pub show_model_details: bool,
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

fn build_status_card<'a>(props: &MLPredictionsProps<'a>) -> Element<'a, Message> {
    let mut col = column![
        text("Forecast Model").size(14).color(style::TEXT_MUTED),
        Space::new().height(15),
    ];

    let train_label = if props.ml_has_model {
        "Retrain"
    } else {
        "Train now"
    };
    let train_button: Element<'a, Message> = if props.retrain_pending {
        text("Retrain requested\u{2026}")
            .size(11)
            .color(style::ACCENT_ORANGE)
            .into()
    } else {
        build_action_button(train_label, Message::TrainModelRequested)
    };

    let mut status_row = row![].align_y(Alignment::Center);
    match props.model {
        None => {
            status_row = status_row.push(
                text("No model yet \u{2014} the daemon trains one automatically; showing averages")
                    .size(12)
                    .color(style::TEXT_MUTED),
            );
        }
        Some(model) => {
            let improvement = model.improvement();
            status_row = status_row
                .push(
                    text(format!("Active ({})", model.algorithm))
                        .size(12)
                        .color(style::ACCENT_GREEN),
                )
                .push(Space::new().width(12))
                .push(
                    text(format!("{:.0}% better than averages", improvement * 100.0))
                        .size(12)
                        .color(improvement_color(improvement)),
                )
                .push(Space::new().width(Length::Fill))
                .push(text("Trained:").size(12).color(style::TEXT_MUTED))
                .push(Space::new().width(6))
                .push(
                    text(
                        model
                            .trained_at
                            .with_timezone(&props.timezone)
                            .format("%a %H:%M")
                            .to_string(),
                    )
                    .size(12)
                    .color(style::TEXT_BRIGHT),
                )
                .push(Space::new().width(12));
        }
    }
    if props.model.is_none() {
        status_row = status_row.push(Space::new().width(Length::Fill));
    }
    col = col.push(status_row.push(train_button));

    if let Some(error) = props.last_training_error {
        col = col.push(Space::new().height(6)).push(
            text(format!("Last training: {error}"))
                .size(11)
                .color(style::ACCENT_ORANGE),
        );
    }

    if let Some(model) = props.model {
        let toggle_label = if props.show_model_details {
            "Hide model details \u{25b2}"
        } else {
            "Show model details \u{25bc}"
        };
        col = col.push(Space::new().height(10)).push(
            button(text(toggle_label).size(11).color(style::ACCENT_BLUE))
                .on_press(Message::ModelDetailsToggled(!props.show_model_details))
                .padding(0)
                .style(|_theme, _status| button::Style {
                    background: None,
                    ..Default::default()
                }),
        );
        if props.show_model_details {
            col = col
                .push(Space::new().height(10))
                .push(build_details_content(model));
        }
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
                timezone: props.timezone,
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
        Some(h) => build_highlights_grid(h, props.timezone),
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

fn build_highlights_grid(h: PredictionHighlights, tz: hardy_core::Tz) -> Element<'static, Message> {
    // Top row: Next Hour + Peak
    let next_hour_col = build_highlight_item("Next Hour", h.next_hour, HighlightFormat::WithRange);
    let peak_col = build_highlight_item("Peak Predicted", h.peak, HighlightFormat::WithTime(tz));

    let top_row = row![
        next_hour_col.width(Length::FillPortion(1)),
        Space::new().width(20),
        peak_col.width(Length::FillPortion(1)),
    ];

    // Bottom row: Quietest + Avg Confidence
    let quietest_col = build_highlight_item(
        "Quietest Predicted",
        h.quietest,
        HighlightFormat::WithTime(tz),
    );

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
    /// Show the entry's time in the given (gym) timezone.
    WithTime(hardy_core::Tz),
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
            HighlightFormat::WithTime(tz) => {
                let time_str = e.time.with_timezone(&tz).format("%H:%M").to_string();
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

fn build_details_content(model: &ModelSummary) -> Element<'_, Message> {
    let detail = |label: &'static str, value: String| {
        row![
            text(label).size(12).color(style::TEXT_MUTED),
            Space::new().width(6),
            text(value).size(12).color(style::TEXT_BRIGHT),
        ]
        .align_y(Alignment::Center)
    };

    let horizons = model
        .mae_by_horizon
        .iter()
        .enumerate()
        .filter(|(_, mae)| mae.is_finite())
        .map(|(i, mae)| format!("+{}h {mae:.1}", i + 1))
        .collect::<Vec<_>>()
        .join("   ");

    column![
        row![
            detail("Samples:", model.training_samples.to_string()),
            Space::new().width(Length::Fill),
            detail(
                "Hyperparameters:",
                if model.tuned {
                    "tuned this run".to_string()
                } else {
                    "reused".to_string()
                }
            ),
        ],
        Space::new().height(4),
        detail(
            "Error on last 7 days:",
            format!(
                "{:.1} pts (averages: {:.1} pts)",
                model.holdout_mae, model.baseline_mae
            )
        ),
        Space::new().height(4),
        detail("Error by hours ahead:", horizons),
    ]
    .into()
}

// ── Color helpers ─────────────────────────────────────────────────────

/// Green for a clear improvement over averages, orange for a small one.
fn improvement_color(improvement: f64) -> iced::Color {
    if improvement >= 0.15 {
        style::ACCENT_GREEN
    } else if improvement > 0.0 {
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
    use hardy_ml::{PredictionMethod, PredictionWithConfidence};

    use super::*;

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
