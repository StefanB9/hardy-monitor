//! "Model & Data": the forecast model, its live accuracy, data repair and
//! CSV export.

mod accuracy;
mod maintenance;

pub(crate) use accuracy::DAYS as ACCURACY_DAYS;
use hardy_core::{Tz, accuracy::AccuracySummary, error::AppError, repair::RepairSummary};
use iced::{
    Alignment, Border, Element, Length,
    widget::{Space, column, container, row, text},
};

use crate::{
    app::Message,
    forecasting::ModelSummary,
    style,
    views::components::{badge, card_with_actions, empty_state, scroll, small_button, stat},
};

/// Everything the Model & Data view shows.
#[derive(Clone, Copy)]
pub struct ModelDataProps<'a> {
    pub timezone: Tz,
    pub model: Option<&'a ModelSummary>,
    /// Logged forecasts scored against readings; `None` before any.
    pub accuracy: Option<&'a AccuracySummary>,
    /// A retrain was requested and the daemon has not started it yet.
    pub retrain_pending: bool,
    /// Why the daemon's last training produced no model.
    pub last_training_error: Option<&'a str>,
    pub repair_start: &'a str,
    pub repair_end: &'a str,
    pub repair_running: bool,
    pub repair_result: Option<&'a Result<RepairSummary, AppError>>,
    pub export_status: Option<&'a str>,
}

/// Green for a clear improvement over the baseline, amber for a small one.
fn improvement_color(improvement: f64) -> iced::Color {
    if improvement >= 0.15 {
        style::SUCCESS
    } else if improvement > 0.0 {
        style::WARNING
    } else {
        style::DANGER
    }
}

/// `16530` → `"16,530"`.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub(super) fn caption(s: impl text::IntoFragment<'static>) -> text::Text<'static> {
    text(s)
        .size(style::TEXT_CAPTION)
        .color(style::TEXT_TERTIARY)
}

/// One bar per horizon, scaled to the largest error.
fn error_by_horizon(mae_by_horizon: &[f64]) -> Element<'static, Message> {
    let max = mae_by_horizon
        .iter()
        .copied()
        .filter(|m| m.is_finite())
        .fold(0.1_f64, f64::max);
    let mut rows = column![caption("Average error by hours ahead")].spacing(6);
    for (i, mae) in mae_by_horizon.iter().enumerate() {
        if !mae.is_finite() {
            continue;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let filled = ((mae / max) * 100.0).round().clamp(1.0, 100.0) as u16;
        rows = rows.push(
            row![
                text(format!("+{} h", i + 1))
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_SECONDARY)
                    .width(36),
                container(
                    row![
                        container(Space::new().height(8))
                            .width(Length::FillPortion(filled))
                            .style(|_| container::Style {
                                background: Some(style::FORECAST.into()),
                                border: Border {
                                    radius: 3.0.into(),
                                    ..Default::default()
                                },
                                ..Default::default()
                            }),
                        Space::new().width(Length::FillPortion(100 - filled.min(99))),
                    ]
                    .width(Length::Fill)
                )
                .width(Length::Fill),
                text(format!("{mae:.1} pts"))
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_PRIMARY)
                    .width(56)
                    .align_x(Alignment::End),
            ]
            .spacing(style::SPACE_S)
            .align_y(Alignment::Center),
        );
    }
    rows.into()
}

fn model_card<'a>(props: &ModelDataProps<'a>) -> Element<'a, Message> {
    let retrain: Element<'a, Message> = if props.retrain_pending {
        badge("Retrain requested…".to_string(), style::WARNING)
    } else {
        let label = if props.model.is_some() {
            "Retrain now"
        } else {
            "Train now"
        };
        small_button(label, Message::TrainModelRequested).into()
    };
    let mut actions = row![].spacing(style::SPACE_S).align_y(Alignment::Center);
    if let Some(model) = props.model {
        actions = actions.push(badge(model.algorithm.clone(), style::FORECAST));
    }
    let actions = actions.push(retrain);

    let mut body = column![].spacing(style::SPACE_XL);
    match props.model {
        None => {
            body = body.push(empty_state(
                "No model yet. The daemon trains one every night; until then forecasts use the \
                 baseline: typical times, adjusted for how busy it is right now.",
            ));
        }
        Some(model) => {
            let improvement = model.improvement();
            body = body
                .push(
                    row![
                        stat(
                            "Better than baseline",
                            format!("{:.0}%", improvement * 100.0),
                            improvement_color(improvement),
                            Some("lower error on the last 7 days".to_string()),
                        ),
                        stat(
                            "Trained",
                            model
                                .trained_at
                                .with_timezone(&props.timezone)
                                .format("%a %H:%M")
                                .to_string(),
                            style::TEXT_PRIMARY,
                            Some(if model.tuned {
                                "settings tuned in this run".to_string()
                            } else {
                                "settings reused".to_string()
                            }),
                        ),
                        stat(
                            "Training samples",
                            thousands(model.training_samples),
                            style::TEXT_PRIMARY,
                            None,
                        ),
                    ]
                    .spacing(style::SPACE_XXL),
                )
                .push(
                    text(format!(
                        "Typical error {:.1} pts (baseline: {:.1} pts)",
                        model.holdout_mae, model.baseline_mae
                    ))
                    .size(style::TEXT_BODY)
                    .color(style::TEXT_SECONDARY),
                )
                .push(error_by_horizon(&model.mae_by_horizon));
        }
    }
    if let Some(error) = props.last_training_error {
        body = body.push(
            text(format!("Last training failed: {error}"))
                .size(style::TEXT_CAPTION)
                .color(style::WARNING),
        );
    }

    card_with_actions("Forecast model", actions, body)
        .width(Length::Fill)
        .into()
}

/// The Model & Data page.
pub fn view(props: ModelDataProps<'_>) -> Element<'_, Message> {
    let content = row![
        column![model_card(&props), accuracy::card(props.accuracy)]
            .spacing(style::SPACE_L)
            .width(Length::FillPortion(3)),
        column![
            maintenance::repair_card(&props),
            maintenance::export_card(&props)
        ]
        .spacing(style::SPACE_L)
        .width(Length::FillPortion(2)),
    ]
    .spacing(style::SPACE_L);

    scroll(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(16_530), "16,530");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }

    #[test]
    fn test_improvement_color_bands() {
        assert_eq!(improvement_color(0.25), style::SUCCESS);
        assert_eq!(improvement_color(0.05), style::WARNING);
        assert_eq!(improvement_color(-0.1), style::DANGER);
    }
}
