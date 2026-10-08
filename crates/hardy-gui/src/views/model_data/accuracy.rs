//! "Live accuracy": how the daemon's logged forecasts compared with what
//! actually happened, next to the baseline.

use hardy_core::accuracy::{AccuracySummary, ErrorPair};
use iced::{
    Alignment, Border, Color, Element, Length,
    widget::{Space, column, container, row, text},
};

use super::improvement_color;
use crate::{
    app::Message,
    style,
    views::components::{badge, card_with_actions, empty_state, legend_item, stat},
};

/// Days shown, newest last.
pub(crate) const DAYS: i64 = 14;

/// How much lower the forecast error is than the baseline's (0.25 = 25%).
fn improvement(pair: &ErrorPair) -> f64 {
    if pair.baseline_mae > 0.0 {
        1.0 - pair.forecast_mae / pair.baseline_mae
    } else {
        0.0
    }
}

/// Bar length in percent of the longest bar, at least 1 so it stays visible.
fn bar_portion(value: f64, max: f64) -> u16 {
    if max <= 0.0 {
        return 1;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let portion = (value / max * 100.0).round().clamp(1.0, 100.0) as u16;
    portion
}

fn bar<'a>(value: f64, max: f64, color: Color) -> Element<'a, Message> {
    let filled = bar_portion(value, max);
    container(
        row![
            container(Space::new().height(6))
                .width(Length::FillPortion(filled))
                .style(move |_| container::Style {
                    background: Some(color.into()),
                    border: Border {
                        radius: 3.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
            Space::new().width(Length::FillPortion(100 - filled.min(99))),
        ]
        .width(Length::Fill),
    )
    .width(Length::Fill)
    .into()
}

pub(super) fn card(summary: Option<&AccuracySummary>) -> Element<'_, Message> {
    let header = badge(format!("Last {DAYS} days"), style::TEXT_SECONDARY);
    let Some(summary) = summary else {
        return card_with_actions(
            "Live accuracy",
            header,
            empty_state(
                "Nothing scored yet. The daemon logs a forecast every hour while the gym is open; \
                 scores appear once those hours have passed.",
            ),
        )
        .width(Length::Fill)
        .into();
    };

    let overall = summary.overall;
    let gain = improvement(&overall);
    #[allow(clippy::cast_precision_loss)]
    let model_share = summary.from_model as f64 / overall.scored.max(1) as f64;
    let stats = row![
        stat(
            "Forecast error",
            format!("{:.1} pts", overall.forecast_mae),
            style::FORECAST,
            Some("what the app showed".to_string()),
        ),
        stat(
            "Baseline",
            format!("{:.1} pts", overall.baseline_mae),
            style::TEXT_PRIMARY,
            Some("same hours".to_string()),
        ),
        stat(
            "Better by",
            format!("{:.0}%", gain * 100.0),
            improvement_color(gain),
            Some(format!(
                "{} forecasts · {:.0}% from the model",
                overall.scored,
                model_share * 100.0
            )),
        ),
    ]
    .spacing(style::SPACE_XXL);

    let max = summary
        .by_day
        .iter()
        .map(|(_, p)| p.forecast_mae.max(p.baseline_mae))
        .fold(0.1_f64, f64::max);
    let mut days = column![
        row![
            text("Error per day")
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_TERTIARY),
            Space::new().width(Length::Fill),
            legend_item(style::FORECAST, "Forecast"),
            legend_item(style::TEXT_TERTIARY, "Baseline"),
        ]
        .spacing(style::SPACE_L)
        .align_y(Alignment::Center)
    ]
    .spacing(style::SPACE_S);
    for (day, pair) in &summary.by_day {
        days = days.push(
            row![
                text(day.format("%a %d").to_string())
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_SECONDARY)
                    .width(56),
                column![
                    bar(pair.forecast_mae, max, style::FORECAST),
                    bar(pair.baseline_mae, max, style::TEXT_TERTIARY),
                ]
                .spacing(2)
                .width(Length::Fill),
                text(format!(
                    "{:.1} / {:.1}",
                    pair.forecast_mae, pair.baseline_mae
                ))
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_PRIMARY)
                .width(76)
                .align_x(Alignment::End),
            ]
            .spacing(style::SPACE_S)
            .align_y(Alignment::Center),
        );
    }

    card_with_actions(
        "Live accuracy",
        header,
        column![stats, days].spacing(style::SPACE_XL),
    )
    .width(Length::Fill)
    .into()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn pair(forecast: f64, baseline: f64) -> ErrorPair {
        ErrorPair {
            scored: 10,
            forecast_mae: forecast,
            baseline_mae: baseline,
        }
    }

    #[test]
    fn test_improvement_relative_to_baseline() {
        assert_relative_eq!(improvement(&pair(3.0, 4.0)), 0.25);
        assert_relative_eq!(improvement(&pair(5.0, 4.0)), -0.25);
        assert_relative_eq!(improvement(&pair(1.0, 0.0)), 0.0);
    }

    #[test]
    fn test_bar_portion_scales_and_stays_visible() {
        assert_eq!(bar_portion(5.0, 10.0), 50);
        assert_eq!(bar_portion(10.0, 10.0), 100);
        assert_eq!(bar_portion(0.0, 10.0), 1);
        assert_eq!(bar_portion(20.0, 10.0), 100);
        assert_eq!(bar_portion(1.0, 0.0), 1);
    }
}
