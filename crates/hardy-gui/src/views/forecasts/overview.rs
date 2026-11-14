//! The Overview: four headline numbers against the period before, the
//! default forecast's error day by day, every measure side by side, and
//! the hourly log against plain averages.

use hardy_core::accuracy::Measures;
use iced::{
    Alignment, Color, Element, Length,
    widget::{Space, column, container, row, text},
};

use super::{
    ForecastsProps, accuracy,
    charts::{day_ticks, day_x, line_chart, x_day},
    period_label, signed_points,
};
use crate::{
    app::Message,
    performance::Performance,
    style,
    views::components::{caption, card, card_with_actions, empty_state, legend_item, stat},
    widgets::plot::Series,
};

/// What a band should cover.
const TARGET_COVERAGE: f64 = 0.8;

/// Which way a measure gets better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Better {
    Lower,
    Higher,
    /// Nearer the 80 % a band should cover.
    NearTarget,
}

/// The change from the period before, e.g. "▼ 0.4 vs before", coloured by
/// whether it is better; `None` without a period before.
fn change(
    current: f64,
    previous: Option<f64>,
    better: Better,
    scale: f64,
    unit: &str,
) -> Option<(String, Color)> {
    let previous = previous?;
    let diff = current - previous;
    if (diff * scale).abs() < 0.05 {
        return Some(("same as before".to_string(), style::TEXT_TERTIARY));
    }
    let improved = match better {
        Better::Lower => diff < 0.0,
        Better::Higher => diff > 0.0,
        Better::NearTarget => {
            (current - TARGET_COVERAGE).abs() < (previous - TARGET_COVERAGE).abs()
        }
    };
    let arrow = if diff < 0.0 { "▼" } else { "▲" };
    let color = if improved {
        style::SUCCESS
    } else {
        style::DANGER
    };
    Some((
        format!("{arrow} {:.1}{unit} vs before", (diff * scale).abs()),
        color,
    ))
}

fn percent(value: f64) -> String {
    format!("{:.0}%", value * 100.0)
}

/// One headline number with its change.
fn headline(
    label: &str,
    value: String,
    note: String,
    change: Option<(String, Color)>,
) -> Element<'_, Message> {
    let mut body = column![stat(label, value, style::TEXT_PRIMARY, Some(note))].spacing(2);
    if let Some((change, color)) = change {
        body = body.push(text(change).size(style::TEXT_CAPTION).color(color));
    }
    container(body)
        .padding(style::CARD_PADDING)
        .width(Length::Fill)
        .style(|_| container::Style {
            background: Some(style::BG_CARD.into()),
            border: iced::Border {
                color: style::BORDER,
                width: 1.0,
                radius: style::RADIUS_CARD.into(),
            },
            ..Default::default()
        })
        .into()
}

fn headlines<'a>(current: &Measures, previous: Option<&Measures>) -> Element<'a, Message> {
    let mae = headline(
        "Typical error",
        format!("{:.1} pts", current.mae),
        "an hour ahead".to_string(),
        change(
            current.mae,
            previous.map(|p| p.mae),
            Better::Lower,
            1.0,
            " pts",
        ),
    );
    let hits = headline(
        "Within ±5 pts",
        percent(current.within_5),
        format!("of {} forecasts", current.count),
        change(
            current.within_5,
            previous.map(|p| p.within_5),
            Better::Higher,
            100.0,
            " pts",
        ),
    );
    let skill = match current.skill {
        Some(skill) => headline(
            "Better than averages",
            percent(skill),
            "than the same slot's last four weeks".to_string(),
            change(
                skill,
                previous.and_then(|p| p.skill),
                Better::Higher,
                100.0,
                " pts",
            ),
        ),
        None => headline(
            "Better than averages",
            "–".to_string(),
            "no averages yet".to_string(),
            None,
        ),
    };
    let coverage = match current.coverage {
        Some(coverage) => headline(
            "Band coverage",
            percent(coverage),
            "inside the band · target 80%".to_string(),
            change(
                coverage,
                previous.and_then(|p| p.coverage),
                Better::NearTarget,
                100.0,
                " pts",
            ),
        ),
        None => headline(
            "Band coverage",
            "–".to_string(),
            "no bands yet".to_string(),
            None,
        ),
    };
    row![mae, hits, skill, coverage]
        .spacing(style::SPACE_L)
        .into()
}

fn daily_chart<'a>(props: &ForecastsProps<'a>, performance: &Performance) -> Element<'a, Message> {
    let (Some(first), Some(last)) = (performance.daily.first(), performance.daily.last()) else {
        return card(
            "Error over time",
            empty_state("No scored forecasts in this period yet."),
        )
        .width(Length::Fill)
        .into();
    };
    let first = first.day;
    let x = |day| day_x(first, day);
    let series = vec![
        Series {
            name: "Averages".to_string(),
            color: style::TEXT_TERTIARY,
            width: 1.5,
            dashed: true,
            points: performance
                .daily
                .iter()
                .filter_map(|d| Some((x(d.day), d.baseline?)))
                .collect(),
        },
        Series {
            name: "Day".to_string(),
            color: style::tint(style::FORECAST, 0.55),
            width: 1.5,
            dashed: false,
            points: performance
                .daily
                .iter()
                .map(|d| (x(d.day), d.mae))
                .collect(),
        },
        Series {
            name: "7 days".to_string(),
            color: style::FORECAST,
            width: 2.5,
            dashed: false,
            points: performance
                .daily
                .iter()
                .map(|d| (x(d.day), d.weekly))
                .collect(),
        },
    ];
    let legend = row![
        legend_item(style::FORECAST, "Forecast, 7 days"),
        legend_item(style::tint(style::FORECAST, 0.55), "per day"),
        legend_item(style::TEXT_TERTIARY, "Averages"),
    ]
    .spacing(style::SPACE_L);
    card_with_actions(
        "Error over time",
        legend,
        column![
            line_chart(
                series,
                day_ticks(first, last.day),
                move |v| x_day(first, v).format("%a %-d %b").to_string(),
                " pts",
                None,
                &props.caches.daily,
            ),
            caption(
                "Average error of the default forecast made an hour ahead, per day and over the \
                 week ending that day, next to plain averages of the same slots.",
            ),
        ]
        .spacing(style::SPACE_S),
    )
    .width(Length::Fill)
    .into()
}

/// The measures table's rows: label, and the value of one column.
type MeasureRow = (&'static str, fn(&Measures) -> String);

const MEASURES: [MeasureRow; 13] = [
    ("Forecasts scored", |m| m.count.to_string()),
    ("Typical error (MAE)", |m| format!("{:.1} pts", m.mae)),
    ("RMSE", |m| format!("{:.1} pts", m.rmse)),
    ("Bias (+ too high)", |m| signed_points(m.bias)),
    ("Too high", |m| percent(m.over_share)),
    ("WAPE", |m| m.wape.map_or("–".to_string(), percent)),
    ("MASE (vs last week)", |m| {
        m.mase.map_or("–".to_string(), |v| format!("{v:.2}"))
    }),
    ("Better than averages", |m| {
        m.skill.map_or("–".to_string(), percent)
    }),
    ("Within ±5 pts", |m| percent(m.within_5)),
    ("Within ±10 pts", |m| percent(m.within_10)),
    ("Band coverage", |m| {
        m.coverage.map_or("–".to_string(), percent)
    }),
    ("Band width", |m| {
        m.width.map_or("–".to_string(), |w| format!("{w:.1} pts"))
    }),
    ("Interval score", |m| {
        m.interval_score
            .map_or("–".to_string(), |s| format!("{s:.1}"))
    }),
];

fn measures_table<'a>(
    props: &ForecastsProps<'a>,
    performance: &Performance,
) -> Element<'a, Message> {
    let columns = [
        ("An hour ahead", performance.current),
        ("Period before", performance.previous),
        ("Live, 15 min – 6 h", performance.live_default),
    ];
    let cell = |s: String, color: Color| {
        text(s)
            .size(style::TEXT_CAPTION)
            .color(color)
            .width(Length::FillPortion(2))
            .align_x(Alignment::End)
    };
    let mut header = row![Space::new().width(Length::FillPortion(3))].spacing(style::SPACE_S);
    for (title, _) in columns {
        header = header.push(cell(title.to_string(), style::TEXT_TERTIARY));
    }
    let mut table = column![header].spacing(style::SPACE_XS);
    for (label, value) in MEASURES {
        let mut line = row![
            text(label)
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_SECONDARY)
                .width(Length::FillPortion(3))
        ]
        .spacing(style::SPACE_S);
        for (_, measures) in &columns {
            line = line.push(match measures {
                Some(m) => cell(value(m), style::TEXT_PRIMARY),
                None => cell("–".to_string(), style::TEXT_TERTIARY),
            });
        }
        table = table.push(line);
    }
    card(
        "All measures",
        column![
            table,
            caption(format!(
                "The default forecast, {}: made an hour ahead for every open quarter hour, and as \
                 logged live every full hour for 15 minutes to 6 hours ahead. Averages: the same \
                 slot over the four weeks before.",
                props.period_label
            )),
        ]
        .spacing(style::SPACE_M),
    )
    .width(Length::Fill)
    .into()
}

pub(super) fn view<'a>(
    props: &ForecastsProps<'a>,
    performance: &'a Performance,
) -> Element<'a, Message> {
    let top: Element<'a, Message> = match &performance.current {
        Some(current) => headlines(current, performance.previous.as_ref()),
        None => empty_state(format!(
            "Nothing scored in the {} yet.",
            period_label(props.period)
        )),
    };
    column![
        top,
        daily_chart(props, performance),
        row![
            column![measures_table(props, performance)].width(Length::FillPortion(3)),
            column![accuracy::card(props.accuracy)].width(Length::FillPortion(2)),
        ]
        .spacing(style::SPACE_L),
    ]
    .spacing(style::SPACE_L)
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_change_colours_by_direction() {
        let lower = change(3.6, Some(4.0), Better::Lower, 1.0, " pts");
        assert_eq!(
            lower,
            Some(("▼ 0.4 pts vs before".to_string(), style::SUCCESS))
        );
        let worse = change(0.70, Some(0.75), Better::Higher, 100.0, " pts");
        assert_eq!(
            worse,
            Some(("▼ 5.0 pts vs before".to_string(), style::DANGER))
        );
        // Coverage rising past 80 % can be worse.
        let past = change(0.95, Some(0.85), Better::NearTarget, 100.0, " pts");
        assert_eq!(past.map(|(_, c)| c), Some(style::DANGER));
        assert_eq!(change(1.0, None, Better::Lower, 1.0, ""), None);
        assert_eq!(
            change(4.01, Some(4.0), Better::Lower, 1.0, " pts").map(|(s, _)| s),
            Some("same as before".to_string())
        );
    }

    #[test]
    fn test_measure_rows_format_missing_values() {
        let m = Measures {
            count: 12,
            mae: 3.25,
            rmse: 4.0,
            bias: -0.5,
            wape: None,
            mase: Some(0.8),
            skill: Some(0.25),
            coverage: None,
            width: None,
            interval_score: None,
            within_5: 0.75,
            within_10: 1.0,
            over_share: 0.4,
        };
        let values: Vec<String> = MEASURES.iter().map(|(_, f)| f(&m)).collect();
        assert_eq!(values[0], "12");
        assert_eq!(values[1], "3.2 pts");
        assert_eq!(values[3], "-0.5 pts");
        assert_eq!(values[5], "–");
        assert_eq!(values[6], "0.80");
        assert_eq!(values[7], "25%");
    }
}
