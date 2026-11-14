//! Intervals & bias: whether the default forecast runs high or low (overall,
//! by hour and by weekday), and how well its 80 % band works over time and
//! by lead time.

use hardy_core::analytics::weekday_short;
use hardy_ml::live::SELECTED;
use iced::{
    Element, Length,
    widget::{column, row},
};

use super::{
    ForecastsProps,
    charts::{day_ticks, day_x, lead_label, lead_ticks, line_chart, x_day},
    signed_points,
};
use crate::{
    app::Message,
    performance::{Performance, by_hour, by_weekday, lead_curve},
    style,
    views::components::{caption, card, empty_state, stat},
    widgets::plot::Series,
};

/// What the band should cover, in percent.
const TARGET: f64 = 80.0;

fn line(name: &str, color: iced::Color, points: Vec<(f64, f64)>) -> Vec<Series> {
    vec![Series {
        name: name.to_string(),
        color,
        width: 2.0,
        dashed: false,
        points,
    }]
}

fn stats<'a>(performance: &Performance) -> Element<'a, Message> {
    let Some(m) = performance.current else {
        return empty_state("Nothing scored in this period yet.");
    };
    let or_dash = |v: Option<f64>, f: fn(f64) -> String| v.map_or("–".to_string(), f);
    row![
        stat(
            "Bias",
            signed_points(m.bias),
            style::TEXT_PRIMARY,
            Some("+ means too high".to_string()),
        ),
        stat(
            "Too high",
            format!("{:.0}%", m.over_share * 100.0),
            style::TEXT_PRIMARY,
            Some(format!("too low {:.0}%", (1.0 - m.over_share) * 100.0)),
        ),
        stat(
            "Band coverage",
            or_dash(m.coverage, |c| format!("{:.0}%", c * 100.0)),
            style::TEXT_PRIMARY,
            Some("target 80%".to_string()),
        ),
        stat(
            "Band width",
            or_dash(m.width, |w| format!("{w:.1} pts")),
            style::TEXT_PRIMARY,
            Some("on average".to_string()),
        ),
        stat(
            "Interval score",
            or_dash(m.interval_score, |s| format!("{s:.1}")),
            style::TEXT_PRIMARY,
            Some("width + 10 × miss · lower is better".to_string()),
        ),
    ]
    .spacing(style::SPACE_XXL)
    .into()
}

fn bias_by_hour<'a>(props: &ForecastsProps<'a>, performance: &Performance) -> Element<'a, Message> {
    let points: Vec<(f64, f64)> = by_hour(&performance.grid)
        .iter()
        .enumerate()
        .filter_map(|(hour, sums)| {
            #[allow(clippy::cast_precision_loss)]
            Some((hour as f64, sums.measures()?.bias))
        })
        .collect();
    let ticks = points
        .iter()
        .filter(|(h, _)| h.rem_euclid(3.0) == 0.0)
        .map(|(h, _)| (*h, format!("{h:02.0}")))
        .collect();
    card(
        "Bias by hour",
        line_chart(
            line("Bias", style::FORECAST, points),
            ticks,
            |h| format!("{h:02.0}:00–{:02.0}:00", h + 1.0),
            " pts",
            None,
            &props.caches.bias_hour,
        ),
    )
    .width(Length::FillPortion(3))
    .into()
}

fn bias_by_weekday<'a>(
    props: &ForecastsProps<'a>,
    performance: &Performance,
) -> Element<'a, Message> {
    let points: Vec<(f64, f64)> = by_weekday(&performance.grid)
        .iter()
        .enumerate()
        .filter_map(|(day, sums)| {
            #[allow(clippy::cast_precision_loss)]
            Some((day as f64, sums.measures()?.bias))
        })
        .collect();
    let day_name = |x: f64| {
        #[allow(clippy::cast_possible_truncation)]
        weekday_short(x.round() as i32).to_string()
    };
    let ticks = points.iter().map(|(d, _)| (*d, day_name(*d))).collect();
    card(
        "Bias by weekday",
        line_chart(
            line("Bias", style::FORECAST, points),
            ticks,
            day_name,
            " pts",
            None,
            &props.caches.bias_weekday,
        ),
    )
    .width(Length::FillPortion(2))
    .into()
}

fn coverage_over_time<'a>(
    props: &ForecastsProps<'a>,
    performance: &Performance,
) -> Element<'a, Message> {
    let banded: Vec<_> = performance
        .daily
        .iter()
        .filter_map(|d| Some((d.day, d.coverage?, d.width?)))
        .collect();
    let (Some(first), Some(last)) = (banded.first(), banded.last()) else {
        return card(
            "Band over time",
            empty_state("No forecasts with a band in this period."),
        )
        .width(Length::Fill)
        .into();
    };
    let (first, last) = (first.0, last.0);
    let coverage = banded
        .iter()
        .map(|(day, c, _)| (day_x(first, *day), c * 100.0))
        .collect();
    let width = banded
        .iter()
        .map(|(day, _, w)| (day_x(first, *day), *w))
        .collect();
    let label = move |v| x_day(first, v).format("%a %-d %b").to_string();
    row![
        card(
            "Band coverage by day",
            column![
                line_chart(
                    line("In band", style::ACCENT, coverage),
                    day_ticks(first, last),
                    label,
                    "%",
                    Some(TARGET),
                    &props.caches.coverage_daily,
                ),
                caption("Share of readings inside the 80 % band; the dashed line is 80 %."),
            ]
            .spacing(style::SPACE_S),
        )
        .width(Length::FillPortion(1)),
        card(
            "Band width by day",
            column![
                line_chart(
                    line("Width", style::TEXT_SECONDARY, width),
                    day_ticks(first, last),
                    label,
                    " pts",
                    None,
                    &props.caches.width_daily,
                ),
                caption("How wide the band was on average: narrower is more precise."),
            ]
            .spacing(style::SPACE_S),
        )
        .width(Length::FillPortion(1)),
    ]
    .spacing(style::SPACE_L)
    .into()
}

fn coverage_by_lead<'a>(
    props: &ForecastsProps<'a>,
    performance: &Performance,
) -> Element<'a, Message> {
    let points: Vec<(f64, f64)> = lead_curve(&performance.model_errors, SELECTED)
        .into_iter()
        .filter_map(|(step, m)| Some((f64::from(step), m.coverage? * 100.0)))
        .collect();
    let body: Element<'a, Message> = if points.is_empty() {
        empty_state("No live forecasts with a band scored in this period yet.")
    } else {
        column![
            line_chart(
                line("In band", style::ACCENT, points),
                lead_ticks(),
                lead_label,
                "%",
                Some(TARGET),
                &props.caches.coverage_lead,
            ),
            caption(
                "The default forecast's live band by how far ahead it was made; the dashed line \
                 is 80 %.",
            ),
        ]
        .spacing(style::SPACE_S)
        .into()
    };
    card("Band coverage by lead time", body)
        .width(Length::Fill)
        .into()
}

pub(super) fn view<'a>(
    props: &ForecastsProps<'a>,
    performance: &'a Performance,
) -> Element<'a, Message> {
    column![
        card("The default forecast an hour ahead", stats(performance)).width(Length::Fill),
        row![
            bias_by_hour(props, performance),
            bias_by_weekday(props, performance)
        ]
        .spacing(style::SPACE_L),
        coverage_over_time(props, performance),
        coverage_by_lead(props, performance),
        caption(format!(
            "Bias: the average of forecast − reading, {}; above zero the forecast ran high.",
            props.period_label
        )),
    ]
    .spacing(style::SPACE_L)
    .into()
}
