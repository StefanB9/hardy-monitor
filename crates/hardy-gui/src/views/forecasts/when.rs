//! When & where: the default forecast's error by weekday and hour and by
//! occupancy, how it grows with lead time (against the best live models),
//! and forecasts against readings.

use hardy_core::accuracy::{ErrorSums, Measures};
use hardy_ml::{live::SELECTED, suite::ModelId};
use iced::{
    Alignment, Element, Length,
    widget::{canvas, column, row, text},
};

use super::{
    ForecastsProps, accuracy,
    charts::{HEIGHT, lead_label, lead_ticks, line_chart},
};
use crate::{
    app::Message,
    performance::{Performance, lead_curve},
    style,
    views::components::{caption, card, empty_state, legend_item},
    widgets::{
        heatmap::{CellScale, CellValue, HeatmapWidget},
        plot::{Dot, ScatterPlot, Series},
    },
};

/// Most points in the forecast-vs-reading plot; more are thinned evenly.
const MAX_DOTS: usize = 3000;
/// Best live models drawn next to the default forecast's lead-time curve.
const COMPARED: usize = 3;
/// Heatmap colours relative to the period's error: green below this share
/// of it, red above the other.
const GOOD_SHARE: f64 = 0.7;
const BAD_SHARE: f64 = 1.5;

/// The error per weekday and hour on the Week tab's grid; the bias goes
/// where the grid keeps its spread.
pub(crate) fn error_grid(grid: &[[ErrorSums; 24]]) -> [[Option<CellValue>; 24]; 7] {
    std::array::from_fn(|day| {
        std::array::from_fn(|hour| {
            let m = grid.get(day)?.get(hour)?.measures()?;
            Some(CellValue {
                value: m.mae,
                days: m.count,
                spread: m.bias,
            })
        })
    })
}

/// Every `step`-th item so that at most `max` remain.
pub(super) fn thin<T: Clone>(items: &[T], max: usize) -> Vec<T> {
    let step = items.len().div_ceil(max.max(1)).max(1);
    items.iter().step_by(step).cloned().collect()
}

fn heatmap<'a>(props: &ForecastsProps<'a>, mae: f64) -> Element<'a, Message> {
    card(
        "When it's off",
        column![
            canvas::Canvas::new(HeatmapWidget {
                grid: props.error_grid,
                scale: CellScale::Error {
                    good: mae * GOOD_SHARE,
                    bad: mae * BAD_SHARE,
                },
                cache: &props.caches.heatmap,
                tooltip_cache: &props.caches.heatmap_tooltip,
                busy_ranges: &[],
                quiet_ranges: &[],
            })
            .width(Length::Fill)
            .height(260),
            caption(format!(
                "Average error of the default forecast an hour ahead, per weekday and hour. Green \
                 is well below its usual {mae:.1} points, red well above."
            )),
        ]
        .spacing(style::SPACE_S),
    )
    .width(Length::Fill)
    .into()
}

fn levels<'a>(performance: &Performance) -> Element<'a, Message> {
    let rows: Vec<(f64, Measures)> = performance
        .levels
        .iter()
        .filter_map(|(lower, sums)| Some((*lower, sums.measures()?)))
        .collect();
    let max = rows.iter().map(|(_, m)| m.mae).fold(0.1_f64, f64::max);
    let mut list = column![].spacing(style::SPACE_S);
    for (lower, m) in rows {
        list = list.push(
            row![
                text(format!("{lower:.0}–{:.0}%", lower + 10.0))
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_SECONDARY)
                    .width(64),
                accuracy::bar(m.mae, max, style::FORECAST),
                text(format!("{:.1} pts · {}", m.mae, m.count))
                    .size(style::TEXT_CAPTION)
                    .color(style::TEXT_PRIMARY)
                    .width(96)
                    .align_x(Alignment::End),
            ]
            .spacing(style::SPACE_S)
            .align_y(Alignment::Center),
        );
    }
    card(
        "By occupancy",
        column![
            list,
            caption("Average error by how busy the gym actually was."),
        ]
        .spacing(style::SPACE_M),
    )
    .width(Length::Fill)
    .into()
}

fn lead_times<'a>(props: &ForecastsProps<'a>, performance: &Performance) -> Element<'a, Message> {
    let mut best: Vec<(&String, f64)> = performance
        .live
        .iter()
        .filter(|(model, _)| model.as_str() != SELECTED)
        .filter_map(|(model, score)| Some((model, score.mae[0]?)))
        .collect();
    best.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut models: Vec<(&str, &'static str)> = vec![(SELECTED, "Default")];
    models.extend(
        best.into_iter()
            .filter_map(|(key, _)| Some((key.as_str(), ModelId::from_key(key)?.name())))
            .take(COMPARED),
    );
    if let Some(id) = props.picked
        && !models.iter().any(|(key, _)| *key == id.key())
    {
        models.push((id.key(), id.name()));
    }
    let lines: Vec<(&'static str, Series)> = models
        .iter()
        .enumerate()
        .map(|(i, (key, name))| {
            (
                *name,
                Series {
                    name: (*name).to_string(),
                    color: if i == 0 {
                        style::FORECAST
                    } else {
                        style::COMPARISON[(i - 1) % style::COMPARISON.len()]
                    },
                    width: if i == 0 { 2.5 } else { 1.5 },
                    dashed: false,
                    points: lead_curve(&performance.model_errors, key)
                        .into_iter()
                        .map(|(step, m)| (f64::from(step), m.mae))
                        .collect(),
                },
            )
        })
        .filter(|(_, s)| !s.points.is_empty())
        .collect();
    let body: Element<'a, Message> = if lines.is_empty() {
        empty_state("No live forecasts scored in this period yet.")
    } else {
        let legend = lines
            .iter()
            .fold(row![].spacing(style::SPACE_M), |legend, (name, s)| {
                legend.push(legend_item(s.color, name))
            });
        let series = lines.into_iter().map(|(_, s)| s).collect();
        column![
            line_chart(
                series,
                lead_ticks(),
                lead_label,
                " pts",
                None,
                &props.caches.lead_compare,
            ),
            legend,
            caption(
                "Live error by how far ahead the forecast was made: the default forecast, the \
                 three best models live, and the model picked on the Models page.",
            ),
        ]
        .spacing(style::SPACE_S)
        .into()
    };
    card("By lead time", body).width(Length::Fill).into()
}

fn forecast_vs_actual<'a>(
    props: &ForecastsProps<'a>,
    performance: &Performance,
) -> Element<'a, Message> {
    let tz = props.timezone;
    let dots = thin(&performance.scored, MAX_DOTS)
        .into_iter()
        .map(|s| Dot {
            x: s.pair.actual,
            y: s.pair.forecast,
            label: s
                .target
                .with_timezone(&tz)
                .format("%a %-d %b %H:%M")
                .to_string(),
            color: style::tint(style::FORECAST, 0.45),
            large: false,
        })
        .collect();
    let histogram = vec![Series {
        name: "Forecasts".to_string(),
        color: style::ACCENT,
        width: 2.0,
        dashed: false,
        points: performance
            .histogram
            .iter()
            .map(|(lower, n)| {
                #[allow(clippy::cast_precision_loss)]
                let count = *n as f64;
                (lower + 1.0, count)
            })
            .collect(),
    }];
    row![
        card(
            "Forecast vs reading",
            column![
                canvas::Canvas::new(ScatterPlot {
                    dots,
                    x_title: "Reading",
                    y_title: "Forecast",
                    suffix: "%",
                    cache: &props.caches.forecast_vs_actual,
                })
                .width(Length::Fill)
                .height(HEIGHT + 60.0),
                caption(
                    "Each forecast an hour ahead against the reading; on the line it was exact."
                ),
            ]
            .spacing(style::SPACE_S),
        )
        .width(Length::FillPortion(1)),
        card(
            "How far off",
            column![
                line_chart(
                    histogram,
                    [-20.0, -10.0, 0.0, 10.0, 20.0]
                        .map(|x: f64| (x, format!("{x:+.0}")))
                        .to_vec(),
                    |x| format!("{:+.0} to {:+.0} pts", x - 1.0, x + 1.0),
                    "",
                    None,
                    &props.caches.histogram,
                ),
                caption(
                    "Forecasts by error in points: left of zero too low, right too high; errors \
                     beyond 20 points count at the edges.",
                ),
            ]
            .spacing(style::SPACE_S),
        )
        .width(Length::FillPortion(1)),
    ]
    .spacing(style::SPACE_L)
    .into()
}

pub(super) fn view<'a>(
    props: &ForecastsProps<'a>,
    performance: &'a Performance,
) -> Element<'a, Message> {
    let Some(current) = performance.current else {
        return empty_state("Nothing scored in this period yet.");
    };
    column![
        heatmap(props, current.mae),
        row![
            column![levels(performance)].width(Length::FillPortion(2)),
            column![lead_times(props, performance)].width(Length::FillPortion(3)),
        ]
        .spacing(style::SPACE_L),
        forecast_vs_actual(props, performance),
    ]
    .spacing(style::SPACE_L)
    .into()
}

#[cfg(test)]
mod tests {
    use hardy_core::accuracy::Pair;

    use super::*;

    #[test]
    fn test_error_grid_puts_the_bias_in_the_spread() {
        let mut grid = vec![[ErrorSums::default(); 24]; 7];
        grid[1][9].add(&Pair {
            actual: 30.0,
            forecast: 27.0,
            band: None,
            baseline: None,
            last_week: None,
        });
        let cells = error_grid(&grid);
        assert_eq!(
            cells[1][9],
            Some(CellValue {
                value: 3.0,
                days: 1,
                spread: -3.0
            })
        );
        assert_eq!(cells[1][10], None);
    }

    #[test]
    fn test_thin_keeps_at_most_max_evenly() {
        let items: Vec<u32> = (0..10).collect();
        assert_eq!(thin(&items, 4), vec![0, 3, 6, 9]);
        assert_eq!(thin(&items, 20), items);
        assert_eq!(thin(&Vec::<u32>::new(), 3), Vec::<u32>::new());
    }
}
