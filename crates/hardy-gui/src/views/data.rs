//! "Data": every stored reading of a day or week, as a table. Only the rows
//! in view are built; spacers stand in for the rest.

use chrono::{DateTime, Utc};
use hardy_core::{
    Tz,
    db::{DataSource, OccupancyLog},
};
use iced::{
    Alignment, Color, Element, Length,
    widget::{Space, column, container, row, text},
};

use crate::{
    app::{DATA_TABLE_ID, Message},
    data_table::{DataPeriod, ROW_HEIGHT, SourceCounts, source_label, visible_rows},
    style,
    time_range::Step,
    views::components::{card_with_actions, empty_state, scroll_area, secondary_button, segmented},
};

const PERIODS: [(&str, DataPeriod); 2] = [("Day", DataPeriod::Day), ("Week", DataPeriod::Week)];

/// Column widths: time, value; the source takes the rest.
const TIME: f32 = 150.0;
const VALUE: f32 = 90.0;
/// Width of the bar drawn for 100%.
const BAR: f32 = 160.0;

/// Everything the Data tab shows.
#[derive(Clone, Copy)]
pub struct DataProps<'a> {
    pub rows: &'a [OccupancyLog],
    pub period: DataPeriod,
    pub window: (DateTime<Utc>, DateTime<Utc>),
    pub can_step_back: bool,
    pub can_step_forward: bool,
    /// The table's scroll offset and visible height.
    pub scroll: (f32, f32),
    pub timezone: Tz,
    pub low_threshold: f64,
    pub high_threshold: f64,
}

pub fn view(props: DataProps<'_>) -> Element<'_, Message> {
    let body: Element<'_, Message> = if props.rows.is_empty() {
        empty_state("No readings in this period")
    } else {
        column![header(), table(props)]
            .spacing(style::SPACE_S)
            .into()
    };
    card_with_actions(
        "Readings",
        segmented(&PERIODS, props.period, Message::DataPeriodSelected),
        column![toolbar(props), body]
            .spacing(style::SPACE_M)
            .height(Length::Fill),
    )
    .height(Length::Fill)
    .into()
}

/// Period stepping, its name and what its readings are made of.
fn toolbar(props: DataProps<'_>) -> Element<'_, Message> {
    let step = |glyph, step, enabled: bool| {
        secondary_button(glyph, Message::ChartStep(step))
            .on_press_maybe(enabled.then_some(Message::ChartStep(step)))
            .padding([4, 10])
    };
    row![
        step("‹", Step::Back, props.can_step_back),
        text(props.period.label(props.window, props.timezone))
            .size(style::TEXT_BODY)
            .color(style::TEXT_PRIMARY),
        step("›", Step::Forward, props.can_step_forward),
        Space::new().width(Length::Fill),
        text(summary(props.rows))
            .size(style::TEXT_CAPTION)
            .color(style::TEXT_TERTIARY),
    ]
    .spacing(style::SPACE_M)
    .align_y(Alignment::Center)
    .into()
}

/// "1,021 readings · 980 measured · 30 filled in · 11 smoothed".
fn summary(rows: &[OccupancyLog]) -> String {
    let counts = SourceCounts::of(rows);
    let mut parts = vec![
        format!("{} readings", thousands(rows.len())),
        format!("{} measured", thousands(counts.measured)),
    ];
    if counts.filled > 0 {
        parts.push(format!("{} filled in", thousands(counts.filled)));
    }
    if counts.changed > 0 {
        parts.push(format!("{} smoothed", thousands(counts.changed)));
    }
    parts.join(" · ")
}

/// `n` with thousands separated by commas.
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

fn caption(label: &str, width: Length, align: Alignment) -> Element<'_, Message> {
    text(label)
        .size(style::TEXT_CAPTION)
        .color(style::TEXT_TERTIARY)
        .width(width)
        .align_x(align)
        .into()
}

fn header() -> Element<'static, Message> {
    row![
        caption("Time", Length::Fixed(TIME), Alignment::Start),
        caption("Occupancy", Length::Fixed(VALUE), Alignment::End),
        Space::new().width(BAR + style::SPACE_M),
        caption("Source", Length::Fill, Alignment::Start),
    ]
    .spacing(style::SPACE_M)
    .padding([0.0, style::SPACE_S])
    .into()
}

/// The rows in view, between spacers as tall as the rows left out.
fn table(props: DataProps<'_>) -> Element<'_, Message> {
    let (offset, height) = props.scroll;
    let shown = visible_rows(props.rows.len(), offset, height);
    #[allow(clippy::cast_precision_loss)]
    let spacer = |rows: usize| Space::new().height(rows as f32 * ROW_HEIGHT);
    let mut list = column![spacer(shown.start)];
    for i in shown.clone() {
        list = list.push(table_row(props, i));
    }
    list = list.push(spacer(props.rows.len() - shown.end));
    scroll_area(list.width(Length::Fill))
        .id(DATA_TABLE_ID)
        .on_scroll(|viewport| Message::DataScrolled {
            offset: viewport.absolute_offset().y,
            height: viewport.bounds().height,
        })
        .into()
}

fn table_row(props: DataProps<'_>, index: usize) -> Element<'_, Message> {
    let Some(log) = props.rows.get(index) else {
        return Space::new().height(ROW_HEIGHT).into();
    };
    let local = log.timestamp.with_timezone(&props.timezone);
    let time = match props.period {
        DataPeriod::Day => local.format("%H:%M").to_string(),
        DataPeriod::Week => local.format("%a %-d · %H:%M").to_string(),
    };
    let color = style::occupancy_color(log.percentage, props.low_threshold, props.high_threshold);
    #[allow(clippy::cast_possible_truncation)]
    let bar_width = (log.percentage.clamp(0.0, 100.0) / 100.0 * f64::from(BAR)) as f32;
    let bar = container(Space::new().width(bar_width.max(1.0)).height(6)).style(move |_| {
        container::Style {
            background: Some(color.into()),
            border: iced::Border {
                radius: 3.0.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    });
    let striped = index % 2 == 1;
    container(
        row![
            text(time)
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_SECONDARY)
                .width(TIME),
            text(format!("{:.1}%", log.percentage))
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_PRIMARY)
                .width(VALUE)
                .align_x(Alignment::End),
            container(bar).width(BAR + style::SPACE_M),
            text(source_label(log.source))
                .size(style::TEXT_CAPTION)
                .color(source_color(log.source)),
        ]
        .spacing(style::SPACE_M)
        .align_y(Alignment::Center),
    )
    .width(Length::Fill)
    .height(ROW_HEIGHT)
    .padding([0.0, style::SPACE_S])
    .center_y(ROW_HEIGHT)
    .style(move |_| container::Style {
        background: striped.then(|| style::BG_ELEVATED.into()),
        ..Default::default()
    })
    .into()
}

fn source_color(source: DataSource) -> Color {
    match source {
        DataSource::Measured => style::TEXT_SECONDARY,
        DataSource::Interpolated => style::WARNING,
        DataSource::Boundary => style::TEXT_TERTIARY,
        DataSource::Smoothed => style::FORECAST,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log(source: DataSource) -> OccupancyLog {
        OccupancyLog {
            id: 0,
            timestamp: DateTime::default(),
            percentage: 20.0,
            source,
        }
    }

    #[test]
    fn test_thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_021), "1,021");
        assert_eq!(thousands(7_146_000), "7,146,000");
    }

    #[test]
    fn test_summary_names_only_the_sources_present() {
        let measured = vec![log(DataSource::Measured); 3];
        assert_eq!(summary(&measured), "3 readings · 3 measured");
        let mut mixed = measured;
        mixed.push(log(DataSource::Interpolated));
        mixed.push(log(DataSource::Smoothed));
        assert_eq!(
            summary(&mixed),
            "5 readings · 3 measured · 1 filled in · 1 smoothed"
        );
    }
}
