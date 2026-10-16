//! The forecast models card: every model's benchmark error per horizon and
//! where it stands.

use chrono::{DateTime, Utc};
use hardy_core::Tz;
use hardy_ml::{benchmark::Bucket, suite::Family};
use iced::{
    Alignment, Element, Length,
    widget::{button, column, row, text},
};

use super::{ModelDataProps, caption};
use crate::{
    app::Message,
    leaderboard::{Row, Sort, SortKey, Standing, family_label},
    style,
    views::components::{badge, card_with_actions, empty_state, segmented, small_button},
};

/// Width of a number column.
const NUMBER: f32 = 76.0;
/// Shares of the remaining width: names with their family tag need more
/// room than the status.
const NAME: u16 = 5;
const STATUS: u16 = 3;

/// The family chips: every model, or one family.
fn family_filter(selected: Option<Family>) -> Element<'static, Message> {
    let mut options: Vec<(&'static str, Option<Family>)> =
        Vec::with_capacity(Family::ALL.len() + 1);
    options.push(("All", None));
    options.extend(Family::ALL.iter().map(|&f| (family_label(f), Some(f))));
    segmented(&options, selected, Message::LeaderboardFiltered)
}

/// A column title that sorts by its column; the active one shows the
/// direction.
fn sort_header(
    label: &'static str,
    key: SortKey,
    sort: Sort,
    width: Length,
    align: Alignment,
) -> Element<'static, Message> {
    let active = sort.key == key;
    let title = match (active, sort.descending) {
        (false, _) => label.to_string(),
        (true, false) => format!("{label} ↑"),
        (true, true) => format!("{label} ↓"),
    };
    button(
        text(title)
            .size(style::TEXT_CAPTION)
            .width(Length::Fill)
            .align_x(align),
    )
    .on_press(Message::LeaderboardSorted(key))
    .padding([2, 0])
    .width(width)
    .style(move |_, status| button::Style {
        text_color: if active || status == button::Status::Hovered {
            style::TEXT_PRIMARY
        } else {
            style::TEXT_TERTIARY
        },
        ..Default::default()
    })
    .into()
}

fn header(sort: Sort) -> Element<'static, Message> {
    let mut line = row![sort_header(
        "Model",
        SortKey::Name,
        sort,
        Length::FillPortion(NAME),
        Alignment::Start
    )]
    .spacing(style::SPACE_S)
    .align_y(Alignment::Center);
    for bucket in [
        Bucket::NextHour,
        Bucket::OneToThreeHours,
        Bucket::ThreeToDay,
        Bucket::Overall,
    ] {
        line = line.push(sort_header(
            bucket.label(),
            SortKey::Error(bucket),
            sort,
            Length::Fixed(NUMBER),
            Alignment::End,
        ));
    }
    line.push(sort_header(
        "In band",
        SortKey::Coverage,
        sort,
        Length::Fixed(NUMBER),
        Alignment::End,
    ))
    .push(caption("Status").width(Length::FillPortion(STATUS)))
    .into()
}

fn number(value: Option<f64>, selected: bool) -> Element<'static, Message> {
    let (label, color) = match value {
        Some(v) if selected => (format!("★ {v:.1}"), style::FORECAST),
        Some(v) => (format!("{v:.1}"), style::TEXT_PRIMARY),
        None => ("–".to_string(), style::TEXT_TERTIARY),
    };
    text(label)
        .size(style::TEXT_CAPTION)
        .color(color)
        .width(NUMBER)
        .align_x(Alignment::End)
        .into()
}

/// The status column: how much data a fitted model saw and when it was
/// last fitted, or why it is not ready.
fn standing(
    standing: &Standing,
    fitted_at: Option<DateTime<Utc>>,
    timezone: Tz,
) -> (String, iced::Color) {
    let fitted = |days: i64| match fitted_at {
        Some(at) => format!(
            "{days} d of data · fit {}",
            at.with_timezone(&timezone).format("%a %H:%M")
        ),
        None => format!("{days} d of data"),
    };
    match standing {
        Standing::Ready { days } => (fitted(*days), style::TEXT_SECONDARY),
        Standing::Learning { days, forecasts } => (
            format!("{} · {forecasts} test forecasts so far", fitted(*days)),
            style::TEXT_SECONDARY,
        ),
        Standing::Waiting(message) => (message.clone(), style::TEXT_TERTIARY),
        Standing::Failed(message) => (format!("failed: {message}"), style::DANGER),
        Standing::Pending => ("not run yet".to_string(), style::TEXT_TERTIARY),
    }
}

fn line(r: &Row, timezone: Tz) -> Element<'static, Message> {
    let is_selected = |bucket: Bucket| r.selected.contains(&bucket);
    let (status, status_color) = standing(&r.standing, r.fitted_at, timezone);
    row![
        row![
            text(r.name)
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_PRIMARY),
            text(family_label(r.family))
                .size(style::TEXT_CAPTION)
                .color(style::TEXT_TERTIARY),
        ]
        .spacing(style::SPACE_S)
        .width(Length::FillPortion(NAME)),
        number(r.mae[1], is_selected(Bucket::NextHour)),
        number(r.mae[2], is_selected(Bucket::OneToThreeHours)),
        number(r.mae[3], is_selected(Bucket::ThreeToDay)),
        number(r.mae[0], false),
        text(
            r.coverage
                .map_or("–".to_string(), |c| format!("{:.0}%", c * 100.0))
        )
        .size(style::TEXT_CAPTION)
        .color(style::TEXT_SECONDARY)
        .width(NUMBER)
        .align_x(Alignment::End),
        text(status)
            .size(style::TEXT_CAPTION)
            .color(status_color)
            .width(Length::FillPortion(STATUS)),
    ]
    .spacing(style::SPACE_S)
    .align_y(Alignment::Center)
    .into()
}

pub(super) fn card<'a>(props: &ModelDataProps<'a>) -> Element<'a, Message> {
    let actions: Element<'a, Message> = if props.run_pending {
        badge("Run requested…".to_string(), style::WARNING)
    } else {
        small_button("Run now", Message::ForecastRunRequested).into()
    };
    let board = props.leaderboard;
    let mut body = column![].spacing(style::SPACE_M);
    match (&board.period, board.finished_at) {
        (Some(period), Some(finished)) => {
            body = body.push(caption(format!(
                "Backtest {period} · run {}",
                finished.with_timezone(&props.timezone).format("%a %H:%M")
            )));
        }
        _ => {
            body = body.push(empty_state(
                "No benchmark yet. The forecaster fits and tests every model each night, and \
                 right away when a model is new; the table fills in after its first run.",
            ));
        }
    }
    let mut table = column![header(props.leaderboard_sort)].spacing(style::SPACE_XS);
    for r in board.view(props.leaderboard_sort, props.leaderboard_family) {
        table = table.push(line(r, props.timezone));
    }
    body = body
        .push(family_filter(props.leaderboard_family))
        .push(table)
        .push(caption(
            "Average error in percentage points over the backtest; click a column to sort. ★ \
             marks the model shown for that horizon. In band: share of actual values inside the \
             80 % band.",
        ));
    if let Some(error) = props.last_run_error {
        body = body.push(
            text(format!("Last run failed: {error}"))
                .size(style::TEXT_CAPTION)
                .color(style::WARNING),
        );
    }
    card_with_actions("Forecast models", actions, body)
        .width(Length::Fill)
        .into()
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn test_status_shows_data_and_last_fit_in_gym_time() {
        let tz: Tz = "Europe/Berlin".parse().unwrap_or(Tz::UTC);
        // Wednesday 21:36 UTC is 23:36 in Berlin in summer.
        let at = Utc.with_ymd_and_hms(2024, 6, 5, 21, 36, 0).single();
        let (ready, _) = standing(&Standing::Ready { days: 53 }, at, tz);
        assert_eq!(ready, "53 d of data · fit Wed 23:36");
        let (learning, _) = standing(
            &Standing::Learning {
                days: 9,
                forecasts: 10,
            },
            at,
            tz,
        );
        assert_eq!(
            learning,
            "9 d of data · fit Wed 23:36 · 10 test forecasts so far"
        );
        let (unknown, _) = standing(&Standing::Ready { days: 3 }, None, tz);
        assert_eq!(unknown, "3 d of data");
        let (failed, color) = standing(&Standing::Failed("boom".to_string()), at, tz);
        assert_eq!(failed, "failed: boom");
        assert_eq!(color, style::DANGER);
    }
}
