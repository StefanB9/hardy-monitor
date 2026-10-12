//! "Model & Data": the forecast models and their benchmark, live accuracy,
//! data repair and CSV export.

mod accuracy;
mod leaderboard;
mod maintenance;

pub(crate) use accuracy::DAYS as ACCURACY_DAYS;
use hardy_core::{Tz, accuracy::AccuracySummary, error::AppError, repair::RepairSummary};
use iced::{
    Element, Length,
    widget::{column, row, text},
};

use crate::{app::Message, leaderboard::Leaderboard, style, views::components::scroll};

/// Everything the Model & Data view shows.
#[derive(Clone, Copy)]
pub struct ModelDataProps<'a> {
    pub timezone: Tz,
    pub leaderboard: &'a Leaderboard,
    /// Logged forecasts scored against readings; `None` before any.
    pub accuracy: Option<&'a AccuracySummary>,
    /// A run was requested and the daemon has not started it yet.
    pub run_pending: bool,
    /// Why the daemon's last forecast run failed.
    pub last_run_error: Option<&'a str>,
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

pub(super) fn caption(s: impl text::IntoFragment<'static>) -> text::Text<'static> {
    text(s)
        .size(style::TEXT_CAPTION)
        .color(style::TEXT_TERTIARY)
}

/// The Model & Data page.
pub fn view(props: ModelDataProps<'_>) -> Element<'_, Message> {
    // The leaderboard is a wide table: it gets the full width.
    let content = column![
        leaderboard::card(&props),
        row![
            column![accuracy::card(props.accuracy)]
                .spacing(style::SPACE_L)
                .width(Length::FillPortion(3)),
            column![
                maintenance::repair_card(&props),
                maintenance::export_card(&props)
            ]
            .spacing(style::SPACE_L)
            .width(Length::FillPortion(2)),
        ]
        .spacing(style::SPACE_L),
    ]
    .spacing(style::SPACE_L);

    scroll(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_improvement_color_bands() {
        assert_eq!(improvement_color(0.25), style::SUCCESS);
        assert_eq!(improvement_color(0.05), style::WARNING);
        assert_eq!(improvement_color(-0.1), style::DANGER);
    }
}
