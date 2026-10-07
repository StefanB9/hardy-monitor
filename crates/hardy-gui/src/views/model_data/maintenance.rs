//! Data maintenance cards: repair and CSV export.

use hardy_core::{error::AppError, repair::RepairSummary};
use iced::{
    Alignment, Element, Length,
    widget::{column, row, text},
};

use super::{ModelDataProps, caption};
use crate::{
    app::{Message, RepairPreset},
    style,
    views::components::{badge, card, date_input, primary_button, secondary_button, small_button},
};

fn repair_result(result: &Result<RepairSummary, AppError>) -> Element<'_, Message> {
    match result {
        Ok(summary) => {
            let line = |label: &'static str, value: u32| {
                row![
                    text(label)
                        .size(style::TEXT_CAPTION)
                        .color(style::TEXT_SECONDARY)
                        .width(Length::Fill),
                    text(value.to_string())
                        .size(style::TEXT_CAPTION)
                        .color(style::TEXT_PRIMARY),
                ]
            };
            column![
                text("Repair finished")
                    .size(style::TEXT_BODY)
                    .color(style::SUCCESS),
                line("Days processed", summary.days_processed),
                line("Gaps filled", summary.gaps_filled),
                line("Outliers removed", summary.records_deleted),
                line("Spikes smoothed", summary.records_smoothed),
                line(
                    "Opening/closing anchors added",
                    summary.boundary_entries_added
                ),
            ]
            .spacing(style::SPACE_XS)
            .into()
        }
        Err(e) => column![
            text("Repair failed")
                .size(style::TEXT_BODY)
                .color(style::DANGER),
            caption(e.to_string()),
        ]
        .spacing(style::SPACE_XS)
        .into(),
    }
}

pub(super) fn repair_card<'a>(props: &ModelDataProps<'a>) -> Element<'a, Message> {
    let dates = row![
        date_input(props.repair_start, Message::RepairStartDateChanged),
        caption("to"),
        date_input(props.repair_end, Message::RepairEndDateChanged),
    ]
    .spacing(style::SPACE_S)
    .align_y(Alignment::Center);

    let presets = row![
        small_button(
            "Last 7 days",
            Message::RepairPresetSelected(RepairPreset::Last7Days)
        ),
        small_button(
            "Last 30 days",
            Message::RepairPresetSelected(RepairPreset::Last30Days)
        ),
        small_button("All", Message::RepairPresetSelected(RepairPreset::AllData)),
    ]
    .spacing(style::SPACE_S);

    let start: Element<'a, Message> = if props.repair_running {
        badge("Repairing…".to_string(), style::ACCENT)
    } else {
        primary_button("Start repair", Message::StartRepairJob).into()
    };

    let mut body = column![
        text(
            "Removes readings outside opening hours, anchors opening and closing at 0%, fills \
             gaps up to 5 minutes and smooths spikes."
        )
        .size(style::TEXT_CAPTION)
        .color(style::TEXT_SECONDARY),
        dates,
        presets,
        start,
    ]
    .spacing(style::SPACE_M);
    if let Some(result) = props.repair_result {
        body = body.push(repair_result(result));
    }
    card("Data repair", body).width(Length::Fill).into()
}

pub(super) fn export_card<'a>(props: &ModelDataProps<'a>) -> Element<'a, Message> {
    let mut body = column![
        text("Every reading as CSV, saved to your Downloads folder.")
            .size(style::TEXT_CAPTION)
            .color(style::TEXT_SECONDARY),
        secondary_button("Export CSV", Message::ExportCsv),
    ]
    .spacing(style::SPACE_M);
    if let Some(status) = props.export_status {
        body = body.push(caption(status.to_string()));
    }
    card("Export", body).width(Length::Fill).into()
}
