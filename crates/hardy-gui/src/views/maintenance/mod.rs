//! "Maintenance": data repair and CSV export.

mod cards;

use hardy_core::{error::AppError, repair::RepairSummary};
use iced::{Element, Length, widget::row};

use crate::{app::Message, style, views::components::scroll};

/// Everything the Maintenance view shows.
#[derive(Clone, Copy)]
pub struct MaintenanceProps<'a> {
    pub repair_start: &'a str,
    pub repair_end: &'a str,
    pub repair_running: bool,
    pub repair_result: Option<&'a Result<RepairSummary, AppError>>,
    pub export_status: Option<&'a str>,
}

/// The Maintenance page.
pub fn view(props: MaintenanceProps<'_>) -> Element<'_, Message> {
    scroll(
        row![cards::repair_card(&props), cards::export_card(&props)]
            .spacing(style::SPACE_L)
            .width(Length::Fill),
    )
}
