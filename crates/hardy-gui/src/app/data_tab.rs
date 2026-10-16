//! The Data tab: loading the readings of its period, choosing and stepping
//! the period, and tracking the table's scroll position.

use hardy_core::{db::OccupancyLog, error::AppError};
use iced::{
    Task,
    widget::{Id, operation},
};

use super::{HardyMonitorApp, Message, RequestId, loads::ErrorSource, tasks::db_err};
use crate::time_range::{Step, can_step_back};

/// The Data tab's scrollable table, reset to the top for a new period.
pub(crate) const TABLE_ID: &str = "data-table";

impl HardyMonitorApp {
    /// `[start, end)` of the Data tab's period.
    pub(super) fn data_window(
        &self,
    ) -> (chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>) {
        self.ui.data_period.window(
            self.clock.now_utc(),
            self.schedule.timezone(),
            self.ui.data_offset,
        )
    }

    /// Loads the readings of the Data tab's period.
    pub(super) fn load_data_table(&mut self) -> Task<Message> {
        let (start, end) = self.data_window();
        let db = self.db.clone();
        let id = self.ui.data_requests.issue();
        // The query's end is inclusive; the window's is not.
        let end = end - chrono::TimeDelta::microseconds(1);
        self.load(
            async move { db.get_history_range(start, end).await },
            move |r: anyhow::Result<Vec<OccupancyLog>>| {
                Message::DataLoaded(id, r.map_err(db_err("load_data_table")))
            },
        )
    }

    /// Period choice, scrolling and loaded readings of the Data tab.
    pub(super) fn update_data(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::DataPeriodSelected(period) => {
                self.ui.data_period = period;
                self.ui.data_offset = 0;
                self.show_new_period()
            }
            Message::DataScrolled { offset, height } => {
                self.ui.data_scroll = (offset, height);
                Task::none()
            }
            Message::DataLoaded(id, result) => {
                self.apply_data_rows(id, result);
                Task::none()
            }
            _ => Task::none(),
        }
    }

    /// Shows the previous or next period, within the data and up to the
    /// current one.
    pub(super) fn step_data(&mut self, step: Step) -> Task<Message> {
        let (start, _) = self.data_window();
        self.ui.data_offset = match step {
            Step::Back if can_step_back(start, self.data.first_reading) => {
                self.ui.data_offset.saturating_add(1)
            }
            Step::Forward if self.ui.data_offset > 0 => self.ui.data_offset - 1,
            _ => return Task::none(),
        };
        self.show_new_period()
    }

    fn show_new_period(&mut self) -> Task<Message> {
        self.ui.data_scroll.0 = 0.0;
        Task::batch([
            operation::snap_to(Id::new(TABLE_ID), operation::RelativeOffset::START),
            self.load_data_table(),
        ])
    }

    fn apply_data_rows(&mut self, id: RequestId, result: Result<Vec<OccupancyLog>, AppError>) {
        // A newer load for another period is on its way.
        if self.ui.data_requests.is_current(id)
            && let Some(rows) = self.errors.record(ErrorSource::DataTable, result)
        {
            self.data.data_rows = rows;
        }
    }
}
