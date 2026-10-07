//! Full-window notice shown while the database schema doesn't match this
//! build; the app runs no queries until it does.

use hardy_core::db::SchemaStatus;
use iced::{
    Alignment, Element, Length,
    widget::{column, container, text},
};

use crate::{
    app::Message,
    style,
    views::components::{card, primary_button},
};

/// Where the app stands with the database schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaGate {
    Checking,
    Ready,
    Mismatch(SchemaStatus),
    /// The check itself failed (e.g. database unreachable).
    CheckFailed(String),
}

/// Title and explanation for a gate that blocks the app; `None` when ready.
pub fn notice(gate: &SchemaGate) -> Option<(&'static str, String)> {
    Some(match gate {
        SchemaGate::Ready => return None,
        SchemaGate::Checking => ("Checking the database…", String::new()),
        SchemaGate::Mismatch(SchemaStatus::DbOlder { .. } | SchemaStatus::Current) => (
            "Database not upgraded yet",
            "This version of the app needs a newer database. The daemon upgrades it: update and \
             restart the daemon, then retry."
                .to_string(),
        ),
        SchemaGate::Mismatch(SchemaStatus::DbNewer { .. }) => (
            "This app is out of date",
            "The database was upgraded by a newer version. Update this app and start it again."
                .to_string(),
        ),
        SchemaGate::CheckFailed(error) => (
            "Database not reachable",
            format!("Could not check the database: {error}"),
        ),
    })
}

pub fn view(gate: &SchemaGate) -> Element<'_, Message> {
    let (title, body) = notice(gate).unwrap_or(("", String::new()));
    let mut content = column![
        text(body)
            .size(style::TEXT_BODY)
            .color(style::TEXT_SECONDARY)
    ]
    .spacing(style::SPACE_L);
    if *gate != SchemaGate::Checking {
        content = content.push(primary_button("Retry", Message::RetrySchemaCheck));
    }
    container(card(title, content).max_width(520))
        .center(Length::Fill)
        .align_x(Alignment::Center)
        .style(|_| container::Style {
            background: Some(style::BG_APP.into()),
            ..Default::default()
        })
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notice_none_when_ready() {
        assert_eq!(notice(&SchemaGate::Ready), None);
    }

    #[test]
    fn test_notice_tells_which_program_to_update() {
        let older = notice(&SchemaGate::Mismatch(SchemaStatus::DbOlder {
            db: Some(1),
            app: 2,
        }));
        assert!(older.is_some_and(|(_, body)| body.contains("restart the daemon")));

        let newer = notice(&SchemaGate::Mismatch(SchemaStatus::DbNewer {
            db: 3,
            app: 2,
        }));
        assert!(newer.is_some_and(|(_, body)| body.contains("Update this app")));

        let failed = notice(&SchemaGate::CheckFailed("connection refused".to_string()));
        assert!(failed.is_some_and(|(_, body)| body.contains("connection refused")));

        assert!(notice(&SchemaGate::Checking).is_some());
    }
}
