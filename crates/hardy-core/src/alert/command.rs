//! Commands sent from the phone to the ntfy control topic.
//!
//! Grammar (case-insensitive, any token order after `on`):
//! `on [<0-100>[%]] [closing | always | <1-12>h]`, `off`, `status`.

use std::fmt;

use super::settings::AlertDuration;
use crate::error::AppError;

/// A parsed control command.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlCommand {
    /// Arm alerts; `threshold_percent` `None` keeps the current threshold.
    On {
        threshold_percent: Option<f64>,
        duration: AlertDuration,
    },
    Off,
    Status,
}

/// Reply sent for text that is not a command.
pub const USAGE: &str = "Commands: on [25] [2h|always|closing], off, status";

impl ControlCommand {
    /// Parses a control message.
    pub fn parse(text: &str) -> Result<Self, AppError> {
        let lowered = text.trim().to_lowercase();
        let mut tokens = lowered.split_whitespace();
        let command = match tokens.next() {
            Some("on") => {
                let mut threshold_percent = None;
                let mut duration = None;
                for token in tokens {
                    if let Some(parsed) = parse_duration(token)? {
                        set_once(&mut duration, parsed, "duration")?;
                    } else {
                        set_once(&mut threshold_percent, parse_threshold(token)?, "threshold")?;
                    }
                }
                return Ok(ControlCommand::On {
                    threshold_percent,
                    duration: duration.unwrap_or(AlertDuration::UntilClosing),
                });
            }
            Some("off") => ControlCommand::Off,
            Some("status") => ControlCommand::Status,
            _ => return Err(AppError::validation(format!("unknown command {text:?}"))),
        };
        match tokens.next() {
            None => Ok(command),
            Some(extra) => Err(AppError::validation(format!(
                "unexpected argument {extra:?}"
            ))),
        }
    }
}

impl fmt::Display for ControlCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ControlCommand::On {
                threshold_percent,
                duration,
            } => {
                f.write_str("on")?;
                if let Some(t) = threshold_percent {
                    write!(f, " {t}")?;
                }
                write!(f, " {duration}")
            }
            ControlCommand::Off => f.write_str("off"),
            ControlCommand::Status => f.write_str("status"),
        }
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, what: &str) -> Result<(), AppError> {
    if slot.replace(value).is_some() {
        Err(AppError::validation(format!("{what} given twice")))
    } else {
        Ok(())
    }
}

/// `Ok(None)` when the token is not a duration at all.
fn parse_duration(token: &str) -> Result<Option<AlertDuration>, AppError> {
    match token {
        "closing" | "close" => Ok(Some(AlertDuration::UntilClosing)),
        "always" => Ok(Some(AlertDuration::Always)),
        _ => match token.strip_suffix('h') {
            Some(hours) => {
                let hours: u8 = hours
                    .parse()
                    .map_err(|_| AppError::validation(format!("invalid duration {token:?}")))?;
                AlertDuration::hours(hours).map(Some)
            }
            None => Ok(None),
        },
    }
}

fn parse_threshold(token: &str) -> Result<f64, AppError> {
    let number = token.strip_suffix('%').unwrap_or(token);
    let value: f64 = number
        .parse()
        .map_err(|_| AppError::validation(format!("invalid threshold {token:?}")))?;
    if (0.0..=100.0).contains(&value) {
        Ok(value)
    } else {
        Err(AppError::validation(format!(
            "threshold must be within 0–100%, got {token:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use proptest::prelude::*;

    use super::*;
    use crate::alert::settings::MAX_ARM_HOURS;

    fn on(threshold_percent: Option<f64>, duration: AlertDuration) -> ControlCommand {
        ControlCommand::On {
            threshold_percent,
            duration,
        }
    }

    #[test]
    fn test_control_command_parses_valid_forms() -> Result<()> {
        let cases = [
            ("on", on(None, AlertDuration::UntilClosing)),
            ("ON", on(None, AlertDuration::UntilClosing)),
            ("on 25", on(Some(25.0), AlertDuration::UntilClosing)),
            ("on 25%", on(Some(25.0), AlertDuration::UntilClosing)),
            ("on 25 2h", on(Some(25.0), AlertDuration::Hours(2))),
            ("on 2h 25", on(Some(25.0), AlertDuration::Hours(2))),
            ("on always", on(None, AlertDuration::Always)),
            (
                "  on   17.5   closing ",
                on(Some(17.5), AlertDuration::UntilClosing),
            ),
            ("off", ControlCommand::Off),
            ("Status", ControlCommand::Status),
        ];
        for (text, expected) in cases {
            assert_eq!(ControlCommand::parse(text)?, expected, "{text:?}");
        }
        Ok(())
    }

    #[test]
    fn test_control_command_rejects_invalid_input() {
        for text in [
            "",
            "hello",
            "on 120",
            "on -5",
            "on 0h",
            "on 13h",
            "on xh",
            "on 25 30",
            "on 2h always",
            "off now",
            "status please",
            "on banana",
        ] {
            assert!(
                matches!(ControlCommand::parse(text), Err(AppError::Validation(_))),
                "{text:?} should be rejected"
            );
        }
    }

    fn arb_duration() -> impl Strategy<Value = AlertDuration> {
        prop_oneof![
            Just(AlertDuration::UntilClosing),
            Just(AlertDuration::Always),
            (1..=MAX_ARM_HOURS).prop_map(AlertDuration::Hours),
        ]
    }

    fn arb_command() -> impl Strategy<Value = ControlCommand> {
        prop_oneof![
            Just(ControlCommand::Off),
            Just(ControlCommand::Status),
            (prop::option::of(0u32..=100), arb_duration()).prop_map(|(t, duration)| {
                ControlCommand::On {
                    threshold_percent: t.map(f64::from),
                    duration,
                }
            }),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn control_command_display_round_trips(command in arb_command()) {
            let parsed = ControlCommand::parse(&command.to_string())
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert_eq!(parsed, command);
        }

        #[test]
        fn control_command_parse_never_panics(text in ".{0,40}") {
            let _ = ControlCommand::parse(&text);
        }
    }
}
