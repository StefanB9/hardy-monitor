//! Times of the week in which alerts may fire, configured in gym-local time.

use chrono::{DateTime, Datelike, NaiveTime};
use chrono_tz::Tz;
use serde::{Deserialize, Deserializer, de};

use crate::{error::AppError, schedule::is_bavarian_holiday};

/// Which days an [`AlertWindow`] applies to. Bavarian holidays count as
/// weekend days, matching the opening schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WindowDays {
    Daily,
    Weekdays,
    Weekends,
}

/// A daily time range, e.g. weekdays 16:00–21:00 (start inclusive, end
/// exclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct AlertWindow {
    days: WindowDays,
    #[serde(deserialize_with = "deserialize_hh_mm")]
    start: NaiveTime,
    #[serde(deserialize_with = "deserialize_hh_mm")]
    end: NaiveTime,
}

impl AlertWindow {
    /// A validated window; `start` must be before `end`.
    pub fn new(days: WindowDays, start: NaiveTime, end: NaiveTime) -> Result<Self, AppError> {
        let window = Self { days, start, end };
        window.validate()?;
        Ok(window)
    }

    /// Rejects empty or inverted ranges (also for windows read from config).
    pub fn validate(&self) -> Result<(), AppError> {
        if self.start < self.end {
            Ok(())
        } else {
            Err(AppError::Config(format!(
                "alert window start {} must be before end {}",
                self.start.format("%H:%M"),
                self.end.format("%H:%M")
            )))
        }
    }

    /// Whether the gym-local time falls inside this window.
    pub fn contains(&self, local: &DateTime<Tz>) -> bool {
        let date = local.date_naive();
        let weekend = is_bavarian_holiday(date) || date.weekday().number_from_monday() > 5;
        let day_matches = match self.days {
            WindowDays::Daily => true,
            WindowDays::Weekdays => !weekend,
            WindowDays::Weekends => weekend,
        };
        let time = local.time();
        day_matches && self.start <= time && time < self.end
    }
}

fn deserialize_hh_mm<'de, D: Deserializer<'de>>(deserializer: D) -> Result<NaiveTime, D::Error> {
    let text = String::deserialize(deserializer)?;
    NaiveTime::parse_from_str(&text, "%H:%M")
        .map_err(|e| de::Error::custom(format!("expected HH:MM, got {text:?}: {e}")))
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeZone;
    use chrono_tz::Europe::Berlin;

    use super::*;

    fn hm(h: u32, m: u32) -> Result<NaiveTime> {
        NaiveTime::from_hms_opt(h, m, 0).context("valid time")
    }

    fn berlin(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> Result<DateTime<Tz>> {
        Berlin
            .with_ymd_and_hms(y, mo, d, h, mi, 0)
            .single()
            .context("valid local time")
    }

    #[test]
    fn test_alert_window_rejects_inverted_range() -> Result<()> {
        assert!(AlertWindow::new(WindowDays::Daily, hm(21, 0)?, hm(16, 0)?).is_err());
        assert!(AlertWindow::new(WindowDays::Daily, hm(16, 0)?, hm(16, 0)?).is_err());
        Ok(())
    }

    #[test]
    fn test_alert_window_bounds_are_start_inclusive_end_exclusive() -> Result<()> {
        let w = AlertWindow::new(WindowDays::Daily, hm(16, 0)?, hm(21, 0)?)?;
        assert!(!w.contains(&berlin(2024, 6, 17, 15, 59)?));
        assert!(w.contains(&berlin(2024, 6, 17, 16, 0)?));
        assert!(w.contains(&berlin(2024, 6, 17, 20, 59)?));
        assert!(!w.contains(&berlin(2024, 6, 17, 21, 0)?));
        Ok(())
    }

    #[test]
    fn test_alert_window_days_and_holidays() -> Result<()> {
        let weekdays = AlertWindow::new(WindowDays::Weekdays, hm(10, 0)?, hm(12, 0)?)?;
        let weekends = AlertWindow::new(WindowDays::Weekends, hm(10, 0)?, hm(12, 0)?)?;
        let monday = berlin(2024, 6, 17, 11, 0)?;
        let saturday = berlin(2024, 6, 15, 11, 0)?;
        // Wednesday 2024-12-25 is a holiday → counts as weekend.
        let christmas = berlin(2024, 12, 25, 11, 0)?;
        assert!(weekdays.contains(&monday));
        assert!(!weekdays.contains(&saturday));
        assert!(!weekdays.contains(&christmas));
        assert!(weekends.contains(&saturday));
        assert!(weekends.contains(&christmas));
        Ok(())
    }

    #[test]
    fn test_alert_window_deserializes_from_toml() -> Result<()> {
        #[derive(Deserialize)]
        struct Wrapper {
            windows: Vec<AlertWindow>,
        }
        let parsed: Wrapper =
            toml::from_str(r#"windows = [{ days = "weekdays", start = "16:00", end = "21:30" }]"#)?;
        assert_eq!(
            parsed.windows,
            vec![AlertWindow::new(
                WindowDays::Weekdays,
                hm(16, 0)?,
                hm(21, 30)?
            )?]
        );

        let bad: Result<Wrapper, _> =
            toml::from_str(r#"windows = [{ days = "weekdays", start = "4pm", end = "21:00" }]"#);
        assert!(bad.is_err());
        Ok(())
    }
}
