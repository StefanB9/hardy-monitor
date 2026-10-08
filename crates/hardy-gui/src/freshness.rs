//! Whether the newest stored reading is recent enough to show as live.

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::schedule::GymSchedule;

/// The daemon stores a reading every minute; a few missed minutes are normal.
pub(crate) const STALE_AFTER: TimeDelta = TimeDelta::minutes(5);

/// Age state of the newest reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Freshness {
    /// Recent, or the gym is closed (no readings expected).
    Live,
    /// Open, but nothing new since this instant: the daemon is not storing.
    Stale(DateTime<Utc>),
    /// Open and no reading at all.
    Missing,
}

impl Freshness {
    /// Warning for the user, `None` while readings are live.
    pub(crate) fn warning(self, tz: hardy_core::Tz) -> Option<String> {
        match self {
            Freshness::Live => None,
            Freshness::Stale(t) => Some(format!(
                "No new readings since {} – is the daemon running?",
                t.with_timezone(&tz).format("%H:%M")
            )),
            Freshness::Missing => Some("No readings yet – is the daemon running?".to_string()),
        }
    }
}

/// Judges the newest reading at `now`.
pub(crate) fn freshness(
    latest: Option<DateTime<Utc>>,
    now: DateTime<Utc>,
    schedule: &GymSchedule,
) -> Freshness {
    if !schedule.is_open(&now) {
        return Freshness::Live;
    }
    match latest {
        None => Freshness::Missing,
        Some(t) if now - t > STALE_AFTER => Freshness::Stale(t),
        Some(_) => Freshness::Live,
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeZone;

    use super::*;

    /// Monday 2024-06-17 `h:mi` gym-local.
    fn local(h: u32, mi: u32) -> Result<DateTime<Utc>> {
        Ok(GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, 17, h, mi, 0)
            .single()
            .context("valid local time")?
            .with_timezone(&Utc))
    }

    #[test]
    fn test_freshness_recent_reading_is_live() -> Result<()> {
        let s = GymSchedule::default();
        assert_eq!(
            freshness(Some(local(20, 37)?), local(20, 39)?, &s),
            Freshness::Live
        );
        Ok(())
    }

    #[test]
    fn test_freshness_old_reading_while_open_is_stale() -> Result<()> {
        let s = GymSchedule::default();
        let last = local(20, 39)?;
        assert_eq!(
            freshness(Some(last), local(22, 39)?, &s),
            Freshness::Stale(last)
        );
        Ok(())
    }

    #[test]
    fn test_freshness_missing_reading_while_open() -> Result<()> {
        let s = GymSchedule::default();
        assert_eq!(freshness(None, local(12, 0)?, &s), Freshness::Missing);
        Ok(())
    }

    #[test]
    fn test_freshness_warning_names_last_reading_in_gym_time() -> Result<()> {
        let tz = GymSchedule::default().timezone();
        assert_eq!(Freshness::Live.warning(tz), None);
        assert_eq!(
            Freshness::Stale(local(20, 39)?).warning(tz).as_deref(),
            Some("No new readings since 20:39 – is the daemon running?")
        );
        assert!(Freshness::Missing.warning(tz).is_some());
        Ok(())
    }

    #[test]
    fn test_freshness_closed_gym_is_never_stale() -> Result<()> {
        let s = GymSchedule::default();
        assert_eq!(
            freshness(Some(local(22, 59)?), local(23, 30)?, &s),
            Freshness::Live
        );
        assert_eq!(freshness(None, local(3, 0)?, &s), Freshness::Live);
        Ok(())
    }
}
