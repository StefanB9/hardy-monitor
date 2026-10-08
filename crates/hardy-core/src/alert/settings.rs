//! Alert settings shared through the database by the daemon, the GUI and
//! phone commands.

use std::fmt;

use chrono::{DateTime, SubsecRound, TimeDelta, Utc};

use crate::{error::AppError, schedule::GymSchedule};

/// Who last changed the alert settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "text", rename_all = "lowercase")]
pub enum SettingsSource {
    /// Default row created by the migration.
    Migration,
    /// The desktop GUI.
    Gui,
    /// A command sent from the phone via the ntfy control topic.
    Phone,
}

impl SettingsSource {
    /// The value stored in the `updated_by` column.
    pub fn as_str(self) -> &'static str {
        match self {
            SettingsSource::Migration => "migration",
            SettingsSource::Gui => "gui",
            SettingsSource::Phone => "phone",
        }
    }
}

/// Longest finite arming duration accepted, in hours.
pub const MAX_ARM_HOURS: u8 = 12;

/// How long alerts stay armed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertDuration {
    /// Until the gym's next closing time.
    UntilClosing,
    /// For a number of hours (`1..=MAX_ARM_HOURS`).
    Hours(u8),
    /// Until switched off.
    Always,
}

impl AlertDuration {
    /// Validated `Hours` duration.
    pub fn hours(hours: u8) -> Result<Self, AppError> {
        if (1..=MAX_ARM_HOURS).contains(&hours) {
            Ok(AlertDuration::Hours(hours))
        } else {
            Err(AppError::validation(format!(
                "duration must be 1–{MAX_ARM_HOURS} hours, got {hours}"
            )))
        }
    }

    /// The expiry instant when armed at `now`; `None` means no expiry.
    pub fn expiry(self, now: DateTime<Utc>, schedule: &GymSchedule) -> Option<DateTime<Utc>> {
        match self {
            AlertDuration::UntilClosing => Some(schedule.next_closing_after(now)),
            AlertDuration::Hours(h) => Some(now + TimeDelta::hours(i64::from(h))),
            AlertDuration::Always => None,
        }
    }
}

impl fmt::Display for AlertDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlertDuration::UntilClosing => f.write_str("closing"),
            AlertDuration::Hours(h) => write!(f, "{h}h"),
            AlertDuration::Always => f.write_str("always"),
        }
    }
}

/// The single row of alert settings.
#[derive(Debug, Clone, PartialEq)]
pub struct AlertSettings {
    enabled: bool,
    threshold_percent: f64,
    active_until: Option<DateTime<Utc>>,
    updated_at: DateTime<Utc>,
    updated_by: SettingsSource,
}

impl AlertSettings {
    /// Settings as stored; validates the threshold.
    pub fn new(
        enabled: bool,
        threshold_percent: f64,
        active_until: Option<DateTime<Utc>>,
        updated_at: DateTime<Utc>,
        updated_by: SettingsSource,
    ) -> Result<Self, AppError> {
        validate_threshold(threshold_percent)?;
        Ok(Self {
            enabled,
            threshold_percent,
            active_until: active_until.map(to_db_precision),
            updated_at: to_db_precision(updated_at),
            updated_by,
        })
    }

    /// Settings that arm alerts below `threshold_percent` for `duration`.
    pub fn armed(
        threshold_percent: f64,
        duration: AlertDuration,
        now: DateTime<Utc>,
        schedule: &GymSchedule,
        updated_by: SettingsSource,
    ) -> Result<Self, AppError> {
        Self::new(
            true,
            threshold_percent,
            duration.expiry(now, schedule),
            now,
            updated_by,
        )
    }

    /// These settings switched off, keeping the threshold.
    #[must_use]
    pub fn disarmed(&self, now: DateTime<Utc>, updated_by: SettingsSource) -> Self {
        Self {
            enabled: false,
            active_until: None,
            updated_at: to_db_precision(now),
            updated_by,
            ..*self
        }
    }

    /// These settings with a new threshold, keeping the armed state and
    /// expiry.
    pub fn with_threshold(
        &self,
        threshold_percent: f64,
        now: DateTime<Utc>,
        updated_by: SettingsSource,
    ) -> Result<Self, AppError> {
        validate_threshold(threshold_percent)?;
        Ok(Self {
            threshold_percent,
            updated_at: to_db_precision(now),
            updated_by,
            ..*self
        })
    }

    /// Whether alerts are armed and not yet expired at `now`.
    pub fn is_active(&self, now: DateTime<Utc>) -> bool {
        self.enabled && self.active_until.is_none_or(|until| now < until)
    }

    /// Whether alerts are armed (until [`Self::active_until`], if set).
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Occupancy below which an alert fires.
    pub fn threshold_percent(&self) -> f64 {
        self.threshold_percent
    }

    /// When the armed state ends; `None` means no end.
    pub fn active_until(&self) -> Option<DateTime<Utc>> {
        self.active_until
    }

    /// When the settings were last changed.
    pub fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }

    /// Who last changed the settings.
    pub fn updated_by(&self) -> SettingsSource {
        self.updated_by
    }

    /// One-line human description in the gym's timezone, e.g.
    /// "On below 25% until 21:00".
    pub fn describe(&self, now: DateTime<Utc>, schedule: &GymSchedule) -> String {
        if !self.is_active(now) {
            return format!("Off (threshold {:.0}%)", self.threshold_percent);
        }
        match self.active_until {
            Some(until) => format!(
                "On below {:.0}% until {}",
                self.threshold_percent,
                until.with_timezone(&schedule.timezone()).format("%H:%M")
            ),
            None => format!("On below {:.0}% (no expiry)", self.threshold_percent),
        }
    }
}

/// Postgres stores microseconds. Settings use `updated_at` as a version, so
/// it must survive a database round-trip unchanged.
fn to_db_precision(t: DateTime<Utc>) -> DateTime<Utc> {
    t.trunc_subsecs(6)
}

fn validate_threshold(threshold_percent: f64) -> Result<(), AppError> {
    if (0.0..=100.0).contains(&threshold_percent) {
        Ok(())
    } else {
        Err(AppError::validation(format!(
            "alert threshold must be within 0–100%, got {threshold_percent}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeZone;

    use super::*;

    fn utc(h: u32, mi: u32) -> Result<DateTime<Utc>> {
        // Monday 2024-06-17, CEST = UTC+2.
        Utc.with_ymd_and_hms(2024, 6, 17, h, mi, 0)
            .single()
            .context("valid time")
    }

    #[test]
    fn test_alert_settings_rejects_invalid_threshold() -> Result<()> {
        for t in [-1.0, 100.5, f64::NAN] {
            let result = AlertSettings::new(true, t, None, utc(8, 0)?, SettingsSource::Gui);
            assert!(result.is_err(), "{t} must be rejected");
        }
        Ok(())
    }

    #[test]
    fn test_alert_duration_hours_validates_range() {
        assert!(AlertDuration::hours(0).is_err());
        assert!(AlertDuration::hours(MAX_ARM_HOURS + 1).is_err());
        assert_eq!(AlertDuration::hours(2).ok(), Some(AlertDuration::Hours(2)));
    }

    #[test]
    fn test_armed_until_closing_expires_at_gym_closing() -> Result<()> {
        let schedule = GymSchedule::default();
        let s = AlertSettings::armed(
            25.0,
            AlertDuration::UntilClosing,
            utc(8, 0)?,
            &schedule,
            SettingsSource::Phone,
        )?;
        // Closing 23:00 CEST = 21:00 UTC.
        assert_eq!(s.active_until(), Some(utc(21, 0)?));
        assert!(s.is_active(utc(20, 59)?));
        assert!(!s.is_active(utc(21, 0)?));
        Ok(())
    }

    #[test]
    fn test_armed_for_hours_and_always() -> Result<()> {
        let schedule = GymSchedule::default();
        let two_h = AlertSettings::armed(
            25.0,
            AlertDuration::Hours(2),
            utc(8, 0)?,
            &schedule,
            SettingsSource::Gui,
        )?;
        assert!(two_h.is_active(utc(9, 59)?));
        assert!(!two_h.is_active(utc(10, 0)?));

        let always = AlertSettings::armed(
            25.0,
            AlertDuration::Always,
            utc(8, 0)?,
            &schedule,
            SettingsSource::Gui,
        )?;
        assert_eq!(always.active_until(), None);
        assert!(always.is_active(utc(23, 59)?));
        Ok(())
    }

    #[test]
    fn test_disarmed_keeps_threshold_and_is_inactive() -> Result<()> {
        let schedule = GymSchedule::default();
        let armed = AlertSettings::armed(
            25.0,
            AlertDuration::Always,
            utc(8, 0)?,
            &schedule,
            SettingsSource::Gui,
        )?;
        let off = armed.disarmed(utc(9, 0)?, SettingsSource::Phone);
        assert!(!off.is_active(utc(9, 0)?));
        assert_eq!(off.threshold_percent(), 25.0);
        assert_eq!(off.updated_by(), SettingsSource::Phone);
        Ok(())
    }

    #[test]
    fn test_with_threshold_keeps_armed_state() -> Result<()> {
        let schedule = GymSchedule::default();
        let armed = AlertSettings::armed(
            25.0,
            AlertDuration::UntilClosing,
            utc(8, 0)?,
            &schedule,
            SettingsSource::Gui,
        )?;
        let changed = armed.with_threshold(40.0, utc(9, 0)?, SettingsSource::Gui)?;
        assert!(changed.is_active(utc(9, 0)?));
        assert_eq!(changed.active_until(), armed.active_until());
        assert_eq!(changed.threshold_percent(), 40.0);
        assert!(
            armed
                .with_threshold(101.0, utc(9, 0)?, SettingsSource::Gui)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn test_describe_uses_gym_time() -> Result<()> {
        let schedule = GymSchedule::default();
        let armed = AlertSettings::armed(
            25.0,
            AlertDuration::UntilClosing,
            utc(8, 0)?,
            &schedule,
            SettingsSource::Gui,
        )?;
        assert_eq!(
            armed.describe(utc(8, 0)?, &schedule),
            "On below 25% until 23:00"
        );
        assert_eq!(
            armed.describe(utc(21, 0)?, &schedule),
            "Off (threshold 25%)"
        );
        Ok(())
    }

    #[test]
    fn test_settings_timestamps_use_database_precision() -> Result<()> {
        let now = utc(8, 0)? + chrono::Duration::nanoseconds(123_456_789);
        let s = AlertSettings::new(true, 25.0, Some(now), now, SettingsSource::Gui)?;
        assert_eq!(s.updated_at().timestamp_subsec_nanos(), 123_456_000);
        assert_eq!(
            s.active_until().map(|t| t.timestamp_subsec_nanos()),
            Some(123_456_000)
        );
        let off = s.disarmed(now, SettingsSource::Gui);
        assert_eq!(off.updated_at().timestamp_subsec_nanos(), 123_456_000);
        Ok(())
    }

    #[test]
    fn test_settings_source_as_str() {
        assert_eq!(SettingsSource::Migration.as_str(), "migration");
        assert_eq!(SettingsSource::Gui.as_str(), "gui");
        assert_eq!(SettingsSource::Phone.as_str(), "phone");
    }
}
