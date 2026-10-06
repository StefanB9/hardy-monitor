//! Decides when a low-occupancy alert fires. Pure: time comes in as an
//! argument, so the rules are fully testable.

use chrono::{DateTime, TimeDelta, Utc};

use super::{settings::AlertSettings, window::AlertWindow};
use crate::{error::AppError, schedule::GymSchedule};

/// Static rules from configuration.
#[derive(Debug, Clone)]
pub struct AlertRules {
    cooldown: TimeDelta,
    opening_grace: TimeDelta,
    windows: Vec<AlertWindow>,
}

impl AlertRules {
    /// `windows` empty means "any time the gym is open".
    pub fn new(
        cooldown_secs: u64,
        opening_grace_minutes: u32,
        windows: Vec<AlertWindow>,
    ) -> Result<Self, AppError> {
        for window in &windows {
            window.validate()?;
        }
        let cooldown_secs = i64::try_from(cooldown_secs)
            .map_err(|_| AppError::Config(format!("cooldown_secs too large: {cooldown_secs}")))?;
        Ok(Self {
            cooldown: TimeDelta::seconds(cooldown_secs),
            opening_grace: TimeDelta::minutes(i64::from(opening_grace_minutes)),
            windows,
        })
    }

    /// Whether alerts may fire at `now`: gym open, past the opening grace
    /// period and inside a configured window (if any).
    pub fn allows(&self, now: DateTime<Utc>, schedule: &GymSchedule) -> bool {
        if !schedule.is_open(&now) {
            return false;
        }
        let local = now.with_timezone(&schedule.timezone());
        if now < schedule.opening_time_on(local.date_naive()) + self.opening_grace {
            return false;
        }
        self.windows.is_empty() || self.windows.iter().any(|w| w.contains(&local))
    }
}

/// A low-occupancy alert to deliver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Alert {
    pub percentage: f64,
    pub threshold_percent: f64,
}

impl Alert {
    pub const TITLE: &'static str = "Hardy's Gym Monitor";

    /// Notification body, e.g. "Quiet now: 18% (below 25%)".
    pub fn body(&self) -> String {
        format!(
            "Quiet now: {:.0}% (below {:.0}%)",
            self.percentage, self.threshold_percent
        )
    }
}

/// Tracks one occupancy "dip" below the threshold and fires at most one
/// alert per dip, respecting the cooldown.
///
/// Changing the settings (arming, new threshold) starts a fresh dip, so
/// arming while it is already quiet alerts right away.
#[derive(Debug, Clone)]
pub struct AlertEngine {
    rules: AlertRules,
    alerted_in_dip: bool,
    settings_version: Option<DateTime<Utc>>,
    last_alert_at: Option<DateTime<Utc>>,
}

impl AlertEngine {
    pub fn new(rules: AlertRules) -> Self {
        Self {
            rules,
            alerted_in_dip: false,
            settings_version: None,
            last_alert_at: None,
        }
    }

    /// Feeds one reading; returns the alert to send, if any.
    pub fn observe(
        &mut self,
        percentage: f64,
        now: DateTime<Utc>,
        settings: &AlertSettings,
        schedule: &GymSchedule,
    ) -> Option<Alert> {
        if self.settings_version != Some(settings.updated_at()) {
            self.settings_version = Some(settings.updated_at());
            self.alerted_in_dip = false;
        }

        let threshold_percent = settings.threshold_percent();
        if percentage >= threshold_percent {
            self.alerted_in_dip = false;
            return None;
        }

        let cooling_down = self
            .last_alert_at
            .is_some_and(|last| now - last < self.rules.cooldown);
        if self.alerted_in_dip
            || cooling_down
            || !settings.is_active(now)
            || !self.rules.allows(now, schedule)
        {
            return None;
        }

        self.alerted_in_dip = true;
        self.last_alert_at = Some(now);
        Some(Alert {
            percentage,
            threshold_percent,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::{NaiveTime, TimeZone};
    use proptest::prelude::*;

    use super::*;
    use crate::alert::{
        settings::{AlertDuration, SettingsSource},
        window::WindowDays,
    };

    /// Monday 2024-06-17 at `h:mi` CEST, as UTC.
    fn local(h: u32, mi: u32) -> Result<DateTime<Utc>> {
        chrono_tz::Europe::Berlin
            .with_ymd_and_hms(2024, 6, 17, h, mi, 0)
            .single()
            .map(|t| t.with_timezone(&Utc))
            .context("valid local time")
    }

    fn armed(threshold: f64, at: DateTime<Utc>) -> Result<AlertSettings> {
        Ok(AlertSettings::armed(
            threshold,
            AlertDuration::Always,
            at,
            &GymSchedule::default(),
            SettingsSource::Gui,
        )?)
    }

    fn engine(cooldown_secs: u64, grace_minutes: u32) -> Result<AlertEngine> {
        Ok(AlertEngine::new(AlertRules::new(
            cooldown_secs,
            grace_minutes,
            Vec::new(),
        )?))
    }

    #[test]
    fn test_alert_fires_on_crossing_below_threshold() -> Result<()> {
        let schedule = GymSchedule::default();
        let settings = armed(30.0, local(9, 0)?)?;
        let mut e = engine(0, 60)?;
        assert_eq!(e.observe(45.0, local(10, 0)?, &settings, &schedule), None);
        let alert = e
            .observe(25.0, local(10, 1)?, &settings, &schedule)
            .context("crossing should alert")?;
        assert_eq!(alert.body(), "Quiet now: 25% (below 30%)");
        Ok(())
    }

    #[test]
    fn test_alert_fires_once_per_dip() -> Result<()> {
        let schedule = GymSchedule::default();
        let settings = armed(30.0, local(9, 0)?)?;
        let mut e = engine(0, 60)?;
        assert!(
            e.observe(25.0, local(10, 0)?, &settings, &schedule)
                .is_some()
        );
        assert!(
            e.observe(20.0, local(10, 1)?, &settings, &schedule)
                .is_none()
        );
        assert!(
            e.observe(35.0, local(10, 2)?, &settings, &schedule)
                .is_none()
        );
        assert!(
            e.observe(28.0, local(10, 3)?, &settings, &schedule)
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn test_alert_respects_cooldown_across_dips() -> Result<()> {
        let schedule = GymSchedule::default();
        let settings = armed(30.0, local(9, 0)?)?;
        let mut e = engine(300, 60)?;
        assert!(
            e.observe(25.0, local(10, 0)?, &settings, &schedule)
                .is_some()
        );
        assert!(
            e.observe(35.0, local(10, 1)?, &settings, &schedule)
                .is_none()
        );
        assert!(
            e.observe(25.0, local(10, 2)?, &settings, &schedule)
                .is_none()
        );
        // Still in the same dip once the cooldown has passed → alerts.
        assert!(
            e.observe(25.0, local(10, 5)?, &settings, &schedule)
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn test_arming_while_quiet_alerts_immediately() -> Result<()> {
        let schedule = GymSchedule::default();
        let off = armed(30.0, local(9, 0)?)?.disarmed(local(9, 0)?, SettingsSource::Gui);
        let mut e = engine(0, 60)?;
        assert!(e.observe(20.0, local(10, 0)?, &off, &schedule).is_none());
        let on = armed(30.0, local(10, 1)?)?;
        assert!(e.observe(20.0, local(10, 1)?, &on, &schedule).is_some());
        Ok(())
    }

    #[test]
    fn test_no_alert_when_expired_or_off() -> Result<()> {
        let schedule = GymSchedule::default();
        let two_hours = AlertSettings::armed(
            30.0,
            AlertDuration::Hours(2),
            local(9, 0)?,
            &schedule,
            SettingsSource::Phone,
        )?;
        let mut e = engine(0, 0)?;
        assert!(
            e.observe(20.0, local(11, 0)?, &two_hours, &schedule)
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn test_no_alert_during_opening_grace() -> Result<()> {
        let schedule = GymSchedule::default();
        let settings = armed(30.0, local(5, 0)?)?;
        let mut e = engine(0, 60)?;
        // Opens 06:00; grace until 07:00.
        assert!(e.observe(0.0, local(6, 0)?, &settings, &schedule).is_none());
        assert!(
            e.observe(5.0, local(6, 59)?, &settings, &schedule)
                .is_none()
        );
        assert!(e.observe(5.0, local(7, 0)?, &settings, &schedule).is_some());
        Ok(())
    }

    #[test]
    fn test_no_alert_when_closed() -> Result<()> {
        let schedule = GymSchedule::default();
        let settings = armed(30.0, local(5, 0)?)?;
        let mut e = engine(0, 0)?;
        assert!(
            e.observe(0.0, local(23, 30)?, &settings, &schedule)
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn test_alert_only_inside_windows() -> Result<()> {
        let schedule = GymSchedule::default();
        let settings = armed(30.0, local(5, 0)?)?;
        let window = AlertWindow::new(
            WindowDays::Weekdays,
            NaiveTime::from_hms_opt(16, 0, 0).context("time")?,
            NaiveTime::from_hms_opt(21, 0, 0).context("time")?,
        )?;
        let mut e = AlertEngine::new(AlertRules::new(0, 0, vec![window])?);
        assert!(
            e.observe(10.0, local(15, 59)?, &settings, &schedule)
                .is_none()
        );
        // Still quiet when the window opens → one alert.
        assert!(
            e.observe(10.0, local(16, 0)?, &settings, &schedule)
                .is_some()
        );
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn alerts_are_rare_and_valid(
            readings in prop::collection::vec((1i64..30, 0.0f64..100.0), 1..120),
            threshold in 0.0f64..100.0,
            cooldown_secs in 0u64..3600,
        ) {
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);
            let start = Utc.with_ymd_and_hms(2024, 6, 17, 0, 0, 0).single()
                .ok_or_else(|| TestCaseError::fail("valid start"))?;
            let settings = AlertSettings::armed(
                threshold, AlertDuration::Always, start, &schedule, SettingsSource::Gui,
            ).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let rules = AlertRules::new(cooldown_secs, 0, Vec::new())
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let mut engine = AlertEngine::new(rules);

            let mut now = start;
            let mut last_alert: Option<DateTime<Utc>> = None;
            let mut alerted_since_recovery = false;
            for (step_minutes, pct) in readings {
                now += TimeDelta::minutes(step_minutes);
                let alert = engine.observe(pct, now, &settings, &schedule);
                if pct >= threshold {
                    prop_assert!(alert.is_none(), "no alert at or above threshold");
                    alerted_since_recovery = false;
                }
                if let Some(a) = alert {
                    prop_assert!(a.percentage < a.threshold_percent);
                    prop_assert!(!alerted_since_recovery, "at most one alert per dip");
                    if let Some(prev) = last_alert {
                        prop_assert!(
                            (now - prev).num_seconds() >= i64::try_from(cooldown_secs).unwrap_or(i64::MAX),
                            "alerts respect the cooldown"
                        );
                    }
                    last_alert = Some(now);
                    alerted_since_recovery = true;
                }
            }
        }

        #[test]
        fn inactive_settings_never_alert(
            readings in prop::collection::vec(0.0f64..100.0, 1..60),
        ) {
            let schedule = GymSchedule::new_for_test(0, 24, 0, 24);
            let start = Utc.with_ymd_and_hms(2024, 6, 17, 0, 0, 0).single()
                .ok_or_else(|| TestCaseError::fail("valid start"))?;
            let off = AlertSettings::new(false, 100.0, None, start, SettingsSource::Gui)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let rules = AlertRules::new(0, 0, Vec::new())
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let mut engine = AlertEngine::new(rules);
            for (i, pct) in readings.into_iter().enumerate() {
                let now = start + TimeDelta::minutes(i64::try_from(i).unwrap_or(0));
                prop_assert!(engine.observe(pct, now, &off, &schedule).is_none());
            }
        }
    }
}
