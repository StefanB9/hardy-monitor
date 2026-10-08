//! Dashboard alert controls: the GUI's view of the shared alert settings and
//! its desktop-popup engine.

use chrono::{DateTime, Utc};
use hardy_core::{
    AppError,
    alert::{Alert, AlertDuration, AlertEngine, AlertRules, AlertSettings, SettingsSource},
    schedule::GymSchedule,
};

/// Threshold shown before the settings have loaded.
const FALLBACK_THRESHOLD: f64 = 30.0;

/// Alert state held by the GUI. `settings` mirrors the database row; every
/// change returns new settings for the caller to save.
#[derive(Debug)]
pub(crate) struct AlertControls {
    settings: Option<AlertSettings>,
    engine: AlertEngine,
    duration: AlertDuration,
    pending_threshold: Option<f64>,
}

impl AlertControls {
    pub(crate) fn new(rules: AlertRules) -> Self {
        Self {
            settings: None,
            engine: AlertEngine::new(rules),
            duration: AlertDuration::UntilClosing,
            pending_threshold: None,
        }
    }

    /// Adopts settings loaded from (or just saved to) the database.
    pub(crate) fn set_settings(&mut self, settings: AlertSettings) {
        self.settings = Some(settings);
    }

    pub(crate) fn duration(&self) -> AlertDuration {
        self.duration
    }

    /// Threshold to display: the slider position while dragging, else the
    /// stored value.
    pub(crate) fn threshold(&self) -> f64 {
        self.pending_threshold
            .or_else(|| self.settings.as_ref().map(AlertSettings::threshold_percent))
            .unwrap_or(FALLBACK_THRESHOLD)
    }

    pub(crate) fn is_active(&self, now: DateTime<Utc>) -> bool {
        self.settings.as_ref().is_some_and(|s| s.is_active(now))
    }

    /// e.g. "On below 25% until 21:00 · set from phone".
    pub(crate) fn status_line(&self, now: DateTime<Utc>, schedule: &GymSchedule) -> String {
        let Some(settings) = &self.settings else {
            return "Loading…".to_string();
        };
        let mut line = settings.describe(now, schedule);
        if settings.updated_by() == SettingsSource::Phone {
            line.push_str(" · set from phone");
        }
        line
    }

    /// Arms with the selected duration or disarms. `None` until settings
    /// have loaded.
    pub(crate) fn toggle(
        &mut self,
        enabled: bool,
        now: DateTime<Utc>,
        schedule: &GymSchedule,
    ) -> Option<Result<AlertSettings, AppError>> {
        let current = self.settings.as_ref()?;
        Some(if enabled {
            AlertSettings::armed(
                self.threshold(),
                self.duration,
                now,
                schedule,
                SettingsSource::Gui,
            )
        } else {
            Ok(current.disarmed(now, SettingsSource::Gui))
        })
    }

    /// Picks the duration; re-arms right away if alerts are on.
    pub(crate) fn select_duration(
        &mut self,
        duration: AlertDuration,
        now: DateTime<Utc>,
        schedule: &GymSchedule,
    ) -> Option<Result<AlertSettings, AppError>> {
        self.duration = duration;
        if self.is_active(now) {
            self.toggle(true, now, schedule)
        } else {
            None
        }
    }

    /// Slider moved; saved on release.
    pub(crate) fn drag_threshold(&mut self, threshold: f64) {
        self.pending_threshold = Some(threshold);
    }

    /// Slider released: the new threshold to save, keeping armed state.
    pub(crate) fn release_threshold(
        &mut self,
        now: DateTime<Utc>,
    ) -> Option<Result<AlertSettings, AppError>> {
        let threshold = self.pending_threshold.take()?;
        let current = self.settings.as_ref()?;
        if (current.threshold_percent() - threshold).abs() < f64::EPSILON {
            return None;
        }
        Some(current.with_threshold(threshold, now, SettingsSource::Gui))
    }

    /// Desktop popup decision for a new reading.
    pub(crate) fn observe(
        &mut self,
        percentage: f64,
        now: DateTime<Utc>,
        schedule: &GymSchedule,
    ) -> Option<Alert> {
        let settings = self.settings.as_ref()?;
        self.engine.observe(percentage, now, settings, schedule)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeZone;

    use super::*;

    /// Monday 2024-06-17 at `h:mi` CEST.
    fn local(h: u32, mi: u32) -> Result<DateTime<Utc>> {
        GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, 17, h, mi, 0)
            .single()
            .map(|t| t.with_timezone(&Utc))
            .context("valid local time")
    }

    fn controls_with(settings: AlertSettings) -> Result<AlertControls> {
        let mut c = AlertControls::new(AlertRules::new(0, 60, Vec::new())?);
        c.set_settings(settings);
        Ok(c)
    }

    fn off(at: DateTime<Utc>) -> Result<AlertSettings> {
        Ok(AlertSettings::new(
            false,
            30.0,
            None,
            at,
            SettingsSource::Migration,
        )?)
    }

    #[test]
    fn test_alert_controls_need_loaded_settings() -> Result<()> {
        let mut c = AlertControls::new(AlertRules::new(0, 60, Vec::new())?);
        let schedule = GymSchedule::default();
        assert!(c.toggle(true, local(10, 0)?, &schedule).is_none());
        assert_eq!(c.status_line(local(10, 0)?, &schedule), "Loading…");
        assert!(c.observe(1.0, local(10, 0)?, &schedule).is_none());
        Ok(())
    }

    #[test]
    fn test_alert_controls_toggle_arms_until_closing_by_default() -> Result<()> {
        let schedule = GymSchedule::default();
        let mut c = controls_with(off(local(9, 0)?)?)?;
        let armed = c
            .toggle(true, local(10, 0)?, &schedule)
            .context("settings loaded")??;
        assert!(armed.is_active(local(22, 59)?));
        assert!(!armed.is_active(local(23, 0)?));
        assert_eq!(armed.updated_by(), SettingsSource::Gui);

        c.set_settings(armed);
        let disarmed = c
            .toggle(false, local(10, 5)?, &schedule)
            .context("settings loaded")??;
        assert!(!disarmed.enabled());
        Ok(())
    }

    #[test]
    fn test_alert_controls_select_duration_rearms_only_when_active() -> Result<()> {
        let schedule = GymSchedule::default();
        let mut c = controls_with(off(local(9, 0)?)?)?;
        assert!(
            c.select_duration(AlertDuration::Hours(2), local(10, 0)?, &schedule)
                .is_none()
        );
        assert_eq!(c.duration(), AlertDuration::Hours(2));

        let armed = c
            .toggle(true, local(10, 0)?, &schedule)
            .context("loaded")??;
        assert!(!armed.is_active(local(12, 0)?));
        c.set_settings(armed);
        let always = c
            .select_duration(AlertDuration::Always, local(10, 1)?, &schedule)
            .context("re-armed")??;
        assert_eq!(always.active_until(), None);
        Ok(())
    }

    #[test]
    fn test_alert_controls_threshold_saved_on_release_only_if_changed() -> Result<()> {
        let mut c = controls_with(off(local(9, 0)?)?)?;
        c.drag_threshold(30.0);
        assert!(c.release_threshold(local(10, 0)?).is_none());

        c.drag_threshold(45.0);
        assert!((c.threshold() - 45.0).abs() < f64::EPSILON);
        let saved = c.release_threshold(local(10, 0)?).context("changed")??;
        assert!((saved.threshold_percent() - 45.0).abs() < f64::EPSILON);
        Ok(())
    }

    #[test]
    fn test_alert_controls_status_line_marks_phone_changes() -> Result<()> {
        let schedule = GymSchedule::default();
        let from_phone = AlertSettings::armed(
            25.0,
            AlertDuration::UntilClosing,
            local(10, 0)?,
            &schedule,
            SettingsSource::Phone,
        )?;
        let c = controls_with(from_phone)?;
        assert_eq!(
            c.status_line(local(10, 0)?, &schedule),
            "On below 25% until 23:00 · set from phone"
        );
        Ok(())
    }
}
