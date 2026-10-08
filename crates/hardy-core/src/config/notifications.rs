//! Alert delivery settings (ntfy).

use anyhow::Result;
use serde::Deserialize;

use super::SecretString;
use crate::{alert::AlertWindow, error::AppError};

/// Alert delivery settings. Whether alerts are armed and their threshold
/// live in the database (`alert_settings`), changed from the GUI or phone.
#[derive(Debug, Deserialize, Clone)]
pub struct NotificationConfig {
    /// Topic the daemon publishes alerts and command replies to.
    pub ntfy_topic: Option<String>,
    pub ntfy_server: String,
    /// ntfy access token. Set via `HARDY__NOTIFICATIONS__NTFY_TOKEN`, never
    /// in a committed config file.
    #[serde(default)]
    pub ntfy_token: Option<SecretString>,
    /// Topic the daemon reads phone commands from (`on 25`, `off`, ...).
    #[serde(default)]
    pub control_topic: Option<String>,
    /// Minimum time between two alerts.
    pub cooldown_secs: u64,
    /// No alerts for this long after the gym opens.
    #[serde(default = "default_opening_grace_minutes")]
    pub opening_grace_minutes: u32,
    /// Report an outage after this many minutes without stored readings
    /// while the gym is open.
    #[serde(default = "default_health_after_minutes")]
    pub health_after_minutes: u32,
    /// Gym-local times in which alerts may fire; empty = any opening hour.
    #[serde(default)]
    pub windows: Vec<AlertWindow>,
}

fn default_opening_grace_minutes() -> u32 {
    60
}

fn default_health_after_minutes() -> u32 {
    5
}

impl NotificationConfig {
    pub(super) fn validate(&self) -> Result<(), AppError> {
        for window in &self.windows {
            window.validate()?;
        }
        if self.health_after_minutes == 0 {
            return Err(AppError::Config(
                "notifications.health_after_minutes must be > 0".to_string(),
            ));
        }
        if self.control_topic.is_some() && self.control_topic == self.ntfy_topic {
            // The daemon's own replies would be read back as commands.
            return Err(AppError::Config(
                "notifications.control_topic must differ from ntfy_topic".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            ntfy_topic: None,
            ntfy_server: "https://ntfy.sh".to_string(),
            ntfy_token: None,
            control_topic: None,
            cooldown_secs: 300,
            opening_grace_minutes: default_opening_grace_minutes(),
            health_after_minutes: default_health_after_minutes(),
            windows: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

    #[test]
    fn test_notification_config_defaults() {
        let config = NotificationConfig::default();
        assert_eq!(config.ntfy_server, "https://ntfy.sh");
        assert_eq!(config.cooldown_secs, 300);
        assert_eq!(config.opening_grace_minutes, 60);
        assert_eq!(config.health_after_minutes, 5);
        assert!(config.control_topic.is_none());
        assert!(config.ntfy_token.is_none());
        assert_eq!(config.windows, Vec::new());
    }

    #[test]
    fn test_notification_config_deserializes_new_fields() -> Result<()> {
        let config: NotificationConfig = toml::from_str(
            r#"
            ntfy_server = "https://ntfy.example"
            cooldown_secs = 120
            control_topic = "hardy-ctl"
            opening_grace_minutes = 30
            health_after_minutes = 10
            windows = [{ days = "weekends", start = "10:00", end = "18:00" }]
            "#,
        )?;
        assert_eq!(config.control_topic.as_deref(), Some("hardy-ctl"));
        assert_eq!(config.opening_grace_minutes, 30);
        assert_eq!(config.health_after_minutes, 10);
        assert_eq!(config.windows.len(), 1);
        Ok(())
    }

    #[test]
    fn test_secret_string_debug_is_redacted() -> Result<()> {
        let config: NotificationConfig = toml::from_str(
            r#"
            ntfy_server = "https://ntfy.sh"
            cooldown_secs = 300
            ntfy_token = "tk_very_secret"
            "#,
        )?;
        let debug = format!("{config:?}");
        assert!(!debug.contains("tk_very_secret"), "{debug}");
        assert_eq!(
            config.ntfy_token.as_ref().map(SecretString::expose),
            Some("tk_very_secret")
        );
        Ok(())
    }
}
