use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use tracing::warn;
use config::{Config, Environment, File};
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub database: DatabaseConfig,
    pub gym: GymConfig,
    pub network: NetworkConfig,
    pub window: WindowConfig,
    pub refresh: RefreshConfig,
    pub notifications: NotificationConfig,
    pub thresholds: ThresholdsConfig,
    pub analytics: AnalyticsConfig,
    pub schedule: ScheduleConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DatabaseConfig {
    pub url: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GymConfig {
    pub api_url: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct NetworkConfig {
    pub request_timeout_secs: u64,
    pub connect_timeout_secs: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            request_timeout_secs: 30,
            connect_timeout_secs: 10,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct WindowConfig {
    #[allow(dead_code)]
    pub title: String,
    pub width: f32,
    pub height: f32,
    pub sidebar_width: f32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Hardy's Gym Monitor".to_string(),
            width: 1200.0,
            height: 850.0,
            sidebar_width: 250.0,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct RefreshConfig {
    pub ui_interval_secs: u64,
    pub data_fetch_interval_secs: u64,
    pub tray_poll_interval_ms: u64,
}

impl Default for RefreshConfig {
    fn default() -> Self {
        Self {
            ui_interval_secs: 30,
            data_fetch_interval_secs: 60,
            tray_poll_interval_ms: 50,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct NotificationConfig {
    pub enabled: bool,
    pub threshold_percent: f64,
    /// Ntfy.sh topic for phone notifications (e.g., "hardys-occupancy-1993")
    pub ntfy_topic: Option<String>,
    /// Ntfy server base URL. Override to point at a self-hosted instance.
    pub ntfy_server: String,
    /// Minimum seconds between consecutive notifications (prevents spam).
    pub cooldown_secs: u64,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold_percent: 30.0,
            ntfy_topic: None,
            ntfy_server: "https://ntfy.sh".to_string(),
            cooldown_secs: 300,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ThresholdsConfig {
    pub low_occupancy_percent: f64,
    pub high_occupancy_percent: f64,
}

impl Default for ThresholdsConfig {
    fn default() -> Self {
        Self {
            low_occupancy_percent: 40.0,
            high_occupancy_percent: 75.0,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AnalyticsConfig {
    pub prediction_window_days: i64,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            prediction_window_days: 28,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ScheduleConfig {
    pub weekday: ScheduleHours,
    pub weekend: ScheduleHours,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            weekday: ScheduleHours {
                open_hour: 6,
                close_hour: 23,
            },
            weekend: ScheduleHours {
                open_hour: 9,
                close_hour: 21,
            },
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct ScheduleHours {
    pub open_hour: u32,
    pub close_hour: u32,
}

impl AppConfig {
    /// Validate all configuration values, returning a descriptive error on the
    /// first constraint violation.
    ///
    /// Called automatically by [`load`](Self::load) after deserialisation.
    pub fn validate(&self) -> Result<()> {
        for (label, hours) in [
            ("schedule.weekday", self.schedule.weekday),
            ("schedule.weekend", self.schedule.weekend),
        ] {
            if hours.open_hour > 24 {
                bail!("{label}.open_hour must be <= 24, got {}", hours.open_hour);
            }
            if hours.close_hour > 24 {
                bail!("{label}.close_hour must be <= 24, got {}", hours.close_hour);
            }
            if hours.open_hour >= hours.close_hour {
                bail!(
                    "{label}.open_hour ({}) must be < close_hour ({})",
                    hours.open_hour,
                    hours.close_hour
                );
            }
        }

        if !(0.0..=100.0).contains(&self.thresholds.low_occupancy_percent) {
            bail!(
                "thresholds.low_occupancy_percent must be in [0.0, 100.0], got {}",
                self.thresholds.low_occupancy_percent
            );
        }
        if !(0.0..=100.0).contains(&self.thresholds.high_occupancy_percent) {
            bail!(
                "thresholds.high_occupancy_percent must be in [0.0, 100.0], got {}",
                self.thresholds.high_occupancy_percent
            );
        }
        if self.thresholds.low_occupancy_percent >= self.thresholds.high_occupancy_percent {
            bail!(
                "thresholds.low_occupancy_percent ({}) must be < high_occupancy_percent ({})",
                self.thresholds.low_occupancy_percent,
                self.thresholds.high_occupancy_percent
            );
        }

        if !(0.0..=100.0).contains(&self.notifications.threshold_percent) {
            bail!(
                "notifications.threshold_percent must be in [0.0, 100.0], got {}",
                self.notifications.threshold_percent
            );
        }

        if self.refresh.data_fetch_interval_secs == 0 {
            bail!("refresh.data_fetch_interval_secs must be > 0");
        }
        if self.refresh.ui_interval_secs == 0 {
            bail!("refresh.ui_interval_secs must be > 0");
        }

        if self.analytics.prediction_window_days <= 0 {
            bail!(
                "analytics.prediction_window_days must be > 0, got {}",
                self.analytics.prediction_window_days
            );
        }

        Ok(())
    }

    pub fn load() -> Result<Self> {
        // Load .env file (silently ignore if not present - production uses env vars
        // directly)
        let _ = dotenvy::dotenv();

        // Read DATABASE_URL from environment (required)
        let database_url = std::env::var("DATABASE_URL")
            .context("DATABASE_URL must be set (via .env file or environment variable)")?;

        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| {
                warn!("could not determine OS config directory, skipping user config file");
                PathBuf::from(".")
            })
            .join("hardy-monitor");

        let builder = Config::builder()
            // 1. Load default values
            // Database (loaded from environment above)
            .set_default("database.url", database_url)?
            // Gym
            .set_default("gym.api_url", "https://portal.aidoo-online.de/workload?mandant=202300180_fuerstenfeldbruck&stud_nr=3&jsonResponse=1")?
            // Network
            .set_default("network.request_timeout_secs", 30)?
            .set_default("network.connect_timeout_secs", 10)?
            // Window
            .set_default("window.title", "Hardy's Gym Monitor")?
            .set_default("window.width", 1200.0)?
            .set_default("window.height", 850.0)?
            .set_default("window.sidebar_width", 250.0)?
            // Refresh
            .set_default("refresh.ui_interval_secs", 30)?
            .set_default("refresh.data_fetch_interval_secs", 60)?
            .set_default("refresh.tray_poll_interval_ms", 50)?
            // Notifications
            .set_default("notifications.enabled", false)?
            .set_default("notifications.threshold_percent", 30.0)?
            .set_default("notifications.ntfy_topic", None::<String>)?
            .set_default("notifications.ntfy_server", "https://ntfy.sh")?
            .set_default("notifications.cooldown_secs", 300)?
            // Thresholds
            .set_default("thresholds.low_occupancy_percent", 40.0)?
            .set_default("thresholds.high_occupancy_percent", 75.0)?
            // Analytics
            .set_default("analytics.prediction_window_days", 28)?
            // Schedule
            .set_default("schedule.weekday.open_hour", 6)?
            .set_default("schedule.weekday.close_hour", 23)?
            .set_default("schedule.weekend.open_hour", 9)?
            .set_default("schedule.weekend.close_hour", 21)?

            // 2. Load from local config file (optional; overrides defaults,
            //    but is itself overridden by the user config dir and HARDY__* env vars)
            .add_source(File::from(PathBuf::from("config.toml")).required(false))

            // 3. Load from user config directory (optional, overrides local)
            .add_source(File::from(config_dir.join("config.toml")).required(false))

            // 4. Load from Environment variables (HARDY_DATABASE__PATH=...)
            .add_source(Environment::with_prefix("HARDY").separator("__"));

        let s = builder.build()?;
        let config: Self = s.try_deserialize()?;
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Default Value Tests ====================

    #[test]
    fn test_network_config_defaults() {
        let config = NetworkConfig::default();
        assert_eq!(config.request_timeout_secs, 30);
        assert_eq!(config.connect_timeout_secs, 10);
    }

    #[test]
    fn test_window_config_defaults() {
        let config = WindowConfig::default();
        assert_eq!(config.title, "Hardy's Gym Monitor");
        assert_eq!(config.width, 1200.0);
        assert_eq!(config.height, 850.0);
        assert_eq!(config.sidebar_width, 250.0);
    }

    #[test]
    fn test_refresh_config_defaults() {
        let config = RefreshConfig::default();
        assert_eq!(config.ui_interval_secs, 30);
        assert_eq!(config.data_fetch_interval_secs, 60);
        assert_eq!(config.tray_poll_interval_ms, 50);
    }

    #[test]
    fn test_notification_config_defaults() {
        let config = NotificationConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.threshold_percent, 30.0);
    }

    #[test]
    fn test_thresholds_config_defaults() {
        let config = ThresholdsConfig::default();
        assert_eq!(config.low_occupancy_percent, 40.0);
        assert_eq!(config.high_occupancy_percent, 75.0);
    }

    #[test]
    fn test_analytics_config_defaults() {
        let config = AnalyticsConfig::default();
        assert_eq!(config.prediction_window_days, 28);
    }

    #[test]
    fn test_schedule_config_defaults() {
        let config = ScheduleConfig::default();
        assert_eq!(config.weekday.open_hour, 6);
        assert_eq!(config.weekday.close_hour, 23);
        assert_eq!(config.weekend.open_hour, 9);
        assert_eq!(config.weekend.close_hour, 21);
    }

    // ==================== Config Loading Tests ====================

    #[test]
    fn test_config_load_with_defaults() {
        // This test verifies that config can be loaded with defaults
        // when no config file exists
        let result = AppConfig::load();
        assert!(result.is_ok());
    }

    #[test]
    fn test_loaded_config_has_expected_structure() {
        let config = AppConfig::load().expect("Config should load");

        assert!(!config.gym.api_url.is_empty());
        assert!(config.network.request_timeout_secs > 0);
        assert!(config.window.width > 0.0);
        assert!(config.refresh.data_fetch_interval_secs > 0);
        assert!(config.thresholds.high_occupancy_percent > config.thresholds.low_occupancy_percent);
        assert!(config.analytics.prediction_window_days > 0);
    }

    // ==================== Struct Field Tests ====================

    #[test]
    fn test_schedule_hours_copy() {
        let hours = ScheduleHours {
            open_hour: 8,
            close_hour: 20,
        };
        let copy = hours;
        assert_eq!(copy.open_hour, 8);
        assert_eq!(copy.close_hour, 20);
    }

    #[test]
    fn test_config_structs_are_clone() {
        let network = NetworkConfig::default();
        let cloned = network.clone();
        assert_eq!(cloned.request_timeout_secs, network.request_timeout_secs);

        let thresholds = ThresholdsConfig::default();
        let cloned = thresholds.clone();
        assert_eq!(
            cloned.low_occupancy_percent,
            thresholds.low_occupancy_percent
        );
    }

    #[test]
    fn test_config_structs_are_debug() {
        let config = NetworkConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("NetworkConfig"));
        assert!(debug_str.contains("request_timeout_secs"));
    }

    // ==================== Environment Variable Override Tests ====================

    #[test]
    fn test_env_var_overrides_gym_api_url() {
        let env_key = "HARDY__GYM__API_URL";
        let test_url = "https://test.example.com/api";

        // SAFETY: temp_env handles the unsafe environment manipulation and thread
        // locking
        temp_env::with_var(env_key, Some(test_url), || {
            let config = AppConfig::load().expect("Config should load");
            assert_eq!(
                config.gym.api_url, test_url,
                "Environment variable should override gym.api_url"
            );
        });
    }

    #[test]
    fn test_env_var_overrides_network_timeout() {
        let env_key = "HARDY__NETWORK__REQUEST_TIMEOUT_SECS";

        temp_env::with_var(env_key, Some("120"), || {
            let config = AppConfig::load().expect("Config should load");
            assert_eq!(
                config.network.request_timeout_secs, 120,
                "Environment variable should override network.request_timeout_secs"
            );
        });
    }

    #[test]
    fn test_env_var_overrides_thresholds() {
        temp_env::with_vars(
            vec![
                ("HARDY__THRESHOLDS__LOW_OCCUPANCY_PERCENT", Some("25.0")),
                ("HARDY__THRESHOLDS__HIGH_OCCUPANCY_PERCENT", Some("85.0")),
            ],
            || {
                let config = AppConfig::load().expect("Config should load");
                assert_eq!(config.thresholds.low_occupancy_percent, 25.0);
                assert_eq!(config.thresholds.high_occupancy_percent, 85.0);
            },
        );
    }

    #[test]
    fn test_env_var_overrides_notifications() {
        temp_env::with_vars(
            vec![
                ("HARDY__NOTIFICATIONS__ENABLED", Some("true")),
                ("HARDY__NOTIFICATIONS__THRESHOLD_PERCENT", Some("15.5")),
            ],
            || {
                let config = AppConfig::load().expect("Config should load");
                assert!(config.notifications.enabled);
                assert_eq!(config.notifications.threshold_percent, 15.5);
            },
        );
    }

    // ==================== AppConfig::validate() Tests ====================

    /// Construct a fully-valid `AppConfig` using safe defaults for all fields.
    /// Individual tests mutate the one field they want to test.
    fn valid_app_config() -> AppConfig {
        AppConfig {
            database: DatabaseConfig {
                url: "postgres://localhost/test".to_string(),
            },
            gym: GymConfig {
                api_url: "https://example.com".to_string(),
            },
            network: NetworkConfig::default(),
            window: WindowConfig::default(),
            refresh: RefreshConfig::default(),
            notifications: NotificationConfig::default(),
            thresholds: ThresholdsConfig::default(),
            analytics: AnalyticsConfig::default(),
            schedule: ScheduleConfig::default(),
        }
    }

    #[test]
    fn test_validate_defaults_pass() {
        assert!(valid_app_config().validate().is_ok());
    }

    #[test]
    fn test_validate_schedule_open_ge_close_fails() {
        let mut cfg = valid_app_config();
        cfg.schedule.weekday.open_hour = 10;
        cfg.schedule.weekday.close_hour = 10;
        assert!(cfg.validate().is_err());

        cfg.schedule.weekday.open_hour = 12;
        cfg.schedule.weekday.close_hour = 10;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_schedule_hour_exceeds_24_fails() {
        let mut cfg = valid_app_config();
        cfg.schedule.weekend.close_hour = 25;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_thresholds_low_ge_high_fails() {
        let mut cfg = valid_app_config();
        cfg.thresholds.low_occupancy_percent = 75.0;
        cfg.thresholds.high_occupancy_percent = 40.0;
        assert!(cfg.validate().is_err());

        cfg.thresholds.low_occupancy_percent = 50.0;
        cfg.thresholds.high_occupancy_percent = 50.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_thresholds_out_of_range_fails() {
        let mut cfg = valid_app_config();
        cfg.thresholds.low_occupancy_percent = -1.0;
        assert!(cfg.validate().is_err());

        let mut cfg = valid_app_config();
        cfg.thresholds.high_occupancy_percent = 101.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_notifications_threshold_out_of_range_fails() {
        let mut cfg = valid_app_config();
        cfg.notifications.threshold_percent = -0.1;
        assert!(cfg.validate().is_err());

        let mut cfg = valid_app_config();
        cfg.notifications.threshold_percent = 100.1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_intervals_fail() {
        let mut cfg = valid_app_config();
        cfg.refresh.data_fetch_interval_secs = 0;
        assert!(cfg.validate().is_err());

        let mut cfg = valid_app_config();
        cfg.refresh.ui_interval_secs = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_nonpositive_prediction_window_fails() {
        let mut cfg = valid_app_config();
        cfg.analytics.prediction_window_days = 0;
        assert!(cfg.validate().is_err());

        cfg.analytics.prediction_window_days = -1;
        assert!(cfg.validate().is_err());
    }

    // ==================== Config Value Validation Tests ====================

    #[test]
    fn test_config_default_values_are_reasonable() {
        let network = NetworkConfig::default();
        assert!(network.request_timeout_secs > 0);
        assert!(network.connect_timeout_secs > 0);
        assert!(network.request_timeout_secs >= network.connect_timeout_secs);

        let thresholds = ThresholdsConfig::default();
        assert!(thresholds.low_occupancy_percent < thresholds.high_occupancy_percent);
        assert!(thresholds.low_occupancy_percent >= 0.0);
        assert!(thresholds.high_occupancy_percent <= 100.0);

        let schedule = ScheduleConfig::default();
        assert!(schedule.weekday.open_hour < schedule.weekday.close_hour);
        assert!(schedule.weekend.open_hour < schedule.weekend.close_hour);
    }

    #[test]
    fn test_config_threshold_relationship() {
        let config = AppConfig::load().expect("Config should load");

        assert!(
            config.thresholds.low_occupancy_percent <= config.thresholds.high_occupancy_percent,
            "Low threshold ({}) should be <= high threshold ({})",
            config.thresholds.low_occupancy_percent,
            config.thresholds.high_occupancy_percent
        );
    }

    #[test]
    fn test_config_schedule_hours_in_valid_range() {
        let config = AppConfig::load().expect("Config should load");

        assert!(config.schedule.weekday.open_hour < 24);
        assert!(config.schedule.weekday.close_hour <= 24);
        assert!(config.schedule.weekend.open_hour < 24);
        assert!(config.schedule.weekend.close_hour <= 24);
    }

    #[test]
    fn test_config_refresh_intervals_are_positive() {
        let config = AppConfig::load().expect("Config should load");

        assert!(config.refresh.ui_interval_secs > 0);
        assert!(config.refresh.data_fetch_interval_secs > 0);
        assert!(config.refresh.tray_poll_interval_ms > 0);
    }

    #[test]
    fn test_config_prediction_window_is_positive() {
        let config = AppConfig::load().expect("Config should load");
        assert!(config.analytics.prediction_window_days > 0);
    }
}
