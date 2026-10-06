use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono_tz::Tz;
use config::{Config, Environment, File};
use serde::Deserialize;
use tracing::warn;

use crate::{alert::AlertWindow, error::AppError};

/// Which ML algorithm to use for occupancy prediction.
#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
pub enum MlAlgorithm {
    #[default]
    RandomForest,
    LinearRegression,
}

/// Configuration for the ML prediction pipeline.
#[derive(Debug, Clone, Deserialize)]
pub struct MlConfig {
    pub enabled: bool,
    pub training_window_days: i64,
    pub retrain_interval_hours: i64,
    pub prediction_horizon_hours: i64,
    pub min_samples_for_training: usize,
    pub model_path: Option<PathBuf>,
    pub fallback_on_error: bool,
    #[serde(default)]
    pub algorithm: MlAlgorithm,
    #[serde(default = "default_cv_folds")]
    pub cv_folds: usize,
    #[serde(default = "default_cv_gap_hours")]
    pub cv_gap_hours: i64,
    #[serde(default = "default_tune_hyperparameters")]
    pub tune_hyperparameters: bool,
}

fn default_cv_folds() -> usize {
    4
}

fn default_cv_gap_hours() -> i64 {
    24
}

fn default_tune_hyperparameters() -> bool {
    true
}

impl MlConfig {
    /// Resolve the model persistence path.
    ///
    /// Uses `model_path` if set, otherwise derives a default using
    /// `dirs::data_dir()` (e.g., `AppData/Roaming/hardy-monitor/model.bin`).
    /// Returns `None` only if `dirs::data_dir()` is unavailable.
    pub fn resolve_model_path(&self) -> Option<PathBuf> {
        if let Some(ref path) = self.model_path {
            return Some(path.clone());
        }
        dirs::data_dir().map(|d| d.join("hardy-monitor").join("model.bin"))
    }
}

impl Default for MlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            training_window_days: 56,
            retrain_interval_hours: 24,
            prediction_horizon_hours: 6,
            min_samples_for_training: 500,
            model_path: None,
            fallback_on_error: true,
            algorithm: MlAlgorithm::default(),
            cv_folds: default_cv_folds(),
            cv_gap_hours: default_cv_gap_hours(),
            tune_hyperparameters: default_tune_hyperparameters(),
        }
    }
}

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
    pub ml: MlConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DatabaseConfig {
    pub url: String,
    /// Upper bound on pooled connections; keep small for hosted free tiers.
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
    /// How long to wait for a free pooled connection before failing.
    #[serde(default = "default_acquire_timeout_secs")]
    pub acquire_timeout_secs: u64,
}

impl DatabaseConfig {
    /// Config for `url` with default pool settings.
    pub fn with_url(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            max_connections: default_max_connections(),
            acquire_timeout_secs: default_acquire_timeout_secs(),
        }
    }
}

fn default_max_connections() -> u32 {
    5
}

fn default_acquire_timeout_secs() -> u64 {
    10
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

/// A secret read from configuration; never shown by `Debug`.
#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct SecretString(String);

impl SecretString {
    /// The secret value, for the one place that needs it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

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
    /// Gym-local times in which alerts may fire; empty = any opening hour.
    #[serde(default)]
    pub windows: Vec<AlertWindow>,
}

fn default_opening_grace_minutes() -> u32 {
    60
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
            windows: Vec::new(),
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
    /// IANA timezone the gym operates in. Opening hours, holidays and all
    /// local-time aggregation are interpreted in this zone, so results do not
    /// depend on the timezone of the host running the binary.
    #[serde(default = "default_gym_timezone")]
    pub timezone: Tz,
    pub weekday: ScheduleHours,
    pub weekend: ScheduleHours,
}

fn default_gym_timezone() -> Tz {
    chrono_tz::Europe::Berlin
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            timezone: default_gym_timezone(),
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
    pub fn validate(&self) -> Result<(), AppError> {
        if self.database.max_connections == 0 {
            return Err(AppError::Config(
                "database.max_connections must be > 0".to_string(),
            ));
        }
        if self.database.acquire_timeout_secs == 0 {
            return Err(AppError::Config(
                "database.acquire_timeout_secs must be > 0".to_string(),
            ));
        }

        for (label, hours) in [
            ("schedule.weekday", self.schedule.weekday),
            ("schedule.weekend", self.schedule.weekend),
        ] {
            if hours.open_hour > 24 {
                return Err(AppError::Config(format!(
                    "{label}.open_hour must be <= 24, got {}",
                    hours.open_hour
                )));
            }
            if hours.close_hour > 24 {
                return Err(AppError::Config(format!(
                    "{label}.close_hour must be <= 24, got {}",
                    hours.close_hour
                )));
            }
            if hours.open_hour >= hours.close_hour {
                return Err(AppError::Config(format!(
                    "{label}.open_hour ({}) must be < close_hour ({})",
                    hours.open_hour, hours.close_hour
                )));
            }
        }

        if !(0.0..=100.0).contains(&self.thresholds.low_occupancy_percent) {
            return Err(AppError::Config(format!(
                "thresholds.low_occupancy_percent must be in [0.0, 100.0], got {}",
                self.thresholds.low_occupancy_percent
            )));
        }
        if !(0.0..=100.0).contains(&self.thresholds.high_occupancy_percent) {
            return Err(AppError::Config(format!(
                "thresholds.high_occupancy_percent must be in [0.0, 100.0], got {}",
                self.thresholds.high_occupancy_percent
            )));
        }
        if self.thresholds.low_occupancy_percent >= self.thresholds.high_occupancy_percent {
            return Err(AppError::Config(format!(
                "thresholds.low_occupancy_percent ({}) must be < high_occupancy_percent ({})",
                self.thresholds.low_occupancy_percent, self.thresholds.high_occupancy_percent
            )));
        }

        for window in &self.notifications.windows {
            window.validate()?;
        }
        if self.notifications.control_topic.is_some()
            && self.notifications.control_topic == self.notifications.ntfy_topic
        {
            // The daemon's own replies would be read back as commands.
            return Err(AppError::Config(
                "notifications.control_topic must differ from ntfy_topic".to_string(),
            ));
        }

        if self.refresh.data_fetch_interval_secs == 0 {
            return Err(AppError::Config(
                "refresh.data_fetch_interval_secs must be > 0".to_string(),
            ));
        }
        if self.refresh.ui_interval_secs == 0 {
            return Err(AppError::Config(
                "refresh.ui_interval_secs must be > 0".to_string(),
            ));
        }

        if self.analytics.prediction_window_days <= 0 {
            return Err(AppError::Config(format!(
                "analytics.prediction_window_days must be > 0, got {}",
                self.analytics.prediction_window_days
            )));
        }

        Ok(())
    }

    pub fn load() -> Result<Self> {
        let _ = dotenvy::dotenv();

        let database_url = std::env::var("DATABASE_URL")
            .context("DATABASE_URL must be set (via .env file or environment variable)")?;

        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| {
                warn!("could not determine OS config directory, skipping user config file");
                PathBuf::from(".")
            })
            .join("hardy-monitor");

        let builder = Config::builder()
            .set_default("database.url", database_url)?
            .set_default("database.max_connections", default_max_connections())?
            .set_default("database.acquire_timeout_secs", default_acquire_timeout_secs())?
            .set_default("gym.api_url", "https://portal.aidoo-online.de/workload?mandant=202300180_fuerstenfeldbruck&stud_nr=3&jsonResponse=1")?
            .set_default("network.request_timeout_secs", 30)?
            .set_default("network.connect_timeout_secs", 10)?
            .set_default("window.title", "Hardy's Gym Monitor")?
            .set_default("window.width", 1200.0)?
            .set_default("window.height", 850.0)?
            .set_default("window.sidebar_width", 250.0)?
            .set_default("refresh.ui_interval_secs", 30)?
            .set_default("refresh.data_fetch_interval_secs", 60)?
            .set_default("refresh.tray_poll_interval_ms", 50)?
            .set_default("notifications.ntfy_topic", None::<String>)?
            .set_default("notifications.ntfy_server", "https://ntfy.sh")?
            .set_default("notifications.cooldown_secs", 300)?
            .set_default("thresholds.low_occupancy_percent", 40.0)?
            .set_default("thresholds.high_occupancy_percent", 75.0)?
            .set_default("analytics.prediction_window_days", 28)?
            .set_default("schedule.timezone", "Europe/Berlin")?
            .set_default("schedule.weekday.open_hour", 6)?
            .set_default("schedule.weekday.close_hour", 23)?
            .set_default("schedule.weekend.open_hour", 9)?
            .set_default("schedule.weekend.close_hour", 21)?

            .set_default("ml.enabled", true)?
            .set_default("ml.training_window_days", 56_i64)?
            .set_default("ml.retrain_interval_hours", 24_i64)?
            .set_default("ml.prediction_horizon_hours", 6_i64)?
            .set_default("ml.min_samples_for_training", 500_i64)?
            .set_default("ml.model_path", None::<String>)?
            .set_default("ml.fallback_on_error", true)?

            .add_source(File::from(PathBuf::from("config.toml")).required(false))

            .add_source(File::from(config_dir.join("config.toml")).required(false))

            .add_source(Environment::with_prefix("HARDY").separator("__"));

        let s = builder.build()?;
        let config: Self = s.try_deserialize()?;
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

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
        assert_relative_eq!(config.width, 1200.0);
        assert_relative_eq!(config.height, 850.0);
        assert_relative_eq!(config.sidebar_width, 250.0);
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
        assert_eq!(config.ntfy_server, "https://ntfy.sh");
        assert_eq!(config.cooldown_secs, 300);
        assert_eq!(config.opening_grace_minutes, 60);
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
            windows = [{ days = "weekends", start = "10:00", end = "18:00" }]
            "#,
        )?;
        assert_eq!(config.control_topic.as_deref(), Some("hardy-ctl"));
        assert_eq!(config.opening_grace_minutes, 30);
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

    #[test]
    fn test_thresholds_config_defaults() {
        let config = ThresholdsConfig::default();
        assert_relative_eq!(config.low_occupancy_percent, 40.0);
        assert_relative_eq!(config.high_occupancy_percent, 75.0);
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

    #[test]
    fn test_config_load_with_defaults() {
        let result = AppConfig::load();
        assert!(result.is_ok());
    }

    #[test]
    fn test_loaded_config_has_expected_structure() -> Result<()> {
        let config = AppConfig::load()?;

        assert_ne!(config.gym.api_url, "");
        assert!(config.network.request_timeout_secs > 0);
        assert!(config.window.width > 0.0);
        assert!(config.refresh.data_fetch_interval_secs > 0);
        assert!(config.thresholds.high_occupancy_percent > config.thresholds.low_occupancy_percent);
        assert!(config.analytics.prediction_window_days > 0);

        Ok(())
    }

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
        assert_relative_eq!(
            cloned.low_occupancy_percent,
            thresholds.low_occupancy_percent
        );
    }

    #[test]
    fn test_config_structs_are_debug() {
        let config = NetworkConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("NetworkConfig"));
        assert!(debug_str.contains("request_timeout_secs"));
    }

    #[test]
    fn test_env_var_overrides_gym_api_url() -> Result<()> {
        let env_key = "HARDY__GYM__API_URL";
        let test_url = "https://test.example.com/api";

        temp_env::with_var(env_key, Some(test_url), || -> Result<()> {
            let config = AppConfig::load()?;
            assert_eq!(
                config.gym.api_url, test_url,
                "Environment variable should override gym.api_url"
            );
            Ok(())
        })?;

        Ok(())
    }

    #[test]
    fn test_env_var_overrides_network_timeout() -> Result<()> {
        let env_key = "HARDY__NETWORK__REQUEST_TIMEOUT_SECS";

        temp_env::with_var(env_key, Some("120"), || -> Result<()> {
            let config = AppConfig::load()?;
            assert_eq!(
                config.network.request_timeout_secs, 120,
                "Environment variable should override network.request_timeout_secs"
            );
            Ok(())
        })?;

        Ok(())
    }

    #[test]
    fn test_env_var_overrides_thresholds() -> Result<()> {
        temp_env::with_vars(
            vec![
                ("HARDY__THRESHOLDS__LOW_OCCUPANCY_PERCENT", Some("25.0")),
                ("HARDY__THRESHOLDS__HIGH_OCCUPANCY_PERCENT", Some("85.0")),
            ],
            || -> Result<()> {
                let config = AppConfig::load()?;
                assert_relative_eq!(config.thresholds.low_occupancy_percent, 25.0);
                assert_relative_eq!(config.thresholds.high_occupancy_percent, 85.0);
                Ok(())
            },
        )?;

        Ok(())
    }

    #[test]
    fn test_env_var_overrides_notifications() -> Result<()> {
        temp_env::with_vars(
            vec![
                ("HARDY__NOTIFICATIONS__NTFY_TOKEN", Some("tk_from_env")),
                ("HARDY__NOTIFICATIONS__CONTROL_TOPIC", Some("ctl-from-env")),
            ],
            || -> Result<()> {
                let config = AppConfig::load()?;
                assert_eq!(
                    config
                        .notifications
                        .ntfy_token
                        .as_ref()
                        .map(SecretString::expose),
                    Some("tk_from_env")
                );
                assert_eq!(
                    config.notifications.control_topic.as_deref(),
                    Some("ctl-from-env")
                );
                Ok(())
            },
        )?;

        Ok(())
    }

    fn valid_app_config() -> AppConfig {
        AppConfig {
            database: DatabaseConfig {
                url: "postgres://localhost/test".to_string(),
                max_connections: 5,
                acquire_timeout_secs: 10,
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
            ml: MlConfig::default(),
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
    fn test_validate_rejects_control_topic_equal_to_alert_topic() {
        let mut cfg = valid_app_config();
        cfg.notifications.ntfy_topic = Some("same".to_string());
        cfg.notifications.control_topic = Some("same".to_string());
        assert!(cfg.validate().is_err());

        cfg.notifications.control_topic = Some("other".to_string());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_inverted_alert_window() -> Result<()> {
        let mut cfg = valid_app_config();
        let parsed: NotificationConfig = toml::from_str(
            r#"
            ntfy_server = "https://ntfy.sh"
            cooldown_secs = 300
            windows = [{ days = "daily", start = "21:00", end = "16:00" }]
            "#,
        )?;
        cfg.notifications = parsed;
        assert!(cfg.validate().is_err());
        Ok(())
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
    fn test_config_threshold_relationship() -> Result<()> {
        let config = AppConfig::load()?;

        assert!(
            config.thresholds.low_occupancy_percent <= config.thresholds.high_occupancy_percent,
            "Low threshold ({}) should be <= high threshold ({})",
            config.thresholds.low_occupancy_percent,
            config.thresholds.high_occupancy_percent
        );

        Ok(())
    }

    #[test]
    fn test_config_schedule_hours_in_valid_range() -> Result<()> {
        let config = AppConfig::load()?;

        assert!(config.schedule.weekday.open_hour < 24);
        assert!(config.schedule.weekday.close_hour <= 24);
        assert!(config.schedule.weekend.open_hour < 24);
        assert!(config.schedule.weekend.close_hour <= 24);

        Ok(())
    }

    #[test]
    fn test_config_refresh_intervals_are_positive() -> Result<()> {
        let config = AppConfig::load()?;

        assert!(config.refresh.ui_interval_secs > 0);
        assert!(config.refresh.data_fetch_interval_secs > 0);
        assert!(config.refresh.tray_poll_interval_ms > 0);

        Ok(())
    }

    #[test]
    fn test_config_prediction_window_is_positive() -> Result<()> {
        let config = AppConfig::load()?;
        assert!(config.analytics.prediction_window_days > 0);

        Ok(())
    }

    // ── MlConfig tests ──────────────────────────────────────────────

    #[test]
    fn test_ml_config_defaults() {
        let config = MlConfig::default();

        assert!(config.enabled);
        assert_eq!(config.training_window_days, 56);
        assert_eq!(config.retrain_interval_hours, 24);
        assert_eq!(config.prediction_horizon_hours, 6);
        assert_eq!(config.min_samples_for_training, 500);
        assert!(config.fallback_on_error);
    }

    #[test]
    fn test_ml_algorithm_default() {
        let algo = MlAlgorithm::default();
        assert_eq!(algo, MlAlgorithm::RandomForest);
    }

    #[test]
    fn test_ml_config_new_fields_default() {
        let config = MlConfig::default();
        assert_eq!(config.algorithm, MlAlgorithm::RandomForest);
        assert_eq!(config.cv_folds, 4);
        assert_eq!(config.cv_gap_hours, 24);
        assert!(config.tune_hyperparameters);
    }

    #[test]
    fn test_ml_algorithm_deserialize() -> Result<()> {
        #[derive(Deserialize)]
        struct Wrapper {
            algorithm: MlAlgorithm,
        }

        let rf: Wrapper = toml::from_str("algorithm = \"RandomForest\"")?;
        assert_eq!(rf.algorithm, MlAlgorithm::RandomForest);

        let lr: Wrapper = toml::from_str("algorithm = \"LinearRegression\"")?;
        assert_eq!(lr.algorithm, MlAlgorithm::LinearRegression);

        Ok(())
    }

    #[test]
    fn test_resolve_model_path_explicit() {
        let config = MlConfig {
            model_path: Some(PathBuf::from("/custom/path/model.bin")),
            ..MlConfig::default()
        };

        let resolved = config.resolve_model_path();
        assert_eq!(resolved, Some(PathBuf::from("/custom/path/model.bin")));
    }

    #[test]
    fn test_resolve_model_path_default() {
        let config = MlConfig::default();
        assert!(config.model_path.is_none());

        let resolved = config.resolve_model_path();
        assert!(resolved.is_some());

        let path = resolved.unwrap_or_else(|| unreachable!());
        let path_str = path.to_string_lossy();
        assert!(
            path_str.contains("hardy-monitor"),
            "default path should contain 'hardy-monitor': {path_str}"
        );
        assert!(
            path_str.ends_with("model.bin"),
            "default path should end with 'model.bin': {path_str}"
        );
    }

    #[test]
    fn test_resolve_model_path_default_not_none() {
        // dirs::data_dir() should return Some on Windows/macOS/Linux
        let data_dir = dirs::data_dir();
        assert!(
            data_dir.is_some(),
            "dirs::data_dir() should be available on this platform"
        );
    }

    #[test]
    fn test_database_config_pool_defaults() -> Result<()> {
        let config: DatabaseConfig = toml::from_str(r#"url = "postgres://x/y""#)?;
        assert_eq!(config.max_connections, 5);
        assert_eq!(config.acquire_timeout_secs, 10);
        Ok(())
    }

    #[test]
    fn test_validate_rejects_zero_max_connections() {
        let mut config = valid_app_config();
        config.database.max_connections = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_acquire_timeout() {
        let mut config = valid_app_config();
        config.database.acquire_timeout_secs = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_schedule_config_default_timezone_is_europe_berlin() {
        let config = ScheduleConfig::default();
        assert_eq!(config.timezone, chrono_tz::Europe::Berlin);
    }

    #[test]
    fn test_schedule_config_deserialize_timezone() -> Result<()> {
        let toml_str = r#"
            timezone = "America/New_York"
            [weekday]
            open_hour = 6
            close_hour = 23
            [weekend]
            open_hour = 9
            close_hour = 21
        "#;

        let config: ScheduleConfig = toml::from_str(toml_str)?;
        assert_eq!(config.timezone, chrono_tz::America::New_York);
        Ok(())
    }

    #[test]
    fn test_schedule_config_deserialize_without_timezone_uses_default() -> Result<()> {
        let toml_str = r"
            [weekday]
            open_hour = 6
            close_hour = 23
            [weekend]
            open_hour = 9
            close_hour = 21
        ";

        let config: ScheduleConfig = toml::from_str(toml_str)?;
        assert_eq!(config.timezone, chrono_tz::Europe::Berlin);
        Ok(())
    }

    #[test]
    fn test_schedule_config_rejects_unknown_timezone() {
        let toml_str = r#"
            timezone = "Mars/Olympus_Mons"
            [weekday]
            open_hour = 6
            close_hour = 23
            [weekend]
            open_hour = 9
            close_hour = 21
        "#;

        let result: Result<ScheduleConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_ml_config_deserialize_without_new_fields() -> Result<()> {
        let toml_str = r"
            enabled = true
            training_window_days = 56
            retrain_interval_hours = 24
            prediction_horizon_hours = 6
            min_samples_for_training = 500
            fallback_on_error = true
        ";

        let config: MlConfig = toml::from_str(toml_str)?;
        assert_eq!(config.algorithm, MlAlgorithm::RandomForest);
        assert_eq!(config.cv_folds, 4);
        assert_eq!(config.cv_gap_hours, 24);
        assert!(config.tune_hyperparameters);

        Ok(())
    }
}
