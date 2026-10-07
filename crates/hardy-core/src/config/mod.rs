mod connection;
mod display;
mod ml;
mod notifications;
mod schedule;
mod validation;

use std::path::PathBuf;

use anyhow::{Context, Result};
use config::{Config, Environment, File};
pub use connection::{DatabaseConfig, GymConfig, NetworkConfig};
use connection::{default_acquire_timeout_secs, default_max_connections};
pub use display::{RefreshConfig, ThresholdsConfig, WindowConfig};
pub use ml::{MlAlgorithm, MlConfig};
pub use notifications::{NotificationConfig, SecretString};
pub use schedule::{ScheduleConfig, ScheduleHours};
use serde::Deserialize;
use tracing::warn;

/// The whole configuration: defaults, then `config.toml`, then `HARDY__…`
/// environment\nvariables.
#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub database: DatabaseConfig,
    pub gym: GymConfig,
    pub network: NetworkConfig,
    pub window: WindowConfig,
    pub refresh: RefreshConfig,
    pub notifications: NotificationConfig,
    pub thresholds: ThresholdsConfig,
    pub schedule: ScheduleConfig,
    pub ml: MlConfig,
}

impl AppConfig {
    /// Loads and validates the configuration.
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
            .set_default("schedule.timezone", "Europe/Berlin")?
            .set_default("schedule.weekday.open_hour", 6)?
            .set_default("schedule.weekday.close_hour", 23)?
            .set_default("schedule.weekend.open_hour", 9)?
            .set_default("schedule.weekend.close_hour", 21)?

            .set_default("ml.enabled", true)?
            .set_default("ml.training_window_days", 56_i64)?
            .set_default("ml.prediction_horizon_hours", 6_i64)?
            .set_default("ml.min_samples_for_training", 500_i64)?

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

        Ok(())
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

    // ── MlConfig tests ──────────────────────────────────────────────
}
