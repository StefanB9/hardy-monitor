//! Checks of a loaded configuration beyond what deserialisation catches.

use super::AppConfig;
use crate::error::AppError;

impl AppConfig {
    /// Rejects values that would make the app misbehave (e.g. inverted opening
    /// hours or thresholds, zero intervals).
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

        self.notifications.validate()?;

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

        if self.ml.holdout_days <= 0 {
            return Err(AppError::Config(format!(
                "ml.holdout_days must be > 0, got {}",
                self.ml.holdout_days
            )));
        }
        if self.ml.training_window_days <= self.ml.holdout_days {
            return Err(AppError::Config(format!(
                "ml.training_window_days ({}) must exceed ml.holdout_days ({})",
                self.ml.training_window_days, self.ml.holdout_days
            )));
        }
        if !(1..=24).contains(&self.ml.prediction_horizon_hours) {
            return Err(AppError::Config(format!(
                "ml.prediction_horizon_hours must be 1–24, got {}",
                self.ml.prediction_horizon_hours
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::config::{
        DatabaseConfig, GymConfig, MlConfig, NetworkConfig, NotificationConfig, RefreshConfig,
        ScheduleConfig, ThresholdsConfig, WindowConfig,
    };

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
    fn test_validate_rejects_zero_health_after_minutes() {
        let mut cfg = valid_app_config();
        cfg.notifications.health_after_minutes = 0;
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
    fn test_validate_rejects_bad_ml_config() {
        let mut config = valid_app_config();
        config.ml.holdout_days = 0;
        assert!(config.validate().is_err());

        let mut config = valid_app_config();
        config.ml.training_window_days = 7;
        assert!(config.validate().is_err(), "window must exceed holdout");

        let mut config = valid_app_config();
        config.ml.prediction_horizon_hours = 25;
        assert!(config.validate().is_err());
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
}
