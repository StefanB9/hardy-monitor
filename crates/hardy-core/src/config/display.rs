//! Desktop window, refresh intervals and occupancy thresholds.

use serde::Deserialize;

/// Desktop window size.
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

/// How often the GUI redraws, fetches and polls the tray.
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

/// Occupancy levels: below `low` is quiet, from `high` on busy.
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

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;

    use super::*;

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
    fn test_thresholds_config_defaults() {
        let config = ThresholdsConfig::default();
        assert_relative_eq!(config.low_occupancy_percent, 40.0);
        assert_relative_eq!(config.high_occupancy_percent, 75.0);
    }
}
