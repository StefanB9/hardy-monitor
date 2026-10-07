//! Opening hours and the gym's timezone.

use chrono_tz::Tz;
use serde::Deserialize;

/// Opening hours on weekdays and weekends (holidays count as weekends).
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

/// Opening and closing hour of one kind of day.
#[derive(Debug, Deserialize, Clone, Copy)]
pub struct ScheduleHours {
    pub open_hour: u32,
    pub close_hour: u32,
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

    #[test]
    fn test_schedule_config_defaults() {
        let config = ScheduleConfig::default();
        assert_eq!(config.weekday.open_hour, 6);
        assert_eq!(config.weekday.close_hour, 23);
        assert_eq!(config.weekend.open_hour, 9);
        assert_eq!(config.weekend.close_hour, 21);
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
}
