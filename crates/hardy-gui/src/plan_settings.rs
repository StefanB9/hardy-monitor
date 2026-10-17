//! The Plan tab's remembered choices: visit length and time ranges, kept in
//! `plan.json` next to the user `config.toml` (per computer).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::TimeDelta;
use serde::{Deserialize, Serialize};

/// How long a visit takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum VisitLength {
    OneHour,
    #[default]
    NinetyMinutes,
    TwoHours,
}

impl VisitLength {
    /// Every length, shortest first.
    pub const ALL: [Self; 3] = [Self::OneHour, Self::NinetyMinutes, Self::TwoHours];

    pub fn duration(self) -> TimeDelta {
        match self {
            Self::OneHour => TimeDelta::hours(1),
            Self::NinetyMinutes => TimeDelta::minutes(90),
            Self::TwoHours => TimeDelta::hours(2),
        }
    }

    /// Button label, e.g. "1.5 h".
    pub fn label(self) -> &'static str {
        match self {
            Self::OneHour => "1 h",
            Self::NinetyMinutes => "1.5 h",
            Self::TwoHours => "2 h",
        }
    }
}

/// Earliest start and latest end, in whole gym-local hours (24: midnight).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HourRange {
    pub earliest: u32,
    pub latest: u32,
}

impl Default for HourRange {
    /// The whole day, so the opening hours decide.
    fn default() -> Self {
        Self {
            earliest: 0,
            latest: 24,
        }
    }
}

impl HourRange {
    /// The range with `earliest` changed, keeping at least an hour.
    #[must_use]
    pub fn with_earliest(self, earliest: u32) -> Self {
        let earliest = earliest.min(23);
        Self {
            earliest,
            latest: self.latest.max(earliest + 1),
        }
    }

    /// The range with `latest` changed, keeping at least an hour.
    #[must_use]
    pub fn with_latest(self, latest: u32) -> Self {
        let latest = latest.clamp(1, 24);
        Self {
            earliest: self.earliest.min(latest - 1),
            latest,
        }
    }
}

/// Everything remembered between app starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PlanSettings {
    pub length: VisitLength,
    /// Monday to Friday.
    pub workday: HourRange,
    /// Weekends and holidays.
    pub weekend: HourRange,
}

/// `plan.json` in the user config directory (`None` if the OS has none).
pub fn default_path() -> Option<PathBuf> {
    dirs::config_dir().map(|dir| dir.join("hardy-monitor").join("plan.json"))
}

/// The settings in `path`; defaults if the file is missing or unreadable
/// (logged, so a broken file never blocks the app).
pub fn load(path: &Path) -> PlanSettings {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return PlanSettings::default(),
        Err(e) => {
            tracing::warn!(path = %path.display(), "Plan settings unreadable, using defaults: {e}");
            return PlanSettings::default();
        }
    };
    serde_json::from_str(&text).unwrap_or_else(|e| {
        tracing::warn!(path = %path.display(), "Plan settings invalid, using defaults: {e}");
        PlanSettings::default()
    })
}

/// Writes `settings` to `path`, creating its directory.
pub fn save(path: &Path, settings: &PlanSettings) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create {}", dir.display()))?;
    }
    let json = serde_json::to_string_pretty(settings).context("Failed to encode plan settings")?;
    std::fs::write(path, json).with_context(|| format!("Failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty directory for one test.
    fn temp_dir(name: &str) -> Result<PathBuf> {
        let dir =
            std::env::temp_dir().join(format!("hardy-plan-settings-{name}-{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).context("clear temp dir")?;
        }
        Ok(dir)
    }

    #[test]
    fn test_visit_lengths() {
        let durations: Vec<_> = VisitLength::ALL.iter().map(|l| l.duration()).collect();
        assert_eq!(
            durations,
            vec![
                TimeDelta::hours(1),
                TimeDelta::minutes(90),
                TimeDelta::hours(2)
            ]
        );
        let labels: Vec<_> = VisitLength::ALL.iter().map(|l| l.label()).collect();
        assert_eq!(labels, vec!["1 h", "1.5 h", "2 h"]);
        assert_eq!(VisitLength::default(), VisitLength::NinetyMinutes);
    }

    #[test]
    fn test_hour_range_keeps_at_least_an_hour() {
        let range = HourRange {
            earliest: 17,
            latest: 21,
        };
        assert_eq!(range.with_earliest(19).earliest, 19);
        // Moving the start past the end pushes the end along.
        assert_eq!(
            range.with_earliest(22),
            HourRange {
                earliest: 22,
                latest: 23
            }
        );
        assert_eq!(
            range.with_latest(16),
            HourRange {
                earliest: 15,
                latest: 16
            }
        );
        // Clamped to the day.
        assert_eq!(
            HourRange::default().with_earliest(30),
            HourRange {
                earliest: 23,
                latest: 24
            }
        );
        assert_eq!(
            HourRange::default().with_latest(0),
            HourRange {
                earliest: 0,
                latest: 1
            }
        );
    }

    #[test]
    fn test_settings_survive_a_save_and_load() -> Result<()> {
        let path = temp_dir("roundtrip")?.join("nested").join("plan.json");
        let settings = PlanSettings {
            length: VisitLength::TwoHours,
            workday: HourRange {
                earliest: 17,
                latest: 22,
            },
            weekend: HourRange {
                earliest: 10,
                latest: 18,
            },
        };
        save(&path, &settings)?;
        assert_eq!(load(&path), settings);
        Ok(())
    }

    #[test]
    fn test_missing_or_broken_file_gives_defaults() -> Result<()> {
        let dir = temp_dir("broken")?;
        assert_eq!(load(&dir.join("plan.json")), PlanSettings::default());
        std::fs::create_dir_all(&dir).context("create dir")?;
        let path = dir.join("plan.json");
        std::fs::write(&path, "{ not json").context("write")?;
        assert_eq!(load(&path), PlanSettings::default());
        Ok(())
    }

    #[test]
    fn test_missing_fields_keep_their_defaults() -> Result<()> {
        let dir = temp_dir("partial")?;
        std::fs::create_dir_all(&dir).context("create dir")?;
        let path = dir.join("plan.json");
        std::fs::write(&path, r#"{ "length": "OneHour" }"#).context("write")?;
        let settings = load(&path);
        assert_eq!(settings.length, VisitLength::OneHour);
        assert_eq!(settings.workday, HourRange::default());
        Ok(())
    }

    #[test]
    fn test_default_path_is_next_to_the_config() {
        if let Some(path) = default_path() {
            assert!(path.ends_with("hardy-monitor/plan.json"), "{path:?}");
        }
    }
}
