//! Typical occupancy per gym-local (weekday, hour) slot.

use chrono::{DateTime, Datelike, Timelike, Utc};
use hardy_core::Tz;
use serde::{Deserialize, Serialize};

use crate::history::HistoryView;

const SLOTS: usize = 7 * 24;

/// Mean and spread of one slot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SlotStat {
    pub mean: f64,
    pub std_dev: f64,
    pub count: u32,
}

/// Slot statistics learned from history. Stored with each model so
/// training and prediction use the same profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotProfile {
    slots: Vec<Option<SlotStat>>,
    overall_mean: f64,
}

impl SlotProfile {
    /// Builds the profile from every reading in `history`, bucketed in `tz`.
    pub fn from_history(history: HistoryView<'_>, tz: Tz) -> Self {
        let mut sums = vec![(0.0_f64, 0.0_f64, 0_u32); SLOTS];
        let (mut total, mut n) = (0.0, 0_u32);
        for (t, v) in history.iter() {
            let slot = &mut sums[slot_index(t, tz)];
            slot.0 += v;
            slot.1 += v * v;
            slot.2 += 1;
            total += v;
            n += 1;
        }
        let slots = sums
            .into_iter()
            .map(|(sum, sum_sq, count)| {
                (count > 0).then(|| {
                    let c = f64::from(count);
                    let mean = sum / c;
                    let variance = (sum_sq / c - mean * mean).max(0.0);
                    SlotStat {
                        mean,
                        std_dev: variance.sqrt(),
                        count,
                    }
                })
            })
            .collect();
        Self {
            slots,
            overall_mean: if n > 0 { total / f64::from(n) } else { 50.0 },
        }
    }

    /// The slot containing `t`, if any reading fell into it.
    pub fn stat(&self, t: DateTime<Utc>, tz: Tz) -> Option<SlotStat> {
        self.slots.get(slot_index(t, tz)).copied().flatten()
    }

    /// Slot mean, or the overall mean for a slot never observed.
    pub fn mean_at(&self, t: DateTime<Utc>, tz: Tz) -> f64 {
        self.stat(t, tz).map_or(self.overall_mean, |s| s.mean)
    }

    /// Mean of all readings; the fallback for slots without data.
    pub fn overall_mean(&self) -> f64 {
        self.overall_mean
    }
}

fn slot_index(t: DateTime<Utc>, tz: Tz) -> usize {
    let local = t.with_timezone(&tz);
    let weekday = local.weekday().num_days_from_monday() as usize;
    weekday * 24 + local.hour() as usize
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::TimeZone;
    use hardy_core::GymSchedule;

    use super::*;
    use crate::history::History;

    #[test]
    fn test_slot_profile_buckets_in_gym_time() -> Result<()> {
        let tz = GymSchedule::default().timezone();
        // Monday 08:10 and 08:40 UTC = 10:10 / 10:40 CEST.
        let t1 = Utc
            .with_ymd_and_hms(2024, 6, 17, 8, 10, 0)
            .single()
            .context("time")?;
        let t2 = t1 + chrono::TimeDelta::minutes(30);
        let h = History::new(vec![(t1, 20.0), (t2, 40.0)]);
        let profile = SlotProfile::from_history(h.view(), tz);

        let stat = profile.stat(t1, tz).context("slot observed")?;
        assert_relative_eq!(stat.mean, 30.0);
        assert_relative_eq!(stat.std_dev, 10.0);
        assert_eq!(stat.count, 2);

        let other = t1 + chrono::TimeDelta::hours(3);
        assert_eq!(profile.stat(other, tz), None);
        assert_relative_eq!(profile.mean_at(other, tz), 30.0);
        Ok(())
    }

    #[test]
    fn test_slot_profile_empty_history_defaults() {
        let tz = GymSchedule::default().timezone();
        let profile = SlotProfile::from_history(History::default().view(), tz);
        assert_relative_eq!(profile.overall_mean(), 50.0);
    }
}
