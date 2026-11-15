//! Decision quality: whether the hour a guide (a forecast, or plain
//! averages) said would be quietest actually turned out quiet.

use std::collections::BTreeMap;

use chrono::{NaiveDate, Timelike};
use chrono_tz::Tz;

use crate::db::ScoredForecast;

/// Quarter-hour slots in the one-hour window looked for.
pub const HOUR_SLOTS: usize = 4;
/// A pick counts as a hit when it was at most this many points busier
/// than the quietest hour that actually happened.
pub const QUIET_TOLERANCE: f64 = 5.0;
/// Quarter hours in a day.
const DAY_SLOTS: usize = 96;

/// A day's guide and readings per quarter hour.
type DaySlots = (Vec<Option<f64>>, Vec<Option<f64>>);

/// One judged pick: the window a guide chose and the actual quietest one,
/// by the index of their first slot, with their mean actual occupancy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    pub picked: usize,
    pub picked_actual: f64,
    pub best: usize,
    pub best_actual: f64,
}

impl Decision {
    /// How many points busier the pick was than the quietest hour.
    pub fn lost(&self) -> f64 {
        self.picked_actual - self.best_actual
    }

    /// Whether the pick was close enough to the quietest hour.
    pub fn hit(&self) -> bool {
        self.lost() <= QUIET_TOLERANCE
    }
}

/// Tallies of judged picks.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DecisionSums {
    pub count: i64,
    pub hits: i64,
    /// Picks that chose exactly the quietest hour.
    pub exact: i64,
    /// Sum of [`Decision::lost`].
    pub lost: f64,
}

impl DecisionSums {
    pub fn add(&mut self, decision: &Decision) {
        self.count += 1;
        self.hits += i64::from(decision.hit());
        self.exact += i64::from(decision.picked == decision.best);
        self.lost += decision.lost();
    }

    /// Share of picks within [`QUIET_TOLERANCE`]; `None` without picks.
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> Option<f64> {
        (self.count > 0).then_some(self.hits as f64 / self.count as f64)
    }

    /// Share of picks that chose the quietest hour exactly.
    #[allow(clippy::cast_precision_loss)]
    pub fn exact_rate(&self) -> Option<f64> {
        (self.count > 0).then_some(self.exact as f64 / self.count as f64)
    }

    /// Mean points lost per pick.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_lost(&self) -> Option<f64> {
        (self.count > 0).then_some(self.lost / self.count as f64)
    }
}

/// The mean of `width` slots from `start`, if every one has a value.
#[allow(clippy::cast_precision_loss)]
fn window_mean(values: &[Option<f64>], start: usize, width: usize) -> Option<f64> {
    let window = values.get(start..start + width)?;
    let sum = window.iter().try_fold(0.0, |sum, v| Some(sum + (*v)?))?;
    Some(sum / width as f64)
}

/// The window of `width` consecutive slots, all with a value, with the
/// lowest mean: its first slot and the mean. The earliest wins a tie.
pub fn quietest_window(values: &[Option<f64>], width: usize) -> Option<(usize, f64)> {
    if width == 0 {
        return None;
    }
    (0..values.len().saturating_sub(width - 1))
        .filter_map(|start| Some((start, window_mean(values, start, width)?)))
        .fold(None, |best: Option<(usize, f64)>, candidate| match best {
            Some(b) if b.1 <= candidate.1 => Some(b),
            _ => Some(candidate),
        })
}

/// Judges the window `guide` says is quietest against `actual`: `None`
/// when either has no full window or the pick has no full readings.
pub fn judge(guide: &[Option<f64>], actual: &[Option<f64>], width: usize) -> Option<Decision> {
    let (picked, _) = quietest_window(guide, width)?;
    let picked_actual = window_mean(actual, picked, width)?;
    let (best, best_actual) = quietest_window(actual, width)?;
    Some(Decision {
        picked,
        picked_actual,
        best,
        best_actual,
    })
}

/// Each gym-local day's pick by plain averages (the quietest hour by the
/// same slots over the four weeks before) judged against the readings.
pub fn average_picks(rows: &[ScoredForecast], tz: Tz) -> Vec<(NaiveDate, Decision)> {
    let mut days: BTreeMap<NaiveDate, DaySlots> = BTreeMap::new();
    for row in rows {
        let local = row.target.with_timezone(&tz);
        let slot = (local.hour() * 4 + local.minute() / 15) as usize;
        let (guide, actual) = days
            .entry(local.date_naive())
            .or_insert_with(|| (vec![None; DAY_SLOTS], vec![None; DAY_SLOTS]));
        if let (Some(g), Some(a)) = (guide.get_mut(slot), actual.get_mut(slot)) {
            *g = row.pair.baseline;
            *a = Some(row.pair.actual);
        }
    }
    days.into_iter()
        .filter_map(|(day, (guide, actual))| Some((day, judge(&guide, &actual, HOUR_SLOTS)?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::{DateTime, TimeZone, Utc};
    use proptest::prelude::*;

    use super::*;
    use crate::accuracy::Pair;

    fn some(values: &[f64]) -> Vec<Option<f64>> {
        values.iter().copied().map(Some).collect()
    }

    #[test]
    fn test_quietest_window_needs_every_slot() {
        let values = vec![Some(9.0), Some(1.0), None, Some(2.0), Some(2.0), Some(4.0)];
        assert_eq!(quietest_window(&values, 2), Some((3, 2.0)));
        assert_eq!(quietest_window(&values, 4), None);
        assert_eq!(quietest_window(&values, 0), None);
        assert_eq!(quietest_window(&[], 1), None);
    }

    #[test]
    fn test_quietest_window_prefers_the_earliest_tie() {
        assert_eq!(
            quietest_window(&some(&[3.0, 1.0, 5.0, 1.0]), 1),
            Some((1, 1.0))
        );
    }

    #[test]
    fn test_judge_scores_the_pick_by_the_readings() -> Result<()> {
        // The guide picks slots 0–1; it really was quietest at 2–3.
        let guide = some(&[10.0, 10.0, 30.0, 30.0]);
        let actual = some(&[20.0, 24.0, 12.0, 12.0]);
        let decision = judge(&guide, &actual, 2).context("judged")?;
        assert_eq!((decision.picked, decision.best), (0, 2));
        assert_relative_eq!(decision.picked_actual, 22.0);
        assert_relative_eq!(decision.best_actual, 12.0);
        assert_relative_eq!(decision.lost(), 10.0);
        assert!(!decision.hit());
        // No reading in the picked window: nothing to judge.
        let gap = vec![None, Some(24.0), Some(12.0), Some(12.0)];
        assert_eq!(judge(&guide, &gap, 2), None);
        Ok(())
    }

    #[test]
    fn test_decision_sums_rates() -> Result<()> {
        let mut sums = DecisionSums::default();
        assert_eq!(sums.hit_rate(), None);
        let exact = Decision {
            picked: 3,
            picked_actual: 10.0,
            best: 3,
            best_actual: 10.0,
        };
        let close = Decision {
            picked: 1,
            picked_actual: 14.0,
            ..exact
        };
        let off = Decision {
            picked: 5,
            picked_actual: 22.0,
            ..exact
        };
        for d in [exact, close, off, exact] {
            sums.add(&d);
        }
        assert_eq!((sums.count, sums.hits, sums.exact), (4, 3, 2));
        assert_relative_eq!(sums.hit_rate().context("rate")?, 0.75);
        assert_relative_eq!(sums.exact_rate().context("exact")?, 0.5);
        assert_relative_eq!(sums.mean_lost().context("lost")?, 4.0);
        Ok(())
    }

    fn scored(target: DateTime<Utc>, actual: f64, baseline: f64) -> ScoredForecast {
        ScoredForecast {
            target,
            made_at: target,
            model: "naive".to_string(),
            pair: Pair {
                actual,
                forecast: actual,
                band: None,
                baseline: Some(baseline),
                last_week: None,
            },
        }
    }

    #[test]
    fn test_average_picks_judge_each_local_day() -> Result<()> {
        let tz: Tz = "Europe/Berlin".parse().unwrap_or(Tz::UTC);
        let at = |d: u32, h: u32, m: u32| {
            Utc.with_ymd_and_hms(2024, 6, d, h, m, 0)
                .single()
                .context("time")
        };
        let mut rows = Vec::new();
        // 17 June, 10:00–11:45 local (08:00 UTC on): averages say the first
        // hour is quiet, the readings the second.
        for i in 0..8 {
            let t = at(17, 8, 0)? + chrono::TimeDelta::minutes(15 * i);
            let (actual, baseline) = if i < 4 { (30.0, 5.0) } else { (10.0, 40.0) };
            rows.push(scored(t, actual, baseline));
        }
        // 18 June: only three slots, no full hour.
        for i in 0..3 {
            rows.push(scored(
                at(18, 8, 0)? + chrono::TimeDelta::minutes(15 * i),
                10.0,
                10.0,
            ));
        }
        let picks = average_picks(&rows, tz);
        assert_eq!(picks.len(), 1);
        let (day, decision) = picks[0];
        assert_eq!(day, NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?);
        // Slot 40 is 10:00 local.
        assert_eq!((decision.picked, decision.best), (40, 44));
        assert_relative_eq!(decision.lost(), 20.0);
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_judging_the_readings_themselves_loses_nothing(
            values in prop::collection::vec(prop::option::of(0.0_f64..100.0), 0..40),
            width in 1_usize..6,
        ) {
            if let Some(decision) = judge(&values, &values, width) {
                prop_assert_eq!(decision.picked, decision.best);
                prop_assert!(decision.lost().abs() < 1e-9);
            }
        }

        #[test]
        fn prop_no_pick_beats_the_quietest_hour(
            guide in prop::collection::vec(prop::option::of(0.0_f64..100.0), 0..40),
            actual in prop::collection::vec(prop::option::of(0.0_f64..100.0), 0..40),
        ) {
            if let Some(decision) = judge(&guide, &actual, HOUR_SLOTS) {
                prop_assert!(decision.lost() >= -1e-9);
            }
        }
    }
}
