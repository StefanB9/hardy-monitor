//! The non-ML forecast: a recency-weighted weekday × quarter-hour profile,
//! corrected by how much busier or quieter than usual it is right now.

use chrono::{DateTime, Datelike, DurationRound, TimeDelta, Timelike, Utc};
use hardy_core::{GymSchedule, Tz};

use crate::{
    confidence::{PredictionMethod, PredictionWithConfidence},
    history::History,
    samples::{ANCHOR_STEP, TARGET_TOLERANCE},
};

/// Days of history the baseline should be fitted on: enough weeks for the
/// recency weighting to matter.
pub const BASELINE_HISTORY_DAYS: i64 = 28;

/// A reading this old counts half as much as one from today.
const HALF_LIFE_DAYS: f64 = 14.0;
/// "Right now" is the mean of the readings in this window before `now`.
const CURRENT_WINDOW: TimeDelta = TimeDelta::minutes(15);
/// Fewer (anomaly, outcome) pairs than this leave a horizon uncorrected.
const MIN_CARRY_PAIRS: u32 = 20;
/// Overall mean of an empty history.
const NEUTRAL: f64 = 50.0;

const QUARTERS: usize = 7 * 24 * 4;
const HOURS: usize = 7 * 24;

#[derive(Debug, Clone, Copy, Default)]
struct Weighted {
    weight: f64,
    sum: f64,
    sum_sq: f64,
}

impl Weighted {
    fn add(&mut self, weight: f64, value: f64) {
        self.weight += weight;
        self.sum += weight * value;
        self.sum_sq += weight * value * value;
    }

    /// Mean and standard deviation, if anything was added.
    fn stat(self) -> Option<(f64, f64)> {
        (self.weight > 0.0).then(|| {
            let mean = self.sum / self.weight;
            let variance = (self.sum_sq / self.weight - mean * mean).max(0.0);
            (mean, variance.sqrt())
        })
    }
}

/// Fitted baseline; see the module docs.
#[derive(Debug, Clone)]
pub struct Baseline {
    tz: Tz,
    quarters: Vec<Option<(f64, f64)>>,
    hours: Vec<Option<(f64, f64)>>,
    overall: (f64, f64),
    /// Share of the current anomaly still present `h` hours later, at
    /// index `h - 1`.
    carry: Vec<f64>,
}

impl Baseline {
    /// Fits the profile and the per-horizon carry-over on `history`.
    pub fn fit(history: &History, schedule: &GymSchedule, max_hours_ahead: u32) -> Self {
        let tz = schedule.timezone();
        let newest = history.last_time();
        let mut quarters = vec![Weighted::default(); QUARTERS];
        let mut hours = vec![Weighted::default(); HOURS];
        let mut overall = Weighted::default();
        for (t, value) in history.view().iter() {
            #[allow(clippy::cast_precision_loss)]
            let age_days = newest.map_or(0.0, |n| (n - t).num_seconds() as f64 / 86_400.0);
            let weight = 0.5_f64.powf(age_days / HALF_LIFE_DAYS);
            quarters[quarter_index(t, tz)].add(weight, value);
            hours[quarter_index(t, tz) / 4].add(weight, value);
            overall.add(weight, value);
        }
        let mut baseline = Self {
            tz,
            quarters: quarters.into_iter().map(Weighted::stat).collect(),
            hours: hours.into_iter().map(Weighted::stat).collect(),
            overall: overall.stat().unwrap_or((NEUTRAL, 0.0)),
            carry: Vec::new(),
        };
        baseline.carry = baseline.fit_carry(history, schedule, max_hours_ahead);
        baseline
    }

    /// Least-squares share of the anomaly at each anchor that remains at
    /// the target, per horizon, clamped to `[0, 1]`.
    fn fit_carry(
        &self,
        history: &History,
        schedule: &GymSchedule,
        max_hours_ahead: u32,
    ) -> Vec<f64> {
        let horizons = max_hours_ahead as usize;
        let (Some(first), Some(last)) = (history.first_time(), history.last_time()) else {
            return vec![0.0; horizons];
        };
        let all = history.view();
        let mut sums = vec![(0.0_f64, 0.0_f64, 0_u32); horizons];
        let mut anchor = first.duration_trunc(ANCHOR_STEP).unwrap_or(first) + ANCHOR_STEP;
        while anchor <= last {
            if schedule.is_open(&anchor)
                && let Some(anomaly) = self.anomaly(history, anchor)
            {
                for (h, (sxy, sxx, n)) in (1_i64..).zip(sums.iter_mut()) {
                    let target = anchor + TimeDelta::hours(h);
                    if !schedule.is_open(&target) {
                        continue;
                    }
                    if let Some(actual) = all.value_near(target, TARGET_TOLERANCE) {
                        *sxy += anomaly * (actual - self.typical(target).0);
                        *sxx += anomaly * anomaly;
                        *n += 1;
                    }
                }
            }
            anchor += ANCHOR_STEP;
        }
        sums.into_iter()
            .map(|(sxy, sxx, n)| {
                if n < MIN_CARRY_PAIRS || sxx <= f64::EPSILON {
                    0.0
                } else {
                    (sxy / sxx).clamp(0.0, 1.0)
                }
            })
            .collect()
    }

    /// Typical value and spread at `t`: its quarter-hour slot, else its
    /// hour slot, else all readings.
    pub fn typical(&self, t: DateTime<Utc>) -> (f64, f64) {
        let quarter = quarter_index(t, self.tz);
        self.quarters[quarter]
            .or(self.hours[quarter / 4])
            .unwrap_or(self.overall)
    }

    /// Share of the current anomaly assumed to persist `hours_ahead` hours.
    pub fn carry(&self, hours_ahead: u32) -> f64 {
        (hours_ahead as usize)
            .checked_sub(1)
            .and_then(|i| self.carry.get(i))
            .copied()
            .unwrap_or(0.0)
    }

    /// How far the readings just before `now` are above (positive) or below
    /// typical; `None` without a reading in that window.
    fn anomaly(&self, history: &History, now: DateTime<Utc>) -> Option<f64> {
        let current = history
            .view_until(now)
            .mean_between(now - CURRENT_WINDOW, now)?;
        Some(current - self.typical(now).0)
    }

    /// Forecast for `now + hours_ahead`, using only readings up to `now`.
    pub fn predict(&self, history: &History, now: DateTime<Utc>, hours_ahead: u32) -> f64 {
        let target = now + TimeDelta::hours(i64::from(hours_ahead));
        let correction = self
            .anomaly(history, now)
            .map_or(0.0, |anomaly| self.carry(hours_ahead) * anomaly);
        (self.typical(target).0 + correction).clamp(0.0, 100.0)
    }
}

/// Baseline forecasts for each open hour `now + 1h ..= now + max_hours_ahead`,
/// fitted on `history` (ideally [`BASELINE_HISTORY_DAYS`] long).
pub fn baseline_forecast(
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    max_hours_ahead: u32,
) -> Vec<PredictionWithConfidence> {
    let baseline = Baseline::fit(history, schedule, max_hours_ahead);
    (1..=max_hours_ahead)
        .filter_map(|h| {
            let target = now + TimeDelta::hours(i64::from(h));
            schedule.is_open(&target).then(|| {
                let predicted = baseline.predict(history, now, h);
                let (_, spread) = baseline.typical(target);
                PredictionWithConfidence::new(
                    target,
                    predicted,
                    predicted - spread,
                    predicted + spread,
                    0.5,
                    PredictionMethod::HistoricalAverage,
                )
            })
        })
        .collect()
}

fn quarter_index(t: DateTime<Utc>, tz: Tz) -> usize {
    let local = t.with_timezone(&tz);
    let weekday = local.weekday().num_days_from_monday() as usize;
    (weekday * 24 + local.hour() as usize) * 4 + local.minute() as usize / 15
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::{DateTime, TimeDelta, TimeZone, Utc};
    use hardy_core::GymSchedule;
    use proptest::prelude::*;

    use super::*;

    /// 2024-06-`d` `h:mi` gym-local (Monday 2024-06-03 starts the month's
    /// first full week).
    fn local(d: u32, h: u32, mi: u32) -> Result<DateTime<Utc>> {
        Ok(GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, d, h, mi, 0)
            .single()
            .context("valid local time")?
            .with_timezone(&Utc))
    }

    /// Readings every 5 minutes from 10:00 to 18:00 local on days
    /// `first..=last`, valued by `value(day, hour, minute)`.
    fn readings(
        first: u32,
        last: u32,
        value: impl Fn(u32, u32, u32) -> f64,
    ) -> Result<Vec<(DateTime<Utc>, f64)>> {
        let mut points = Vec::new();
        for day in first..=last {
            for hour in 10..18 {
                for minute in (0..60).step_by(5) {
                    points.push((local(day, hour, minute)?, value(day, hour, minute)));
                }
            }
        }
        Ok(points)
    }

    #[test]
    fn test_baseline_resolves_quarter_hours() -> Result<()> {
        // Every hour: quiet first half, busy second half.
        let history = History::new(readings(3, 16, |_, _, m| if m < 30 { 10.0 } else { 50.0 })?);
        let baseline = Baseline::fit(&history, &GymSchedule::default(), 6);
        let (early, _) = baseline.typical(local(17, 12, 5)?);
        let (late, _) = baseline.typical(local(17, 12, 45)?);
        assert!((early - 10.0).abs() < 1e-9, "{early}");
        assert!((late - 50.0).abs() < 1e-9, "{late}");
        Ok(())
    }

    #[test]
    fn test_baseline_weights_recent_weeks_more() -> Result<()> {
        // Two old weeks at 10, then two recent weeks at 50.
        let history = History::new(readings(3, 30, |d, _, _| if d < 17 { 10.0 } else { 50.0 })?);
        let baseline = Baseline::fit(&history, &GymSchedule::default(), 6);
        let (mean, _) = baseline.typical(local(17, 12, 0)?);
        assert!(mean > 35.0, "recent weeks should dominate, got {mean}");
        assert!(mean < 50.0, "older weeks still count, got {mean}");
        Ok(())
    }

    #[test]
    fn test_baseline_carries_a_persistent_anomaly() -> Result<()> {
        // Days alternate between quiet (20) and busy (40) all day long, so
        // being busier than usual now means busier later today too.
        let level = |d: u32| if d.is_multiple_of(2) { 40.0 } else { 20.0 };
        let mut points = readings(3, 28, |d, _, _| level(d))?;
        // Today (an even day: busy) until 12:00.
        points.extend(
            readings(30, 30, |_, _, _| 40.0)?
                .into_iter()
                .filter(|(t, _)| local(30, 12, 0).is_ok_and(|now| *t <= now)),
        );
        let history = History::new(points);
        let baseline = Baseline::fit(&history, &GymSchedule::default(), 6);
        assert!(baseline.carry(1) > 0.8, "carry {}", baseline.carry(1));

        let now = local(30, 12, 0)?;
        let (typical, _) = baseline.typical(now + TimeDelta::hours(2));
        let predicted = baseline.predict(&history, now, 2);
        assert!(
            predicted > typical + 5.0,
            "{predicted} vs typical {typical}"
        );
        assert!((predicted - 40.0).abs() < 3.0, "{predicted}");
        Ok(())
    }

    #[test]
    fn test_baseline_without_a_current_reading_is_the_typical_value() -> Result<()> {
        let history = History::new(readings(3, 16, |d, _, _| f64::from(d))?);
        let baseline = Baseline::fit(&history, &GymSchedule::default(), 6);
        // The last reading is days before `now`.
        let now = local(20, 12, 0)?;
        let (typical, _) = baseline.typical(now + TimeDelta::hours(1));
        assert!((baseline.predict(&history, now, 1) - typical).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn test_baseline_of_empty_history_is_neutral() -> Result<()> {
        let history = History::new(Vec::new());
        let baseline = Baseline::fit(&history, &GymSchedule::default(), 6);
        let now = local(20, 12, 0)?;
        assert!((baseline.predict(&history, now, 1) - 50.0).abs() < 1e-9);
        assert!((baseline.carry(1)).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn test_baseline_forecast_covers_open_hours_only() -> Result<()> {
        let history = History::new(readings(3, 16, |_, _, _| 30.0)?);
        let schedule = GymSchedule::default();
        // 20:30 on a Monday: 21:30 and 22:30 are open, later hours are not.
        let now = local(17, 20, 30)?;
        let points = baseline_forecast(&history, &schedule, now, 6);
        assert_eq!(points.len(), 2);
        assert!(points.iter().all(|p| schedule.is_open(&p.timestamp)));
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn baseline_forecasts_stay_within_bounds(
            raw in prop::collection::vec((0i64..(28 * 24 * 60), 0.0f64..=100.0), 0..300),
            now_minute in 0i64..(29 * 24 * 60),
            hours_ahead in 1u32..=6,
        ) {
            let start = Utc.with_ymd_and_hms(2024, 6, 3, 0, 0, 0).single().unwrap_or_default();
            let history = History::new(
                raw.iter().map(|&(m, v)| (start + TimeDelta::minutes(m), v)).collect(),
            );
            let baseline = Baseline::fit(&history, &GymSchedule::default(), 6);
            let now = start + TimeDelta::minutes(now_minute);
            let predicted = baseline.predict(&history, now, hours_ahead);
            prop_assert!((0.0..=100.0).contains(&predicted), "{predicted}");
            let carry = baseline.carry(hours_ahead);
            prop_assert!((0.0..=1.0).contains(&carry), "{carry}");
            let (mean, spread) = baseline.typical(now);
            prop_assert!((0.0..=100.0).contains(&mean) && spread >= 0.0);
        }
    }
}
