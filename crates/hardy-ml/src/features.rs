//! Features for forecasting occupancy `h` hours after an anchor time `t`.
//!
//! Every value is computed from readings at or before `t` (via
//! [`History::view_until`]) plus calendar facts about the target time, so the
//! same code serves training and live prediction without leaking the target.

use std::f64::consts::TAU;

use chrono::{DateTime, Datelike, NaiveTime, TimeDelta, TimeZone, Timelike, Utc};
use hardy_core::{GymSchedule, schedule::is_bavarian_holiday};

use crate::{history::History, profile::SlotProfile};

/// Version of the feature definition. Models trained with a different
/// version are not used.
pub const FEATURE_VERSION: u32 = 2;

/// Number of features per row.
pub const NUM_FEATURES: usize = 18;

/// Names in row order, for logs and diagnostics.
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "target_hour_sin",
    "target_hour_cos",
    "target_weekday_sin",
    "target_weekday_cos",
    "target_is_weekend",
    "target_is_holiday",
    "target_hours_to_close",
    "hours_ahead",
    "target_slot_mean",
    "target_slot_std",
    "current",
    "mean_1h",
    "mean_3h",
    "trend_1h",
    "day_mean_so_far",
    "current_deviation",
    "yesterday_same_time",
    "last_week_same_time",
];

/// How old the latest reading may be for a forecast anchored at `t`.
pub const MAX_CURRENT_AGE: TimeDelta = TimeDelta::minutes(10);

/// Tolerance when looking up "same time yesterday / last week".
const SAME_TIME_TOLERANCE: TimeDelta = TimeDelta::minutes(10);

/// One feature vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureRow(pub [f64; NUM_FEATURES]);

impl FeatureRow {
    /// The feature values in model input order.
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    /// The `hours_ahead` feature.
    pub fn hours_ahead(&self) -> f64 {
        self.0[7]
    }

    /// The target slot mean (the baseline forecast).
    pub fn target_slot_mean(&self) -> f64 {
        self.0[8]
    }
}

/// Features for forecasting `target = anchor + hours_ahead`.
///
/// Returns `None` when there is no reading within [`MAX_CURRENT_AGE`] before
/// the anchor — without a current state there is nothing to forecast from.
pub fn extract(
    history: &History,
    profile: &SlotProfile,
    schedule: &GymSchedule,
    anchor: DateTime<Utc>,
    hours_ahead: u32,
) -> Option<FeatureRow> {
    let tz = schedule.timezone();
    let past = history.view_until(anchor);
    let (current_time, current) = past.latest()?;
    if anchor - current_time > MAX_CURRENT_AGE {
        return None;
    }

    let target = anchor + TimeDelta::hours(i64::from(hours_ahead));
    let local_target = target.with_timezone(&tz);
    let target_date = local_target.date_naive();

    let hour_fraction = f64::from(local_target.hour()) + f64::from(local_target.minute()) / 60.0;
    let weekday = f64::from(local_target.weekday().num_days_from_monday());
    let is_weekend = local_target.weekday().number_from_monday() > 5;
    #[allow(clippy::cast_precision_loss)]
    let hours_to_close =
        ((schedule.next_closing_after(target) - target).num_minutes() as f64 / 60.0).max(0.0);

    let target_stat = profile.stat(target, tz);
    let target_slot_mean = target_stat.map_or(profile.overall_mean(), |s| s.mean);
    let target_slot_std = target_stat.map_or(0.0, |s| s.std_dev);

    let mean_1h = past
        .mean_between(anchor - TimeDelta::hours(1), anchor)
        .unwrap_or(current);
    let mean_3h = past
        .mean_between(anchor - TimeDelta::hours(3), anchor)
        .unwrap_or(mean_1h);
    let trend_1h = past
        .slope_per_hour(anchor - TimeDelta::hours(1), anchor)
        .unwrap_or(0.0);
    let day_start = local_midnight(anchor, schedule);
    let day_mean_so_far = past.mean_between(day_start, anchor).unwrap_or(current);
    let current_deviation = current - profile.mean_at(current_time, tz);

    // Both lookups end before the anchor for any horizon under 24 h; the view
    // enforces it regardless.
    let yesterday_same_time = past
        .value_near(target - TimeDelta::days(1), SAME_TIME_TOLERANCE)
        .unwrap_or(target_slot_mean);
    let last_week_same_time = past
        .value_near(target - TimeDelta::days(7), SAME_TIME_TOLERANCE)
        .unwrap_or(target_slot_mean);

    Some(FeatureRow([
        (TAU * hour_fraction / 24.0).sin(),
        (TAU * hour_fraction / 24.0).cos(),
        (TAU * weekday / 7.0).sin(),
        (TAU * weekday / 7.0).cos(),
        flag(is_weekend),
        flag(is_bavarian_holiday(target_date)),
        hours_to_close,
        f64::from(hours_ahead),
        target_slot_mean,
        target_slot_std,
        current,
        mean_1h,
        mean_3h,
        trend_1h,
        day_mean_so_far,
        current_deviation,
        yesterday_same_time,
        last_week_same_time,
    ]))
}

fn flag(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
}

/// Start of the gym-local calendar day containing `t`.
fn local_midnight(t: DateTime<Utc>, schedule: &GymSchedule) -> DateTime<Utc> {
    let tz = schedule.timezone();
    let date = t.with_timezone(&tz).date_naive();
    tz.from_local_datetime(&date.and_time(NaiveTime::MIN))
        .earliest()
        .map_or(t, |midnight| midnight.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    /// Monday 2024-06-17 `h:mi` CEST plus `days`.
    fn local(days: i64, h: u32, mi: u32) -> DateTime<Utc> {
        GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, 17, h, mi, 0)
            .single()
            .map_or_else(DateTime::default, |t| t.with_timezone(&Utc))
            + TimeDelta::days(days)
    }

    /// One reading per minute for `days` days, 06:00–23:00 local, with a
    /// simple daily pattern.
    fn synthetic(days: i64) -> History {
        let mut points = Vec::new();
        for d in 0..days {
            for minute in 0..(17 * 60) {
                let t = local(d, 6, 0) + TimeDelta::minutes(minute);
                #[allow(clippy::cast_precision_loss)]
                let v = 20.0 + 30.0 * ((minute as f64) / 1020.0 * std::f64::consts::PI).sin();
                points.push((t, v));
            }
        }
        History::new(points)
    }

    #[test]
    fn test_feature_names_match_row_length() {
        assert_eq!(FEATURE_NAMES.len(), NUM_FEATURES);
    }

    #[test]
    fn test_extract_requires_recent_reading() {
        let schedule = GymSchedule::default();
        let h = History::new(vec![(local(0, 10, 0), 30.0)]);
        let profile = SlotProfile::from_history(h.view(), schedule.timezone());
        assert!(extract(&h, &profile, &schedule, local(0, 10, 10), 1).is_some());
        assert!(extract(&h, &profile, &schedule, local(0, 10, 11), 1).is_none());
        assert!(extract(&h, &profile, &schedule, local(0, 9, 59), 1).is_none());
    }

    #[test]
    fn test_extract_values_for_known_series() -> Result<()> {
        let schedule = GymSchedule::default();
        let h = synthetic(8);
        let profile = SlotProfile::from_history(h.view(), schedule.timezone());
        let anchor = local(7, 12, 0);
        let row = extract(&h, &profile, &schedule, anchor, 2).context("features")?;

        assert_relative_eq!(row.hours_ahead(), 2.0);
        // Target 14:00 → 9 h to closing at 23:00.
        assert_relative_eq!(row.0[6], 9.0);
        let current = h
            .view()
            .value_near(anchor, TimeDelta::zero())
            .context("reading at anchor")?;
        assert_relative_eq!(row.0[10], current);
        // The pattern repeats daily, so yesterday/last week equal the target.
        let target_value = h
            .view()
            .value_near(anchor + TimeDelta::hours(2), TimeDelta::zero())
            .context("target reading")?;
        assert_relative_eq!(row.0[16], target_value, epsilon = 1e-9);
        assert_relative_eq!(row.0[17], target_value, epsilon = 1e-9);
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Features depend only on readings at or before the anchor: adding
        /// or changing later readings never changes them.
        #[test]
        fn features_never_read_after_anchor(
            values in prop::collection::vec(0.0f64..100.0, 30..400),
            anchor_minute in 0i64..400,
            future in prop::collection::vec(0.0f64..100.0, 1..50),
            hours_ahead in 1u32..=6,
        ) {
            let schedule = GymSchedule::default();
            let start = local(0, 6, 0);
            let points: Vec<_> = values.iter().enumerate()
                .map(|(i, &v)| (start + TimeDelta::minutes(i64::try_from(i).unwrap_or(0) * 3), v))
                .collect();
            let anchor = start + TimeDelta::minutes(anchor_minute * 3);

            let past_only: Vec<_> = points.iter().copied().filter(|(t, _)| *t <= anchor).collect();
            let mut with_other_future = past_only.clone();
            with_other_future.extend(future.iter().enumerate().map(|(i, &v)| {
                (anchor + TimeDelta::minutes(i64::try_from(i).unwrap_or(0) + 1), v)
            }));

            let profile = SlotProfile::from_history(History::new(past_only.clone()).view(), schedule.timezone());
            let a = extract(&History::new(points), &profile, &schedule, anchor, hours_ahead);
            let b = extract(&History::new(past_only), &profile, &schedule, anchor, hours_ahead);
            let c = extract(&History::new(with_other_future), &profile, &schedule, anchor, hours_ahead);
            prop_assert_eq!(a, b);
            prop_assert_eq!(b, c);
        }
    }
}
