//! Training samples: (features at anchor `t`, actual value at `t + h`).

use chrono::{DateTime, DurationRound, TimeDelta, Utc};
use hardy_core::GymSchedule;

use crate::{
    features::{self, FeatureRow},
    history::History,
    profile::SlotProfile,
};

/// Spacing of forecast anchors in training data.
pub const ANCHOR_STEP: TimeDelta = TimeDelta::minutes(15);

/// How far a reading may be from `t + h` to serve as its target.
pub const TARGET_TOLERANCE: TimeDelta = TimeDelta::minutes(5);

/// One training example.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub anchor: DateTime<Utc>,
    pub hours_ahead: u32,
    pub features: FeatureRow,
    pub target: f64,
}

/// Which anchors and targets to use.
#[derive(Debug, Clone, Copy)]
pub struct SampleWindow {
    /// First anchor (inclusive).
    pub anchors_from: DateTime<Utc>,
    /// Last anchor (exclusive).
    pub anchors_until: DateTime<Utc>,
    /// Targets must lie before this instant (keeps holdout data out of
    /// training samples).
    pub targets_until: DateTime<Utc>,
}

/// Builds samples for anchors every [`ANCHOR_STEP`] while the gym is open,
/// for each horizon `1..=max_hours_ahead` whose target time is open and has
/// a reading. Samples are ordered by anchor time.
pub fn build_samples(
    history: &History,
    profile: &SlotProfile,
    schedule: &GymSchedule,
    window: SampleWindow,
    max_hours_ahead: u32,
) -> Vec<Sample> {
    let all = history.view();
    let mut samples = Vec::new();
    let mut anchor = window
        .anchors_from
        .duration_trunc(ANCHOR_STEP)
        .unwrap_or(window.anchors_from);
    if anchor < window.anchors_from {
        anchor += ANCHOR_STEP;
    }

    while anchor < window.anchors_until {
        if schedule.is_open(&anchor) {
            for hours_ahead in 1..=max_hours_ahead {
                let target_time = anchor + TimeDelta::hours(i64::from(hours_ahead));
                if target_time >= window.targets_until || !schedule.is_open(&target_time) {
                    continue;
                }
                let Some(target) = all.value_near(target_time, TARGET_TOLERANCE) else {
                    continue;
                };
                if let Some(features) =
                    features::extract(history, profile, schedule, anchor, hours_ahead)
                {
                    samples.push(Sample {
                        anchor,
                        hours_ahead,
                        features,
                        target,
                    });
                }
            }
        }
        anchor += ANCHOR_STEP;
    }
    samples
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use proptest::prelude::*;

    use super::*;

    fn local(days: i64, h: u32, mi: u32) -> DateTime<Utc> {
        GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, 17, h, mi, 0)
            .single()
            .map_or_else(DateTime::default, |t| t.with_timezone(&Utc))
            + TimeDelta::days(days)
    }

    fn day_of_readings(days: i64) -> History {
        let mut points = Vec::new();
        for d in 0..days {
            for minute in 0..(17 * 60) {
                points.push((local(d, 6, 0) + TimeDelta::minutes(minute), 30.0));
            }
        }
        History::new(points)
    }

    #[test]
    fn test_build_samples_only_open_targets_with_readings() {
        let schedule = GymSchedule::default();
        let history = day_of_readings(1);
        let profile = SlotProfile::from_history(history.view(), schedule.timezone());
        let window = SampleWindow {
            anchors_from: local(0, 20, 0),
            anchors_until: local(0, 20, 1),
            targets_until: local(1, 0, 0),
        };
        let samples = build_samples(&history, &profile, &schedule, window, 6);
        // Anchor 20:00 → targets 21:00, 22:00, 23:00 (closing minute) are
        // open; 00:00+ are closed.
        let horizons: Vec<u32> = samples.iter().map(|s| s.hours_ahead).collect();
        assert_eq!(horizons, vec![1, 2, 3]);
    }

    #[test]
    fn test_build_samples_respects_targets_until() {
        let schedule = GymSchedule::default();
        let history = day_of_readings(1);
        let profile = SlotProfile::from_history(history.view(), schedule.timezone());
        let window = SampleWindow {
            anchors_from: local(0, 10, 0),
            anchors_until: local(0, 10, 1),
            targets_until: local(0, 12, 30),
        };
        let horizons: Vec<u32> = build_samples(&history, &profile, &schedule, window, 6)
            .iter()
            .map(|s| s.hours_ahead)
            .collect();
        assert_eq!(horizons, vec![1, 2]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn samples_are_well_formed(
            from_minute in 0i64..(17 * 60),
            span_minutes in 0i64..300,
            max_hours_ahead in 1u32..=6,
        ) {
            let schedule = GymSchedule::default();
            let history = day_of_readings(1);
            let profile = SlotProfile::from_history(history.view(), schedule.timezone());
            let anchors_from = local(0, 6, 0) + TimeDelta::minutes(from_minute);
            let window = SampleWindow {
                anchors_from,
                anchors_until: anchors_from + TimeDelta::minutes(span_minutes),
                targets_until: local(1, 0, 0),
            };
            let samples = build_samples(&history, &profile, &schedule, window, max_hours_ahead);
            for pair in samples.windows(2) {
                prop_assert!(pair[0].anchor <= pair[1].anchor);
            }
            for s in &samples {
                prop_assert!(s.anchor >= window.anchors_from && s.anchor < window.anchors_until);
                prop_assert!((1..=max_hours_ahead).contains(&s.hours_ahead));
                let target_time = s.anchor + TimeDelta::hours(i64::from(s.hours_ahead));
                prop_assert!(schedule.is_open(&target_time));
                prop_assert!(target_time < window.targets_until);
                prop_assert!((s.features.hours_ahead() - f64::from(s.hours_ahead)).abs() < f64::EPSILON);
            }
        }
    }
}
