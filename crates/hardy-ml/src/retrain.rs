//! When the daemon should (re)train, and with which tuning.

use chrono::{DateTime, Datelike, TimeDelta, Utc, Weekday};
use hardy_core::{GymSchedule, db::MlState};

use crate::{model::RfParams, training::Tuning};

/// Nightly training starts this long after closing.
pub const AFTER_CLOSE: TimeDelta = TimeDelta::minutes(15);
/// Without any model, failed attempts are retried at this interval.
pub const MISSING_MODEL_RETRY: TimeDelta = TimeDelta::hours(6);

/// Why training is due.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrainReason {
    /// No model for the current feature version exists.
    MissingModel,
    /// Requested from the GUI.
    Requested,
    /// The gym closed and the newest model predates that closing.
    Nightly { closed_at: DateTime<Utc> },
}

/// Decides whether to train now. `latest_model_at` is when the newest usable
/// model was trained; `state.last_attempt_at` prevents retry storms after a
/// failed attempt.
pub fn retrain_due(
    now: DateTime<Utc>,
    schedule: &GymSchedule,
    latest_model_at: Option<DateTime<Utc>>,
    state: &MlState,
) -> Option<RetrainReason> {
    let attempted_since = |t: DateTime<Utc>| state.last_attempt_at.is_some_and(|a| a >= t);

    if let Some(requested) = state.retrain_requested_at
        && !attempted_since(requested)
    {
        return Some(RetrainReason::Requested);
    }

    let Some(latest_model_at) = latest_model_at else {
        let retry_due = state
            .last_attempt_at
            .is_none_or(|a| now - a >= MISSING_MODEL_RETRY);
        return retry_due.then_some(RetrainReason::MissingModel);
    };

    let closed_at = schedule.last_closing_at_or_before(now);
    let nightly_due = now >= closed_at + AFTER_CLOSE
        && latest_model_at < closed_at
        && !attempted_since(closed_at);
    nightly_due.then_some(RetrainReason::Nightly { closed_at })
}

/// Grid search on Sunday nights and when no model exists yet; otherwise
/// reuse the newest model's hyperparameters (or the defaults).
pub fn tuning_for(
    reason: RetrainReason,
    schedule: &GymSchedule,
    previous_params: Option<RfParams>,
) -> Tuning {
    let reuse = Tuning::Fixed(previous_params.unwrap_or(RfParams::DEFAULT));
    match reason {
        RetrainReason::MissingModel => Tuning::GridSearch,
        RetrainReason::Requested => reuse,
        RetrainReason::Nightly { closed_at } => {
            let weekday = closed_at.with_timezone(&schedule.timezone()).weekday();
            if weekday == Weekday::Sun {
                Tuning::GridSearch
            } else {
                reuse
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use proptest::prelude::*;

    use super::*;

    /// 2024-06-17 (Monday) + `days` at `h:mi` CEST.
    fn local(days: i64, h: u32, mi: u32) -> DateTime<Utc> {
        GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, 17, h, mi, 0)
            .single()
            .map_or_else(DateTime::default, |t| t.with_timezone(&Utc))
            + TimeDelta::days(days)
    }

    fn state(requested: Option<DateTime<Utc>>, attempted: Option<DateTime<Utc>>) -> MlState {
        MlState {
            retrain_requested_at: requested,
            last_attempt_at: attempted,
            last_error: None,
        }
    }

    #[test]
    fn test_missing_model_trains_and_backs_off() {
        let s = GymSchedule::default();
        let now = local(0, 12, 0);
        assert_eq!(
            retrain_due(now, &s, None, &state(None, None)),
            Some(RetrainReason::MissingModel)
        );
        let recent = state(None, Some(now - TimeDelta::hours(1)));
        assert_eq!(retrain_due(now, &s, None, &recent), None);
        let old = state(None, Some(now - MISSING_MODEL_RETRY));
        assert_eq!(
            retrain_due(now, &s, None, &old),
            Some(RetrainReason::MissingModel)
        );
    }

    #[test]
    fn test_nightly_after_closing_once() {
        let s = GymSchedule::default();
        let trained_this_morning = Some(local(0, 1, 0));
        // Monday closes 23:00; due from 23:15.
        assert_eq!(
            retrain_due(
                local(0, 23, 14),
                &s,
                trained_this_morning,
                &state(None, None)
            ),
            None
        );
        assert_eq!(
            retrain_due(
                local(0, 23, 15),
                &s,
                trained_this_morning,
                &state(None, None)
            ),
            Some(RetrainReason::Nightly {
                closed_at: local(0, 23, 0)
            })
        );
        // Already attempted after closing (e.g. failed): no retry tonight.
        let attempted = state(None, Some(local(0, 23, 15)));
        assert_eq!(
            retrain_due(local(0, 23, 30), &s, trained_this_morning, &attempted),
            None
        );
        // Model newer than closing: nothing to do.
        assert_eq!(
            retrain_due(
                local(0, 23, 30),
                &s,
                Some(local(0, 23, 20)),
                &state(None, None)
            ),
            None
        );
    }

    #[test]
    fn test_nightly_catches_up_after_downtime() {
        let s = GymSchedule::default();
        // Daemon was down overnight; Tuesday 08:00 still owes Monday's run.
        assert_eq!(
            retrain_due(local(1, 8, 0), &s, Some(local(0, 1, 0)), &state(None, None)),
            Some(RetrainReason::Nightly {
                closed_at: local(0, 23, 0)
            })
        );
    }

    #[test]
    fn test_request_wins_until_attempted() {
        let s = GymSchedule::default();
        let now = local(0, 12, 0);
        let requested = state(Some(now - TimeDelta::minutes(1)), None);
        assert_eq!(
            retrain_due(now, &s, Some(local(0, 1, 0)), &requested),
            Some(RetrainReason::Requested)
        );
        let consumed = state(Some(now - TimeDelta::minutes(1)), Some(now));
        assert_eq!(retrain_due(now, &s, Some(local(0, 1, 0)), &consumed), None);
    }

    #[test]
    fn test_tuning_choice() {
        let s = GymSchedule::default();
        let params = RfParams::new(50, 8, 20).ok();
        assert_eq!(
            tuning_for(RetrainReason::MissingModel, &s, params),
            Tuning::GridSearch
        );
        // 2024-06-23 is a Sunday (weekend closing 21:00).
        let sunday_close = local(6, 21, 0);
        assert_eq!(
            tuning_for(
                RetrainReason::Nightly {
                    closed_at: sunday_close
                },
                &s,
                params
            ),
            Tuning::GridSearch
        );
        let monday_close = local(0, 23, 0);
        assert_eq!(
            tuning_for(
                RetrainReason::Nightly {
                    closed_at: monday_close
                },
                &s,
                params
            ),
            Tuning::Fixed(params.unwrap_or(RfParams::DEFAULT))
        );
        assert_eq!(
            tuning_for(RetrainReason::Requested, &s, None),
            Tuning::Fixed(RfParams::DEFAULT)
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        /// Simulating a week minute by minute with successful training,
        /// nightly runs happen at most once per closing and never while
        /// the gym is open.
        #[test]
        fn nightly_runs_once_per_closing(start_offset in 0i64..(24 * 60)) {
            let s = GymSchedule::default();
            let start = local(0, 0, 0) + TimeDelta::minutes(start_offset);
            let mut latest = Some(start);
            let mut st = state(None, None);
            let mut runs: Vec<DateTime<Utc>> = Vec::new();
            let mut now = start;
            while now < start + TimeDelta::days(7) {
                if let Some(reason) = retrain_due(now, &s, latest, &st) {
                    let is_nightly = matches!(reason, RetrainReason::Nightly { .. });
                    prop_assert!(is_nightly);
                    prop_assert!(!s.is_open(&now), "trained while open at {now}");
                    if let RetrainReason::Nightly { closed_at } = reason {
                        prop_assert!(!runs.contains(&closed_at));
                        runs.push(closed_at);
                    }
                    st.last_attempt_at = Some(now);
                    latest = Some(now);
                }
                now += TimeDelta::minutes(5);
            }
            prop_assert!((6..=8).contains(&runs.len()), "{} runs", runs.len());
        }
    }
}
