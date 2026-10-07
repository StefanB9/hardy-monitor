//! The daemon's nightly automatic repair: which days are due, and running it.

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Days, NaiveDate, TimeDelta, Utc};

use super::{DataRepairer, RepairSummary};
use crate::{
    db::{Database, RepairState},
    schedule::GymSchedule,
};

/// A day is repaired this long after the gym closed.
pub const AFTER_CLOSE: TimeDelta = TimeDelta::minutes(15);
/// Days missed during downtime that are caught up at most.
pub const MAX_CATCH_UP_DAYS: u64 = 30;
/// Wait after a failed attempt before trying again.
pub const RETRY_AFTER: TimeDelta = TimeDelta::minutes(30);

/// The gym-local days (inclusive) to repair now, if any: the days after the
/// last repaired one up to the day that closed most recently.
pub fn repair_due(
    now: DateTime<Utc>,
    schedule: &GymSchedule,
    state: &RepairState,
) -> Option<(NaiveDate, NaiveDate)> {
    let failed_recently = state.last_error.is_some()
        && state
            .last_attempt_at
            .is_some_and(|at| now - at < RETRY_AFTER);
    if failed_recently {
        return None;
    }

    // The gym-local day of the most recent closing that is AFTER_CLOSE old;
    // a midnight closing belongs to the day before.
    let closed_at = schedule.last_closing_at_or_before(now - AFTER_CLOSE);
    let last = (closed_at - TimeDelta::minutes(1))
        .with_timezone(&schedule.timezone())
        .date_naive();

    let earliest = last.checked_sub_days(Days::new(MAX_CATCH_UP_DAYS - 1))?;
    let first = match state.repaired_through {
        Some(through) => through.succ_opt()?.max(earliest),
        None => last,
    };
    (first <= last).then_some((first, last))
}

/// Repairs the due days, if any, and records the progress. Returns the
/// repaired range and what was changed.
#[tracing::instrument(skip(db, schedule))]
pub async fn run_nightly_repair(
    db: &Database,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
) -> Result<Option<((NaiveDate, NaiveDate), RepairSummary)>> {
    let state = db.get_repair_state().await?;
    let Some((first, last)) = repair_due(now, schedule, &state) else {
        return Ok(None);
    };
    tracing::info!(%first, %last, "running nightly data repair");

    let repairer = DataRepairer::new(Arc::new(db.clone()), schedule.clone());
    match repairer.repair_date_range(first, last, None).await {
        Ok(summary) => {
            db.record_repair_success(last, now).await?;
            Ok(Some(((first, last), summary)))
        }
        Err(e) => {
            db.record_repair_failure(now, &format!("{e:#}")).await?;
            Err(e).with_context(|| format!("nightly repair of {first}..={last} failed"))
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeZone;
    use proptest::prelude::*;

    use super::*;

    /// 2024-06-`d` `h:mi` gym-local (17th is a Monday).
    fn local(d: u32, h: u32, mi: u32) -> Result<DateTime<Utc>> {
        Ok(GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, d, h, mi, 0)
            .single()
            .context("valid local time")?
            .with_timezone(&Utc))
    }

    fn day(d: u32) -> Result<NaiveDate> {
        NaiveDate::from_ymd_opt(2024, 6, d).context("valid date")
    }

    fn through(d: u32) -> Result<RepairState> {
        Ok(RepairState {
            repaired_through: Some(day(d)?),
            ..RepairState::default()
        })
    }

    #[test]
    fn test_repair_due_first_run_repairs_the_day_that_closed() -> Result<()> {
        let s = GymSchedule::default();
        // Monday closes at 23:00; 15 minutes later Monday is due.
        assert_eq!(
            repair_due(local(17, 23, 20)?, &s, &RepairState::default()),
            Some((day(17)?, day(17)?))
        );
        // Before that, the latest closed day is Sunday.
        assert_eq!(
            repair_due(local(17, 23, 10)?, &s, &RepairState::default()),
            Some((day(16)?, day(16)?))
        );
        Ok(())
    }

    #[test]
    fn test_repair_due_catches_up_missed_days() -> Result<()> {
        let s = GymSchedule::default();
        assert_eq!(
            repair_due(local(18, 10, 0)?, &s, &through(14)?),
            Some((day(15)?, day(17)?))
        );
        assert_eq!(repair_due(local(18, 10, 0)?, &s, &through(17)?), None);
        Ok(())
    }

    #[test]
    fn test_repair_due_caps_catch_up() -> Result<()> {
        let s = GymSchedule::default();
        let old = RepairState {
            repaired_through: NaiveDate::from_ymd_opt(2024, 1, 1),
            ..RepairState::default()
        };
        let (first, last) = repair_due(local(18, 10, 0)?, &s, &old).context("due")?;
        assert_eq!(last, day(17)?);
        assert_eq!((last - first).num_days(), 29);
        Ok(())
    }

    #[test]
    fn test_repair_due_waits_after_failure() -> Result<()> {
        let s = GymSchedule::default();
        let now = local(18, 10, 0)?;
        let failed = |minutes_ago| -> Result<RepairState> {
            Ok(RepairState {
                last_attempt_at: Some(now - TimeDelta::minutes(minutes_ago)),
                last_error: Some("db down".to_string()),
                ..through(14)?
            })
        };
        assert_eq!(repair_due(now, &s, &failed(10)?), None);
        assert!(repair_due(now, &s, &failed(40)?).is_some());
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn repair_due_only_covers_closed_days_after_progress(
            minute in 0i64..(366 * 24 * 60),
            behind in -3i64..100,
        ) {
            let s = GymSchedule::default();
            let tz = s.timezone();
            let now = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).single().unwrap_or_default()
                + TimeDelta::minutes(minute);
            let today = now.with_timezone(&tz).date_naive();
            let state = RepairState {
                repaired_through: Some(today - TimeDelta::days(behind)),
                ..RepairState::default()
            };
            if let Some((first, last)) = repair_due(now, &s, &state) {
                prop_assert!(first <= last);
                prop_assert!((last - first).num_days() < 30);
                prop_assert!(state.repaired_through.is_some_and(|t| t < first));
                // The last day's closing (+ margin) has passed.
                let closing = s.next_closing_after(s.opening_time_on(last));
                prop_assert!(closing + AFTER_CLOSE <= now);
            }
        }
    }
}
