//! Keeping fetch cycles on full minutes.

use std::time::Duration;

const DRIFT_THRESHOLD_SECS: i64 = 5;

pub(crate) fn new_interval(interval_secs: u64) -> tokio::time::Interval {
    let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    interval
}

/// Re-aligns to the next full minute if the timer drifted; returns whether it
/// did.
pub(crate) async fn realign_if_drifted() -> bool {
    let seconds_into_minute = chrono::Utc::now().timestamp() % 60;
    let drift = if seconds_into_minute <= 30 {
        seconds_into_minute
    } else {
        60 - seconds_into_minute
    };

    if drift > DRIFT_THRESHOLD_SECS {
        tracing::warn!(
            drift_secs = drift,
            threshold_secs = DRIFT_THRESHOLD_SECS,
            "timer drift detected, re-syncing"
        );
        wait_for_minute_alignment().await;
        true
    } else {
        tracing::debug!(
            drift_secs = drift,
            threshold_secs = DRIFT_THRESHOLD_SECS,
            "alignment check passed"
        );
        false
    }
}

pub(crate) async fn wait_for_minute_alignment() {
    let now = chrono::Utc::now();
    let seconds_until_next_minute = 60 - (now.timestamp() % 60);
    if seconds_until_next_minute > 0 && seconds_until_next_minute < 60 {
        tracing::info!(
            wait_secs = seconds_until_next_minute,
            "waiting for next full minute"
        );
        let sleep_secs = seconds_until_next_minute.try_into().unwrap_or(0);
        tokio::time::sleep(Duration::from_secs(sleep_secs)).await;
    }
}
