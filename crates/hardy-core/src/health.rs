//! Data-collection health: turns the daemon's cycle outcomes into one "down"
//! message per outage and a "resumed" message when readings return.

use chrono::{DateTime, TimeDelta, Utc};
use chrono_tz::Tz;

/// ntfy title of health messages.
pub const HEALTH_TITLE: &str = "Hardy Monitor: data collection";

/// Longest failure reason included in a message.
const MAX_REASON_CHARS: usize = 200;

/// Something the user should be told about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthEvent {
    /// Nothing was stored since `since`.
    Down {
        since: DateTime<Utc>,
        reason: String,
    },
    /// Readings are stored again after an outage that began at `since`.
    Recovered {
        since: DateTime<Utc>,
        at: DateTime<Utc>,
    },
    /// The database was migrated by a newer build.
    SchemaAhead { db: i64, app: i64 },
}

impl HealthEvent {
    /// Message text with times in the gym's timezone.
    pub fn body(&self, tz: Tz) -> String {
        let local = |t: DateTime<Utc>| t.with_timezone(&tz).format("%H:%M").to_string();
        match self {
            HealthEvent::Down { since, reason } => {
                let reason: String = reason.chars().take(MAX_REASON_CHARS).collect();
                format!("No readings stored since {}: {reason}", local(*since))
            }
            HealthEvent::Recovered { since, at } => format!(
                "Readings resumed at {} after a gap since {}.",
                local(*at),
                local(*since)
            ),
            HealthEvent::SchemaAhead { db, app } => format!(
                "The database schema ({db}) is newer than this daemon ({app}): update and restart \
                 the daemon."
            ),
        }
    }
}

/// Tracks consecutive failed cycles. Feed it only while the gym is open:
/// closed hours neither count towards an outage nor end one.
#[derive(Debug, Clone)]
pub struct HealthMonitor {
    after_cycles: u32,
    after: TimeDelta,
    failing_since: Option<DateTime<Utc>>,
    failures: u32,
    notified: bool,
}

impl HealthMonitor {
    /// Reports an outage after `after_minutes` failed one-minute cycles (or
    /// that much time, for slower retries such as reconnecting).
    pub fn new(after_minutes: u32) -> Self {
        Self {
            after_cycles: after_minutes.max(1),
            after: TimeDelta::minutes(i64::from(after_minutes.max(1))),
            failing_since: None,
            failures: 0,
            notified: false,
        }
    }

    /// A cycle stored nothing.
    pub fn failure(&mut self, at: DateTime<Utc>, reason: &str) -> Option<HealthEvent> {
        let since = *self.failing_since.get_or_insert(at);
        self.failures += 1;
        if self.notified || (self.failures < self.after_cycles && at - since < self.after) {
            return None;
        }
        self.notified = true;
        Some(HealthEvent::Down {
            since,
            reason: reason.to_string(),
        })
    }

    /// A cycle stored (or found) a reading.
    pub fn success(&mut self, at: DateTime<Utc>) -> Option<HealthEvent> {
        let since = self.failing_since.take();
        let was_reported = std::mem::take(&mut self.notified);
        self.failures = 0;
        since
            .filter(|_| was_reported)
            .map(|since| HealthEvent::Recovered { since, at })
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use proptest::prelude::*;

    use super::*;

    fn tz() -> Tz {
        chrono_tz::Europe::Berlin
    }

    /// 2024-06-17 18:00 UTC (20:00 in Berlin) plus `minutes`.
    fn at(minutes: i64) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2024, 6, 17, 18, 0, 0)
            .single()
            .unwrap_or_default()
            + TimeDelta::minutes(minutes)
    }

    #[test]
    fn test_health_reports_once_after_threshold() {
        let mut h = HealthMonitor::new(5);
        for m in 0..4 {
            assert_eq!(h.failure(at(m), "insert failed"), None, "minute {m}");
        }
        assert_eq!(
            h.failure(at(4), "insert failed"),
            Some(HealthEvent::Down {
                since: at(0),
                reason: "insert failed".to_string()
            })
        );
        for m in 5..30 {
            assert_eq!(h.failure(at(m), "insert failed"), None);
        }
    }

    #[test]
    fn test_health_recovery_only_after_reported_outage() {
        let mut h = HealthMonitor::new(5);
        h.failure(at(0), "x");
        h.failure(at(1), "x");
        assert_eq!(h.success(at(2)), None, "short blip is not reported");

        for m in 3..10 {
            h.failure(at(m), "x");
        }
        assert_eq!(
            h.success(at(10)),
            Some(HealthEvent::Recovered {
                since: at(3),
                at: at(10)
            })
        );
        assert_eq!(h.success(at(11)), None);
    }

    #[test]
    fn test_health_time_threshold_for_slow_retries() {
        // Reconnect attempts back off: 3 failures over 6 minutes is an
        // outage.
        let mut h = HealthMonitor::new(5);
        assert_eq!(h.failure(at(0), "db down"), None);
        assert_eq!(h.failure(at(2), "db down"), None);
        assert!(matches!(
            h.failure(at(6), "db down"),
            Some(HealthEvent::Down { .. })
        ));
    }

    #[test]
    fn test_health_body_texts() {
        let down = HealthEvent::Down {
            since: at(39),
            reason: "x".repeat(500),
        };
        let body = down.body(tz());
        assert!(
            body.starts_with("No readings stored since 20:39: xxx"),
            "{body}"
        );
        assert!(body.chars().count() < 260, "reason is truncated");

        let up = HealthEvent::Recovered {
            since: at(39),
            at: at(161),
        };
        assert_eq!(
            up.body(tz()),
            "Readings resumed at 22:41 after a gap since 20:39."
        );

        let ahead = HealthEvent::SchemaAhead { db: 7, app: 5 }.body(tz());
        assert!(ahead.contains("update and restart"), "{ahead}");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn health_events_alternate_down_then_recovered(
            outcomes in proptest::collection::vec(any::<bool>(), 0..200),
            after in 1u32..10,
        ) {
            let mut h = HealthMonitor::new(after);
            let mut down = false;
            let mut run = 0u32;
            for (m, ok) in outcomes.iter().enumerate() {
                let t = at(i64::try_from(m).unwrap_or_default());
                if *ok {
                    let event = h.success(t);
                    prop_assert_eq!(event.is_some(), down);
                    down = false;
                    run = 0;
                } else {
                    run += 1;
                    let event = h.failure(t, "x");
                    let expect_down = !down && run >= after;
                    prop_assert_eq!(event.is_some(), expect_down);
                    if expect_down {
                        down = true;
                    }
                }
            }
        }
    }
}
