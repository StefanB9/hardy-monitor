//! The forecaster lease: one forecaster per database. It is renewed every
//! minute; a forecaster that loses it, or cannot renew it for as long as it
//! takes to count as abandoned, stops rather than run alongside another.

use chrono::{DateTime, TimeDelta, Utc};

/// A lease not renewed for this long counts as abandoned (its forecaster
/// crashed or lost its connection) and may be taken over.
pub(crate) const STALE_AFTER: TimeDelta = TimeDelta::minutes(5);

/// This process's holder id: host, process and start time, so a restarted
/// forecaster is a new holder.
pub(crate) fn holder_id(now: DateTime<Utc>) -> String {
    let host = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown host".to_string());
    format!(
        "{host} · pid {} · started {}",
        std::process::id(),
        now.format("%Y-%m-%d %H:%M:%S%.3f UTC")
    )
}

/// Why the forecaster must stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lost {
    /// Another forecaster took the lease over.
    TakenOver,
    /// Renewing failed for so long that the lease may be taken over.
    NotRenewed,
}

impl Lost {
    /// Why, for the log.
    pub(crate) fn reason(self) -> &'static str {
        match self {
            Lost::TakenOver => "another forecaster took over the lease",
            Lost::NotRenewed => "the lease could not be renewed for 5 minutes",
        }
    }
}

/// Tracks the lease's renewals.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Keeper {
    last_renewed: DateTime<Utc>,
}

impl Keeper {
    /// A lease claimed at `now`.
    pub(crate) fn new(now: DateTime<Utc>) -> Self {
        Self { last_renewed: now }
    }

    /// Takes in one renewal attempt at `now`: `Ok(true)` renewed,
    /// `Ok(false)` the lease belongs to someone else, `Err` the attempt
    /// failed. A failed attempt is tolerated until the lease could count as
    /// abandoned.
    pub(crate) fn renewed(
        &mut self,
        now: DateTime<Utc>,
        attempt: Result<bool, ()>,
    ) -> Result<(), Lost> {
        match attempt {
            Ok(true) => {
                self.last_renewed = now;
                Ok(())
            }
            Ok(false) => Err(Lost::TakenOver),
            // Stop a minute early: the next renewal would come too late.
            Err(()) if now - self.last_renewed >= STALE_AFTER - TimeDelta::minutes(1) => {
                Err(Lost::NotRenewed)
            }
            Err(()) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::TimeZone;

    use super::*;

    fn at(mi: u32) -> Result<DateTime<Utc>> {
        Utc.with_ymd_and_hms(2026, 10, 16, 12, mi, 0)
            .single()
            .context("valid time")
    }

    #[test]
    fn test_keeper_keeps_a_renewed_lease() -> Result<()> {
        let mut keeper = Keeper::new(at(0)?);
        for mi in 1..20 {
            assert_eq!(keeper.renewed(at(mi)?, Ok(true)), Ok(()));
        }
        Ok(())
    }

    #[test]
    fn test_keeper_stops_when_taken_over() -> Result<()> {
        let mut keeper = Keeper::new(at(0)?);
        assert_eq!(keeper.renewed(at(1)?, Ok(false)), Err(Lost::TakenOver));
        Ok(())
    }

    #[test]
    fn test_keeper_tolerates_failures_until_the_lease_would_go_stale() -> Result<()> {
        let mut keeper = Keeper::new(at(0)?);
        assert_eq!(keeper.renewed(at(1)?, Err(())), Ok(()));
        assert_eq!(keeper.renewed(at(3)?, Err(())), Ok(()));
        // A success in between starts the count again.
        assert_eq!(keeper.renewed(at(4)?, Ok(true)), Ok(()));
        assert_eq!(keeper.renewed(at(7)?, Err(())), Ok(()));
        assert_eq!(keeper.renewed(at(8)?, Err(())), Err(Lost::NotRenewed));
        Ok(())
    }

    #[test]
    fn test_lost_reasons_and_holder_id() -> Result<()> {
        assert!(Lost::TakenOver.reason().contains("took over"));
        assert!(Lost::NotRenewed.reason().contains("5 minutes"));
        let id = holder_id(at(0)?);
        assert!(id.contains(&format!("pid {}", std::process::id())), "{id}");
        assert!(id.contains("2026-10-16 12:00:00.000 UTC"), "{id}");
        Ok(())
    }
}
