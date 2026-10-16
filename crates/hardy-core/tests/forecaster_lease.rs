//! Only one forecaster may run against a database at a time.

mod common;

use anyhow::{Context, Result};
use chrono::TimeDelta;

const STALE: TimeDelta = TimeDelta::minutes(5);

#[tokio::test]
async fn test_forecaster_lease_is_exclusive_until_released() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let db = &tdb.db;
    let first = db.claim_forecaster_lease("first", STALE).await?;
    let second = db.claim_forecaster_lease("second", STALE).await?;
    let renewed = match &first {
        Some(lease) => db.renew_forecaster_lease(lease).await?,
        None => false,
    };
    // The holder may claim again (a restart that kept its id).
    let again = db.claim_forecaster_lease("first", STALE).await?;
    if let Some(lease) = first {
        db.release_forecaster_lease(lease).await?;
    }
    let after_release = db.claim_forecaster_lease("third", STALE).await?;
    tdb.cleanup().await;

    assert!(again.is_some(), "the holder can claim its own lease");
    assert!(renewed, "the holder renews its lease");
    assert!(second.is_none(), "a second forecaster is refused");
    assert!(after_release.is_some(), "a released lease is free");
    Ok(())
}

#[tokio::test]
async fn test_forecaster_lease_is_taken_over_once_stale() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let db = &tdb.db;
    let first = db
        .claim_forecaster_lease("crashed", STALE)
        .await?
        .context("first claim")?;
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    // Not renewed for longer than the (here tiny) staleness limit.
    let takeover = db
        .claim_forecaster_lease("new", TimeDelta::milliseconds(10))
        .await?;
    let old_renews = db.renew_forecaster_lease(&first).await?;
    // Releasing a lease that was taken over leaves the new holder alone.
    db.release_forecaster_lease(first).await?;
    let blocked = db.claim_forecaster_lease("third", STALE).await?;
    tdb.cleanup().await;

    assert!(takeover.is_some(), "a stale lease is taken over");
    assert!(!old_renews, "the old holder learns it lost the lease");
    assert!(blocked.is_none(), "the new holder keeps the lease");
    Ok(())
}
