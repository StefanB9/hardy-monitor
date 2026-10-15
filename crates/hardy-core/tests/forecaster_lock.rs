//! Only one forecaster may run against a database at a time.

mod common;

use std::time::Duration;

use anyhow::{Context, Result};

#[tokio::test]
async fn test_forecaster_lock_is_exclusive_until_dropped() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    let db = &tdb.db;
    let first = db.lock_forecaster().await?;
    let second = db.lock_forecaster().await?;
    let held = first.is_some();
    let refused = second.is_none();
    drop(first);
    // The lock goes with the first holder's connection, which the server
    // notices shortly after it closes.
    let third = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Some(lock) = db.lock_forecaster().await? {
                return anyhow::Ok(lock);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .context("lock not released after drop")??;
    drop(third);
    tdb.cleanup().await;
    assert!(held, "the first forecaster gets the lock");
    assert!(refused, "a second one is refused while it is held");
    Ok(())
}
