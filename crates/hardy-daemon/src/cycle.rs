//! One fetch cycle: fetch the reading, store it in its minute slot, log the
//! outcome.

use std::time::Duration;

use anyhow::Result;
use hardy_core::{AppError, retry};

use crate::Worker;

/// How long an in-flight fetch may continue after a shutdown request; stays
/// below Docker's default 10 s stop timeout.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Fetches and stores one reading within the cycle budget. On shutdown the
/// in-flight cycle gets a short grace period; `true` means stop afterwards.
pub(crate) async fn run_cycle(
    worker: &Worker<'_>,
    slot: chrono::DateTime<chrono::Utc>,
    mut shutdown: std::pin::Pin<&mut impl std::future::Future<Output = ()>>,
) -> (
    Result<Result<Stored, AppError>, tokio::time::error::Elapsed>,
    bool,
) {
    let cycle = fetch_and_store(worker, slot);
    tokio::pin!(cycle);
    let first = tokio::select! {
        result = tokio::time::timeout(worker.cycle_budget, &mut cycle) => Some(result),
        () = &mut shutdown => None,
    };
    if let Some(result) = first {
        return (result, false);
    }
    tracing::info!("shutdown requested, finishing in-flight fetch");
    // SAFETY: if the grace period also expires, the cycle is dropped. A
    // single INSERT commits atomically or not at all and the minute slot is
    // unique, so no partial or duplicate row can result.
    (tokio::time::timeout(SHUTDOWN_GRACE, &mut cycle).await, true)
}

pub(crate) fn log_outcome(
    slot: chrono::DateTime<chrono::Utc>,
    outcome: Result<Result<Stored, AppError>, tokio::time::error::Elapsed>,
) {
    match outcome {
        Ok(Ok(Stored::Inserted(percentage))) => {
            tracing::info!(%slot, occupancy_pct = percentage, "recorded occupancy");
        }
        Ok(Ok(Stored::SlotTaken(percentage))) => {
            tracing::debug!(%slot, occupancy_pct = percentage, "minute slot already filled");
        }
        Ok(Err(AppError::Validation(reason))) => {
            tracing::warn!(%slot, %reason, "rejected invalid reading");
        }
        Ok(Err(e)) => {
            tracing::error!(%slot, error = %e, "failed to fetch/store data");
        }
        Err(_) => {
            tracing::error!(%slot, "fetch cycle timed out");
        }
    }
}

/// What happened to a fetched reading.
pub(crate) enum Stored {
    Inserted(f64),
    SlotTaken(f64),
}

#[tracing::instrument(skip_all, fields(%slot))]
async fn fetch_and_store(
    worker: &Worker<'_>,
    slot: chrono::DateTime<chrono::Utc>,
) -> Result<Stored, AppError> {
    let response = retry(&worker.fetch_policy, "fetch_occupancy", || {
        worker.api_client.fetch_occupancy()
    })
    .await?;
    let percentage = response.occupancy_percentage()?;

    let inserted = retry(&worker.fetch_policy, "insert_record", || async {
        worker
            .database
            .insert_record(slot, percentage)
            .await
            .map_err(|e| AppError::from_anyhow_sqlx(&e, "insert_record"))
    })
    .await?;

    Ok(if inserted.is_some() {
        Stored::Inserted(percentage)
    } else {
        Stored::SlotTaken(percentage)
    })
}
