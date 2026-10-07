//! CSV export of all readings.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use futures::TryStreamExt;

use super::{DataSource, Database, OccupancyLog};
use crate::traits::Clock;

impl Database {
    /// Writes every reading to a timestamped CSV file in `output_dir`; returns
    /// its path.
    #[tracing::instrument(skip_all, fields(db.operation = "export_csv", output_dir = %output_dir.display()))]
    pub async fn export_to_csv(&self, output_dir: &Path, clock: &dyn Clock) -> Result<PathBuf> {
        let export_time = clock.now_utc();
        let filename = format!(
            "hardy_monitor_export_{}.csv",
            export_time.format("%Y%m%d_%H%M%S")
        );
        let output_path = output_dir.join(&filename);

        let (tx, mut rx) = tokio::sync::mpsc::channel::<OccupancyLog>(256);

        let path = output_path.clone();
        let writer_task = tokio::task::spawn_blocking(move || -> Result<()> {
            let mut wtr = csv::Writer::from_path(&path).context("Failed to create CSV writer")?;
            while let Some(log) = rx.blocking_recv() {
                wtr.serialize(log)
                    .context("Failed to serialize log entry")?;
            }
            wtr.flush().context("Failed to flush CSV writer")
        });

        let mut stream = sqlx::query_as!(
            OccupancyLog,
            r#"
            SELECT
                id as "id!",
                timestamp as "timestamp!",
                percentage as "percentage!",
                source as "source!: DataSource"
            FROM occupancy_logs
            ORDER BY timestamp ASC
            "#
        )
        .fetch(&self.pool);

        while let Some(log) = stream
            .try_next()
            .await
            .context("Failed to stream record during export")?
        {
            if tx.send(log).await.is_err() {
                break;
            }
        }

        drop(tx);

        writer_task
            .await
            .context("CSV export writer task panicked")??;

        Ok(output_path)
    }
}
