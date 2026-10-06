//! Persistence of the single `alert_settings` row.

use anyhow::{Context, Result};

use super::Database;
use crate::alert::{AlertSettings, SettingsSource};

impl Database {
    /// Loads the alert settings (the migration guarantees the row exists).
    #[tracing::instrument(skip_all, fields(db.operation = "get_alert_settings"))]
    pub async fn get_alert_settings(&self) -> Result<AlertSettings> {
        let row = sqlx::query!(
            r#"
            SELECT
                enabled,
                threshold_percent,
                active_until,
                updated_at,
                updated_by as "updated_by: SettingsSource"
            FROM alert_settings
            WHERE id = 1
            "#
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to load alert settings")?;

        AlertSettings::new(
            row.enabled,
            row.threshold_percent,
            row.active_until,
            row.updated_at,
            row.updated_by,
        )
        .context("Stored alert settings are invalid")
    }

    /// Replaces the alert settings; the last writer wins.
    #[tracing::instrument(skip_all, fields(
        db.operation = "save_alert_settings",
        enabled = settings.enabled(),
        updated_by = settings.updated_by().as_str(),
    ))]
    pub async fn save_alert_settings(&self, settings: &AlertSettings) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE alert_settings
            SET enabled = $1,
                threshold_percent = $2,
                active_until = $3,
                updated_at = $4,
                updated_by = $5
            WHERE id = 1
            "#,
            settings.enabled(),
            settings.threshold_percent(),
            settings.active_until(),
            settings.updated_at(),
            settings.updated_by().as_str()
        )
        .execute(&self.pool)
        .await
        .context("Failed to save alert settings")?;
        Ok(())
    }
}
