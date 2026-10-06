//! Runs alerts in the daemon: applies phone commands to the shared settings
//! and delivers alerts via ntfy.

use chrono::{DateTime, Utc};

use super::{
    command::{ControlCommand, USAGE},
    engine::{Alert, AlertEngine, AlertRules},
    settings::{AlertSettings, SettingsSource},
};
use crate::{
    config::{NetworkConfig, NotificationConfig},
    db::Database,
    error::AppError,
    ntfy::{NtfyClient, PollSince},
    schedule::GymSchedule,
};

/// Alert state for one daemon process.
#[derive(Debug)]
pub struct AlertService {
    engine: AlertEngine,
    schedule: GymSchedule,
    ntfy: Option<NtfyClient>,
    alert_topic: Option<String>,
    control_topic: Option<String>,
    control_cursor: PollSince,
}

impl AlertService {
    /// `now` is the start time: commands sent before it are never replayed.
    pub fn new(
        config: &NotificationConfig,
        network: &NetworkConfig,
        schedule: GymSchedule,
        now: DateTime<Utc>,
    ) -> Result<Self, AppError> {
        let rules = AlertRules::new(
            config.cooldown_secs,
            config.opening_grace_minutes,
            config.windows.clone(),
        )?;
        let ntfy = if config.ntfy_topic.is_some() || config.control_topic.is_some() {
            let token = config.ntfy_token.as_ref().map(|t| t.expose().to_string());
            Some(NtfyClient::new(&config.ntfy_server, token, network)?)
        } else {
            None
        };
        Ok(Self {
            engine: AlertEngine::new(rules),
            schedule,
            ntfy,
            alert_topic: config.ntfy_topic.clone(),
            control_topic: config.control_topic.clone(),
            control_cursor: PollSince::Time(now),
        })
    }

    /// Fetches new phone commands, applies them and replies on the alert
    /// topic. Returns the number of commands applied.
    #[tracing::instrument(skip_all)]
    pub async fn process_commands(
        &mut self,
        db: &Database,
        now: DateTime<Utc>,
    ) -> Result<usize, AppError> {
        let (Some(ntfy), Some(control_topic)) = (&self.ntfy, &self.control_topic) else {
            return Ok(0);
        };
        let messages = ntfy.poll(control_topic, &self.control_cursor).await?;

        let mut applied = 0;
        for message in messages {
            let reply = match ControlCommand::parse(&message.message) {
                Ok(command) => {
                    tracing::info!(%command, "applying phone command");
                    let settings = self.apply(db, command, now).await?;
                    applied += 1;
                    format!("Alerts: {}", settings.describe(now, &self.schedule))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "ignoring invalid phone command");
                    USAGE.to_string()
                }
            };
            self.publish(&reply).await;
            // Advance only once handled: if applying fails above, the next
            // poll retries this command.
            self.control_cursor = PollSince::Id(message.id);
        }
        Ok(applied)
    }

    async fn apply(
        &self,
        db: &Database,
        command: ControlCommand,
        now: DateTime<Utc>,
    ) -> Result<AlertSettings, AppError> {
        let current = load_settings(db).await?;
        let updated = match command {
            ControlCommand::On {
                threshold_percent,
                duration,
            } => AlertSettings::armed(
                threshold_percent.unwrap_or(current.threshold_percent()),
                duration,
                now,
                &self.schedule,
                SettingsSource::Phone,
            )?,
            ControlCommand::Off => current.disarmed(now, SettingsSource::Phone),
            ControlCommand::Status => return Ok(current),
        };
        db.save_alert_settings(&updated)
            .await
            .map_err(|e| AppError::from_anyhow_sqlx(&e, "save_alert_settings"))?;
        Ok(updated)
    }

    /// Feeds a stored reading to the engine and publishes the alert it
    /// produces, if any.
    #[tracing::instrument(skip(self, db))]
    pub async fn process_reading(
        &mut self,
        db: &Database,
        percentage: f64,
        now: DateTime<Utc>,
    ) -> Result<Option<Alert>, AppError> {
        let settings = load_settings(db).await?;
        let alert = self
            .engine
            .observe(percentage, now, &settings, &self.schedule);
        if let Some(alert) = alert {
            tracing::info!(percentage, "sending low-occupancy alert");
            self.publish(&alert.body()).await;
        }
        Ok(alert)
    }

    /// Delivery failures are logged, never propagated: alerts must not
    /// disturb data collection.
    async fn publish(&self, body: &str) {
        let (Some(ntfy), Some(topic)) = (&self.ntfy, &self.alert_topic) else {
            return;
        };
        if let Err(e) = ntfy.publish(topic, Alert::TITLE, body).await {
            tracing::warn!(error = %e, "failed to publish to ntfy");
        }
    }
}

async fn load_settings(db: &Database) -> Result<AlertSettings, AppError> {
    db.get_alert_settings()
        .await
        .map_err(|e| AppError::from_anyhow_sqlx(&e, "get_alert_settings"))
}
