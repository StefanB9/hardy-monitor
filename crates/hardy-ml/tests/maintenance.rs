//! Training against a real database: storage, state and the background tick.

#[path = "../../hardy-core/tests/common/mod.rs"]
mod common;

use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, TimeDelta, TimeZone, Utc};
use hardy_core::{GymSchedule, MlConfig, db::DataSource};
use hardy_ml::{
    ModelArtifact,
    features::FEATURE_VERSION,
    maintenance::{ModelMaintenance, TrainingOutcome},
    retrain::RetrainReason,
};

/// 2024-06-03 (Monday) + `days`, `h:mi` CEST.
fn local(days: i64, h: u32, mi: u32) -> Result<DateTime<Utc>> {
    Ok(GymSchedule::default()
        .timezone()
        .with_ymd_and_hms(2024, 6, 3, h, mi, 0)
        .single()
        .context("unambiguous local time")?
        .with_timezone(&Utc)
        + TimeDelta::days(days))
}

/// Daily curve with a per-day level, every 5 minutes while open.
async fn insert_learnable_history(db: &hardy_core::Database, days: i64) -> Result<()> {
    let schedule = GymSchedule::default();
    let mut rows = Vec::new();
    for d in 0..days {
        #[allow(clippy::cast_precision_loss)]
        let level = ((d * 37) % 23) as f64 - 11.0;
        for step in 0..(24 * 12) {
            let t = local(d, 0, 0)? + TimeDelta::minutes(step * 5);
            if schedule.is_open(&t) {
                #[allow(clippy::cast_precision_loss)]
                let hour = step as f64 / 12.0;
                let curve = 35.0 + 20.0 * ((hour - 6.0) / 17.0 * std::f64::consts::PI).sin();
                rows.push((t, (curve + level).clamp(0.0, 100.0)));
            }
        }
    }
    db.batch_insert(&rows, DataSource::Measured)
        .await
        .context("insert history")?;
    Ok(())
}

fn config() -> MlConfig {
    MlConfig {
        training_window_days: 56,
        min_samples_for_training: 200,
        ..MlConfig::default()
    }
}

#[tokio::test]
async fn test_train_now_stores_model_and_clears_error() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    insert_learnable_history(&tdb.db, 21).await?;

    let mut maintenance = ModelMaintenance::new(config(), GymSchedule::default());
    let now = local(21, 0, 30)?;
    let outcome = tokio::time::timeout(
        Duration::from_secs(300),
        maintenance.train_now(&tdb.db, RetrainReason::Requested, now),
    )
    .await
    .context("training finishes")?
    .context("training succeeds")?;
    assert!(
        matches!(outcome, TrainingOutcome::Stored { .. }),
        "expected a stored model, got {outcome:?}"
    );
    let TrainingOutcome::Stored { id, artifact } = outcome else {
        return Ok(());
    };
    assert!(artifact.metrics.holdout_mae < artifact.metrics.baseline_mae);

    let info = tdb
        .db
        .latest_model_info(FEATURE_VERSION)
        .await?
        .context("a model is stored")?;
    assert_eq!(info.id, id);
    let restored = ModelArtifact::from_bytes(&tdb.db.load_model(id).await?)?;
    assert_eq!(restored.metrics, artifact.metrics);

    let state = tdb.db.get_ml_state().await?;
    assert_eq!(state.last_attempt_at, Some(now));
    assert_eq!(state.last_error, None);

    tdb.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_train_now_records_rejection() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    insert_learnable_history(&tdb.db, 2).await?;

    let mut maintenance = ModelMaintenance::new(config(), GymSchedule::default());
    let outcome = maintenance
        .train_now(&tdb.db, RetrainReason::Requested, local(2, 0, 30)?)
        .await
        .context("a rejection is not an error")?;
    assert!(matches!(outcome, TrainingOutcome::Rejected(_)));

    assert!(tdb.db.latest_model_info(FEATURE_VERSION).await?.is_none());
    let error = tdb
        .db
        .get_ml_state()
        .await?
        .last_error
        .context("rejection is recorded")?;
    assert!(error.contains("not enough data"), "{error}");

    tdb.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_tick_trains_missing_model_in_background() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    insert_learnable_history(&tdb.db, 21).await?;

    let mut maintenance = ModelMaintenance::new(config(), GymSchedule::default());
    let now = local(21, 0, 30)?;
    maintenance.tick(&tdb.db, now).await?;
    assert!(
        maintenance.is_running(),
        "missing model should start training"
    );

    tokio::time::timeout(Duration::from_secs(600), async {
        while maintenance.is_running() {
            tokio::time::sleep(Duration::from_millis(200)).await;
            maintenance.tick(&tdb.db, now).await?;
        }
        Ok::<(), anyhow::Error>(())
    })
    .await
    .context("training finishes")??;

    let info = tdb.db.latest_model_info(FEATURE_VERSION).await?;
    assert!(
        info.is_some_and(|m| m.tuned),
        "first model is grid-searched"
    );
    // Nothing further is due right after a fresh model.
    maintenance.tick(&tdb.db, now).await?;
    assert!(!maintenance.is_running());

    tdb.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn test_current_model_after_training_and_restart() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    insert_learnable_history(&tdb.db, 21).await?;

    let mut maintenance = ModelMaintenance::new(config(), GymSchedule::default());
    let before = maintenance.current_model().map(|(id, _)| id);
    let outcome = tokio::time::timeout(
        Duration::from_secs(300),
        maintenance.train_now(&tdb.db, RetrainReason::Requested, local(21, 0, 30)?),
    )
    .await
    .context("training finishes")??;
    let TrainingOutcome::Stored { id, .. } = outcome else {
        anyhow::bail!("expected a stored model");
    };
    let after_training = maintenance.current_model().map(|(id, _)| id);

    let mut restarted = ModelMaintenance::new(config(), GymSchedule::default());
    restarted.load_previous(&tdb.db).await?;
    let after_restart = restarted.current_model().map(|(id, _)| id);
    tdb.cleanup().await;

    assert_eq!(before, None);
    assert_eq!(after_training, Some(id));
    assert_eq!(after_restart, Some(id));
    Ok(())
}
