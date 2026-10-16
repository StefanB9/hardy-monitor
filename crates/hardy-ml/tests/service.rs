//! The forecasting service against a real database: the nightly run, its
//! stored results and live publishing.

#[path = "../../hardy-core/tests/common/mod.rs"]
mod common;

use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result};
use chrono::{DateTime, TimeDelta, TimeZone, Timelike, Utc};
use hardy_core::{
    GymSchedule, MlConfig,
    db::{DataSource, FitStatus, ModelUpdate},
};
use hardy_ml::{
    live::SELECTED,
    progress::{Progress, Stage},
    service::ForecastService,
    suite::{ARTIFACT_VERSION, ModelId},
};

const MODELS: [ModelId; 4] = [
    ModelId::Naive,
    ModelId::DailySeasonalNaive,
    ModelId::WeeklySeasonalNaive,
    ModelId::ProfileDecayingOffset,
];

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

/// The same daily curve every day, one reading per 5 minutes while open.
async fn insert_days(db: &hardy_core::Database, days: i64) -> Result<()> {
    let schedule = GymSchedule::default();
    let tz = schedule.timezone();
    let start = local(0, 0, 0)?;
    let mut rows = Vec::new();
    for minute in (0..days * 24 * 60).step_by(5) {
        let t = start + TimeDelta::minutes(minute);
        if schedule.is_open(&t) {
            // The hour of the slot the reading falls into, so slot means
            // repeat exactly from day to day.
            let slot = hardy_ml::grid::slot_end_of(t) - TimeDelta::minutes(1);
            let hour = f64::from(slot.with_timezone(&tz).hour());
            rows.push((t, 10.0 + 3.0 * hour));
        }
    }
    db.batch_insert(&rows, DataSource::Measured)
        .await
        .context("insert history")?;
    Ok(())
}

fn service() -> ForecastService {
    ForecastService::with_models(&MlConfig::default(), GymSchedule::default(), &MODELS)
}

#[tokio::test]
async fn test_run_now_stores_states_benchmark_selection_and_forecasts() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    insert_days(&tdb.db, 5).await?;
    let mut service = service();
    service.load(&tdb.db).await?;
    let now = local(5, 0, 30)?;
    assert_eq!((service.last_run_at(), service.fitted_count()), (None, 0));
    tokio::time::timeout(Duration::from_secs(120), service.run_now(&tdb.db, now))
        .await
        .context("run finishes")??;
    assert_eq!(service.last_run_at(), Some(now));
    // The weekly model waits for more history.
    assert_eq!(service.fitted_count(), MODELS.len() - 1);

    let records = tdb.db.model_records().await?;
    let run = tdb.db.latest_benchmark().await?;
    let selection = tdb.db.forecast_selection().await?;
    let state = tdb.db.get_ml_state().await?;
    let live = tdb.db.live_forecasts(&[SELECTED.to_string()]).await?;
    tdb.cleanup().await;

    assert_eq!(records.len(), MODELS.len());
    let status = |key: &str| records.iter().find(|r| r.model == key).map(|r| r.status);
    assert_eq!(status("naive"), Some(FitStatus::Fitted));
    // Five days: too little for the weekly model.
    assert_eq!(status("weekly_seasonal_naive"), Some(FitStatus::Waiting));
    let run = run.context("benchmark stored")?;
    assert_eq!(run.days, 5, "complete days before now");
    assert!(run.origins > 0);
    assert_eq!(selection.len(), 3);
    assert!(
        selection.iter().all(|(_, m)| m == "daily_seasonal_naive"),
        "{selection:?}"
    );
    assert_eq!(state.last_error, None);
    assert_ne!(live, Vec::new());
    assert!(
        live.iter()
            .all(|f| f.source.as_deref() == Some("daily_seasonal_naive"))
    );
    Ok(())
}

/// Counts the steps of each stage reported to it.
#[derive(Default)]
struct Steps(std::sync::Mutex<Vec<(Stage, usize, usize)>>);

impl Progress for Steps {
    fn start(&self, stage: Stage, total: usize) {
        if let Ok(mut steps) = self.0.lock() {
            steps.push((stage, total, 0));
        }
    }

    fn advance(&self, stage: Stage) {
        if let Ok(mut steps) = self.0.lock()
            && let Some(entry) = steps.iter_mut().rev().find(|(s, _, _)| *s == stage)
        {
            entry.2 += 1;
        }
    }

    fn finish(&self, _: Stage) {}
}

#[tokio::test]
async fn test_a_run_reports_its_progress() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    insert_days(&tdb.db, 5).await?;
    let steps = Arc::new(Steps::default());
    let mut service = service().with_progress(steps.clone());
    service.load(&tdb.db).await?;
    tokio::time::timeout(
        Duration::from_secs(120),
        service.run_now(&tdb.db, local(5, 0, 30)?),
    )
    .await
    .context("run finishes")??;
    tdb.cleanup().await;
    let steps = steps
        .0
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned"))?
        .clone();
    assert!(steps.contains(&(Stage::Backtest, 5, 5)), "{steps:?}");
    assert!(
        steps.contains(&(Stage::Fit, MODELS.len(), MODELS.len())),
        "{steps:?}"
    );
    Ok(())
}

#[tokio::test]
async fn test_tick_runs_in_the_background_when_models_are_new() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    insert_days(&tdb.db, 3).await?;
    let mut service = service();
    service.load(&tdb.db).await?;
    let now = local(3, 12, 0)?;

    service.tick(&tdb.db, now).await?;
    let started = service.is_running();
    let finished = tokio::time::timeout(Duration::from_secs(120), async {
        while service.is_running() {
            tokio::time::sleep(Duration::from_millis(50)).await;
            service.tick(&tdb.db, now).await?;
        }
        anyhow::Ok(())
    })
    .await;
    let records = tdb.db.model_records().await?;
    tdb.cleanup().await;

    assert!(started, "unknown models start a run at once");
    finished.context("run finishes")??;
    assert_eq!(records.len(), MODELS.len());
    Ok(())
}

#[tokio::test]
async fn test_an_unreadable_stored_model_is_refitted() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    insert_days(&tdb.db, 3).await?;
    let now = local(3, 12, 0)?;
    service().run_now(&tdb.db, now).await?;
    // A stored model this build cannot read must not count as fitted.
    tdb.db
        .save_model_states(
            &[ModelUpdate {
                model: "naive",
                status: FitStatus::Fitted,
                message: None,
                fitted_at: Some(now),
                data_from: Some(now),
                data_until: Some(now),
                artifact: Some((ARTIFACT_VERSION, b"not a model")),
            }],
            now,
        )
        .await?;
    let mut restarted = service();
    restarted.load(&tdb.db).await?;
    restarted.tick(&tdb.db, now + TimeDelta::hours(7)).await?;
    let started = restarted.is_running();
    tdb.cleanup().await;

    assert!(started, "a model without a usable fit triggers a run");
    Ok(())
}

#[tokio::test]
async fn test_a_run_interrupted_by_a_restart_is_rerun() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    insert_days(&tdb.db, 3).await?;
    service().run_now(&tdb.db, local(3, 12, 0)?).await?;
    // After closing a run starts, then the daemon stops before it ends.
    let started = local(3, 23, 20)?;
    tdb.db.start_training_attempt(started).await?;

    let mut restarted = service();
    restarted.load(&tdb.db).await?;
    restarted
        .tick(&tdb.db, started + TimeDelta::minutes(5))
        .await?;
    let running = restarted.is_running();
    tdb.cleanup().await;

    assert!(running, "the interrupted run starts again");
    Ok(())
}

#[tokio::test]
async fn test_models_added_by_an_update_run_at_once() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    insert_days(&tdb.db, 3).await?;
    let now = local(3, 12, 0)?;
    // An older build knew only some of the models.
    ForecastService::with_models(&MlConfig::default(), GymSchedule::default(), &MODELS[..2])
        .run_now(&tdb.db, now)
        .await?;

    let mut updated = service();
    updated.load(&tdb.db).await?;
    // Well within the retry spacing for missing models.
    updated.tick(&tdb.db, now + TimeDelta::minutes(30)).await?;
    let running = updated.is_running();
    tdb.cleanup().await;

    assert!(running, "new models are fitted without waiting");
    Ok(())
}

#[tokio::test]
async fn test_ensembles_are_stored_reloaded_and_published() -> Result<()> {
    let tdb = common::TestDatabase::new().await?;
    insert_days(&tdb.db, 8).await?;
    let models = [
        ModelId::Naive,
        ModelId::DailySeasonalNaive,
        ModelId::BestMemberEnsemble,
        ModelId::StackedEnsemble,
    ];
    let make =
        || ForecastService::with_models(&MlConfig::default(), GymSchedule::default(), &models);
    let now = local(8, 0, 30)?;
    tokio::time::timeout(Duration::from_secs(120), make().run_now(&tdb.db, now))
        .await
        .context("run finishes")??;
    let records = tdb.db.model_records().await?;

    // A restarted daemon reads the stored ensembles back: no new run is
    // needed, and they are published from the members' forecasts.
    let mut restarted = make();
    restarted.load(&tdb.db).await?;
    restarted
        .tick(&tdb.db, now + TimeDelta::minutes(30))
        .await?;
    let running = restarted.is_running();
    let later = local(8, 10, 0)?;
    restarted.publish(&tdb.db, later).await?;
    let keys = ["ensemble_best_member", "daily_seasonal_naive"].map(String::from);
    let live = tdb.db.live_forecasts(&keys).await?;
    tdb.cleanup().await;

    let status = |key: &str| records.iter().find(|r| r.model == key).map(|r| r.status);
    assert_eq!(status("ensemble_best_member"), Some(FitStatus::Fitted));
    assert_eq!(status("ensemble_stacked"), Some(FitStatus::Fitted));
    assert!(!running, "stored ensembles count as known");
    let values = |key: &str| -> Vec<(DateTime<Utc>, f64)> {
        live.iter()
            .filter(|f| f.model == key)
            .map(|f| (f.target, f.value))
            .collect()
    };
    // Every day repeats exactly, so the daily seasonal naive is the best
    // member in every bucket.
    assert_ne!(values("ensemble_best_member"), Vec::new());
    assert_eq!(
        values("ensemble_best_member"),
        values("daily_seasonal_naive")
    );
    Ok(())
}
