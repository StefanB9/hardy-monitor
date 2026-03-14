pub mod cross_validation;
pub(crate) mod data_prep;
pub(crate) mod hyperparameter;

use chrono::Duration;

use self::{
    cross_validation::{CrossValidationScores, FoldScores, TimeSeriesSplit},
    data_prep::{TrainingDataPreparer, estimate_samples_per_hour},
    hyperparameter::{
        GridSearchResult, HyperparameterSet, collect_cv_residuals, grid_search_with_grid,
    },
};
use super::{
    MlConfig,
    config::MlAlgorithm,
    evaluation,
    features::FeatureExtractor,
    model::{ModelBuilder, TrainedModel, TrainingError},
    persistence::{ModelSummary, PersistedModel, SerializedSlotStats},
    residuals::ResidualQuantiles,
};
use crate::{
    db::{Database, HourlyAverage, OccupancyLog},
    schedule::GymSchedule,
    traits::Clock,
};

#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub model: TrainedModel,
    pub feature_extractor: FeatureExtractor,
    pub persisted: PersistedModel,
    pub cv_scores: Option<CrossValidationScores>,
    pub best_hyperparameters: Option<HyperparameterSet>,
    pub feature_importance: Option<Vec<(String, f64)>>,
    pub oob_error: Option<f64>,
    pub residual_quantiles: Option<ResidualQuantiles>,
}

/// Build `PersistedModel` and `FeatureExtractor` from training artifacts.
fn build_training_result(
    model: TrainedModel,
    baseline: &[HourlyAverage],
    config: &MlConfig,
    cv_scores: Option<CrossValidationScores>,
    best_hyperparameters: Option<HyperparameterSet>,
    residual_quantiles: Option<ResidualQuantiles>,
) -> TrainingResult {
    let mut feature_extractor = FeatureExtractor::new();
    feature_extractor.update_historical_stats(baseline);

    let slot_stats: Vec<SerializedSlotStats> = baseline
        .iter()
        .map(|avg| SerializedSlotStats {
            weekday: avg.weekday.cast_unsigned(),
            hour: avg.hour.cast_unsigned(),
            mean: avg.avg_percentage,
            std_dev: 10.0,
            sample_count: avg.sample_count,
        })
        .collect();

    let persisted = PersistedModel::new(
        config.training_window_days,
        model.training_samples,
        model.training_mse,
        model.validation_mse,
        slot_stats,
        ModelSummary {
            model_type: model.model_type().to_string(),
            max_depth: best_hyperparameters
                .as_ref()
                .map(|p| p.max_depth)
                .or(Some(10)),
            feature_importance: model.feature_importance(),
        },
        residual_quantiles.as_ref(),
    );

    TrainingResult {
        model,
        feature_extractor,
        persisted,
        cv_scores,
        best_hyperparameters,
        feature_importance: None,
        oob_error: None,
        residual_quantiles,
    }
}

/// Train a Random Forest model with optional grid-search hyperparameter
/// tuning.
fn train_rf_with_tuning(
    features: &[super::features::PredictionFeatures],
    targets: &[f64],
    config: &MlConfig,
    logs: &[OccupancyLog],
    grid: &[HyperparameterSet],
) -> Result<
    (
        TrainedModel,
        Option<CrossValidationScores>,
        Option<HyperparameterSet>,
        Option<ResidualQuantiles>,
    ),
    TrainingError,
> {
    if config.tune_hyperparameters {
        let sph = estimate_samples_per_hour(logs);
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let gap_samples = (config.cv_gap_hours.unsigned_abs() as usize) * sph;
        let splitter = TimeSeriesSplit::new(config.cv_folds, gap_samples);
        let folds = splitter.and_then(|s| s.split(features.len()));

        if let Some(folds) = folds {
            let GridSearchResult {
                best_params,
                best_cv_scores,
                ..
            } = grid_search_with_grid(features, targets, &folds, grid)?;

            // Collect residuals from CV with best config
            let residual_quantiles = collect_cv_residuals(&best_params, features, targets, &folds)
                .ok()
                .and_then(|r| ResidualQuantiles::from_residuals(&r));

            // Retrain on full data with best params
            let builder = ModelBuilder::new()
                .n_trees(best_params.n_trees)
                .max_depth(best_params.max_depth)
                .min_samples_leaf(best_params.min_samples_leaf)
                .max_features(best_params.max_features);

            let model = builder.train_rf(features, targets)?;
            Ok((
                model,
                Some(best_cv_scores),
                Some(best_params),
                residual_quantiles,
            ))
        } else {
            // Not enough data for CV — train with defaults
            let model = ModelBuilder::new()
                .n_trees(100)
                .max_depth(10)
                .min_samples_leaf(2)
                .train_rf(features, targets)?;
            Ok((model, None, None, None))
        }
    } else {
        // No tuning — train RF with defaults
        let model = ModelBuilder::new()
            .n_trees(100)
            .max_depth(10)
            .min_samples_leaf(2)
            .train_rf(features, targets)?;
        Ok((model, None, None, None))
    }
}

/// Run cross-validation for linear regression and collect residuals.
#[allow(
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn compute_lr_cv_scores(
    features: &[super::features::PredictionFeatures],
    targets: &[f64],
    config: &MlConfig,
    logs: &[OccupancyLog],
) -> Option<(CrossValidationScores, Option<ResidualQuantiles>)> {
    let sph = estimate_samples_per_hour(logs);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let gap_samples = (config.cv_gap_hours.unsigned_abs() as usize) * sph;
    let splitter = TimeSeriesSplit::new(config.cv_folds, gap_samples)?;
    let folds = splitter.split(features.len())?;

    let mut mse_scores = Vec::with_capacity(folds.len());
    let mut rmse_scores = Vec::with_capacity(folds.len());
    let mut mae_scores = Vec::with_capacity(folds.len());
    let mut r2_scores = Vec::with_capacity(folds.len());

    let total_val: usize = folds.iter().map(|f| f.val_end - f.val_start).sum();
    let mut residuals = Vec::with_capacity(total_val);

    for fold in &folds {
        let train_features = &features[fold.train_start..fold.train_end];
        let train_targets = &targets[fold.train_start..fold.train_end];
        let val_features = &features[fold.val_start..fold.val_end];
        let val_targets = &targets[fold.val_start..fold.val_end];

        let builder = ModelBuilder::new().ridge_lambda(1e-3);
        let model = builder.train(train_features, train_targets).ok()?;
        let predictions = model.predict_batch(val_features);

        mse_scores.push(evaluation::mse(&predictions, val_targets).unwrap_or(f64::MAX));
        rmse_scores.push(evaluation::rmse(&predictions, val_targets).unwrap_or(f64::MAX));
        mae_scores.push(evaluation::mae(&predictions, val_targets).unwrap_or(f64::MAX));
        r2_scores
            .push(evaluation::r_squared(&predictions, val_targets).unwrap_or(f64::NEG_INFINITY));

        // Collect residuals for quantile computation
        for (i, (&actual, predicted)) in val_targets.iter().zip(predictions.iter()).enumerate() {
            let feature = &val_features[i];
            let weekday = feature.raw_weekday as u32;
            let hour = feature.raw_hour as u32;
            residuals.push((weekday, hour, actual - predicted));
        }
    }

    let cv_scores = CrossValidationScores {
        mse: FoldScores::from_scores(mse_scores),
        rmse: FoldScores::from_scores(rmse_scores),
        mae: FoldScores::from_scores(mae_scores),
        r_squared: FoldScores::from_scores(r2_scores),
    };

    let quantiles = ResidualQuantiles::from_residuals(&residuals);

    Some((cv_scores, quantiles))
}

pub async fn train_model(
    db: &Database,
    clock: &dyn Clock,
    schedule: &GymSchedule,
    config: &MlConfig,
) -> Result<TrainingResult, TrainingError> {
    let end = clock.now_utc();
    let start = end - Duration::days(config.training_window_days);

    let logs = db
        .get_history_range(start, end)
        .await
        .map_err(|e| TrainingError::FitError(format!("Database error: {e}")))?;

    if logs.len() < config.min_samples_for_training {
        return Err(TrainingError::InsufficientData(logs.len()));
    }

    let baseline = db
        .get_averages_range(start, end)
        .await
        .map_err(|e| TrainingError::FitError(format!("Database error: {e}")))?;

    let schedule = schedule.clone();
    let config = config.clone();

    tokio::task::spawn_blocking(move || train_model_sync(&logs, &baseline, &schedule, &config))
        .await
        .map_err(|e| TrainingError::FitError(format!("Task join error: {e}")))?
}

pub fn train_model_sync(
    logs: &[OccupancyLog],
    baseline: &[HourlyAverage],
    schedule: &GymSchedule,
    config: &MlConfig,
) -> Result<TrainingResult, TrainingError> {
    if logs.len() < config.min_samples_for_training {
        return Err(TrainingError::InsufficientData(logs.len()));
    }

    let preparer = TrainingDataPreparer::new(config.clone());
    let (features, targets) = preparer.prepare(logs, baseline, schedule)?;

    let default_grid = HyperparameterSet::default_grid();
    let (model, cv_scores, best_params, residual_quantiles) = match config.algorithm {
        MlAlgorithm::RandomForest => {
            train_rf_with_tuning(&features, &targets, config, logs, &default_grid)?
        }
        MlAlgorithm::LinearRegression => {
            let builder = ModelBuilder::new().ridge_lambda(1e-3);
            let model = builder.train_with_validation(&features, &targets, 0.2)?;
            let (cv_scores, quantiles) = compute_lr_cv_scores(&features, &targets, config, logs)
                .map_or((None, None), |(scores, q)| (Some(scores), q));
            (model, cv_scores, None, quantiles)
        }
    };

    Ok(build_training_result(
        model,
        baseline,
        config,
        cv_scores,
        best_params,
        residual_quantiles,
    ))
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use chrono::{TimeZone, Utc};

    use super::*;
    use crate::ml::config::MlAlgorithm;

    fn create_test_logs(n: i32) -> Vec<OccupancyLog> {
        let base_time = Utc.with_ymd_and_hms(2024, 6, 1, 6, 0, 0).unwrap();

        (0..n)
            .map(|i| {
                let timestamp = base_time + Duration::hours(i64::from(i));
                let hour = (6 + i) % 24;
                let weekday = f64::from((i / 24) % 7);
                let percentage =
                    30.0 + (f64::from(hour) * 2.0) + (weekday * 3.0) + f64::from(i % 10);
                OccupancyLog {
                    id: i64::from(i),
                    timestamp,
                    percentage: percentage.min(95.0),
                }
            })
            .collect()
    }

    fn create_test_baseline() -> Vec<HourlyAverage> {
        let mut baseline = Vec::new();
        for weekday in 0..7 {
            for hour in 0..24 {
                baseline.push(HourlyAverage {
                    weekday,
                    hour,
                    avg_percentage: 40.0 + f64::from(hour) + (f64::from(weekday) * 2.0),
                    sample_count: 10,
                });
            }
        }
        baseline
    }

    // ── train_model_sync — RF path ───────────────────────────────────

    #[test]
    fn test_train_rf_with_tuning_small_grid() -> Result<()> {
        let config = MlConfig {
            min_samples_for_training: 100,
            training_window_days: 28,
            algorithm: MlAlgorithm::RandomForest,
            tune_hyperparameters: true,
            cv_folds: 3,
            cv_gap_hours: 0,
            ..Default::default()
        };

        let logs = create_test_logs(1000);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let preparer = TrainingDataPreparer::new(config.clone());
        let (features, targets) = preparer.prepare(&logs, &baseline, &schedule)?;

        let small_grid = HyperparameterSet::small_grid();
        let (model, cv_scores, best_params, residual_quantiles) =
            train_rf_with_tuning(&features, &targets, &config, &logs, &small_grid)?;

        assert!(model.training_samples >= 100);
        assert_eq!(model.model_type(), "RandomForest");
        assert!(cv_scores.is_some());
        assert!(best_params.is_some());

        // Wrap in full TrainingResult to verify field integration
        let result = build_training_result(
            model,
            &baseline,
            &config,
            cv_scores,
            best_params,
            residual_quantiles,
        );
        assert!(result.cv_scores.is_some());
        assert!(result.best_hyperparameters.is_some());
        assert!(result.feature_importance.is_none());
        assert!(result.oob_error.is_none());

        Ok(())
    }

    #[test]
    fn test_train_model_sync_rf_no_tuning() -> Result<()> {
        let config = MlConfig {
            min_samples_for_training: 100,
            training_window_days: 28,
            algorithm: MlAlgorithm::RandomForest,
            tune_hyperparameters: false,
            ..Default::default()
        };

        let logs = create_test_logs(1000);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let result = train_model_sync(&logs, &baseline, &schedule, &config)?;

        assert_eq!(result.model.model_type(), "RandomForest");
        assert!(result.cv_scores.is_none());
        assert!(result.best_hyperparameters.is_none());

        Ok(())
    }

    // ── train_model_sync — LR path ───────────────────────────────────

    #[test]
    fn test_train_model_sync_lr_path() -> Result<()> {
        let config = MlConfig {
            min_samples_for_training: 100,
            training_window_days: 28,
            algorithm: MlAlgorithm::LinearRegression,
            cv_folds: 3,
            cv_gap_hours: 0,
            ..Default::default()
        };

        let logs = create_test_logs(1000);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let result = train_model_sync(&logs, &baseline, &schedule, &config)?;

        assert_eq!(result.model.model_type(), "LinearRegression");
        // LR path should produce CV scores but no best_hyperparameters
        assert!(result.cv_scores.is_some());
        assert!(result.best_hyperparameters.is_none());

        Ok(())
    }

    // ── train_model_sync — insufficient data ─────────────────────────

    #[test]
    fn test_train_model_sync_insufficient_data() {
        let config = MlConfig {
            min_samples_for_training: 1000,
            ..Default::default()
        };

        let logs = create_test_logs(100);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let result = train_model_sync(&logs, &baseline, &schedule, &config);

        assert!(matches!(result, Err(TrainingError::InsufficientData(_))));
    }

    #[test]
    fn test_train_model_sync_insufficient_for_cv() -> Result<()> {
        let config = MlConfig {
            min_samples_for_training: 5,
            training_window_days: 28,
            algorithm: MlAlgorithm::RandomForest,
            tune_hyperparameters: true,
            cv_folds: 4,
            cv_gap_hours: 24,
            ..Default::default()
        };

        // Only 10 logs — enough to train but not enough for 4-fold CV with gap
        let logs = create_test_logs(10);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let result = train_model_sync(&logs, &baseline, &schedule, &config)?;

        // Falls back to default RF training, no CV
        assert_eq!(result.model.model_type(), "RandomForest");
        assert!(result.cv_scores.is_none());
        assert!(result.best_hyperparameters.is_none());

        Ok(())
    }

    // ── LR CV scores ─────────────────────────────────────────────────

    #[test]
    fn test_lr_cv_scores_basic() {
        let config = MlConfig {
            min_samples_for_training: 100,
            cv_folds: 3,
            cv_gap_hours: 0,
            ..Default::default()
        };

        let logs = create_test_logs(1000);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let preparer = TrainingDataPreparer::new(config.clone());
        let (features, targets) = preparer
            .prepare(&logs, &baseline, &schedule)
            .unwrap_or_default();

        let result = compute_lr_cv_scores(&features, &targets, &config, &logs);
        assert!(result.is_some());

        let (scores, _quantiles) = result.unwrap_or_else(|| {
            (
                CrossValidationScores {
                    mse: FoldScores::from_scores(vec![]),
                    rmse: FoldScores::from_scores(vec![]),
                    mae: FoldScores::from_scores(vec![]),
                    r_squared: FoldScores::from_scores(vec![]),
                },
                None,
            )
        });
        assert_eq!(scores.mse.per_fold.len(), 3);
        assert!(scores.mse.mean >= 0.0);
    }

    // ── Residual quantile integration ────────────────────────────────

    #[test]
    fn test_train_rf_with_tuning_produces_residual_quantiles() -> Result<()> {
        let config = MlConfig {
            min_samples_for_training: 100,
            training_window_days: 28,
            algorithm: MlAlgorithm::RandomForest,
            tune_hyperparameters: true,
            cv_folds: 3,
            cv_gap_hours: 0,
            ..Default::default()
        };

        let logs = create_test_logs(1000);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let preparer = TrainingDataPreparer::new(config.clone());
        let (features, targets) = preparer.prepare(&logs, &baseline, &schedule)?;

        let small_grid = HyperparameterSet::small_grid();
        let (_, _, _, residual_quantiles) =
            train_rf_with_tuning(&features, &targets, &config, &logs, &small_grid)?;

        assert!(
            residual_quantiles.is_some(),
            "RF with tuning should produce residual quantiles"
        );

        Ok(())
    }

    #[test]
    fn test_train_rf_no_tuning_no_residual_quantiles() -> Result<()> {
        let config = MlConfig {
            min_samples_for_training: 100,
            training_window_days: 28,
            algorithm: MlAlgorithm::RandomForest,
            tune_hyperparameters: false,
            ..Default::default()
        };

        let logs = create_test_logs(1000);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let preparer = TrainingDataPreparer::new(config.clone());
        let (features, targets) = preparer.prepare(&logs, &baseline, &schedule)?;

        let small_grid = HyperparameterSet::small_grid();
        let (_, _, _, residual_quantiles) =
            train_rf_with_tuning(&features, &targets, &config, &logs, &small_grid)?;

        assert!(
            residual_quantiles.is_none(),
            "RF without tuning should have no residual quantiles"
        );

        Ok(())
    }

    #[test]
    fn test_train_lr_with_cv_produces_residual_quantiles() {
        let config = MlConfig {
            min_samples_for_training: 100,
            cv_folds: 3,
            cv_gap_hours: 0,
            ..Default::default()
        };

        let logs = create_test_logs(1000);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let preparer = TrainingDataPreparer::new(config.clone());
        let (features, targets) = preparer
            .prepare(&logs, &baseline, &schedule)
            .unwrap_or_default();

        let result = compute_lr_cv_scores(&features, &targets, &config, &logs);
        assert!(result.is_some(), "LR CV should succeed");

        let (_scores, quantiles) = result.unwrap_or_else(|| unreachable!());
        assert!(
            quantiles.is_some(),
            "LR with CV should produce residual quantiles"
        );
    }

    #[test]
    fn test_train_model_sync_insufficient_for_cv_no_quantiles() -> Result<()> {
        let config = MlConfig {
            min_samples_for_training: 5,
            training_window_days: 28,
            algorithm: MlAlgorithm::RandomForest,
            tune_hyperparameters: true,
            cv_folds: 4,
            cv_gap_hours: 24,
            ..Default::default()
        };

        let logs = create_test_logs(10);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let result = train_model_sync(&logs, &baseline, &schedule, &config)?;

        assert!(
            result.residual_quantiles.is_none(),
            "Insufficient data for CV should yield no residual quantiles"
        );

        Ok(())
    }

    #[test]
    fn test_lr_cv_scores_insufficient_data() {
        let config = MlConfig {
            min_samples_for_training: 5,
            cv_folds: 4,
            cv_gap_hours: 24,
            ..Default::default()
        };

        let logs = create_test_logs(10);
        let baseline = create_test_baseline();
        let schedule = GymSchedule::default();

        let preparer = TrainingDataPreparer::new(config.clone());
        let result = preparer.prepare(&logs, &baseline, &schedule);

        if let Ok((features, targets)) = result {
            let scores = compute_lr_cv_scores(&features, &targets, &config, &logs);
            assert!(scores.is_none());
        }
        // Err is also acceptable — insufficient data at prep stage
    }
}
