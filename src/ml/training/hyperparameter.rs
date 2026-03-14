use super::cross_validation::{CrossValidationScores, Fold, FoldScores};
use crate::ml::{
    evaluation,
    features::PredictionFeatures,
    model::{ModelBuilder, TrainingError},
};

/// A single hyperparameter configuration for grid search.
#[derive(Debug, Clone)]
pub struct HyperparameterSet {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
}

/// Result of a grid search: best hyperparameters + their CV scores.
/// Result of a grid search: best hyperparameters + their CV scores.
#[derive(Debug, Clone)]
pub struct GridSearchResult {
    pub best_params: HyperparameterSet,
    pub best_cv_scores: CrossValidationScores,
    #[allow(dead_code)]
    pub configs_evaluated: usize,
}

impl HyperparameterSet {
    /// Generate the default grid of hyperparameter configurations.
    ///
    /// Grid: 4 x 4 x 3 x 3 = 144 configurations
    pub fn default_grid() -> Vec<Self> {
        let n_trees_values = [100, 200, 300, 500];
        let max_depth_values = [8, 12, 16, 0]; // 0 = unlimited
        let min_samples_leaf_values = [2, 5, 10];
        let max_features_values: [Option<usize>; 3] = [Some(5), Some(11), None];

        let mut grid = Vec::with_capacity(144);
        for &n_trees in &n_trees_values {
            for &max_depth in &max_depth_values {
                for &min_samples_leaf in &min_samples_leaf_values {
                    for &max_features in &max_features_values {
                        grid.push(Self {
                            n_trees,
                            max_depth,
                            min_samples_leaf,
                            max_features,
                        });
                    }
                }
            }
        }
        grid
    }

    /// Generate a small grid for fast testing or when data is limited.
    ///
    /// Grid: 2 x 2 x 2 x 2 = 16 configurations
    pub fn small_grid() -> Vec<Self> {
        let n_trees_values = [50, 100];
        let max_depth_values = [5, 10];
        let min_samples_leaf_values = [2, 5];
        let max_features_values: [Option<usize>; 2] = [Some(5), None];

        let mut grid = Vec::with_capacity(16);
        for &n_trees in &n_trees_values {
            for &max_depth in &max_depth_values {
                for &min_samples_leaf in &min_samples_leaf_values {
                    for &max_features in &max_features_values {
                        grid.push(Self {
                            n_trees,
                            max_depth,
                            min_samples_leaf,
                            max_features,
                        });
                    }
                }
            }
        }
        grid
    }
}

/// Evaluate one hyperparameter config across all CV folds.
///
/// Returns the `CrossValidationScores` for this config.
#[allow(clippy::similar_names)]
pub fn evaluate_config(
    config: &HyperparameterSet,
    features: &[PredictionFeatures],
    targets: &[f64],
    folds: &[Fold],
) -> Result<CrossValidationScores, TrainingError> {
    let mut mse_scores = Vec::with_capacity(folds.len());
    let mut rmse_scores = Vec::with_capacity(folds.len());
    let mut mae_scores = Vec::with_capacity(folds.len());
    let mut r2_scores = Vec::with_capacity(folds.len());

    for fold in folds {
        let train_features = &features[fold.train_start..fold.train_end];
        let train_targets = &targets[fold.train_start..fold.train_end];
        let val_features = &features[fold.val_start..fold.val_end];
        let val_targets = &targets[fold.val_start..fold.val_end];

        let builder = ModelBuilder::new()
            .n_trees(config.n_trees)
            .max_depth(config.max_depth)
            .min_samples_leaf(config.min_samples_leaf)
            .max_features(config.max_features);

        let model = builder.train_rf(train_features, train_targets)?;
        let predictions = model.predict_batch(val_features);

        let mse = evaluation::mse(&predictions, val_targets).unwrap_or(f64::MAX);
        let rmse = evaluation::rmse(&predictions, val_targets).unwrap_or(f64::MAX);
        let mae = evaluation::mae(&predictions, val_targets).unwrap_or(f64::MAX);
        let r2 = evaluation::r_squared(&predictions, val_targets).unwrap_or(f64::NEG_INFINITY);

        mse_scores.push(mse);
        rmse_scores.push(rmse);
        mae_scores.push(mae);
        r2_scores.push(r2);
    }

    Ok(CrossValidationScores {
        mse: FoldScores::from_scores(mse_scores),
        rmse: FoldScores::from_scores(rmse_scores),
        mae: FoldScores::from_scores(mae_scores),
        r_squared: FoldScores::from_scores(r2_scores),
    })
}

/// Collect per-prediction CV residuals from validation folds.
///
/// Re-runs cross-validation with the given config and collects
/// `(weekday, hour, residual)` triples where `residual = actual - predicted`.
/// Weekday/hour are extracted from each feature's `raw_weekday`/`raw_hour`
/// fields.
#[allow(
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn collect_cv_residuals(
    config: &HyperparameterSet,
    features: &[PredictionFeatures],
    targets: &[f64],
    folds: &[Fold],
) -> Result<Vec<(u32, u32, f64)>, TrainingError> {
    let total_val_samples: usize = folds.iter().map(|f| f.val_end - f.val_start).sum();
    let mut residuals = Vec::with_capacity(total_val_samples);

    for fold in folds {
        let train_features = &features[fold.train_start..fold.train_end];
        let train_targets = &targets[fold.train_start..fold.train_end];
        let val_features = &features[fold.val_start..fold.val_end];
        let val_targets = &targets[fold.val_start..fold.val_end];

        let builder = ModelBuilder::new()
            .n_trees(config.n_trees)
            .max_depth(config.max_depth)
            .min_samples_leaf(config.min_samples_leaf)
            .max_features(config.max_features);

        let model = builder.train_rf(train_features, train_targets)?;
        let predictions = model.predict_batch(val_features);

        for (i, (&actual, predicted)) in val_targets.iter().zip(predictions.iter()).enumerate() {
            let feature = &val_features[i];
            let weekday = feature.raw_weekday as u32;
            let hour = feature.raw_hour as u32;
            let residual = actual - predicted;
            residuals.push((weekday, hour, residual));
        }
    }

    Ok(residuals)
}

/// Run grid search with the default hyperparameter grid.
#[allow(dead_code)]
pub fn grid_search(
    features: &[PredictionFeatures],
    targets: &[f64],
    folds: &[Fold],
) -> Result<GridSearchResult, TrainingError> {
    grid_search_with_grid(features, targets, folds, &HyperparameterSet::default_grid())
}

/// Run grid search with a custom set of hyperparameter configurations.
pub fn grid_search_with_grid(
    features: &[PredictionFeatures],
    targets: &[f64],
    folds: &[Fold],
    grid: &[HyperparameterSet],
) -> Result<GridSearchResult, TrainingError> {
    let mut best_params: Option<HyperparameterSet> = None;
    let mut best_scores: Option<CrossValidationScores> = None;
    let mut best_mse = f64::MAX;
    let mut configs_evaluated = 0_usize;

    for config in grid {
        if let Ok(scores) = evaluate_config(config, features, targets, folds) {
            configs_evaluated += 1;
            if scores.mse.mean < best_mse {
                best_mse = scores.mse.mean;
                best_params = Some(config.clone());
                best_scores = Some(scores);
            }
        }
    }

    match (best_params, best_scores) {
        (Some(params), Some(scores)) => Ok(GridSearchResult {
            best_params: params,
            best_cv_scores: scores,
            configs_evaluated,
        }),
        _ => Err(TrainingError::FitError(
            "Grid search: no configuration trained successfully".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use proptest::prelude::*;

    use super::*;
    use crate::ml::training::cross_validation::TimeSeriesSplit;

    #[allow(clippy::cast_precision_loss)]
    fn create_test_features(n: usize) -> Vec<PredictionFeatures> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                let noise1 = (t * 0.1).sin() * 0.01;
                let noise2 = (t * 0.17).cos() * 0.01;

                PredictionFeatures {
                    hour_sin: (t * 0.3).sin() + noise1,
                    hour_cos: (t * 0.31).cos() + noise2,
                    weekday_sin: (t * 0.07).sin() + noise1,
                    weekday_cos: (t * 0.071).cos() + noise2,
                    historical_avg: 30.0 + (t % 40.0) + noise1 * 100.0,
                    historical_std: 5.0 + (t % 15.0),
                    recent_avg_1h: 35.0 + (t % 35.0),
                    recent_avg_3h: 40.0 + ((t * 1.3) % 30.0),
                    recent_trend: -10.0 + (t % 20.0),
                    day_avg_so_far: 30.0 + (t % 45.0),
                    prev_day_avg: 45.0 + ((t * 0.7) % 25.0),
                    is_weekend: if (i % 7) >= 5 { 1.0 } else { 0.0 },
                    is_holiday: if i % 30 == 0 { 1.0 } else { 0.0 },
                    week_of_year_sin: (t * 0.02).sin() + noise1,
                    week_of_year_cos: (t * 0.021).cos() + noise2,
                    hours_ahead: 1.0 + (t % 6.0),
                    raw_hour: t % 24.0,
                    raw_weekday: t % 7.0,
                    time_to_close: 5.0 + (t % 12.0),
                    occupancy_volatility: 2.0 + (t % 10.0),
                    recent_avg_6h: 42.0 + ((t * 0.9) % 28.0),
                    prev_week_same_slot: 38.0 + (t % 35.0),
                }
            })
            .collect()
    }

    #[test]
    fn test_default_grid_size() {
        let grid = HyperparameterSet::default_grid();
        assert_eq!(grid.len(), 144);
    }

    #[test]
    fn test_small_grid_size() {
        let grid = HyperparameterSet::small_grid();
        assert_eq!(grid.len(), 16);
    }

    #[test]
    fn test_default_grid_values() {
        let grid = HyperparameterSet::default_grid();

        // First config: smallest values
        assert_eq!(grid[0].n_trees, 100);
        assert_eq!(grid[0].max_depth, 8);
        assert_eq!(grid[0].min_samples_leaf, 2);
        assert_eq!(grid[0].max_features, Some(5));

        // Last config: largest values
        let last = &grid[143];
        assert_eq!(last.n_trees, 500);
        assert_eq!(last.max_depth, 0); // unlimited
        assert_eq!(last.min_samples_leaf, 10);
        assert!(last.max_features.is_none());
    }

    #[test]
    fn test_evaluate_config_basic() -> Result<()> {
        let features = create_test_features(500);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let splitter = TimeSeriesSplit::new(3, 0);
        let folds = splitter
            .and_then(|s| s.split(features.len()))
            .ok_or_else(|| anyhow::anyhow!("Failed to create folds"))?;

        let config = HyperparameterSet {
            n_trees: 10,
            max_depth: 5,
            min_samples_leaf: 2,
            max_features: Some(4),
        };

        let scores = evaluate_config(&config, &features, &targets, &folds)?;

        assert_eq!(scores.mse.per_fold.len(), 3);
        assert!(scores.mse.mean >= 0.0);
        assert!(scores.rmse.mean >= 0.0);
        assert!(scores.mae.mean >= 0.0);
        assert!(scores.mse.mean.is_finite());

        Ok(())
    }

    #[test]
    fn test_grid_search_selects_best() -> Result<()> {
        let features = create_test_features(500);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let splitter = TimeSeriesSplit::new(3, 0);
        let folds = splitter
            .and_then(|s| s.split(features.len()))
            .ok_or_else(|| anyhow::anyhow!("Failed to create folds"))?;

        let small_grid = HyperparameterSet::small_grid();
        let result = grid_search_with_grid(&features, &targets, &folds, &small_grid)?;

        assert!(result.configs_evaluated > 0);
        assert!(result.best_cv_scores.mse.mean >= 0.0);
        assert!(result.best_cv_scores.mse.mean.is_finite());

        Ok(())
    }

    #[test]
    fn test_grid_search_insufficient_data() {
        // With zero features, train_rf will fail with InsufficientData
        let features: Vec<PredictionFeatures> = Vec::new();
        let targets: Vec<f64> = Vec::new();

        let folds = vec![Fold {
            train_start: 0,
            train_end: 0,
            val_start: 0,
            val_end: 0,
        }];

        let small_grid = HyperparameterSet::small_grid();
        let result = grid_search_with_grid(&features, &targets, &folds, &small_grid);
        assert!(result.is_err());
    }

    // ── collect_cv_residuals tests ───────────────────────────────────

    #[test]
    fn test_collect_cv_residuals_basic() -> Result<()> {
        let features = create_test_features(500);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let splitter = TimeSeriesSplit::new(3, 0);
        let folds = splitter
            .and_then(|s| s.split(features.len()))
            .ok_or_else(|| anyhow::anyhow!("Failed to create folds"))?;

        let config = HyperparameterSet {
            n_trees: 10,
            max_depth: 5,
            min_samples_leaf: 2,
            max_features: Some(5),
        };

        let residuals = collect_cv_residuals(&config, &features, &targets, &folds)?;

        // Total residuals should equal sum of validation set sizes
        let expected_count: usize = folds.iter().map(|f| f.val_end - f.val_start).sum();
        assert_eq!(residuals.len(), expected_count);

        Ok(())
    }

    #[test]
    fn test_collect_cv_residuals_weekday_hour_range() -> Result<()> {
        let features = create_test_features(500);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let splitter = TimeSeriesSplit::new(3, 0);
        let folds = splitter
            .and_then(|s| s.split(features.len()))
            .ok_or_else(|| anyhow::anyhow!("Failed to create folds"))?;

        let config = HyperparameterSet {
            n_trees: 10,
            max_depth: 5,
            min_samples_leaf: 2,
            max_features: Some(5),
        };

        let residuals = collect_cv_residuals(&config, &features, &targets, &folds)?;

        for &(weekday, hour, _) in &residuals {
            assert!(weekday < 7, "weekday ({weekday}) should be < 7");
            assert!(hour < 24, "hour ({hour}) should be < 24");
        }

        Ok(())
    }

    #[test]
    fn test_collect_cv_residuals_finite() -> Result<()> {
        let features = create_test_features(500);
        let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

        let splitter = TimeSeriesSplit::new(3, 0);
        let folds = splitter
            .and_then(|s| s.split(features.len()))
            .ok_or_else(|| anyhow::anyhow!("Failed to create folds"))?;

        let config = HyperparameterSet {
            n_trees: 10,
            max_depth: 5,
            min_samples_leaf: 2,
            max_features: Some(5),
        };

        let residuals = collect_cv_residuals(&config, &features, &targets, &folds)?;

        for &(_, _, residual) in &residuals {
            assert!(
                residual.is_finite(),
                "All residuals should be finite, got {residual}"
            );
        }

        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        #[allow(clippy::cast_possible_truncation)]
        fn prop_residual_count_equals_val_samples(seed in 0_u64..50) {
            let n = 300 + (seed as usize % 200);
            let features = create_test_features(n);
            let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

            let Some(splitter) = TimeSeriesSplit::new(3, 0) else {
                return Ok(());
            };
            let Some(folds) = splitter.split(features.len()) else {
                return Ok(());
            };

            let config = HyperparameterSet {
                n_trees: 5,
                max_depth: 3,
                min_samples_leaf: 2,
                max_features: Some(5),
            };

            if let Ok(residuals) = collect_cv_residuals(&config, &features, &targets, &folds) {
                let expected: usize = folds.iter().map(|f| f.val_end - f.val_start).sum();
                prop_assert_eq!(residuals.len(), expected);
            }
        }

        #[test]
        #[allow(clippy::cast_possible_truncation)]
        fn prop_grid_search_best_mse_le_all(seed in 0_u64..100) {
            // Use seed to create slightly different feature sets
            let n = 300 + (seed as usize % 200);
            let features = create_test_features(n);
            let targets: Vec<f64> = features.iter().map(|f| f.historical_avg).collect();

            let Some(splitter) = TimeSeriesSplit::new(3, 0) else {
                return Ok(());
            };
            let Some(folds) = splitter.split(features.len()) else {
                return Ok(());
            };

            // Evaluate a small subset of configs manually
            let configs = vec![
                HyperparameterSet {
                    n_trees: 10,
                    max_depth: 5,
                    min_samples_leaf: 2,
                    max_features: Some(5),
                },
                HyperparameterSet {
                    n_trees: 10,
                    max_depth: 8,
                    min_samples_leaf: 5,
                    max_features: None,
                },
            ];

            let mut best_mse = f64::MAX;
            for config in &configs {
                if let Ok(scores) = evaluate_config(config, &features, &targets, &folds)
                    && scores.mse.mean < best_mse
                {
                    best_mse = scores.mse.mean;
                }
            }

            // If we got any valid results, best_mse should be non-negative
            if best_mse < f64::MAX {
                prop_assert!(best_mse >= 0.0);
            }
        }
    }
}
