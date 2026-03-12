# Technical Design Document: ML Model Upgrade

**Status:** Draft
**Author:** Claude Opus 4.6 + User
**Date:** 2026-03-11
**Scope:** Replace linear regression with a tree-based ensemble model for state-of-the-art
occupancy predictions.

---

## 1. Executive Summary

Hardy Monitor's occupancy prediction pipeline currently uses linear regression (`linfa-linear`
v0.8.1), which cannot capture the non-linear temporal patterns inherent in gym occupancy data
(e.g., Tuesday 6 PM is disproportionately busier than a linear hour effect would suggest).
The confidence interval system is heuristic (hand-tuned sigmoid), model weights are not
persisted (only metadata), and there is no cross-validation, hyperparameter tuning, or feature
importance analysis.

This document proposes upgrading to a **Random Forest Regressor** via the `smartcore` crate
(v0.4.9), using its native `DenseMatrix` type to avoid ndarray version conflicts. The upgrade
includes: k-fold time-series cross-validation, hyperparameter grid search, expanded feature
engineering (22 features), quantile-based confidence intervals from per-tree predictions,
proper model persistence with bincode + zstd compression, and comprehensive testing with
property-based tests.

**Constraints:** Pure Rust only (no C++ FFI), `gui` feature-gated, `default-features = false`
on all dependencies, strict lints (`deny(unwrap_used, expect_used, panic)`).

---

## 2. Current State Analysis

### 2.1 Architecture Overview

```
src/ml/ (~2,022 LOC, 6 files)
├── mod.rs         (324 LOC)  OccupancyPredictor coordinator, MlConfig, predict()
├── model.rs       (459 LOC)  TrainedModel wraps FittedLinearRegression, ModelBuilder
├── features.rs    (466 LOC)  PredictionFeatures (16 fields), FeatureExtractor
├── training.rs    (315 LOC)  TrainingDataPreparer, train_model() async/sync
├── confidence.rs  (187 LOC)  PredictionWithConfidence, PredictionMethod enum
└── persistence.rs (277 LOC)  PersistedModel (metadata only), bincode serialization
```

Feature-gated: `#[cfg(feature = "gui")] pub mod ml;` in `src/lib.rs`.
Integration: `src/app.rs` owns `OccupancyPredictor`, triggers training via
`Message::MlTrainingCompleted`.

### 2.2 Current Algorithm

- **Model:** `linfa_linear::FittedLinearRegression<f64>` with manual ridge regularization
  (λ=1e-3 via design matrix augmentation).
- **Features:** 16 — cyclical time encoding (hour/weekday/week-of-year as sin/cos),
  historical slot stats (mean/std), momentum (1h/3h rolling avg + linear trend), day-level
  context (today avg, yesterday avg), binary flags (weekend, holiday), forecast horizon.
- **Training split:** Fixed 80/20 sequential (no shuffle — correct for time series).
- **Metrics:** MSE only. No R², MAE, MAPE.
- **Prediction:** Dual-path — ML if model available, else historical average fallback.

### 2.3 Known Limitations

1. **Linear model cannot capture non-linear interactions.** Hour × weekday interaction effects
   are invisible. A Tuesday at 6 PM cannot be modeled differently from a linear combination of
   "Tuesday effect" + "6 PM effect."
2. **No cross-validation.** Single 80/20 split has high variance for model selection.
3. **Heuristic confidence intervals.** `1.0 / (1.0 + adjusted_std / 20.0)` — sigmoid with
   hand-tuned constants, not statistically grounded.
4. **Model weights not persisted.** `PersistedModel` stores only metadata (MSE, slot stats,
   version). The actual `FittedLinearRegression` is never serialized. Model must retrain on
   every app restart.
5. **No feature importance.** No way to know which features matter.
6. **No hyperparameter tuning.** Ridge lambda is hardcoded at 1e-3.
7. **`ModelBuilder` has unused tree stubs.** `max_depth()`, `min_samples_split()`,
   `min_samples_leaf()` are no-ops, suggesting tree models were anticipated but never built.
8. **Zero property-based tests in ML module.** CLAUDE.md requires proptest with 1000+ cases
   for data pipeline logic.

---

## 3. Crate Evaluation

### 3.1 Evaluation Criteria

| Criterion | Weight | Notes |
|-----------|--------|-------|
| Random Forest Regressor support | Required | Must have regression, not just classification |
| Pure Rust | Required | No C++ FFI |
| ndarray 0.16 compatibility | Preferred | Project uses ndarray 0.16.1 via linfa 0.8.1 |
| `default-features = false` | Required | Per CLAUDE.md dependency rules |
| Feature importance extraction | High | Key deliverable for state-of-the-art pipeline |
| Per-tree predictions | High | Needed for quantile-based confidence intervals |
| Serialization (serde) | High | Must persist trained model weights |
| Hyperparameter search built-in | Medium | Can implement ourselves if needed |
| Active maintenance | Medium | Last update within 12 months |
| License (MIT/Apache-2.0) | Required | Must be compatible |

### 3.2 Candidate Analysis

#### 3.2.1 smartcore v0.4.9

| Property | Value |
|----------|-------|
| Last updated | January 9, 2026 |
| Downloads | 292,928 |
| License | Apache-2.0 |
| RF Regressor | **Yes** — `RandomForestRegressor` |
| GBT | No |
| ndarray | ^0.15 **optional** — not required |
| Native types | `DenseMatrix<f64>` (own matrix type) |
| Feature importance | Undocumented — needs source verification |
| Per-tree predictions | Undocumented — needs source verification |
| Serde | Optional feature |
| Grid search | Built-in `RandomForestRegressorSearchParameters` |

**Key insight:** ndarray is an optional dependency in smartcore. The primary API uses
`DenseMatrix<f64>`, which is smartcore's own matrix type. We can add smartcore WITHOUT enabling
its `ndarray` feature, completely avoiding the ndarray 0.15 vs 0.16 conflict. Conversion
between our `PredictionFeatures` (which produces `Vec<f64>`) and `DenseMatrix` is trivial.

**Verdict: Primary recommendation.** Most mature, most downloaded, actively maintained, has
the exact algorithm we need. Type conversion at boundary is minimal overhead.

#### 3.2.2 linfa-trees v0.8.1 + linfa-ensemble v0.8.1

| Property | Value |
|----------|-------|
| Last updated | December 23, 2025 |
| ndarray | ^0.16 ✓ (fully compatible) |
| License | MIT/Apache-2.0 |
| linfa-trees | Decision tree **classifier only** — no regression |
| linfa-ensemble | `RandomForest` (classifier), `AdaBoost`, `EnsembleLearner` — **no regression** |

**Verdict: Not viable.** Despite perfect ndarray compatibility, neither crate provides
regression variants. Classification-only.

#### 3.2.3 randomforest v0.1.6

| Property | Value |
|----------|-------|
| Last updated | September 18, 2020 |
| Downloads | 13,870 |
| License | MIT |
| RF Regressor | Yes |
| Own types | `TableBuilder` (not ndarray) |

**Verdict: Not recommended.** Abandoned (no updates in 5+ years), low downloads, own types
requiring conversion.

#### 3.2.4 forust-ml v0.4.8

| Property | Value |
|----------|-------|
| Last updated | May 9, 2024 |
| Downloads | 92,661 |
| License | Non-standard (needs verification) |
| GBT | **Yes** — XGBoost algorithm in pure Rust |
| RF | No |
| Own types | serde_json-based, no ndarray |

**Verdict: Strong GBT candidate.** If we decide GBT is the better algorithm, forust-ml is a
solid option. License needs verification before adoption.

#### 3.2.5 gbdt v0.1.3

| Property | Value |
|----------|-------|
| Last updated | January 24, 2024 |
| Downloads | 55,542 |
| License | Apache-2.0 |
| GBT | Yes |
| Own types | Not ndarray |

**Verdict: Backup GBT option.** Less active than forust-ml but has a clean license.

#### 3.2.6 Custom Implementation on ndarray 0.16

| Property | Value |
|----------|-------|
| Effort | ~800-1200 LOC (decision tree regressor + bagging + OOB + feature importance) |
| ndarray | ^0.16 (native) |
| Full control | Complete control over per-tree predictions, feature importance, serialization |

**Verdict: Viable fallback.** Maximum control and zero dependency conflicts, but significant
implementation and testing effort. Consider only if smartcore proves inadequate.

### 3.3 ndarray Version Strategy

**Resolution: Use smartcore without its ndarray feature.**

smartcore's primary API uses `DenseMatrix<f64>`, not ndarray arrays. By adding smartcore with
`default-features = false` and only enabling `serde` (for serialization), we avoid the ndarray
0.15 vs 0.16 conflict entirely. The existing linfa + ndarray 0.16 stack remains untouched.

Conversion at the boundary is a few lines:
```rust
// PredictionFeatures → DenseMatrix (for smartcore)
let flat: Vec<f64> = features.iter().flat_map(|f| f.to_vec()).collect();
let matrix = DenseMatrix::from_2d_vec(&rows);  // smartcore's type

// For prediction output, smartcore returns Vec<f64> — no conversion needed.
```

### 3.4 Recommendation

**Primary: smartcore v0.4.9** with `default-features = false, features = ["serde"]`.

Rationale:
- Only pure Rust crate with a Random Forest Regressor
- Most mature (293K downloads), actively maintained (Jan 2026)
- Built-in grid search for hyperparameter tuning
- Avoids ndarray conflict by using `DenseMatrix`
- Apache-2.0 license

**Items requiring source-level verification before finalizing:**
1. Can we extract feature importance from a fitted `RandomForestRegressor`?
2. Can we access individual tree predictions for quantile-based confidence intervals?
3. What serialization format does the serde feature use?
4. What hyperparameters are available on `RandomForestRegressorParameters`?

> **Decision needed:** Approve smartcore as the primary crate, or should we also evaluate
> forust-ml (GBT) in parallel before committing?

---

## 4. Algorithm Selection

### 4.1 Comparison for Occupancy Forecasting

| Criterion | Linear Regression | Random Forest | Gradient Boosted Trees |
|-----------|-------------------|---------------|------------------------|
| Non-linear pattern capture | No | Yes | Yes |
| Feature interactions | No (manual only) | Automatic | Automatic |
| Outlier robustness | Low | High | Medium |
| Overfitting risk (500-5000 samples) | Low | Low (bagging) | Medium-High |
| Natural confidence intervals | Via residuals | **Per-tree variance** | Not natural |
| OOB error (free validation) | No | **Yes** | No |
| Feature importance | Via coefficients | **Yes (Gini/permutation)** | Yes (gain) |
| Training speed | Very fast | Moderate | Slow |
| Prediction speed | O(features) | O(trees × depth) | O(trees × depth) |
| Interpretability | High (coefficients) | Medium (importance) | Low |
| Hyperparameter sensitivity | Low (just λ) | Low | **High** |

### 4.2 Why Random Forest is the Best Fit

For this specific domain (gym occupancy forecasting with 500-5000 training samples):

1. **Natural prediction intervals.** Each tree produces an independent prediction. The
   distribution of per-tree predictions gives us statistically grounded confidence intervals
   without bootstrapping or calibration hacks. This directly addresses the biggest weakness of
   the current system.

2. **OOB error estimation.** Each tree is trained on a bootstrap sample (~63% of data). The
   remaining ~37% provides "out-of-bag" predictions, giving us essentially free
   cross-validation without having to set aside a validation set.

3. **Robustness with limited data.** Bagging (bootstrap aggregating) reduces variance without
   increasing bias. With only 500-5000 samples, GBT's sequential error correction is more
   prone to overfitting, while RF's parallel independent trees are safer.

4. **Low hyperparameter sensitivity.** RF works well with reasonable defaults (100-500 trees,
   max_features=sqrt(n_features)). GBT requires careful tuning of learning rate, tree depth,
   and regularization — getting these wrong causes significant degradation.

5. **Automatic feature interaction capture.** The current cyclical encoding (sin/cos for
   hour/weekday) was designed to help linear regression understand periodicity. RF naturally
   captures "Tuesday at 18:00" as a distinct pattern without explicit interaction terms.

### 4.3 Recommendation

**Random Forest Regressor** as the primary model.
- Keep linear regression as a permanent fallback (existing code, zero marginal cost).
- Monitor RF vs LR performance via cross-validation metrics during each training run.
- The architecture should allow swapping algorithms via a trait/enum, enabling future
  experimentation with GBT if needed.

> **Decision needed:** Approve Random Forest as the primary algorithm?

---

## 5. Feature Engineering Expansion

### 5.1 Current Feature Set (16 features)

| # | Feature | Category | RF Benefit |
|---|---------|----------|------------|
| 1-2 | hour_sin, hour_cos | Cyclical time | RF can split on raw hour — cyclical encoding still helps for interpolation near period boundary |
| 3-4 | weekday_sin, weekday_cos | Cyclical time | Same as above |
| 5 | historical_avg | Historical | **High value** — direct baseline signal |
| 6 | historical_std | Historical | High value — captures slot volatility |
| 7 | recent_avg_1h | Momentum | High value — captures current state |
| 8 | recent_avg_3h | Momentum | Moderate — may be redundant with 1h |
| 9 | recent_trend | Momentum | Moderate — noisy with limited data |
| 10 | day_avg_so_far | Day-level | Moderate — captures "is today busy?" |
| 11 | prev_day_avg | Day-level | Low-Moderate — loose correlation |
| 12 | is_weekend | Context | **High value** — strong occupancy predictor |
| 13 | is_holiday | Context | Moderate — rare events, limited training signal |
| 14-15 | week_of_year_sin/cos | Cyclical time | Moderate — captures seasonal trends |
| 16 | hours_ahead | Forecast | **Critical** — prediction horizon |

### 5.2 Proposed New Features

| # | Feature | Category | Rationale |
|---|---------|----------|-----------|
| 17 | raw_hour | Direct temporal | RF can split directly on hour boundaries (e.g., hour >= 17). Avoids information loss from cyclical encoding. |
| 18 | raw_weekday | Direct temporal | Same rationale. RF handles categorical splits natively. |
| 19 | time_to_close | Schedule-aware | Hours until gym closes. Captures "winding down" pattern. Available via `GymSchedule::get_close_hour()`. |
| 20 | occupancy_volatility | Rolling stats | Std dev of recent_data window. Captures "is occupancy stable or erratic right now?" |
| 21 | recent_avg_6h | Extended momentum | Broader trend window. Captures half-day patterns. |
| 22 | prev_week_same_slot | Lag | Occupancy at same weekday+hour last week. Strong autoregressive signal. |

**Rationale for NOT including more:**
- Day-of-month cyclical encoding: gym occupancy has weak monthly patterns (no payroll effect).
- Holiday proximity (days since/until): holidays are binary events; proximity adds noise.
- Rate of change / acceleration: second derivatives are very noisy with 5-min sample intervals.

### 5.3 Feature Importance Feedback Loop

After initial training:
1. Extract feature importance scores from the RF model.
2. Store importance in `PersistedModel` for GUI display.
3. Log features with importance < 1% as candidates for removal.
4. Do NOT auto-remove features — human review required. Low importance in aggregate doesn't
   mean unimportant for edge cases (e.g., is_holiday is rare but critical for those days).

### 5.4 `PredictionFeatures` Struct Changes

```rust
pub struct PredictionFeatures {
    // Existing (keep all 16)
    pub hour_sin: f64,
    pub hour_cos: f64,
    pub weekday_sin: f64,
    pub weekday_cos: f64,
    pub historical_avg: f64,
    pub historical_std: f64,
    pub recent_avg_1h: f64,
    pub recent_avg_3h: f64,
    pub recent_trend: f64,
    pub day_avg_so_far: f64,
    pub prev_day_avg: f64,
    pub is_weekend: f64,
    pub is_holiday: f64,
    pub week_of_year_sin: f64,
    pub week_of_year_cos: f64,
    pub hours_ahead: f64,

    // New (6 additions)
    pub raw_hour: f64,
    pub raw_weekday: f64,
    pub time_to_close: f64,
    pub occupancy_volatility: f64,
    pub recent_avg_6h: f64,
    pub prev_week_same_slot: f64,
}

impl PredictionFeatures {
    pub const NUM_FEATURES: usize = 22;
}
```

`to_vec()` and `feature_names()` must be updated accordingly.

> **Decision needed:** Approve the 6 new features, or add/remove any?

---

## 6. Training Pipeline Redesign

### 6.1 Time-Series Cross-Validation

**Why not standard k-fold:** Random k-fold shuffles temporal data, causing data leakage —
the model trains on future data to predict the past. Occupancy data is inherently temporal.

**TimeSeriesSplit** (expanding window with gap):

```
Fold 1: [Train: weeks 1-4] [Gap: 1 day] [Val: week 5]
Fold 2: [Train: weeks 1-5] [Gap: 1 day] [Val: week 6]
Fold 3: [Train: weeks 1-6] [Gap: 1 day] [Val: week 7]
Fold 4: [Train: weeks 1-7] [Gap: 1 day] [Val: week 8]
```

- **k = 4 folds** (with 56-day default window, each validation set ≈ 1 week).
- **1-day gap** between train and validation to prevent label leakage from adjacent samples.
- Training set always precedes validation set temporally.
- Metrics computed per fold: MSE, RMSE, MAE, MAPE, R².
- Final metric: mean ± std across folds.

### 6.2 Hyperparameter Tuning

**Strategy: Grid search** (exhaustive over a small grid).

Rationale: Random search is better for high-dimensional spaces. RF has few critical
hyperparameters and our training time is unconstrained, so grid search is tractable and
deterministic.

**Search grid:**

| Parameter | Values | Count |
|-----------|--------|-------|
| n_trees | 100, 200, 300, 500 | 4 |
| max_depth | 8, 12, 16, None (unlimited) | 4 |
| min_samples_leaf | 2, 5, 10 | 3 |
| max_features | sqrt(22)≈5, log2(22)≈4, 11 (50%) | 3 |

Total: 4 × 4 × 3 × 3 = **144 configurations** × 4 CV folds = 576 model fits.

With ~2000 training samples and 22 features, each fit should take <100ms. Total grid search:
<1 minute.

**Tuning frequency:** Every retraining cycle (default 24h). Since there's no time limit and
the grid is small, always re-tune.

### 6.3 OOB Error Estimation

If smartcore's `RandomForestRegressor` exposes OOB predictions:
- Compute OOB MSE/RMSE after training.
- Compare OOB error with CV error as a consistency check.
- Store OOB error in `TrainingResult` metadata.

If smartcore does not expose OOB: skip this metric. CV provides sufficient validation.

### 6.4 Training Pipeline Changes

**New `TrainingResult` fields:**

```rust
pub struct TrainingResult {
    pub model: TrainedModel,          // Now wraps smartcore RF
    pub feature_extractor: FeatureExtractor,
    pub persisted: PersistedModel,
    // New:
    pub cv_scores: CrossValidationScores,
    pub best_hyperparameters: HyperparameterSet,
    pub feature_importance: Vec<(String, f64)>,
    pub oob_error: Option<f64>,
}

pub struct CrossValidationScores {
    pub mse: FoldScores,     // per-fold + mean + std
    pub rmse: FoldScores,
    pub mae: FoldScores,
    pub mape: FoldScores,
    pub r_squared: FoldScores,
}

pub struct FoldScores {
    pub per_fold: Vec<f64>,
    pub mean: f64,
    pub std_dev: f64,
}
```

**Training orchestration:**

```
1. TrainingDataPreparer::prepare() → (features, targets)
2. TimeSeriesSplit::split(features, targets, k=4, gap=1day)
3. For each hyperparameter config in grid:
   a. For each fold:
      - Train RF on training fold
      - Predict on validation fold
      - Compute metrics (MSE, RMSE, MAE, MAPE, R²)
   b. Average metrics across folds
4. Select best config by mean CV MSE
5. Retrain on full dataset with best hyperparameters
6. Extract feature importance
7. Build TrainingResult
```

**CPU-intensive training:** Use `tokio::task::spawn_blocking()` to avoid blocking the async
runtime during grid search. The current async training already runs in a spawned task, but
the longer grid search makes this more critical.

### 6.5 `MlConfig` Expansion

```rust
pub struct MlConfig {
    // Existing (unchanged):
    pub enabled: bool,
    pub training_window_days: i64,
    pub retrain_interval_hours: i64,
    pub prediction_horizon_hours: i64,
    pub min_samples_for_training: usize,
    pub model_path: Option<PathBuf>,
    pub fallback_on_error: bool,

    // New:
    pub algorithm: MlAlgorithm,       // RandomForest (default) | LinearRegression
    pub cv_folds: usize,              // Default: 4
    pub cv_gap_hours: i64,            // Default: 24 (1 day gap between folds)
    pub tune_hyperparameters: bool,   // Default: true
}

pub enum MlAlgorithm {
    RandomForest,
    LinearRegression,  // Kept as fallback/baseline
}
```

Hyperparameter grid values are NOT user-configurable (they are internal tuning details).

> **Decision needed:**
> 1. Approve TimeSeriesSplit with 4 folds and 1-day gap?
> 2. Grid search with 144 configs — acceptable, or reduce the grid?
> 3. Should `MlAlgorithm` be in the TOML config, or always RandomForest?

---

## 7. Confidence Interval Redesign

### 7.1 Current System (Problems)

```rust
// mod.rs:215-221
let horizon_penalty = 1.0 + (hours_ahead as f64 - 1.0) * 0.15;
let adjusted_std = base_std * horizon_penalty;
let confidence_score = (1.0 / (1.0 + adjusted_std / 20.0)).clamp(0.0, 1.0);
```

- Uses historical slot std dev, not model uncertainty.
- Sigmoid with magic constant (20.0) — not calibrated.
- Horizon penalty (0.15 per hour) — arbitrary.
- No relationship to actual model prediction variance.

### 7.2 Proposed: Per-Tree Prediction Intervals

**If smartcore exposes individual tree predictions:**

Each of the N trees in the forest produces an independent prediction for a given input.
The distribution of these N predictions directly gives us a prediction interval.

```
For input x:
  tree_predictions = [tree_1(x), tree_2(x), ..., tree_N(x)]
  predicted_value  = mean(tree_predictions)
  confidence_low   = percentile(tree_predictions, 10)  // 10th percentile
  confidence_high  = percentile(tree_predictions, 90)  // 90th percentile
  // → 80% prediction interval
```

This is the **Quantile Regression Forest** approach (Meinshausen, 2006). Advantages:
- Statistically grounded — directly reflects model disagreement.
- Automatically widens for uncertain inputs (where trees disagree).
- No hand-tuned constants.
- Naturally captures heteroscedasticity (different uncertainty at different times of day).

### 7.3 Fallback: Residual-Based Intervals

**If smartcore does NOT expose per-tree predictions:**

Calibrate intervals from cross-validation residuals.

```
1. During CV, collect all residuals: residual = actual - predicted
2. Group residuals by (weekday, hour) slot
3. For each slot, compute empirical quantiles:
   q10 = percentile(residuals, 10)
   q90 = percentile(residuals, 90)
4. At prediction time:
   confidence_low  = predicted + q10_for_slot
   confidence_high = predicted + q90_for_slot
5. Scale by horizon: width *= (1 + 0.1 * (hours_ahead - 1))
```

Less elegant than per-tree quantiles but still empirically calibrated rather than heuristic.

### 7.4 Confidence Score

Replace the sigmoid heuristic with **interval-width-based scoring:**

```rust
// Narrower interval = higher confidence
let interval_width = confidence_high - confidence_low;
let confidence_score = (1.0 - interval_width / 100.0).clamp(0.1, 0.95);
```

Or use the coefficient of variation of per-tree predictions:

```rust
let cv = std_dev(tree_predictions) / mean(tree_predictions).abs();
let confidence_score = (1.0 - cv).clamp(0.1, 0.95);
```

### 7.5 `PredictionWithConfidence` Changes

```rust
pub enum PredictionMethod {
    MachineLearning { confidence: f64 },
    HistoricalAverage,
    // New:
    RandomForest { confidence: f64, n_trees: usize },
}
```

The existing fields (`confidence_low`, `confidence_high`, `confidence_score`) remain — only
the computation changes. GUI code remains compatible.

### 7.6 Default Interval Coverage

**Recommendation: 80% prediction interval** (10th to 90th percentile).

Rationale: 95% intervals are too wide to be actionable for gym-goers ("the gym will be
somewhere between 10% and 90% full" is useless). 80% gives a practical range that's useful
for decision-making while still being honest about uncertainty.

> **Decision needed:**
> 1. 80% prediction interval, or prefer 90%?
> 2. If per-tree predictions are unavailable, approve residual-based fallback?

---

## 8. Model Persistence Redesign

### 8.1 Current State

`PersistedModel` (v1) stores:
- Version, created_at, training_window_days, training_samples
- training_mse, validation_mse
- slot_stats (serialized historical stats)
- `ModelSummary` { model_type: String, max_depth: Option, feature_importance: Option }

**The actual model weights (`FittedLinearRegression`) are NOT serialized.** The model must
retrain from scratch on every application restart. This means changing the persistence format
is zero-risk for backward compatibility — there's nothing to break.

### 8.2 V2 Schema

```rust
pub struct PersistedModel {
    pub version: u32,                           // Bump to 2
    pub created_at: DateTime<Utc>,
    pub training_window_days: i64,
    pub training_samples: usize,
    pub algorithm: String,                      // "RandomForest" | "LinearRegression"

    // Training metrics
    pub cv_scores: Option<SerializedCvScores>,  // New: CV results
    pub oob_error: Option<f64>,                 // New: OOB error

    // Model weights (NEW — actually persists the model now)
    pub model_bytes: Vec<u8>,                   // Serialized model (smartcore serde)

    // Feature metadata
    pub feature_names: Vec<String>,             // For validation on load
    pub feature_importance: Option<Vec<(String, f64)>>,  // Named importance scores
    pub slot_stats: Vec<SerializedSlotStats>,

    // Hyperparameters
    pub hyperparameters: SerializedHyperparameters,

    // Residual calibration (for confidence intervals)
    pub residual_quantiles: Option<Vec<SlotResidualQuantiles>>,
}
```

### 8.3 Serialization Format

**bincode** (already a dependency) + **zstd compression** (new dependency).

- bincode is compact and fast for Rust-native serialization.
- RF models can be large (500 trees × ~1000 nodes × ~24 bytes/node ≈ 12MB uncompressed).
- zstd typically achieves 3-5x compression on tree structures → ~3-4MB compressed.
- zstd is pure Rust via the `zstd` crate with `default-features = false`.

**Version migration:** On load, if `version < 2`, return `Err` that triggers retrain.
No complex migration — the V1 format didn't contain model weights anyway.

### 8.4 `zstd` Dependency

```toml
[dependencies]
zstd = { version = "0.13", default-features = false, optional = true }
```

Add to `gui` feature list.

> **Decision needed:**
> 1. Approve bincode + zstd for persistence?
> 2. Is ~3-4MB model file size acceptable?

---

## 9. Architecture Changes

### 9.1 Proposed Module Structure

```
src/ml/
├── mod.rs                  OccupancyPredictor coordinator (stable public API)
├── config.rs               MlConfig + MlAlgorithm (extracted from mod.rs)
├── features/
│   ├── mod.rs              PredictionFeatures (22 fields), FeatureExtractor, re-exports
│   ├── cyclical.rs         cyclical_encode(), temporal feature computation
│   └── momentum.rs         extract_momentum(), extract_day_features(), new rolling stats
├── model/
│   ├── mod.rs              ModelBackend enum, TrainedModel trait or dispatching struct
│   ├── random_forest.rs    smartcore RF wrapper, DenseMatrix conversion
│   └── linear.rs           Existing linfa linear regression (moved, unchanged)
├── training/
│   ├── mod.rs              train_model() async, train_model_sync(), orchestration
│   ├── cross_validation.rs TimeSeriesSplit, FoldScores, CrossValidationScores
│   ├── hyperparameter.rs   HyperparameterSet, grid search logic
│   └── data_prep.rs        TrainingDataPreparer (moved from training.rs)
├── confidence.rs           Redesigned: per-tree quantiles or residual-based
├── persistence.rs          V2 schema, bincode+zstd, version migration
└── evaluation.rs           MSE, RMSE, MAE, MAPE, R², feature importance formatting
```

Estimated: ~3,500–4,000 LOC (up from ~2,022).

### 9.2 Model Dispatch

**Enum dispatch** (not trait-based):

```rust
pub enum ModelBackend {
    RandomForest(RandomForestModel),
    LinearRegression(LinearRegressionModel),
}

impl ModelBackend {
    pub fn predict(&self, features: &PredictionFeatures) -> Option<f64> { ... }
    pub fn predict_batch(&self, features: &[PredictionFeatures]) -> Vec<f64> { ... }
    pub fn feature_importance(&self) -> Option<Vec<(String, f64)>> { ... }
    pub fn per_tree_predictions(&self, features: &PredictionFeatures) -> Option<Vec<f64>> { ... }
}
```

Rationale: Only two variants, no need for dynamic dispatch. Enum is simpler, faster, and
easier to serialize than `Box<dyn ModelTrait>`.

### 9.3 Public API Stability

The `OccupancyPredictor` public API remains stable:
- `new(config)`, `predict()`, `set_model()`, `add_observation()`, `update_baseline()` —
  signatures unchanged.
- `PredictionWithConfidence` gains a new `PredictionMethod` variant but existing fields stay.
- `MlConfig` gains new fields with defaults — non-breaking via `Default`.
- `app.rs` integration points (`Message::MlTrainingCompleted`, `trigger_ml_training()`)
  unchanged.

### 9.4 Dependency Changes

```toml
# New dependencies (all optional, gui-gated):
smartcore = { version = "0.4", default-features = false, features = ["serde"], optional = true }
zstd = { version = "0.13", default-features = false, optional = true }

# Existing (unchanged):
linfa = { version = "0.8", default-features = false, optional = true }
linfa-linear = { version = "0.8", default-features = false, optional = true }
ndarray = { version = "0.16", default-features = false, optional = true }

# Feature gate update:
[features]
gui = [...existing..., "dep:smartcore", "dep:zstd"]
```

`linfa` and `linfa-linear` are kept for the linear regression fallback.

> **Decision needed:**
> 1. Enum dispatch over trait-based dispatch?
> 2. Approve the module restructure, or prefer a flatter layout?

---

## 10. Testing Strategy

### 10.1 Test Categories

| Category | Framework | Count | Location |
|----------|-----------|-------|----------|
| Unit tests | `#[cfg(test)] mod tests` | ~70-80 | Inline in each source file |
| Property-based | `proptest` (1000 cases) | ~12-15 | Inline in features/, model/, confidence.rs |
| Integration | `#[tokio::test]` | ~5-8 | `tests/ml_integration.rs` (new) |
| Regression | Snapshot comparison | ~3-5 | Inline in training/ |

### 10.2 Property-Based Tests (Currently Missing — Required by CLAUDE.md)

```rust
// features/mod.rs
proptest! {
    #[test]
    fn features_to_vec_always_correct_length(hour in 0u32..24, weekday in 0u32..7) {
        let features = /* construct with arbitrary valid inputs */;
        prop_assert_eq!(features.to_vec().len(), PredictionFeatures::NUM_FEATURES);
    }

    #[test]
    fn all_features_are_finite(/* arbitrary valid inputs */) {
        let features = /* construct */;
        for v in features.to_vec() {
            prop_assert!(v.is_finite(), "Feature value was not finite: {}", v);
        }
    }
}

// confidence.rs
proptest! {
    #[test]
    fn predictions_always_in_valid_range(predicted in 0.0f64..100.0) {
        let pred = PredictionWithConfidence::new(/* ... */);
        prop_assert!(pred.predicted_value >= 0.0 && pred.predicted_value <= 100.0);
        prop_assert!(pred.confidence_low <= pred.predicted_value);
        prop_assert!(pred.confidence_high >= pred.predicted_value);
    }
}

// training/cross_validation.rs
proptest! {
    #[test]
    fn time_series_split_no_overlap(n_samples in 50usize..500, k in 2usize..6) {
        let splits = TimeSeriesSplit::new(k, gap).split(n_samples);
        // Assert: no validation index appears in any training set
        // Assert: all training indices < all validation indices per fold
    }
}
```

### 10.3 RF-Specific Tests

- **Prediction determinism:** Same seed + same data → same predictions.
- **Prediction range:** All predictions in [0.0, 100.0] for valid inputs.
- **Feature importance:** Sum of importances ≈ 1.0 (within tolerance).
- **More trees → lower variance:** Prediction std dev across runs decreases with n_trees.
- **Model persistence round-trip:** `save() → load() → predict()` produces identical results.
- **DenseMatrix conversion:** `PredictionFeatures → DenseMatrix → predict()` works correctly.

### 10.4 Cross-Validation Tests

- Folds are temporally ordered (train always before validation).
- No data leakage (no overlap between train and validation sets).
- Gap is respected (min distance between last train sample and first val sample).
- Every data point appears in exactly one validation fold.
- Metrics are computed correctly per fold.

### 10.5 Backward Compatibility Tests

- V1 `PersistedModel` triggers graceful retrain (not crash).
- `MlConfig` with only old fields deserializes correctly (new fields use `Default`).
- `PredictionMethod::MachineLearning { .. }` still renders correctly in GUI.
- Historical average fallback still works when RF model is unavailable.

---

## 11. Migration Path

### 11.1 Linear Regression Retention

Keep linear regression as a **permanent fallback**:
- Zero marginal cost (code already exists).
- Provides baseline for A/B comparison during development.
- Fallback when RF training fails (insufficient data, numerical issues).
- `MlAlgorithm::LinearRegression` selectable via config.

### 11.2 Runtime Algorithm Selection

```rust
// MlConfig
pub algorithm: MlAlgorithm,  // Default: RandomForest

// At training time:
match config.algorithm {
    MlAlgorithm::RandomForest => train_random_forest(...),
    MlAlgorithm::LinearRegression => train_linear_regression(...),
}
```

### 11.3 Validation Strategy

Before switching the default from LR to RF:
1. Train both models on the same data.
2. Compare CV metrics (MSE, RMSE, R²).
3. RF should show meaningful improvement (>10% lower RMSE) to justify the complexity.
4. If RF doesn't improve: keep LR as default, investigate feature engineering.

---

## 12. Implementation Phases

### Phase 1: Foundation (no algorithm change)

**Goal:** Restructure the ML module and add missing infrastructure.

- Extract `MlConfig` to `ml/config.rs`.
- Add `evaluation.rs` with MSE, RMSE, MAE, MAPE, R² functions.
- Implement `TimeSeriesSplit` in `training/cross_validation.rs`.
- Add property-based tests for existing features and confidence.
- Split `features.rs` into `features/` directory module.
- **All existing tests must continue passing.**

**Files created:** `ml/config.rs`, `ml/evaluation.rs`, `ml/training/cross_validation.rs`,
`ml/features/cyclical.rs`, `ml/features/momentum.rs`
**Files modified:** `ml/mod.rs`, `ml/features/mod.rs`, `ml/training/mod.rs`

### Phase 2: Random Forest Core

**Goal:** Add smartcore RF alongside existing LR.

- Add `smartcore` dependency (optional, gui-gated).
- Implement `RandomForestModel` in `ml/model/random_forest.rs`.
- Implement `ModelBackend` enum dispatch in `ml/model/mod.rs`.
- Move existing LR code to `ml/model/linear.rs`.
- DenseMatrix conversion layer.
- Update `ModelBuilder` for RF hyperparameters.
- **Both LR and RF should be trainable and testable.**

**Files created:** `ml/model/random_forest.rs`, `ml/model/linear.rs`, `ml/model/mod.rs`
**Files modified:** `Cargo.toml`, `ml/mod.rs`, `ml/training/mod.rs`

### Phase 3: Training Pipeline Upgrade

**Goal:** Add cross-validation and hyperparameter tuning.

- Implement grid search in `training/hyperparameter.rs`.
- Integrate TimeSeriesSplit with grid search in training orchestration.
- Add `CrossValidationScores`, `HyperparameterSet` types.
- Update `TrainingResult` with CV scores and best hyperparameters.
- Feature importance extraction from trained RF.
- `spawn_blocking` for CPU-intensive grid search.
- Update `MlConfig` with new fields.

**Files created:** `ml/training/hyperparameter.rs`, `ml/training/data_prep.rs`
**Files modified:** `ml/training/mod.rs`, `ml/config.rs`, `src/config.rs`

### Phase 4: Feature Engineering Expansion

**Goal:** Add 6 new features.

- Implement `raw_hour`, `raw_weekday`, `time_to_close`, `occupancy_volatility`,
  `recent_avg_6h`, `prev_week_same_slot`.
- Update `PredictionFeatures` struct and `NUM_FEATURES` constant.
- Update `to_vec()` and `feature_names()`.
- Update `FeatureExtractor::extract()`.
- Add property-based tests for new features.

**Files modified:** `ml/features/mod.rs`, `ml/features/momentum.rs`

### Phase 5: Confidence Interval Overhaul

**Goal:** Replace heuristic confidence with model-derived intervals.

- Implement per-tree prediction extraction (if smartcore supports it).
- Or implement residual-based calibration from CV residuals.
- Update `PredictionWithConfidence` construction in `OccupancyPredictor::ml_predict()`.
- Add `PredictionMethod::RandomForest` variant.
- Update confidence score computation.
- Property-based tests for interval validity.

**Files modified:** `ml/confidence.rs`, `ml/mod.rs`

### Phase 6: Persistence + GUI Integration

**Goal:** Persist model weights and update the GUI.

- Implement V2 `PersistedModel` with model_bytes, feature importance, CV scores.
- Add zstd compression.
- Version migration (V1 → retrain).
- Update `ml_predictions.rs` view:
  - Show algorithm name (RF vs LR).
  - Display feature importance (if available).
  - Show CV scores in model status card.
- Update `app.rs` for new `TrainingResult` fields.

**Files modified:** `ml/persistence.rs`, `src/views/ml_predictions.rs`, `src/app.rs`,
`Cargo.toml` (add zstd)

---

## Appendix A: Open Decisions Summary

| # | Decision | Options | Section |
|---|----------|---------|---------|
| 1 | Approve smartcore as primary crate? | Yes / Evaluate forust-ml too | §3.4 |
| 2 | Approve Random Forest as primary algorithm? | RF / GBT / Both | §4.3 |
| 3 | Approve 6 new features? | Yes / Modify set | §5.2 |
| 4 | TimeSeriesSplit with 4 folds, 1-day gap? | Yes / Different params | §6.1 |
| 5 | Grid search with 144 configs? | Yes / Reduce grid | §6.2 |
| 6 | `MlAlgorithm` in TOML config? | Yes / Always RF | §6.5 |
| 7 | 80% prediction interval? | 80% / 90% | §7.6 |
| 8 | Residual-based CI as fallback? | Yes / Other approach | §7.3 |
| 9 | bincode + zstd for persistence? | Yes / Alternative | §8.3 |
| 10 | ~3-4MB model file acceptable? | Yes / Compress more | §8.3 |
| 11 | Enum dispatch (not trait)? | Enum / Trait | §9.2 |
| 12 | Module restructure as proposed? | Yes / Flatter layout | §9.1 |

## Appendix B: Critical Files Reference

| File | Role in Upgrade |
|------|----------------|
| `src/ml/model.rs` | Core replacement: TrainedModel → ModelBackend enum |
| `src/ml/training.rs` | Major rewrite: add CV, grid search, data prep extraction |
| `src/ml/features.rs` | Expand: 16 → 22 features, split into directory module |
| `src/ml/persistence.rs` | V2 schema: actually persist model weights |
| `src/ml/confidence.rs` | Redesign: per-tree quantiles or residual-based |
| `src/ml/mod.rs` | Extract MlConfig, update predict() for new confidence |
| `Cargo.toml` | Add smartcore, zstd as optional gui deps |
| `src/config.rs` | Add new MlConfig fields with defaults |
| `src/app.rs` | Update TrainingResult handling, new fields |
| `src/views/ml_predictions.rs` | Display algorithm, feature importance, CV scores |
