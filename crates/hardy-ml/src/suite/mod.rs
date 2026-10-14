//! The forecasting suite: every model behind one interface, so the
//! benchmark, the nightly fit and live publishing treat them alike.
//!
//! A model is fitted on everything observed up to a cut and then forecasts
//! [`HORIZON_STEPS`] slots from any later origin, seeing only data up to that
//! origin ([`Observed`]). [`forecast`] turns a model's raw output into a
//! usable forecast: checked, clipped to 0–100 and 0 while the gym is closed.

mod baselines;
mod legacy;
pub(crate) mod linalg;
mod packaged;
mod profiles;
pub mod safety;
mod statistical;
mod tabular;

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::GymSchedule;
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    benchmark::{Combination, Ensemble},
    error::MlError,
    grid::{Grid, GridView, HORIZON_STEPS, is_open_slot},
    history::History,
};

/// Version of the stored model bytes; a stored model with another version
/// is refitted instead of loaded.
pub const ARTIFACT_VERSION: u32 = 1;
const ZSTD_LEVEL: i32 = 3;
/// Safety limit when decompressing a stored model.
const MAX_DECOMPRESSED_BYTES: usize = 64 * 1024 * 1024;
/// Raw forecasts outside this range (before clipping to 0–100) mean the
/// model has gone wrong (e.g. an exploding fit), not that it is a bit off.
const SANE_RANGE: std::ops::RangeInclusive<f64> = -50.0..=150.0;

/// Everything observed up to a moment: the slot grid and the readings.
#[derive(Debug, Clone, Copy)]
pub struct Observed<'a> {
    grid: GridView<'a>,
    history: &'a History,
    schedule: &'a GymSchedule,
}

impl<'a> Observed<'a> {
    /// What a forecast made at `at` may see. `history` must hold no reading
    /// after `at` (see [`History::until`]); the grid is cut here.
    pub fn new(
        grid: &'a Grid,
        history: &'a History,
        schedule: &'a GymSchedule,
        at: DateTime<Utc>,
    ) -> Self {
        Self {
            grid: grid.view_until(at),
            history,
            schedule,
        }
    }

    /// The slots up to the origin.
    pub fn grid(&self) -> GridView<'a> {
        self.grid
    }

    /// The readings up to the origin.
    pub fn history(&self) -> &'a History {
        self.history
    }

    /// Opening hours (known in advance, so they may be read for any time).
    pub fn schedule(&self) -> &'a GymSchedule {
        self.schedule
    }

    /// The moment of the forecast (a slot boundary).
    pub fn origin(&self) -> DateTime<Utc> {
        self.grid.origin()
    }

    /// Time from the first reading to the origin; zero without readings.
    pub fn span(&self) -> TimeDelta {
        self.history
            .first_time()
            .map_or(TimeDelta::zero(), |first| self.origin() - first)
    }
}

/// A forecasting model.
pub trait Forecaster {
    /// History needed before the model can be fitted (at least one reading
    /// is always needed).
    fn min_history(&self) -> TimeDelta;

    /// Backtest days one fit serves: 1 refits every day; more reuses a fit
    /// for the following days, which it forecasts from their own data
    /// without refitting (for models whose fit is slow).
    fn refit_days(&self) -> usize {
        1
    }

    /// Fits the model on everything in `data`.
    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError>;

    /// Raw forecasts for the [`HORIZON_STEPS`] slots after `data.origin()`;
    /// values for closed slots are ignored.
    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError>;
}

/// Broad kind of model, for grouping in the GUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    Baseline,
    Profile,
    Statistical,
    MachineLearning,
    Ensemble,
}

impl Family {
    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Family::Baseline => "Baselines",
            Family::Profile => "Profile-based",
            Family::Statistical => "Statistical",
            Family::MachineLearning => "Regression and machine learning",
            Family::Ensemble => "Ensembles",
        }
    }
}

macro_rules! registry {
    (
        forecasters { $( $variant:ident($ty:ty) = $key:literal, $name:literal, $family:ident; )* }
        ensembles { $( $evariant:ident = $ekey:literal, $ename:literal, $combination:ident; )* }
    ) => {
        /// Identifies a model; stored by its [`ModelId::key`].
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum ModelId { $( $variant, )* $( $evariant, )* }

        impl ModelId {
            /// Every model, in display order.
            pub const ALL: &'static [ModelId] =
                &[ $( ModelId::$variant, )* $( ModelId::$evariant, )* ];
            /// The models that forecast from data (every one but the
            /// ensembles).
            pub const FORECASTERS: &'static [ModelId] = &[ $( ModelId::$variant, )* ];
            /// The ensembles, which combine the forecasters.
            pub const ENSEMBLES: &'static [ModelId] = &[ $( ModelId::$evariant, )* ];

            /// Stable key used in the database.
            pub fn key(self) -> &'static str {
                match self {
                    $( ModelId::$variant => $key, )*
                    $( ModelId::$evariant => $ekey, )*
                }
            }

            /// Display name.
            pub fn name(self) -> &'static str {
                match self {
                    $( ModelId::$variant => $name, )*
                    $( ModelId::$evariant => $ename, )*
                }
            }

            /// Kind of model.
            pub fn family(self) -> Family {
                match self {
                    $( ModelId::$variant => Family::$family, )*
                    $( ModelId::$evariant => Family::Ensemble, )*
                }
            }

            /// How an ensemble combines the forecasters; `None` for a
            /// forecaster.
            pub fn combination(self) -> Option<Combination> {
                match self {
                    $( ModelId::$evariant => Some(Combination::$combination), )*
                    _ => None,
                }
            }

            /// The model with this key.
            pub fn from_key(key: &str) -> Option<Self> {
                match key {
                    $( $key => Some(ModelId::$variant), )*
                    $( $ekey => Some(ModelId::$evariant), )*
                    _ => None,
                }
            }
        }

        /// Any forecaster of the suite.
        #[derive(Debug)]
        pub enum AnyModel { $( $variant($ty), )* }

        impl AnyModel {
            /// An unfitted model; `None` for an ensemble.
            pub fn new(id: ModelId) -> Option<Self> {
                match id {
                    $( ModelId::$variant => Some(AnyModel::$variant(<$ty>::default())), )*
                    _ => None,
                }
            }

            /// Which model this is.
            pub fn id(&self) -> ModelId {
                match self { $( AnyModel::$variant(_) => ModelId::$variant, )* }
            }

            fn inner(&self) -> &dyn Forecaster {
                match self { $( AnyModel::$variant(m) => m, )* }
            }

            fn inner_mut(&mut self) -> &mut dyn Forecaster {
                match self { $( AnyModel::$variant(m) => m, )* }
            }

            /// The fitted model as compressed bytes.
            pub fn to_bytes(&self) -> Result<Vec<u8>, MlError> {
                match self { $( AnyModel::$variant(m) => encode(m), )* }
            }

            /// A fitted model from [`AnyModel::to_bytes`].
            pub fn from_bytes(id: ModelId, bytes: &[u8]) -> Result<Self, MlError> {
                match id {
                    $( ModelId::$variant => decode::<$ty>(bytes).map(AnyModel::$variant), )*
                    _ => Err(MlError::Serialization(format!("{} is an ensemble", id.key()))),
                }
            }
        }
    };
}

registry! {
    forecasters {
        Naive(baselines::Naive) = "naive", "Persistence", Baseline;
        MovingAverage(baselines::MovingAverage) =
            "moving_average", "Recent moving average", Baseline;
        DampedTrend(baselines::DampedTrend) = "damped_trend", "Damped local trend", Baseline;
        DailySeasonalNaive(baselines::DailySeasonalNaive) =
            "daily_seasonal_naive", "Daily seasonal naive", Baseline;
        WeeklySeasonalNaive(baselines::WeeklySeasonalNaive) =
            "weekly_seasonal_naive", "Weekly seasonal naive", Baseline;
        WeeklySeasonalMean(baselines::WeeklySeasonalMean) =
            "weekly_seasonal_mean", "Weekly seasonal mean", Baseline;
        WeeklySeasonalMedian(baselines::WeeklySeasonalMedian) =
            "weekly_seasonal_median", "Weekly seasonal median", Baseline;
        WeeklyRecencyMean(baselines::WeeklyRecencyMean) =
            "weekly_recency_mean", "Recency-weighted weekly mean", Baseline;
        ProfileCurrentOffset(profiles::ProfileCurrentOffset) =
            "profile_current_offset", "Weekly profile + current offset", Profile;
        ProfileDecayingOffset(legacy::ProfileDecayingOffset) =
            "profile_decaying_offset", "Weekly profile + decaying offset", Profile;
        ProfileResidualAr(profiles::ProfileResidualAr) =
            "profile_residual_ar", "Weekly profile + residual autoregression", Profile;
        SimilarDays(profiles::SimilarDays) = "similar_days", "Similar days", Profile;
        DailyPca(profiles::DailyPca) = "daily_pca", "Daily-profile PCA + regression", Profile;
        SimpleSmoothing(statistical::SimpleSmoothing) =
            "simple_smoothing", "Simple exponential smoothing", Statistical;
        HoltSmoothing(statistical::HoltSmoothing) = "holt", "Holt's linear trend", Statistical;
        DampedHolt(statistical::DampedHolt) = "damped_holt", "Damped Holt", Statistical;
        Theta(statistical::Theta) = "theta", "Theta method", Statistical;
        Autoregression(statistical::Autoregression) =
            "autoregression", "Autoregression", Statistical;
        HoltWintersDaily(statistical::HoltWintersDaily) =
            "holt_winters_daily", "Holt–Winters (daily season)", Statistical;
        HoltWintersWeekly(statistical::HoltWintersWeekly) =
            "holt_winters_weekly", "Holt–Winters (weekly season)", Statistical;
        StructuralModel(statistical::StructuralModel) =
            "structural", "Structural model (Kalman filter)", Statistical;
        AutoEts(packaged::AutoEts) = "auto_ets", "AutoETS", Statistical;
        AutoArima(packaged::AutoArima) = "auto_arima", "AutoARIMA", Statistical;
        Sarima(packaged::Sarima) = "sarima", "SARIMA (daily season)", Statistical;
        StlEts(packaged::StlEts) = "stl_ets", "STL + ETS", Statistical;
        StlArima(packaged::StlArima) = "stl_arima", "STL + ARIMA", Statistical;
        MstlEts(packaged::MstlEts) = "mstl_ets", "MSTL + ETS", Statistical;
        MstlArima(packaged::MstlArima) = "mstl_arima", "MSTL + ARIMA", Statistical;
        AutoTbats(packaged::AutoTbats) = "auto_tbats", "AutoTBATS", Statistical;
        Mfles(packaged::Mfles) = "mfles", "MFLES", Statistical;
        AutoTheta(packaged::AutoTheta) = "auto_theta", "AutoTheta", Statistical;
        FourierArima(packaged::FourierArima) =
            "fourier_arima", "Fourier regression + ARIMA errors", Statistical;
        Ridge(tabular::Ridge) = "ridge", "Ridge regression", MachineLearning;
        ElasticNet(tabular::ElasticNet) = "elastic_net", "Elastic-net regression", MachineLearning;
        SplineGam(tabular::SplineGam) = "spline_gam", "Spline regression (GAM)", MachineLearning;
        MedianRegression(tabular::MedianRegression) =
            "median_regression", "Median (quantile) regression", MachineLearning;
        RandomForest(tabular::RandomForest) = "random_forest", "Random forest", MachineLearning;
        BoostedTrees(tabular::BoostedTrees) =
            "boosted_trees", "Gradient-boosted trees", MachineLearning;
        SupportVectors(tabular::SupportVectors) =
            "svr", "Support-vector regression", MachineLearning;
    }
    ensembles {
        MeanEnsemble = "ensemble_mean", "Mean of the best five", Mean;
        MedianEnsemble = "ensemble_median", "Median of the best five", Median;
        WeightedEnsemble = "ensemble_inverse_mae", "Accuracy-weighted ensemble", InverseMae;
        BestMemberEnsemble = "ensemble_best_member", "Best-member ensemble", BestMember;
        StackedEnsemble = "ensemble_stacked", "Stacked ensemble", Stacked;
    }
}

impl Forecaster for AnyModel {
    fn min_history(&self) -> TimeDelta {
        self.inner().min_history()
    }

    fn refit_days(&self) -> usize {
        self.inner().refit_days()
    }

    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
        self.inner_mut().fit(data)
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        self.inner().predict(data)
    }
}

/// Whether `data` holds enough history to fit `model`.
pub fn can_fit(model: &dyn Forecaster, data: &Observed<'_>) -> bool {
    data.history().first_time().is_some() && data.span() >= model.min_history()
}

/// Fits `model` on `data` if it holds enough history.
pub fn fit(model: &mut dyn Forecaster, data: &Observed<'_>) -> Result<(), MlError> {
    if !can_fit(model, data) {
        return Err(MlError::NotEnoughHistory {
            have_hours: data.span().num_hours(),
            need_hours: model.min_history().num_hours(),
        });
    }
    model.fit(data)
}

/// `model`'s forecast for the [`HORIZON_STEPS`] slots after `data.origin()`:
/// 0 for closed slots, clipped to 0–100 otherwise. Fails if the model
/// returns the wrong number of values or anything non-finite or far outside
/// the possible range for an open slot.
pub fn forecast(model: &dyn Forecaster, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
    let raw = model.predict(data)?;
    if raw.len() != HORIZON_STEPS {
        return Err(MlError::BadForecast(format!(
            "{} values instead of {HORIZON_STEPS}",
            raw.len()
        )));
    }
    let grid = data.grid();
    raw.into_iter()
        .enumerate()
        .map(|(i, value)| {
            if !is_open_slot(data.schedule(), grid.step_end(i + 1)) {
                Ok(0.0)
            } else if value.is_finite() && SANE_RANGE.contains(&value) {
                Ok(value.clamp(0.0, 100.0))
            } else {
                Err(MlError::BadForecast(format!("{value} for step {}", i + 1)))
            }
        })
        .collect()
}

impl Ensemble<ModelId> {
    /// The ensemble as compressed bytes (members stored by key).
    pub fn to_bytes(&self) -> Result<Vec<u8>, MlError> {
        let keyed = self
            .try_map(|id| Some(id.key()))
            .ok_or_else(|| MlError::Serialization("unkeyed member".to_string()))?;
        encode(&keyed)
    }

    /// An ensemble from [`Ensemble::to_bytes`]; fails if a member is no
    /// longer known.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MlError> {
        let keyed: Ensemble<String> = decode(bytes)?;
        keyed
            .try_map(|key| ModelId::from_key(&key))
            .ok_or_else(|| MlError::Serialization("unknown ensemble member".to_string()))
    }
}

pub(crate) fn encode<T: Serialize>(model: &T) -> Result<Vec<u8>, MlError> {
    let raw = postcard::to_allocvec(model).map_err(|e| MlError::Serialization(e.to_string()))?;
    zstd::bulk::compress(&raw, ZSTD_LEVEL).map_err(|e| MlError::Serialization(e.to_string()))
}

pub(crate) fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, MlError> {
    let raw = zstd::bulk::decompress(bytes, MAX_DECOMPRESSED_BYTES)
        .map_err(|e| MlError::Serialization(e.to_string()))?;
    postcard::from_bytes(&raw).map_err(|e| MlError::Serialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use anyhow::{Context, Result};

    use super::*;
    use crate::test_support::{local, minutely};

    /// Forecasts a fixed value for every step.
    struct Constant(f64);

    impl Forecaster for Constant {
        fn min_history(&self) -> TimeDelta {
            TimeDelta::zero()
        }

        fn fit(&mut self, _: &Observed<'_>) -> Result<(), MlError> {
            Ok(())
        }

        fn predict(&self, _: &Observed<'_>) -> Result<Vec<f64>, MlError> {
            Ok(vec![self.0; HORIZON_STEPS])
        }
    }

    #[test]
    fn test_model_keys_are_unique_and_round_trip() {
        let keys: HashSet<_> = ModelId::ALL.iter().map(|id| id.key()).collect();
        assert_eq!(keys.len(), ModelId::ALL.len());
        for id in ModelId::ALL {
            assert_eq!(ModelId::from_key(id.key()), Some(*id));
            assert_ne!(id.name(), "");
        }
        for id in ModelId::FORECASTERS {
            assert_eq!(AnyModel::new(*id).map(|m| m.id()), Some(*id));
            assert_eq!(id.combination(), None);
            assert_ne!(id.family(), Family::Ensemble);
        }
        for id in ModelId::ENSEMBLES {
            assert!(AnyModel::new(*id).is_none());
            assert!(id.combination().is_some());
            assert_eq!(id.family(), Family::Ensemble);
            assert!(AnyModel::from_bytes(*id, &[]).is_err());
        }
        assert_eq!(
            ModelId::FORECASTERS.len() + ModelId::ENSEMBLES.len(),
            ModelId::ALL.len()
        );
        assert_eq!(ModelId::from_key("no_such_model"), None);
    }

    #[test]
    fn test_forecast_zeroes_closed_slots_and_clips() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = minutely(1, |_| 30.0);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(1, 0, 0));
        // 22:00 on a weekday: open until 23:00.
        let data = Observed::new(&grid, &history, &schedule, local(0, 22, 0));

        let values = forecast(&Constant(120.0), &data)?;
        assert_eq!(values.len(), HORIZON_STEPS);
        assert_eq!(values[0], 100.0, "22:00–22:15 is open, clipped");
        assert_eq!(values[3], 100.0, "22:45–23:00 is open");
        assert_eq!(values[4], 0.0, "23:00–23:15 is closed");
        Ok(())
    }

    #[test]
    fn test_forecast_rejects_impossible_values() {
        let schedule = GymSchedule::default();
        let history = minutely(1, |_| 30.0);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(1, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(0, 12, 0));

        for bad in [f64::NAN, f64::INFINITY, 1e19, -60.0] {
            let result = forecast(&Constant(bad), &data);
            assert!(
                matches!(result, Err(MlError::BadForecast(_))),
                "{bad} accepted"
            );
        }
    }

    #[test]
    fn test_fit_refuses_short_history() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = minutely(2, |_| 30.0);
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(2, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(2, 0, 0));

        let mut weekly = AnyModel::new(ModelId::WeeklySeasonalNaive).context("forecaster")?;
        assert!(!can_fit(&weekly, &data));
        let error = fit(&mut weekly, &data).err().context("fit should fail")?;
        assert!(matches!(error, MlError::NotEnoughHistory { .. }));

        let mut naive = AnyModel::new(ModelId::Naive).context("forecaster")?;
        assert!(can_fit(&naive, &data));
        fit(&mut naive, &data)?;
        Ok(())
    }

    #[test]
    fn test_ensembles_round_trip_through_bytes() -> Result<()> {
        use crate::benchmark::ensemble::Rows;

        let members = [ModelId::Naive, ModelId::DailyPca];
        let rows: [Rows; 3] = std::array::from_fn(|_| {
            let mut rows = Rows::new(2);
            for i in 0..10 {
                rows.push(
                    f64::from(i),
                    &[Some(f64::from(i) + 1.0), Some(f64::from(i) - 3.0)],
                );
            }
            rows
        });
        let ensemble = Ensemble::fit(Combination::InverseMae, &members, &rows).context("fitted")?;
        let restored = Ensemble::from_bytes(&ensemble.to_bytes()?)?;
        assert_eq!(restored, ensemble);
        assert!(Ensemble::from_bytes(b"junk").is_err());
        Ok(())
    }

    #[test]
    fn test_every_model_round_trips_through_bytes_after_fitting() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = minutely(15, |t| {
            f64::from(chrono::Timelike::hour(
                &t.with_timezone(&schedule.timezone()),
            )) * 3.0
        });
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(15, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(14, 12, 0));

        for id in ModelId::FORECASTERS {
            let mut model = AnyModel::new(*id).context("forecaster")?;
            fit(&mut model, &data).with_context(|| format!("fit {}", id.key()))?;
            let before = forecast(&model, &data)?;
            let restored = AnyModel::from_bytes(*id, &model.to_bytes()?)
                .with_context(|| format!("decode {}", id.key()))?;
            assert_eq!(forecast(&restored, &data)?, before, "{}", id.key());
        }
        Ok(())
    }
}
