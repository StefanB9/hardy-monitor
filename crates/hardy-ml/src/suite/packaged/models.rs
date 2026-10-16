//! The packaged models: one anofox run (search and fit) per forecast origin.

use anofox_forecast::{
    core::TimeSeries,
    models::{MFLES, exponential::AutoETS, tbats::AutoTBATS, theta::AutoTheta as AnofoxAutoTheta},
};
use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};

use super::{
    arima::{auto_arima, sarima},
    decompose::decompose,
    fourier::Fourier,
    series::{deviation_series, grid_series, open_slots},
};
use crate::{
    error::MlError,
    grid::{HORIZON_STEPS, SLOT, SLOTS_PER_DAY, SLOTS_PER_WEEK, is_open_slot},
    suite::{Forecaster, Observed, profiles::WeeklyProfile},
};

/// Which packaged model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    AutoEts,
    AutoArima,
    Sarima,
    StlEts,
    StlArima,
    MstlEts,
    MstlArima,
    AutoTbats,
    Mfles,
    AutoTheta,
    FourierArima,
}

/// Weekly season once the window holds two weeks, daily before.
fn main_season(len: usize) -> usize {
    if len >= 2 * SLOTS_PER_WEEK {
        SLOTS_PER_WEEK
    } else {
        SLOTS_PER_DAY
    }
}

/// Daily and (once the window holds two weeks) weekly seasons.
fn all_seasons(len: usize) -> Vec<usize> {
    if len >= 2 * SLOTS_PER_WEEK {
        vec![SLOTS_PER_DAY, SLOTS_PER_WEEK]
    } else {
        vec![SLOTS_PER_DAY]
    }
}

/// `values` (consecutive slots) as an anofox series.
pub(super) fn time_series(values: &[f64]) -> Result<TimeSeries, MlError> {
    // Only the spacing of the timestamps matters to the models.
    let timestamps = (0..values.len())
        .map(|i| DateTime::UNIX_EPOCH + SLOT * i32::try_from(i).unwrap_or(i32::MAX))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec()).map_err(failed)
}

#[allow(clippy::needless_pass_by_value)] // used as `map_err(failed)`
pub(super) fn failed(e: anofox_forecast::ForecastError) -> MlError {
    MlError::Fit(e.to_string())
}

/// The fitted `model`'s next `horizon` values.
pub(super) fn ahead(
    model: &impl anofox_forecast::models::Forecaster,
    horizon: usize,
) -> Result<Vec<f64>, MlError> {
    let forecast = model.predict(horizon).map_err(failed)?;
    let values = forecast.primary();
    if values.len() == horizon {
        Ok(values.to_vec())
    } else {
        Err(MlError::BadForecast(format!(
            "{} values instead of {horizon}",
            values.len()
        )))
    }
}

/// Fits `model` on `values` (consecutive slots) and forecasts `horizon`
/// steps.
fn run(
    mut model: impl anofox_forecast::models::Forecaster,
    values: &[f64],
    horizon: usize,
) -> Result<Vec<f64>, MlError> {
    model.fit(&time_series(values)?).map_err(failed)?;
    ahead(&model, horizon)
}

/// The forecast of the seasonally adjusted `values` by `inner`, plus their
/// seasonal part (periods `periods`) carried forward.
fn decomposed(
    values: &[f64],
    periods: &[usize],
    inner: impl FnOnce(&[f64], usize) -> Result<Vec<f64>, MlError>,
) -> Result<Vec<f64>, MlError> {
    let (seasonal, adjusted) = decompose(values, periods)?;
    let rest = inner(&adjusted, HORIZON_STEPS)?;
    Ok(rest
        .into_iter()
        .enumerate()
        .map(|(i, v)| v + seasonal.ahead(i + 1))
        .collect())
}

/// Puts the forecasts for consecutive open slots (`open[h - 1]`, h-th open
/// slot ahead) on the horizon's steps, adding `base` for each step's end.
fn on_open_steps(
    data: &Observed<'_>,
    open: &[f64],
    base: impl Fn(DateTime<Utc>) -> f64,
) -> Vec<f64> {
    let grid = data.grid();
    let mut h = 0;
    (1..=HORIZON_STEPS)
        .map(|k| {
            let end = grid.step_end(k);
            if !is_open_slot(data.schedule(), end) {
                return 0.0;
            }
            h += 1;
            base(end) + open.get(h - 1).copied().unwrap_or(0.0)
        })
        .collect()
}

/// Open slots among the horizon's steps.
fn open_steps(data: &Observed<'_>) -> usize {
    let grid = data.grid();
    (1..=HORIZON_STEPS)
        .filter(|&k| is_open_slot(data.schedule(), grid.step_end(k)))
        .count()
        .max(1)
}

/// The forecast of `kind` for the [`HORIZON_STEPS`] slots after `data`'s
/// origin, fitted on the window before it.
fn forecast(kind: Kind, profile: &WeeklyProfile, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
    match kind {
        Kind::AutoArima => {
            let deviations = deviation_series(data, profile);
            let ahead = auto_arima(&deviations, open_steps(data))?;
            Ok(on_open_steps(data, &ahead, |end| profile.at(end)))
        }
        Kind::FourierArima => {
            let tz = data.schedule().timezone();
            let slots = open_slots(data);
            let points: Vec<(DateTime<Utc>, f64)> = slots
                .iter()
                .filter_map(|&(end, v)| v.map(|v| (end, v)))
                .collect();
            let fourier = Fourier::fit(&points, tz)
                .ok_or_else(|| MlError::Fit("Fourier regression is underdetermined".to_string()))?;
            let residuals: Vec<f64> = slots
                .iter()
                .map(|&(end, v)| v.map_or(0.0, |v| v - fourier.at(end)))
                .collect();
            let ahead = auto_arima(&residuals, open_steps(data))?;
            Ok(on_open_steps(data, &ahead, |end| fourier.at(end)))
        }
        _ => {
            let (_, values) = grid_series(data, profile);
            let n = values.len();
            match kind {
                Kind::AutoEts => run(AutoETS::with_period(main_season(n)), &values, HORIZON_STEPS),
                Kind::Sarima => sarima(&values, HORIZON_STEPS),
                Kind::StlEts => decomposed(&values, &[SLOTS_PER_DAY], |v, h| {
                    run(AutoETS::non_seasonal(), v, h)
                }),
                Kind::StlArima => decomposed(&values, &[SLOTS_PER_DAY], auto_arima),
                Kind::MstlEts => decomposed(&values, &all_seasons(n), |v, h| {
                    run(AutoETS::non_seasonal(), v, h)
                }),
                Kind::MstlArima => decomposed(&values, &all_seasons(n), auto_arima),
                Kind::AutoTbats => run(AutoTBATS::new(all_seasons(n)), &values, HORIZON_STEPS),
                Kind::Mfles => run(MFLES::new(all_seasons(n)), &values, HORIZON_STEPS),
                _ => run(
                    AnofoxAutoTheta::seasonal(main_season(n)),
                    &values,
                    HORIZON_STEPS,
                ),
            }
        }
    }
}

macro_rules! packaged_model {
    ($(#[$doc:meta])* $name:ident, $kind:expr, $min_days:expr) => {
        $(#[$doc])*
        #[derive(Debug, Default, Serialize, Deserialize)]
        pub struct $name {
            profile: Option<WeeklyProfile>,
        }

        impl Forecaster for $name {
            fn min_history(&self) -> TimeDelta {
                TimeDelta::days($min_days)
            }

            fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
                self.profile = Some(WeeklyProfile::fit(data));
                Ok(())
            }

            fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
                let profile = self
                    .profile
                    .as_ref()
                    .ok_or_else(|| MlError::Fit("model is not fitted".to_string()))?;
                forecast($kind, profile, data)
            }
        }
    };
}

packaged_model!(
    /// `AutoETS` (anofox) with a weekly season, daily under two weeks.
    AutoEts,
    Kind::AutoEts,
    2
);
packaged_model!(
    /// `AutoARIMA` (anofox) on the deviation from the weekly profile.
    AutoArima,
    Kind::AutoArima,
    1
);
packaged_model!(
    /// `SARIMA(1,0,1)(1,1,1)` with a daily season (anofox).
    Sarima,
    Kind::Sarima,
    2
);
packaged_model!(
    /// STL (daily season, stlrs) + `AutoETS` on the rest.
    StlEts,
    Kind::StlEts,
    2
);
packaged_model!(
    /// STL (daily season, stlrs) + `AutoARIMA` on the rest.
    StlArima,
    Kind::StlArima,
    2
);
packaged_model!(
    /// MSTL (daily and weekly seasons, stlrs) + `AutoETS` on the rest.
    MstlEts,
    Kind::MstlEts,
    14
);
packaged_model!(
    /// MSTL (daily and weekly seasons, stlrs) + `AutoARIMA` on the rest.
    MstlArima,
    Kind::MstlArima,
    14
);
packaged_model!(
    /// `AutoTBATS` (anofox) with daily and weekly seasons.
    AutoTbats,
    Kind::AutoTbats,
    2
);
packaged_model!(
    /// `MFLES` (anofox) with daily and weekly seasons.
    Mfles,
    Kind::Mfles,
    2
);
packaged_model!(
    /// `AutoTheta` (anofox) with a weekly season, daily under two weeks.
    AutoTheta,
    Kind::AutoTheta,
    2
);
packaged_model!(
    /// Fourier regression (faer) + `AutoARIMA` errors (anofox).
    FourierArima,
    Kind::FourierArima,
    7
);

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::{Datelike, Timelike};
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{
        grid::{Grid, slot_end_of},
        history::History,
        suite::{forecast as checked_forecast, packaged::series::LOOKBACK},
        test_support::{local, minutely},
    };

    const ALL: [Kind; 11] = [
        Kind::AutoEts,
        Kind::AutoArima,
        Kind::Sarima,
        Kind::StlEts,
        Kind::StlArima,
        Kind::MstlEts,
        Kind::MstlArima,
        Kind::AutoTbats,
        Kind::Mfles,
        Kind::AutoTheta,
        Kind::FourierArima,
    ];

    /// A smooth weekly curve: busier evenings, quieter weekends.
    fn weekly(t: DateTime<Utc>) -> f64 {
        let local = (slot_end_of(t) - TimeDelta::minutes(1))
            .with_timezone(&GymSchedule::default().timezone());
        let hour = f64::from(local.hour()) + f64::from(local.minute()) / 60.0;
        let weekend = local.weekday().number_from_monday() > 5;
        let peak = 25.0 * (-((hour - 18.5).powi(2)) / 4.0).exp();
        (20.0 + peak) * if weekend { 0.7 } else { 1.0 }
    }

    fn data_for(history: &History, origin: DateTime<Utc>) -> (Grid, History) {
        let schedule = GymSchedule::default();
        (
            Grid::build(history, &schedule, local(0, 0, 0), origin),
            history.until(origin),
        )
    }

    /// Mean absolute error of `kind` over the open slots of the next day.
    fn error_of(kind: Kind, history: &History, origin: DateTime<Utc>) -> Result<f64> {
        let schedule = GymSchedule::default();
        let (grid, seen) = data_for(history, origin);
        let data = Observed::new(&grid, &seen, &schedule, origin);
        let profile = WeeklyProfile::fit(&data);
        let values = forecast(kind, &profile, &data).with_context(|| format!("{kind:?}"))?;
        anyhow::ensure!(
            values.len() == crate::grid::HORIZON_STEPS,
            "{kind:?}: length"
        );
        let (sum, n) = values
            .iter()
            .enumerate()
            .map(|(k, v)| (origin + SLOT * i32::try_from(k + 1).unwrap_or(0), *v))
            .filter(|(end, _)| is_open_slot(&schedule, *end))
            .fold((0.0, 0_u32), |(s, n), (end, v)| {
                (s + (v - weekly(end - TimeDelta::minutes(1))).abs(), n + 1)
            });
        Ok(sum / f64::from(n))
    }

    #[test]
    fn test_every_kind_forecasts_a_repeating_week() -> Result<()> {
        // A Wednesday: daily-season models then look back at a weekday.
        let history = minutely(24, weekly);
        let origin = local(23, 12, 0);
        for kind in ALL {
            let mae = error_of(kind, &history, origin)?;
            assert!(mae < 3.0, "{kind:?}: MAE {mae}");
        }
        Ok(())
    }

    #[test]
    fn test_predict_uses_only_the_lookback_window() -> Result<()> {
        // Garbage before the window must not matter.
        let history = minutely(66, |t| {
            if t < local(66, 0, 0) - LOOKBACK - TimeDelta::days(1) {
                95.0
            } else {
                weekly(t)
            }
        });
        let mae = error_of(Kind::Mfles, &history, local(65, 12, 0))?;
        assert!(mae < 3.0, "MAE {mae}");
        Ok(())
    }

    #[test]
    fn test_models_need_a_fit_and_minimum_history() -> Result<()> {
        let history = minutely(3, |_| 30.0);
        let (grid, seen) = data_for(&history, local(2, 12, 0));
        let schedule = GymSchedule::default();
        let data = Observed::new(&grid, &seen, &schedule, local(2, 12, 0));
        assert!(AutoEts::default().predict(&data).is_err());
        assert_eq!(AutoArima::default().min_history(), TimeDelta::days(1));
        assert_eq!(MstlEts::default().min_history(), TimeDelta::days(14));
        assert_eq!(FourierArima::default().min_history(), TimeDelta::days(7));
        let mut model = AutoArima::default();
        model.fit(&data)?;
        let values = checked_forecast(&model, &data)?;
        assert!(values.iter().all(|v| (0.0..=100.0).contains(v)));
        Ok(())
    }

    /// Forecasts at steps 1, 8 and 40 from Tuesday noon after 15 days of
    /// the weekly curve, as anofox-forecast 0.16.2 and stlrs 0.4.0 produce
    /// them. A dependency update that changes any of them must be checked
    /// against the benchmark before the values here are updated (0.16.0 to
    /// 0.16.2: the ARIMA models' 28-day error fell 6–24%).
    #[allow(clippy::unreadable_literal)]
    const GOLDEN: [(Kind, [f64; 3]); 11] = [
        (
            Kind::AutoEts,
            [20.00061252751475, 20.00061252751475, 20.00061252751475],
        ),
        (
            Kind::AutoArima,
            [20.001361862606124, 20.15240807987595, 21.2037877537554],
        ),
        (
            Kind::Sarima,
            [19.89852100328245, 19.465786486826495, 19.248689474052355],
        ),
        (
            Kind::StlEts,
            [20.027198503122484, 20.16010914324821, 17.1148286590333],
        ),
        (
            Kind::StlArima,
            [20.02719885528378, 20.160109495409504, 17.114829011194594],
        ),
        (
            Kind::MstlEts,
            [20.023236702399554, 20.173263987375943, 21.014207756524847],
        ),
        (
            Kind::MstlArima,
            [20.025108452706924, 20.184913480970927, 21.025202666958354],
        ),
        (
            Kind::AutoTbats,
            [20.839590324375646, 21.618755371309614, 21.139644573589273],
        ),
        (
            Kind::Mfles,
            [19.76675794979764, 20.066185562228142, 17.32902094196144],
        ),
        (
            Kind::AutoTheta,
            [20.001361862606124, 20.15240807987595, 21.203787753755396],
        ),
        (
            Kind::FourierArima,
            [20.028599207146442, 20.426040363503642, 20.87517985499386],
        ),
    ];

    #[test]
    fn test_golden_values_are_unchanged() -> Result<()> {
        let schedule = GymSchedule::default();
        let history = minutely(16, weekly);
        let origin = local(15, 12, 0);
        let (grid, seen) = data_for(&history, origin);
        let data = Observed::new(&grid, &seen, &schedule, origin);
        let profile = WeeklyProfile::fit(&data);
        let mut changed = Vec::new();
        for (kind, expected) in GOLDEN {
            let values = forecast(kind, &profile, &data)?;
            let got = [values[0], values[7], values[39]];
            if got.iter().zip(expected).any(|(g, e)| (g - e).abs() > 1e-6) {
                changed.push(format!("(Kind::{kind:?}, {got:?}),"));
            }
        }
        anyhow::ensure!(changed.is_empty(), "changed:\n{}", changed.join("\n"));
        Ok(())
    }
}
