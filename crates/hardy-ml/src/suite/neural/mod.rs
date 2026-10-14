//! Neural networks (burn, pure-Rust CPU backend): each learns the deviation
//! from the weekly profile over the next [`HORIZON_STEPS`] slots from a
//! window of recent slots, in one multi-output head. A fit is reused for a
//! week of backtest days; the inputs roll forward with the origin.

mod nets;
mod train;
mod windows;

use std::{fmt, marker::PhantomData, sync::Mutex};

use burn::{
    prelude::*,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Bytes,
};
use chrono::TimeDelta;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use self::{
    train::{Net, Training, tensors},
    windows::{DEVIATION_SCALE, forecast_window, training_windows},
};
use crate::{
    error::MlError,
    grid::{HORIZON_STEPS, SLOT, is_open_slot},
    suite::{Forecaster, Observed, profiles::WeeklyProfile},
};

/// Backtest days one fit serves.
const REFIT_DAYS: usize = 7;

/// A network design.
pub trait Architecture {
    type Net: Net;
    /// Input slots before the origin.
    const INPUT: usize;
    /// Days of history needed to fit.
    const MIN_DAYS: i64;

    /// A network with fresh weights on `device`.
    fn build(device: &Device) -> Self::Net;
}

/// A neural network forecaster of design `A`.
pub struct Neural<A: Architecture> {
    fitted: Option<Box<Fitted<A::Net>>>,
    design: PhantomData<fn() -> A>,
}

struct Fitted<N> {
    profile: WeeklyProfile,
    /// Locked only because burn's modules are not `Sync`; forecasts run
    /// one at a time per model.
    net: Mutex<N>,
}

impl<A: Architecture> Default for Neural<A> {
    fn default() -> Self {
        Self {
            fitted: None,
            design: PhantomData,
        }
    }
}

impl<A: Architecture> fmt::Debug for Neural<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Neural")
            .field("fitted", &self.fitted.is_some())
            .finish()
    }
}

impl<A: Architecture> Forecaster for Neural<A> {
    fn min_history(&self) -> TimeDelta {
        TimeDelta::days(A::MIN_DAYS)
    }

    fn refit_days(&self) -> usize {
        REFIT_DAYS
    }

    fn fit(&mut self, data: &Observed<'_>) -> Result<(), MlError> {
        let profile = WeeklyProfile::fit(data);
        // Inputs fully observed where the history allows, but no later than
        // the minimum history.
        let lead = (SLOT * i32::try_from(A::INPUT).unwrap_or(i32::MAX))
            .min(TimeDelta::days(A::MIN_DAYS - 1));
        let (train, validation) = training_windows(data, &profile, A::INPUT, lead);
        let net = train::train(A::build, &train, &validation, &Training::default())?;
        self.fitted = Some(Box::new(Fitted {
            profile,
            net: Mutex::new(net),
        }));
        Ok(())
    }

    fn predict(&self, data: &Observed<'_>) -> Result<Vec<f64>, MlError> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| MlError::Fit("model is not fitted".to_string()))?;
        let window = forecast_window(data, &fitted.profile, A::INPUT);
        let (series, covariates, _, _) = tensors(&window, &Device::flex());
        let net = fitted
            .net
            .lock()
            .map_err(|_| MlError::Fit("network lock poisoned".to_string()))?;
        let output = net
            .forward(series, covariates)
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| MlError::Fit(format!("network output: {e:?}")))?;
        let grid = data.grid();
        Ok(output
            .iter()
            .take(HORIZON_STEPS)
            .enumerate()
            .map(|(i, &deviation)| {
                let end = grid.step_end(i + 1);
                if is_open_slot(data.schedule(), end) {
                    fitted.profile.at(end) + f64::from(deviation) * DEVIATION_SCALE
                } else {
                    0.0
                }
            })
            .collect())
    }
}

/// How a fitted network is stored: the profile and burnpack weights.
#[derive(Serialize, Deserialize)]
struct Stored {
    profile: WeeklyProfile,
    weights: Vec<u8>,
}

impl<A: Architecture> Serialize for Neural<A> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let stored = self
            .fitted
            .as_ref()
            .map(|fitted| {
                let net = fitted
                    .net
                    .lock()
                    .map_err(|_| serde::ser::Error::custom("network lock poisoned"))?;
                let mut store = BurnpackStore::from_bytes(None);
                net.save_into(&mut store)
                    .map_err(|e| serde::ser::Error::custom(format!("{e:?}")))?;
                let weights = store
                    .get_bytes()
                    .map_err(|e| serde::ser::Error::custom(format!("{e:?}")))?
                    .to_vec();
                Ok(Stored {
                    profile: fitted.profile.clone(),
                    weights,
                })
            })
            .transpose()?;
        stored.serialize(serializer)
    }
}

impl<'de, A: Architecture> Deserialize<'de> for Neural<A> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let Some(stored) = Option::<Stored>::deserialize(deserializer)? else {
            return Ok(Self::default());
        };
        let mut net = train::seeded(A::build, &Device::flex());
        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(stored.weights)));
        net.load_from(&mut store)
            .map_err(|e| serde::de::Error::custom(format!("{e:?}")))?;
        Ok(Self {
            fitted: Some(Box::new(Fitted {
                profile: stored.profile,
                net: Mutex::new(net),
            })),
            design: PhantomData,
        })
    }
}

pub use nets::{Gru, Lstm, Mlp, NBeats, NHits, PatchTst, Tcn};

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use chrono::{DateTime, Datelike, Timelike, Utc};
    use hardy_core::GymSchedule;

    use super::*;
    use crate::{
        grid::{Grid, slot_end_of},
        history::History,
        suite::{decode, encode, forecast},
        test_support::{local, minutely},
    };

    /// A weekly curve with a daily wave, shifted by a level that holds for
    /// the whole day.
    fn value(t: DateTime<Utc>) -> f64 {
        let clock = (slot_end_of(t) - TimeDelta::minutes(1))
            .with_timezone(&GymSchedule::default().timezone());
        let day = (t - local(0, 0, 0)).num_days();
        #[allow(clippy::cast_precision_loss)]
        let level = ((day * 37) % 5) as f64 * 6.0;
        20.0 + 2.0 * f64::from(clock.weekday().num_days_from_monday())
            + 15.0
                * (f64::from(clock.hour()) / 24.0 * std::f64::consts::TAU)
                    .sin()
                    .abs()
            + level
    }

    fn setup(days: i64) -> (History, Grid, GymSchedule) {
        let history = minutely(days + 1, value);
        let schedule = GymSchedule::default();
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(days + 1, 0, 0));
        (history, grid, schedule)
    }

    /// Fits on `days` days (a Sunday) and forecasts from the next day, a
    /// Monday, at noon, when its level has been visible since opening; then
    /// checks the forecast and that the stored model forecasts the same.
    fn check<A: Architecture>(days: i64) -> Result<()> {
        let (history, grid, schedule) = setup(days);
        let cut = local(days, 0, 0);
        let fit_history = history.until(cut);
        let mut model = Neural::<A>::default();
        crate::suite::fit(
            &mut model,
            &Observed::new(&grid, &fit_history, &schedule, cut),
        )?;
        let origin = local(days, 12, 0);
        let seen = history.until(origin);
        let data = Observed::new(&grid, &seen, &schedule, origin);
        let values = forecast(&model, &data)?;
        assert_eq!(values.len(), HORIZON_STEPS);
        assert_eq!(forecast(&model, &data)?, values);
        let (mut error, mut open) = (0.0, 0.0);
        for (k, v) in values.iter().enumerate() {
            let end = origin + SLOT * i32::try_from(k + 1)?;
            if !is_open_slot(&schedule, end) {
                assert_eq!(*v, 0.0);
            } else if end <= local(days + 1, 0, 0) {
                error += (v - value(end - TimeDelta::minutes(1))).abs();
                open += 1.0;
            }
        }
        // The weekly profile alone is 6 off on day 14 (its Monday mixes the
        // levels 0 and 24, day 14 is at 18) and 3.75 on day 28; the nets
        // carry today's level.
        let mae = error / open;
        assert!(mae < 3.0, "MAE {mae}");
        let restored: Neural<A> = decode(&encode(&model)?)?;
        assert_eq!(forecast(&restored, &data)?, values);
        Ok(())
    }

    #[test]
    fn test_mlp_forecasts_and_round_trips() -> Result<()> {
        check::<nets::MlpDesign>(14)
    }

    #[test]
    fn test_tcn_forecasts_and_round_trips() -> Result<()> {
        check::<nets::TcnDesign>(28)
    }

    #[test]
    fn test_gru_forecasts_and_round_trips() -> Result<()> {
        check::<nets::GruDesign>(14)
    }

    #[test]
    fn test_lstm_forecasts_and_round_trips() -> Result<()> {
        check::<nets::LstmDesign>(14)
    }

    #[test]
    fn test_n_beats_forecasts_and_round_trips() -> Result<()> {
        check::<nets::NBeatsDesign>(28)
    }

    #[test]
    fn test_n_hits_forecasts_and_round_trips() -> Result<()> {
        check::<nets::NHitsDesign>(28)
    }

    #[test]
    fn test_patch_tst_forecasts_and_round_trips() -> Result<()> {
        check::<nets::PatchTstDesign>(28)
    }

    #[test]
    fn test_unfitted_and_minimum_history() -> Result<()> {
        let history = minutely(3, |_| 30.0);
        let schedule = GymSchedule::default();
        let grid = Grid::build(&history, &schedule, local(0, 0, 0), local(3, 0, 0));
        let data = Observed::new(&grid, &history, &schedule, local(3, 0, 0));
        assert!(Mlp::default().predict(&data).is_err());
        assert_eq!(Mlp::default().min_history(), TimeDelta::days(1));
        assert_eq!(Gru::default().min_history(), TimeDelta::days(2));
        assert_eq!(PatchTst::default().min_history(), TimeDelta::days(8));
        assert_eq!(Tcn::default().refit_days(), REFIT_DAYS);
        // Too little history for a week-long window.
        assert!(crate::suite::fit(&mut NBeats::default(), &data).is_err());
        // An unfitted model stores and restores as unfitted.
        let restored: Mlp = decode(&encode(&Mlp::default())?)?;
        assert!(restored.predict(&data).is_err());
        Ok(())
    }
}
