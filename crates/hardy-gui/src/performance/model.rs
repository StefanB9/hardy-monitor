//! One model's detail: its nightly backtest error, how often it was
//! selected, and its live accuracy by day and by lead time.

use chrono::{DateTime, NaiveDate, Utc};
use hardy_core::{
    accuracy::{ErrorSums, Measures},
    db::{BenchmarkNight, ModelErrors},
};
use hardy_ml::{benchmark::Bucket, suite::ModelId};

/// One model's detail.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDetail {
    pub id: ModelId,
    /// Per night: backtest error per bucket, in [`Bucket::ALL`] order.
    pub backtest: Vec<(DateTime<Utc>, [Option<f64>; 4])>,
    /// Nights it was selected, per horizon bucket (next hour, 1–3 h,
    /// 3–24 h).
    pub selected_nights: [usize; 3],
    /// Nights in the period.
    pub nights: usize,
    /// Live error per gym-local day.
    pub live_daily: Vec<(NaiveDate, f64)>,
    /// Live error per step (1 = 15 min ahead).
    pub lead_time: Vec<(u32, f64)>,
    /// Live accuracy over the period, every lead time.
    pub live: Option<Measures>,
}

/// The detail of model `id`.
pub fn detail(id: ModelId, nights: &[BenchmarkNight], errors: &ModelErrors) -> ModelDetail {
    let key = id.key();
    let backtest = nights
        .iter()
        .filter(|n| n.scores.iter().any(|s| s.model == key))
        .map(|n| {
            let mae = Bucket::ALL.map(|bucket| {
                n.scores
                    .iter()
                    .find(|s| s.model == key && s.bucket == bucket.code())
                    .map(|s| s.mae)
            });
            (n.finished_at, mae)
        })
        .collect();
    let selected_nights = Bucket::HORIZONS.map(|bucket| {
        nights
            .iter()
            .filter(|n| {
                n.selection
                    .iter()
                    .any(|(code, model)| *code == bucket.code() && model == key)
            })
            .count()
    });
    let mut total = ErrorSums::default();
    let lead_time = errors
        .by_step
        .iter()
        .filter(|(model, _, _)| model == key)
        .filter_map(|(_, step, sums)| {
            total.merge(sums);
            Some((*step, sums.measures()?.mae))
        })
        .collect();
    let live_daily = errors
        .by_day
        .iter()
        .filter(|(model, _, _)| model == key)
        .filter_map(|(_, day, sums)| Some((*day, sums.measures()?.mae)))
        .collect();
    ModelDetail {
        id,
        backtest,
        selected_nights,
        nights: nights.len(),
        live_daily,
        lead_time,
        live: total.measures(),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::{NaiveDate, TimeZone, Utc};
    use hardy_core::{
        accuracy::{ErrorSums, Pair},
        db::BenchmarkScore,
    };

    use super::*;

    fn sums(error: f64) -> ErrorSums {
        let mut sums = ErrorSums::default();
        sums.add(&Pair {
            actual: 30.0,
            forecast: 30.0 + error,
            band: None,
            baseline: None,
            last_week: None,
        });
        sums
    }

    fn night(day: u32, mae: f64, selected: &str) -> BenchmarkNight {
        BenchmarkNight {
            run_id: i64::from(day),
            finished_at: Utc
                .with_ymd_and_hms(2024, 6, day, 22, 0, 0)
                .single()
                .unwrap_or_default(),
            selection: vec![(1, selected.to_string()), (2, "theta".to_string())],
            scores: [0_i16, 1]
                .map(|bucket| BenchmarkScore {
                    model: "naive".to_string(),
                    bucket,
                    mae: mae + f64::from(bucket),
                    rmse: mae,
                    mase: None,
                    coverage: None,
                    pinball: None,
                    errors: 10,
                    interval: None,
                })
                .to_vec(),
        }
    }

    #[test]
    fn test_detail_collects_the_models_nights_and_live_errors() -> Result<()> {
        let day = |d| NaiveDate::from_ymd_opt(2024, 6, d).context("date");
        let errors = ModelErrors {
            by_step: vec![
                ("naive".to_string(), 1, sums(2.0)),
                ("naive".to_string(), 2, sums(-4.0)),
                ("theta".to_string(), 1, sums(9.0)),
            ],
            by_day: vec![
                ("naive".to_string(), day(16)?, sums(2.0)),
                ("naive".to_string(), day(17)?, sums(4.0)),
            ],
        };
        let nights = [night(15, 5.0, "naive"), night(16, 6.0, "theta")];

        let detail = detail(ModelId::Naive, &nights, &errors);

        assert_eq!(detail.backtest.len(), 2);
        assert_eq!(detail.backtest[1].1[0], Some(6.0));
        assert_eq!(detail.backtest[1].1[1], Some(7.0));
        assert_eq!(detail.backtest[1].1[2], None);
        assert_eq!(detail.selected_nights, [1, 0, 0]);
        assert_eq!(detail.nights, 2);
        assert_eq!(detail.live_daily, vec![(day(16)?, 2.0), (day(17)?, 4.0)]);
        assert_eq!(detail.lead_time, vec![(1, 2.0), (2, 4.0)]);
        let live = detail.live.context("live")?;
        assert_relative_eq!(live.mae, 3.0);
        assert_relative_eq!(live.bias, -1.0);
        Ok(())
    }

    #[test]
    fn test_detail_of_a_model_without_data_is_empty() {
        let detail = detail(ModelId::Ridge, &[], &ModelErrors::default());
        assert!(detail.backtest.is_empty() && detail.lead_time.is_empty());
        assert_eq!((detail.nights, detail.live), (0, None));
    }
}
