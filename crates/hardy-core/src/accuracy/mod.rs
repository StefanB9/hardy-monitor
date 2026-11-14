//! Summaries of logged forecast accuracy for display.

use std::collections::BTreeMap;

use chrono::NaiveDate;

use crate::db::HorizonAccuracy;

/// Errors of one group of scored forecasts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorPair {
    pub scored: i64,
    /// Mean absolute error of what was forecast.
    pub forecast_mae: f64,
    /// Mean absolute error of plain averages on the same targets.
    pub baseline_mae: f64,
}

/// Accuracy over a period: overall, per day and per horizon.
#[derive(Debug, Clone, PartialEq)]
pub struct AccuracySummary {
    pub overall: ErrorPair,
    /// Forecasts that came from a model (the rest fell back to averages).
    pub from_model: i64,
    pub by_day: Vec<(NaiveDate, ErrorPair)>,
    pub by_horizon: Vec<(u32, ErrorPair)>,
}

/// Combines per-day, per-horizon rows, weighting by scored forecasts.
/// `None` when nothing was scored.
pub fn summarize(rows: &[HorizonAccuracy]) -> Option<AccuracySummary> {
    let mut overall = Sums::default();
    let mut by_day: BTreeMap<NaiveDate, Sums> = BTreeMap::new();
    let mut by_horizon: BTreeMap<u32, Sums> = BTreeMap::new();
    let mut from_model = 0;
    for row in rows.iter().filter(|r| r.scored > 0) {
        overall.add(row);
        by_day.entry(row.day).or_default().add(row);
        by_horizon.entry(row.horizon_hours).or_default().add(row);
        from_model += row.from_model;
    }
    Some(AccuracySummary {
        overall: overall.pair()?,
        from_model,
        by_day: by_day
            .into_iter()
            .filter_map(|(day, sums)| Some((day, sums.pair()?)))
            .collect(),
        by_horizon: by_horizon
            .into_iter()
            .filter_map(|(h, sums)| Some((h, sums.pair()?)))
            .collect(),
    })
}

/// Error sums weighted by the number of scored forecasts.
#[derive(Debug, Default)]
struct Sums {
    scored: i64,
    forecast: f64,
    baseline: f64,
}

impl Sums {
    fn add(&mut self, row: &HorizonAccuracy) {
        #[allow(clippy::cast_precision_loss)]
        let weight = row.scored as f64;
        self.scored += row.scored;
        self.forecast += row.forecast_mae * weight;
        self.baseline += row.baseline_mae * weight;
    }

    fn pair(&self) -> Option<ErrorPair> {
        #[allow(clippy::cast_precision_loss)]
        let n = (self.scored > 0).then_some(self.scored as f64)?;
        Some(ErrorPair {
            scored: self.scored,
            forecast_mae: self.forecast / n,
            baseline_mae: self.baseline / n,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    fn row(day: u32, horizon: u32, scored: i64, forecast: f64, baseline: f64) -> HorizonAccuracy {
        HorizonAccuracy {
            day: NaiveDate::from_ymd_opt(2024, 6, day).unwrap_or_default(),
            horizon_hours: horizon,
            scored,
            from_model: scored / 2,
            forecast_mae: forecast,
            baseline_mae: baseline,
        }
    }

    #[test]
    fn test_summarize_empty_is_none() {
        assert_eq!(summarize(&[]), None);
        assert_eq!(summarize(&[row(17, 1, 0, 1.0, 1.0)]), None);
    }

    #[test]
    fn test_summarize_weights_by_scored() -> Result<()> {
        let s = summarize(&[
            row(17, 1, 3, 2.0, 4.0),
            row(17, 2, 1, 6.0, 4.0),
            row(18, 1, 4, 1.0, 2.0),
        ])
        .context("summary")?;
        assert_eq!(s.overall.scored, 8);
        assert_relative_eq!(s.overall.forecast_mae, (6.0 + 6.0 + 4.0) / 8.0);
        assert_relative_eq!(s.overall.baseline_mae, (12.0 + 4.0 + 8.0) / 8.0);
        // from_model = scored / 2 per row: 1, 0 and 2.
        assert_eq!(s.from_model, 3);

        assert_eq!(s.by_day.len(), 2);
        let (day, first) = s.by_day[0];
        assert_eq!(day, NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?);
        assert_relative_eq!(first.forecast_mae, 3.0);

        assert_eq!(
            s.by_horizon.iter().map(|(h, _)| *h).collect::<Vec<_>>(),
            [1, 2]
        );
        assert_relative_eq!(s.by_horizon[0].1.forecast_mae, 10.0 / 7.0);
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn summarize_overall_lies_within_row_range(
            rows in proptest::collection::vec(
                (1u32..28, 1u32..7, 1i64..50, 0.0f64..50.0, 0.0f64..50.0),
                1..40,
            ),
        ) {
            let rows: Vec<_> = rows
                .into_iter()
                .map(|(d, h, n, f, b)| row(d, h, n, f, b))
                .collect();
            let s = summarize(&rows);
            prop_assert!(s.is_some());
            if let Some(s) = s {
                prop_assert_eq!(s.overall.scored, rows.iter().map(|r| r.scored).sum::<i64>());
                let lo = rows.iter().map(|r| r.forecast_mae).fold(f64::MAX, f64::min);
                let hi = rows.iter().map(|r| r.forecast_mae).fold(f64::MIN, f64::max);
                prop_assert!(s.overall.forecast_mae >= lo - 1e-9 && s.overall.forecast_mae <= hi + 1e-9);
                prop_assert_eq!(
                    s.by_day.iter().map(|(_, e)| e.scored).sum::<i64>(),
                    s.overall.scored
                );
                prop_assert!(s.by_day.windows(2).all(|w| w[0].0 < w[1].0));
                prop_assert!(s.by_horizon.windows(2).all(|w| w[0].0 < w[1].0));
            }
        }
    }
}
