//! Breakdowns of scored forecasts: in total, per gym-local day, and over a
//! trailing window of days.

use std::collections::{BTreeMap, VecDeque};

use chrono::{NaiveDate, TimeDelta};
use chrono_tz::Tz;

use super::ErrorSums;
use crate::db::ScoredForecast;

/// Error sums of every forecast.
pub fn total(rows: &[ScoredForecast]) -> ErrorSums {
    let mut sums = ErrorSums::default();
    for row in rows {
        sums.add(&row.pair);
    }
    sums
}

/// Error sums per gym-local target day, oldest first.
pub fn by_day(rows: &[ScoredForecast], tz: Tz) -> Vec<(NaiveDate, ErrorSums)> {
    let mut days: BTreeMap<NaiveDate, ErrorSums> = BTreeMap::new();
    for row in rows {
        days.entry(row.target.with_timezone(&tz).date_naive())
            .or_default()
            .add(&row.pair);
    }
    days.into_iter().collect()
}

/// For each day, the sums of the `window` calendar days ending with it
/// (days without forecasts count towards the window too).
pub fn trailing(days: &[(NaiveDate, ErrorSums)], window: i64) -> Vec<(NaiveDate, ErrorSums)> {
    let mut rolled = Vec::with_capacity(days.len());
    let mut open: VecDeque<&(NaiveDate, ErrorSums)> = VecDeque::new();
    for entry in days {
        open.push_back(entry);
        let first = entry.0 - TimeDelta::days(window - 1);
        while open.front().is_some_and(|(d, _)| *d < first) {
            open.pop_front();
        }
        let mut sums = ErrorSums::default();
        for (_, s) in &open {
            sums.merge(s);
        }
        rolled.push((entry.0, sums));
    }
    rolled
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::{DateTime, TimeZone, Utc};
    use chrono_tz::Europe::Berlin;
    use proptest::prelude::*;

    use super::*;
    use crate::accuracy::Pair;

    fn scored(target: DateTime<Utc>, actual: f64, forecast: f64) -> ScoredForecast {
        ScoredForecast {
            target,
            made_at: target,
            model: "naive".to_string(),
            pair: Pair {
                actual,
                forecast,
                band: None,
                baseline: None,
                last_week: None,
            },
        }
    }

    fn utc(d: u32, h: u32) -> Result<DateTime<Utc>> {
        Utc.with_ymd_and_hms(2024, 6, d, h, 0, 0)
            .single()
            .context("time")
    }

    fn day(d: u32) -> Result<NaiveDate> {
        NaiveDate::from_ymd_opt(2024, 6, d).context("date")
    }

    #[test]
    fn test_by_day_groups_by_gym_local_day() -> Result<()> {
        // 22:30 UTC on the 17th is 00:30 on the 18th in Berlin.
        let late = utc(17, 22)? + chrono::TimeDelta::minutes(30);
        let rows = [
            scored(utc(17, 10)?, 10.0, 12.0),
            scored(utc(17, 12)?, 10.0, 14.0),
            scored(late, 10.0, 20.0),
        ];
        let days = by_day(&rows, Berlin);
        assert_eq!(days.len(), 2);
        assert_eq!(days[0].0, day(17)?);
        assert_eq!(days[0].1.count, 2);
        assert_relative_eq!(days[0].1.abs, 6.0);
        assert_eq!(days[1].0, day(18)?);
        assert_eq!(total(&rows).count, 3);
        Ok(())
    }

    #[test]
    fn test_trailing_adds_up_the_window_by_date() -> Result<()> {
        let one = |abs: f64| ErrorSums {
            count: 1,
            abs,
            ..ErrorSums::default()
        };
        // The 19th is missing: the window counts days, not entries.
        let days = [
            (day(17)?, one(1.0)),
            (day(18)?, one(2.0)),
            (day(20)?, one(4.0)),
        ];
        let rolled = trailing(&days, 3);
        let abs: Vec<f64> = rolled.iter().map(|(_, s)| s.abs).collect();
        assert_eq!(
            rolled.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
            vec![day(17)?, day(18)?, day(20)?]
        );
        assert_eq!(abs, vec![1.0, 3.0, 6.0]);
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn by_day_keeps_every_forecast_once(
            rows in proptest::collection::vec((0i64..30 * 24, 0.0f64..100.0, 0.0f64..100.0), 0..80),
        ) {
            let start = Utc.with_ymd_and_hms(2024, 3, 20, 0, 0, 0).single().unwrap_or_default();
            let rows: Vec<ScoredForecast> = rows
                .into_iter()
                .map(|(h, a, f)| scored(start + chrono::TimeDelta::hours(h), a, f))
                .collect();
            let days = by_day(&rows, Berlin);
            let n = i64::try_from(rows.len()).unwrap_or(i64::MAX);
            prop_assert_eq!(days.iter().map(|(_, s)| s.count).sum::<i64>(), n);
            prop_assert!(days.windows(2).all(|w| w[0].0 < w[1].0));
            let rolled = trailing(&days, 7);
            prop_assert_eq!(rolled.len(), days.len());
            for ((_, rolled), (_, single)) in rolled.iter().zip(&days) {
                prop_assert!(rolled.count >= single.count);
            }
            let whole = trailing(&days, 10_000);
            prop_assert_eq!(whole.last().map_or(0, |(_, s)| s.count), n);
        }
    }
}
