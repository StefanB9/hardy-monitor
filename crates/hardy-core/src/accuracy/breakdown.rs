//! Breakdowns of scored forecasts: in total, per gym-local day, and over a
//! trailing window of days.

use std::collections::{BTreeMap, VecDeque};

use chrono::{Datelike, NaiveDate, TimeDelta, Timelike};
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

/// Error sums per gym-local weekday (0 = Monday, seven rows) and hour of
/// the target. On the heap: the grid is too large for the stack.
pub fn by_weekday_hour(rows: &[ScoredForecast], tz: Tz) -> Vec<[ErrorSums; 24]> {
    let mut grid = vec![[ErrorSums::default(); 24]; 7];
    for row in rows {
        let local = row.target.with_timezone(&tz);
        let day = local.weekday().num_days_from_monday() as usize;
        let hour = local.hour() as usize;
        grid[day][hour].add(&row.pair);
    }
    grid
}

/// Error sums per occupancy level of the reading, in bands `width` points
/// wide (keyed by the band's lower edge), lowest first; a full gym counts
/// in the top band.
pub fn by_level(rows: &[ScoredForecast], width: f64) -> Vec<(f64, ErrorSums)> {
    #[allow(clippy::cast_possible_truncation)]
    let top = (100.0 / width).ceil() as i64 - 1;
    let mut levels: BTreeMap<i64, ErrorSums> = BTreeMap::new();
    for row in rows {
        #[allow(clippy::cast_possible_truncation)]
        let band = ((row.pair.actual / width).floor() as i64).min(top);
        levels.entry(band).or_default().add(&row.pair);
    }
    #[allow(clippy::cast_precision_loss)]
    levels
        .into_iter()
        .map(|(band, sums)| (band as f64 * width, sums))
        .collect()
}

/// How many forecasts missed by how much (forecast − reading), in bins
/// `bin` points wide from −`limit` to `limit` (keyed by the bin's lower
/// edge); larger errors count in the outermost bins.
pub fn error_histogram(rows: &[ScoredForecast], bin: f64, limit: f64) -> Vec<(f64, i64)> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let bins = ((2.0 * limit / bin).ceil() as usize).max(1);
    let mut counts = vec![0_i64; bins];
    for row in rows {
        let error = row.pair.forecast - row.pair.actual;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let i = ((error + limit) / bin)
            .floor()
            .clamp(0.0, (bins - 1) as f64) as usize;
        counts[i] += 1;
    }
    #[allow(clippy::cast_precision_loss)]
    counts
        .into_iter()
        .enumerate()
        .map(|(i, n)| (-limit + i as f64 * bin, n))
        .collect()
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

    #[test]
    fn test_by_weekday_hour_uses_gym_local_time() -> Result<()> {
        // Monday 17 June 08:30 UTC is 10:30 in Berlin; Sunday 16 June 21:00
        // UTC is 23:00.
        let rows = [
            scored(utc(17, 8)? + chrono::TimeDelta::minutes(30), 10.0, 14.0),
            scored(utc(16, 21)?, 10.0, 11.0),
        ];
        let grid = by_weekday_hour(&rows, Berlin);
        assert_eq!(grid[0][10].count, 1);
        assert_relative_eq!(grid[0][10].abs, 4.0);
        assert_eq!(grid[6][23].count, 1);
        assert_eq!(grid.iter().flatten().map(|s| s.count).sum::<i64>(), 2);
        Ok(())
    }

    #[test]
    fn test_by_level_groups_by_the_reading() -> Result<()> {
        let rows = [
            scored(utc(17, 8)?, 5.0, 9.0),
            scored(utc(17, 9)?, 9.9, 9.0),
            scored(utc(17, 10)?, 35.0, 30.0),
            scored(utc(17, 11)?, 100.0, 90.0),
        ];
        let levels = by_level(&rows, 10.0);
        let summary: Vec<(f64, i64)> = levels.iter().map(|(l, s)| (*l, s.count)).collect();
        assert_eq!(summary, vec![(0.0, 2), (30.0, 1), (90.0, 1)]);
        Ok(())
    }

    #[test]
    fn test_error_histogram_bins_and_clamps() -> Result<()> {
        let rows = [
            scored(utc(17, 8)?, 30.0, 31.0),
            scored(utc(17, 9)?, 30.0, 29.0),
            scored(utc(17, 10)?, 30.0, 30.0),
            // Far beyond the range: counted in the outermost bins.
            scored(utc(17, 11)?, 30.0, 80.0),
            scored(utc(17, 12)?, 30.0, 0.0),
        ];
        let bins = error_histogram(&rows, 2.0, 4.0);
        assert_eq!(bins, vec![(-4.0, 1), (-2.0, 1), (0.0, 2), (2.0, 1)]);
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
            prop_assert_eq!(
                by_weekday_hour(&rows, Berlin).iter().flatten().map(|s| s.count).sum::<i64>(),
                n
            );
            prop_assert_eq!(by_level(&rows, 10.0).iter().map(|(_, s)| s.count).sum::<i64>(), n);
            prop_assert_eq!(error_histogram(&rows, 2.0, 20.0).iter().map(|(_, c)| c).sum::<i64>(), n);
            let whole = trailing(&days, 10_000);
            prop_assert_eq!(whole.last().map_or(0, |(_, s)| s.count), n);
        }
    }
}
