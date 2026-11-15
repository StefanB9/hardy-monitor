//! Misses & days: the biggest misses, the worst days, one day's forecast
//! against the readings, and whether the quiet hours picked turned out
//! quiet.

use std::collections::BTreeMap;

use chrono::{DateTime, NaiveDate, Timelike, Utc};
use hardy_core::{
    Tz,
    accuracy::{Decision, DecisionSums, HOUR_SLOTS, average_picks, judge},
    db::{LoggedOutcome, ScoredForecast},
};

use super::DayPoint;

/// How well the quiet hours picked turned out.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecisionQuality {
    /// The default forecast's quietest hour in the next 6 h, every logged
    /// hour.
    pub forecast: DecisionSums,
    /// Plain averages' quietest hour of each day.
    pub averages: DecisionSums,
    pub daily: Vec<DecisionDay>,
}

/// Mean points lost on one gym-local day.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionDay {
    pub day: NaiveDate,
    /// By the forecast's picks made that day.
    pub forecast: Option<f64>,
    /// By the averages' pick for the day.
    pub averages: Option<f64>,
}

/// One slot of a day's chart, by gym-local hour of the day.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DaySlot {
    pub hour: f64,
    pub actual: f64,
    pub forecast: f64,
    pub baseline: Option<f64>,
    pub band: Option<(f64, f64)>,
}

/// The `n` forecasts furthest off, furthest first (earlier first on ties).
pub fn biggest_misses(scored: &[ScoredForecast], n: usize) -> Vec<ScoredForecast> {
    let mut misses: Vec<&ScoredForecast> = scored.iter().collect();
    misses.sort_by(|a, b| {
        let error = |s: &ScoredForecast| (s.pair.forecast - s.pair.actual).abs();
        error(b)
            .total_cmp(&error(a))
            .then_with(|| a.target.cmp(&b.target))
    });
    misses.into_iter().take(n).cloned().collect()
}

/// The `n` days with the largest error, worst first.
pub fn worst_days(daily: &[DayPoint], n: usize) -> Vec<DayPoint> {
    let mut days = daily.to_vec();
    days.sort_by(|a, b| b.mae.total_cmp(&a.mae).then_with(|| a.day.cmp(&b.day)));
    days.truncate(n);
    days
}

/// The forecasts of one gym-local day, by hour of the day.
pub fn day_slots(scored: &[ScoredForecast], day: NaiveDate, tz: Tz) -> Vec<DaySlot> {
    scored
        .iter()
        .filter_map(|s| {
            let local = s.target.with_timezone(&tz);
            (local.date_naive() == day).then(|| DaySlot {
                hour: f64::from(local.hour()) + f64::from(local.minute()) / 60.0,
                actual: s.pair.actual,
                forecast: s.pair.forecast,
                baseline: s.pair.baseline,
                band: s.pair.band,
            })
        })
        .collect()
}

/// The forecast's quietest hour in each logged forecast, judged.
pub fn forecast_picks(outcomes: &[LoggedOutcome]) -> Vec<(DateTime<Utc>, Decision)> {
    outcomes
        .iter()
        .filter_map(|o| Some((o.made_at, judge(&o.forecast, &o.actual, HOUR_SLOTS)?)))
        .collect()
}

/// Tallies both kinds of picks, overall and per day.
pub fn decision_quality(
    outcomes: &[LoggedOutcome],
    scored: &[ScoredForecast],
    tz: Tz,
) -> DecisionQuality {
    let mut quality = DecisionQuality::default();
    let mut days: BTreeMap<NaiveDate, (DecisionSums, DecisionSums)> = BTreeMap::new();
    for (made_at, decision) in forecast_picks(outcomes) {
        quality.forecast.add(&decision);
        let day = made_at.with_timezone(&tz).date_naive();
        days.entry(day).or_default().0.add(&decision);
    }
    for (day, decision) in average_picks(scored, tz) {
        quality.averages.add(&decision);
        days.entry(day).or_default().1.add(&decision);
    }
    quality.daily = days
        .into_iter()
        .map(|(day, (forecast, averages))| DecisionDay {
            day,
            forecast: forecast.mean_lost(),
            averages: averages.mean_lost(),
        })
        .collect();
    quality
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::{TimeDelta, TimeZone};
    use hardy_core::accuracy::Pair;

    use super::*;

    fn tz() -> Tz {
        "Europe/Berlin".parse().unwrap_or(Tz::UTC)
    }

    fn utc(d: u32, h: u32, m: u32) -> Result<DateTime<Utc>> {
        Utc.with_ymd_and_hms(2024, 6, d, h, m, 0)
            .single()
            .context("time")
    }

    fn scored(target: DateTime<Utc>, actual: f64, forecast: f64) -> ScoredForecast {
        ScoredForecast {
            target,
            made_at: target - TimeDelta::hours(1),
            model: "naive".to_string(),
            pair: Pair {
                actual,
                forecast,
                band: None,
                baseline: Some(actual),
                last_week: None,
            },
        }
    }

    fn day_point(day: u32, mae: f64) -> Result<DayPoint> {
        Ok(DayPoint {
            day: NaiveDate::from_ymd_opt(2024, 6, day).context("date")?,
            mae,
            baseline: None,
            weekly: mae,
            bias: 0.0,
            coverage: None,
            width: None,
        })
    }

    #[test]
    fn test_biggest_misses_sort_by_size_either_way() -> Result<()> {
        let rows = vec![
            scored(utc(17, 8, 0)?, 30.0, 32.0),
            scored(utc(17, 9, 0)?, 30.0, 18.0),
            scored(utc(17, 10, 0)?, 30.0, 42.0),
            scored(utc(17, 11, 0)?, 30.0, 35.0),
        ];
        let misses = biggest_misses(&rows, 3);
        let targets: Vec<_> = misses.iter().map(|s| s.target).collect();
        assert_eq!(
            targets,
            vec![utc(17, 9, 0)?, utc(17, 10, 0)?, utc(17, 11, 0)?],
            "a tie goes to the earlier"
        );
        assert_eq!(biggest_misses(&[], 3), Vec::new());
        Ok(())
    }

    #[test]
    fn test_worst_days_come_first() -> Result<()> {
        let daily = vec![day_point(1, 3.0)?, day_point(2, 7.0)?, day_point(3, 5.0)?];
        let worst: Vec<f64> = worst_days(&daily, 2).iter().map(|d| d.mae).collect();
        assert_eq!(worst, vec![7.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_day_slots_keep_one_local_day() -> Result<()> {
        // 21:45 UTC on the 17th is 23:45 in Berlin; 22:15 is the 18th.
        let rows = vec![
            scored(utc(17, 8, 30)?, 30.0, 32.0),
            scored(utc(17, 21, 45)?, 10.0, 12.0),
            scored(utc(17, 22, 15)?, 10.0, 12.0),
        ];
        let day = NaiveDate::from_ymd_opt(2024, 6, 17).context("date")?;
        let slots = day_slots(&rows, day, tz());
        assert_eq!(slots.len(), 2);
        assert_relative_eq!(slots[0].hour, 10.5);
        assert_relative_eq!(slots[1].hour, 23.75);
        assert_relative_eq!(slots[0].forecast, 32.0);
        Ok(())
    }

    #[test]
    fn test_decision_quality_tallies_both_guides_per_day() -> Result<()> {
        // The forecast at 08:00 UTC picks its first hour, which was 8
        // points busier than the second.
        let forecast = (0..8)
            .map(|i| Some(if i < 4 { 10.0 } else { 20.0 }))
            .collect();
        let actual = (0..8)
            .map(|i| Some(if i < 4 { 28.0 } else { 20.0 }))
            .collect();
        let outcomes = vec![LoggedOutcome {
            made_at: utc(17, 8, 0)?,
            forecast,
            actual,
        }];
        // A day of readings whose averages are the readings: exact.
        let start = utc(17, 8, 0)?;
        let rows: Vec<_> = (0..4)
            .map(|i| scored(start + TimeDelta::minutes(15 * i), 20.0, 20.0))
            .collect();
        let quality = decision_quality(&outcomes, &rows, tz());
        assert_eq!(quality.forecast.count, 1);
        assert_relative_eq!(quality.forecast.mean_lost().context("lost")?, 8.0);
        assert_eq!(quality.averages.count, 1);
        assert_eq!(quality.averages.exact, 1);
        assert_eq!(quality.daily.len(), 1);
        assert_eq!(quality.daily[0].forecast, Some(8.0));
        assert_eq!(quality.daily[0].averages, Some(0.0));
        Ok(())
    }
}
