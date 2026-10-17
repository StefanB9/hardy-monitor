//! A day's estimates from recent weeks, with forecast points laid over.

use chrono::{DateTime, Datelike, NaiveDate, Timelike, Utc};

use super::{EstimateSource, Expected, QUARTER, SlotEstimate};
use crate::{
    analytics::MIN_DAYS,
    db::QuarterHourAverage,
    schedule::{GymSchedule, is_bavarian_holiday},
};

/// One estimate per quarter hour the gym is open on `day`, from `averages`
/// of the day's weekday (of Sunday on a holiday, which keeps weekend hours).
/// The range is one standard deviation between days either side of the
/// mean. Quarter hours seen on fewer than [`MIN_DAYS`] days
/// have no estimate.
pub fn recent_weeks_estimates(
    day: NaiveDate,
    schedule: &GymSchedule,
    averages: &[QuarterHourAverage],
) -> Vec<SlotEstimate> {
    let tz = schedule.timezone();
    let (weekday, source) = if is_bavarian_holiday(day) {
        (6, EstimateSource::RecentSundays)
    } else {
        #[allow(clippy::cast_possible_wrap)] // 0..7
        let weekday = day.weekday().num_days_from_monday() as i32;
        (weekday, EstimateSource::RecentWeeks)
    };
    let mut by_quarter = [None; 96];
    for a in averages
        .iter()
        .filter(|a| a.weekday == weekday && a.days >= MIN_DAYS)
    {
        if let Some(slot) = usize::try_from(a.quarter)
            .ok()
            .and_then(|q| by_quarter.get_mut(q))
        {
            *slot = Some(Expected {
                mean: a.avg_percentage,
                low: (a.avg_percentage - a.std_dev).max(0.0),
                high: (a.avg_percentage + a.std_dev).min(100.0),
            });
        }
    }

    let (open, close) = (schedule.opening_time_on(day), schedule.closing_time_on(day));
    let count = usize::try_from((close - open).num_minutes() / QUARTER.num_minutes()).unwrap_or(0);
    let mut estimates = Vec::with_capacity(count);
    let mut start = open;
    while start < close {
        let local = start.with_timezone(&tz);
        let quarter = (local.hour() * 4 + local.minute() / 15) as usize;
        estimates.push(SlotEstimate {
            start,
            expected: by_quarter.get(quarter).copied().flatten(),
            source,
        });
        start += QUARTER;
    }
    estimates
}

/// Replaces the estimates whose quarter hour starts at a forecast point's
/// time with that point.
pub fn with_forecast(
    estimates: &mut [SlotEstimate],
    points: impl IntoIterator<Item = (DateTime<Utc>, Expected)>,
) {
    // `estimates` are in time order, one per quarter hour.
    for (time, expected) in points {
        if let Ok(i) = estimates.binary_search_by_key(&time, |e| e.start)
            && let Some(slot) = estimates.get_mut(i)
        {
            slot.expected = Some(expected);
            slot.source = EstimateSource::Forecast;
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::TimeZone;

    use super::*;

    fn avg(weekday: i32, quarter: i32, mean: f64, std_dev: f64, days: i64) -> QuarterHourAverage {
        QuarterHourAverage {
            weekday,
            quarter,
            avg_percentage: mean,
            std_dev,
            days,
        }
    }

    fn date(y: i32, m: u32, d: u32) -> Result<NaiveDate> {
        NaiveDate::from_ymd_opt(y, m, d).context("valid date")
    }

    fn berlin(y: i32, m: u32, d: u32, h: u32, mi: u32) -> Result<DateTime<Utc>> {
        chrono_tz::Europe::Berlin
            .with_ymd_and_hms(y, m, d, h, mi, 0)
            .single()
            .map(|t| t.with_timezone(&Utc))
            .context("valid Berlin time")
    }

    /// Every quarter hour of `weekday` at `mean`, seen on four days.
    fn whole_weekday(weekday: i32, mean: f64) -> Vec<QuarterHourAverage> {
        (0..96).map(|q| avg(weekday, q, mean, 5.0, 4)).collect()
    }

    #[test]
    fn test_estimates_cover_the_opening_hours() -> Result<()> {
        // Monday, 06:00–23:00.
        let schedule = GymSchedule::default();
        let estimates =
            recent_weeks_estimates(date(2024, 6, 17)?, &schedule, &whole_weekday(0, 30.0));
        assert_eq!(estimates.len(), 17 * 4);
        assert_eq!(estimates[0].start, berlin(2024, 6, 17, 6, 0)?);
        assert_eq!(
            estimates.last().map(|e| e.start),
            Some(berlin(2024, 6, 17, 22, 45)?)
        );
        assert!(
            estimates
                .iter()
                .all(|e| e.source == EstimateSource::RecentWeeks)
        );
        Ok(())
    }

    #[test]
    fn test_estimates_use_the_quarter_hours_own_average() -> Result<()> {
        let schedule = GymSchedule::default();
        // Monday 10:15 is quarter 41.
        let averages = vec![avg(0, 41, 20.0, 4.0, 3), avg(1, 41, 90.0, 4.0, 3)];
        let estimates = recent_weeks_estimates(date(2024, 6, 17)?, &schedule, &averages);
        let at = |t| estimates.iter().find(|e| e.start == t);
        let e = at(berlin(2024, 6, 17, 10, 15)?).context("10:15")?;
        let expected = e.expected.context("enough days")?;
        assert_relative_eq!(expected.mean, 20.0);
        assert_relative_eq!(expected.low, 16.0);
        assert_relative_eq!(expected.high, 24.0);
        assert_eq!(
            at(berlin(2024, 6, 17, 10, 0)?).and_then(|e| e.expected),
            None
        );
        Ok(())
    }

    #[test]
    fn test_estimate_needs_two_days() -> Result<()> {
        let schedule = GymSchedule::default();
        let averages = vec![avg(0, 41, 20.0, 0.0, 1)];
        let estimates = recent_weeks_estimates(date(2024, 6, 17)?, &schedule, &averages);
        assert!(estimates.iter().all(|e| e.expected.is_none()));
        Ok(())
    }

    #[test]
    fn test_range_stays_between_0_and_100() -> Result<()> {
        let schedule = GymSchedule::default();
        let averages = vec![avg(0, 41, 5.0, 10.0, 3), avg(0, 42, 95.0, 10.0, 3)];
        let estimates = recent_weeks_estimates(date(2024, 6, 17)?, &schedule, &averages);
        let known: Vec<Expected> = estimates.iter().filter_map(|e| e.expected).collect();
        assert_eq!(known.len(), 2);
        assert_relative_eq!(known[0].low, 0.0);
        assert_relative_eq!(known[1].high, 100.0);
        Ok(())
    }

    #[test]
    fn test_holiday_uses_recent_sundays_and_weekend_hours() -> Result<()> {
        // Thursday 3 October 2024 is German Unity Day: open 09:00–21:00.
        let schedule = GymSchedule::default();
        let mut averages = whole_weekday(3, 60.0);
        averages.extend(whole_weekday(6, 15.0));
        let estimates = recent_weeks_estimates(date(2024, 10, 3)?, &schedule, &averages);
        assert_eq!(estimates.len(), 12 * 4);
        assert_eq!(estimates[0].start, berlin(2024, 10, 3, 9, 0)?);
        for e in &estimates {
            assert_eq!(e.source, EstimateSource::RecentSundays);
            assert_relative_eq!(e.expected.context("known")?.mean, 15.0);
        }
        Ok(())
    }

    #[test]
    fn test_estimates_follow_local_quarter_hours_in_winter() -> Result<()> {
        // Saturday 7 December 2024 (CET), 09:00 local is 08:00 UTC.
        let schedule = GymSchedule::default();
        let averages = vec![avg(5, 36, 11.0, 1.0, 2)];
        let estimates = recent_weeks_estimates(date(2024, 12, 7)?, &schedule, &averages);
        assert_eq!(estimates[0].start.hour(), 8);
        assert_relative_eq!(estimates[0].expected.context("09:00")?.mean, 11.0);
        Ok(())
    }

    #[test]
    fn test_forecast_replaces_matching_quarter_hours() -> Result<()> {
        let schedule = GymSchedule::default();
        let mut estimates =
            recent_weeks_estimates(date(2024, 6, 17)?, &schedule, &whole_weekday(0, 30.0));
        let point = Expected {
            mean: 12.0,
            low: 8.0,
            high: 16.0,
        };
        let at = berlin(2024, 6, 17, 18, 30)?;
        // A point outside the day changes nothing.
        with_forecast(
            &mut estimates,
            [(at, point), (berlin(2024, 6, 18, 18, 30)?, point)],
        );
        let forecast: Vec<_> = estimates
            .iter()
            .filter(|e| e.source == EstimateSource::Forecast)
            .collect();
        assert_eq!(forecast.len(), 1);
        assert_eq!(forecast[0].start, at);
        assert_eq!(forecast[0].expected, Some(point));
        Ok(())
    }
}
