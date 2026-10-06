//! "When should I go?": the quietest upcoming hour.

use chrono::{DateTime, DurationRound, NaiveDate, TimeDelta, Utc};
use hardy_core::GymSchedule;
use hardy_ml::{PredictionWithConfidence, SlotProfile};

/// Length of the window searched for.
pub const WINDOW: TimeDelta = TimeDelta::hours(1);
/// Spacing of candidate start times.
const STEP: TimeDelta = TimeDelta::minutes(15);
/// Windows end at least this long before closing; the last hour is too
/// late to start a workout.
pub const CLOSING_MARGIN: TimeDelta = TimeDelta::hours(1);
/// A forecast point counts for times within this distance of it.
const FORECAST_REACH: TimeDelta = TimeDelta::minutes(30);

/// Where the expected value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Forecast,
    Averages,
}

/// The quietest upcoming window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuietWindow {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    /// Expected mean occupancy in percent.
    pub expected: f64,
    pub source: Source,
}

/// Finds the quietest [`WINDOW`]: within the rest of today while the gym is
/// open, otherwise on the next opening day starting `grace` after opening
/// (it is always empty right at opening). Windows end [`CLOSING_MARGIN`]
/// before closing.
pub fn next_quiet_window(
    now: DateTime<Utc>,
    schedule: &GymSchedule,
    forecasts: &[PredictionWithConfidence],
    profile: &SlotProfile,
    grace: TimeDelta,
) -> Option<QuietWindow> {
    let tz = schedule.timezone();
    let today = now.with_timezone(&tz).date_naive();

    let mut candidates = if schedule.is_open(&now) {
        starts(
            ceil_step(now),
            closing_on(schedule, today) - WINDOW - CLOSING_MARGIN,
        )
    } else {
        Vec::new()
    };
    if candidates.is_empty() {
        let day = if now < schedule.opening_time_on(today) {
            today
        } else {
            today.succ_opt()?
        };
        candidates = starts(
            schedule.opening_time_on(day) + grace,
            closing_on(schedule, day) - WINDOW - CLOSING_MARGIN,
        );
    }

    candidates
        .into_iter()
        .map(|start| {
            let (expected, source) = window_expectation(start, schedule, forecasts, profile);
            QuietWindow {
                start,
                end: start + WINDOW,
                expected,
                source,
            }
        })
        .min_by(|a, b| {
            a.expected
                .total_cmp(&b.expected)
                .then(a.start.cmp(&b.start))
        })
}

fn closing_on(schedule: &GymSchedule, day: NaiveDate) -> DateTime<Utc> {
    schedule.next_closing_after(schedule.opening_time_on(day))
}

fn ceil_step(t: DateTime<Utc>) -> DateTime<Utc> {
    let floor = t.duration_trunc(STEP).unwrap_or(t);
    if floor == t { t } else { floor + STEP }
}

fn starts(from: DateTime<Utc>, last: DateTime<Utc>) -> Vec<DateTime<Utc>> {
    let mut out = Vec::new();
    let mut t = ceil_step(from);
    while t <= last {
        out.push(t);
        t += STEP;
    }
    out
}

/// Mean expectation over the window, sampled every [`STEP`]; forecast where
/// a forecast point is close enough, slot averages otherwise.
fn window_expectation(
    start: DateTime<Utc>,
    schedule: &GymSchedule,
    forecasts: &[PredictionWithConfidence],
    profile: &SlotProfile,
) -> (f64, Source) {
    let tz = schedule.timezone();
    let mut total = 0.0;
    let mut samples = 0.0;
    let mut forecast_samples = 0;
    let mut t = start;
    while t < start + WINDOW {
        let forecast = forecasts
            .iter()
            .filter(|f| (f.timestamp - t).abs() <= FORECAST_REACH)
            .min_by_key(|f| (f.timestamp - t).abs());
        total += forecast.map_or_else(|| profile.mean_at(t, tz), |f| f.predicted_value);
        if forecast.is_some() {
            forecast_samples += 1;
        }
        samples += 1.0;
        t += STEP;
    }
    let source = if forecast_samples > 0 {
        Source::Forecast
    } else {
        Source::Averages
    };
    (total / samples, source)
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::TimeZone;
    use hardy_ml::{History, PredictionMethod};
    use proptest::prelude::*;

    use super::*;

    const GRACE: TimeDelta = TimeDelta::hours(1);

    /// Monday 2024-06-17 + `days`, `h:mi` CEST.
    fn local(days: i64, h: u32, mi: u32) -> DateTime<Utc> {
        GymSchedule::default()
            .timezone()
            .with_ymd_and_hms(2024, 6, 17, h, mi, 0)
            .single()
            .map_or_else(DateTime::default, |t| t.with_timezone(&Utc))
            + TimeDelta::days(days)
    }

    /// Two weeks of history: 30% everywhere except a quiet 14:00 hour (10%).
    fn profile() -> SlotProfile {
        let schedule = GymSchedule::default();
        let mut points = Vec::new();
        for d in -14..0 {
            for minute in 0..(24 * 60) {
                let t = local(d, 0, 0) + TimeDelta::minutes(minute);
                if schedule.is_open(&t) {
                    let hour = t
                        .with_timezone(&schedule.timezone())
                        .format("%H")
                        .to_string();
                    points.push((t, if hour == "14" { 10.0 } else { 30.0 }));
                }
            }
        }
        SlotProfile::from_history(History::new(points).view(), schedule.timezone())
    }

    fn forecast(t: DateTime<Utc>, value: f64) -> PredictionWithConfidence {
        PredictionWithConfidence::new(
            t,
            value,
            value - 5.0,
            value + 5.0,
            0.8,
            PredictionMethod::HistoricalAverage,
        )
    }

    #[test]
    fn test_quiet_window_uses_averages_today() -> Result<()> {
        let w = next_quiet_window(
            local(0, 10, 7),
            &GymSchedule::default(),
            &[],
            &profile(),
            GRACE,
        )
        .context("window")?;
        assert_eq!(w.start, local(0, 14, 0));
        assert_eq!(w.end, local(0, 15, 0));
        assert_relative_eq!(w.expected, 10.0);
        assert_eq!(w.source, Source::Averages);
        Ok(())
    }

    #[test]
    fn test_quiet_window_prefers_forecast_where_available() -> Result<()> {
        // The forecast says 12:00 will be nearly empty today.
        let forecasts = [
            forecast(local(0, 12, 0), 2.0),
            forecast(local(0, 13, 0), 40.0),
        ];
        let w = next_quiet_window(
            local(0, 10, 0),
            &GymSchedule::default(),
            &forecasts,
            &profile(),
            GRACE,
        )
        .context("window")?;
        assert_eq!(w.source, Source::Forecast);
        assert!(w.start >= local(0, 11, 30) && w.start <= local(0, 12, 0));
        Ok(())
    }

    #[test]
    fn test_quiet_window_moves_to_tomorrow_near_closing() -> Result<()> {
        // Monday 22:30: no window ends an hour before closing → Tuesday.
        let w = next_quiet_window(
            local(0, 22, 30),
            &GymSchedule::default(),
            &[],
            &profile(),
            GRACE,
        )
        .context("window")?;
        assert_eq!(w.start, local(1, 14, 0));
        Ok(())
    }

    #[test]
    fn test_quiet_window_skips_last_hour_before_closing() -> Result<()> {
        // Everything is busy except the last hour (22:00–23:00 on Monday),
        // which is too late to be useful.
        let schedule = GymSchedule::default();
        let mut points = Vec::new();
        for d in -14..0 {
            for minute in 0..(24 * 60) {
                let t = local(d, 0, 0) + TimeDelta::minutes(minute);
                if schedule.is_open(&t) {
                    let hour = t
                        .with_timezone(&schedule.timezone())
                        .format("%H")
                        .to_string();
                    let value = match hour.as_str() {
                        "22" => 2.0,
                        "15" => 10.0,
                        _ => 40.0,
                    };
                    points.push((t, value));
                }
            }
        }
        let profile = SlotProfile::from_history(History::new(points).view(), schedule.timezone());
        let w = next_quiet_window(local(0, 10, 0), &schedule, &[], &profile, GRACE)
            .context("window")?;
        assert_eq!(w.start, local(0, 15, 0));
        Ok(())
    }

    #[test]
    fn test_quiet_window_before_opening_skips_grace() -> Result<()> {
        // Flat profile: earliest allowed start wins the tie.
        let schedule = GymSchedule::default();
        let flat = SlotProfile::from_history(History::default().view(), schedule.timezone());
        let w =
            next_quiet_window(local(0, 4, 0), &schedule, &[], &flat, GRACE).context("window")?;
        assert_eq!(w.start, local(0, 7, 0));
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn quiet_window_is_upcoming_and_inside_opening_hours(minute in 0i64..(7 * 24 * 60)) {
            let schedule = GymSchedule::default();
            let now = local(0, 0, 0) + TimeDelta::minutes(minute);
            let flat = SlotProfile::from_history(History::default().view(), schedule.timezone());
            let w = next_quiet_window(now, &schedule, &[], &flat, GRACE);
            prop_assert!(w.is_some());
            if let Some(w) = w {
                prop_assert!(w.start >= now);
                prop_assert_eq!(w.end - w.start, WINDOW);
                prop_assert!(schedule.is_open(&w.start));
                prop_assert!(schedule.is_open(&(w.end - TimeDelta::minutes(1))));
                prop_assert!(schedule.is_open(&(w.end + CLOSING_MARGIN - TimeDelta::minutes(1))));
                prop_assert!(w.start - now < TimeDelta::hours(48));
            }
        }
    }
}
