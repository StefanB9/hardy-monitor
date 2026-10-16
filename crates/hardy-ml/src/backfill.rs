//! The forecast history: the default forecast of past slots as it would
//! have been made, for the accuracy log and the history chart. The backfill
//! benchmarks every day with data once; each nightly run then adds its
//! newest benchmark day.

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::{
    GymSchedule,
    db::{ForecastHistoryEntry, ForecastLogEntry},
};

use crate::{
    benchmark::{PastForecast, Settings},
    history::History,
    nightly::benchmark_suite,
    progress::Progress,
    suite::ModelId,
};

/// Slots per hour.
const SLOTS_PER_HOUR: usize = 4;
/// Steps that are an hour ahead in the history chart: the forecast made at
/// the last full hour at least an hour before the slot.
const HOUR_AHEAD: std::ops::RangeInclusive<usize> = SLOTS_PER_HOUR..=2 * SLOTS_PER_HOUR - 1;

/// Database rows made from past forecasts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PastRows {
    /// The default forecast 1, 2, … hours ahead of each origin.
    pub log: Vec<ForecastLogEntry>,
    /// The hour-ahead forecast of each slot.
    pub history: Vec<ForecastHistoryEntry>,
}

/// Rows for the past forecasts made at or after `since`: accuracy log
/// entries for whole hours `1..=log_hours` ahead, and each slot's
/// hour-ahead forecast.
pub fn rows(past: &[PastForecast<ModelId>], log_hours: u32, since: DateTime<Utc>) -> PastRows {
    let mut out = PastRows::default();
    for p in past.iter().filter(|p| p.made_at >= since) {
        let hours = u32::try_from(p.step / SLOTS_PER_HOUR).unwrap_or(u32::MAX);
        if p.step.is_multiple_of(SLOTS_PER_HOUR) && (1..=log_hours).contains(&hours) {
            let (low, high) = p.bounds.unwrap_or((p.value, p.value));
            out.log.push(ForecastLogEntry {
                made_at: p.made_at,
                target: p.made_at + TimeDelta::hours(i64::from(hours)),
                horizon_hours: hours,
                model: Some(p.model.key().to_string()),
                predicted: p.value,
                low,
                high,
                baseline: p.comparison.unwrap_or(p.value),
            });
        }
        if HOUR_AHEAD.contains(&p.step) {
            out.history.push(ForecastHistoryEntry {
                target: p.target,
                made_at: p.made_at,
                model: p.model.key().to_string(),
                value: p.value,
                low: p.bounds.map(|(low, _)| low),
                high: p.bounds.map(|(_, high)| high),
            });
        }
    }
    out
}

/// Backtests `models` on every day of `history` before `now`, recording
/// the default forecast of each as it would have been made.
pub fn run_reporting(
    models: &[ModelId],
    history: &History,
    schedule: &GymSchedule,
    now: DateTime<Utc>,
    settings: &Settings,
    progress: &dyn Progress,
) -> Vec<PastForecast<ModelId>> {
    let all_days = Settings {
        days: u32::MAX,
        ..*settings
    };
    benchmark_suite(models, history, schedule, now, &all_days, progress).past
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use anyhow::{Context, Result};
    use chrono::{DurationRound, Timelike};

    use super::*;
    use crate::{
        grid::SLOT,
        progress::{Stage, counting::Counting},
        test_support::{local, minutely},
    };

    fn past(made_at: DateTime<Utc>, step: usize) -> PastForecast<ModelId> {
        #[allow(clippy::cast_precision_loss)]
        let value = 10.0 + step as f64;
        PastForecast {
            made_at,
            step,
            target: made_at + SLOT * i32::try_from(step).unwrap_or(0),
            model: ModelId::Naive,
            value,
            bounds: (step != 8).then_some((value - 1.0, value + 2.0)),
            comparison: (step != 4).then_some(5.0),
        }
    }

    #[test]
    fn test_rows_log_whole_hours_and_keep_the_hour_ahead_slots() -> Result<()> {
        let made = local(3, 9, 0);
        let all: Vec<PastForecast<ModelId>> = (1..=13).map(|s| past(made, s)).collect();
        let out = rows(&all, 2, made);

        let horizons: Vec<u32> = out.log.iter().map(|e| e.horizon_hours).collect();
        assert_eq!(horizons, vec![1, 2]);
        let first = out.log.first().context("no log")?;
        assert_eq!(first.target, local(3, 10, 0));
        assert_eq!(first.model.as_deref(), Some("naive"));
        assert_eq!((first.low, first.high), (13.0, 16.0));
        // No comparison forecast: compared with itself.
        assert!((first.baseline - 14.0).abs() < 1e-12);
        let second = out.log.get(1).context("no second hour")?;
        // No interval yet: a zero-width band.
        assert_eq!((second.low, second.high), (18.0, 18.0));
        assert!((second.baseline - 5.0).abs() < 1e-12);

        let targets: Vec<DateTime<Utc>> = out.history.iter().map(|e| e.target).collect();
        assert_eq!(
            targets,
            vec![
                local(3, 10, 0),
                local(3, 10, 15),
                local(3, 10, 30),
                local(3, 10, 45)
            ]
        );
        assert_eq!(out.history[0].low, Some(13.0));
        Ok(())
    }

    #[test]
    fn test_rows_skip_forecasts_made_before_since() {
        let all = vec![past(local(3, 9, 0), 4), past(local(3, 10, 0), 4)];
        let out = rows(&all, 6, local(3, 10, 0));
        assert_eq!(out.log.len(), 1);
        assert_eq!(out.history.len(), 1);
        assert_eq!(out.history[0].made_at, local(3, 10, 0));
    }

    #[test]
    fn test_backfill_covers_every_day_with_one_forecast_per_slot() -> Result<()> {
        let history = minutely(40, |t| 20.0 + f64::from(t.minute() % 7));
        let now = local(40, 0, 0);
        let progress = Counting::default();
        let models = [ModelId::Naive, ModelId::MovingAverage];
        let past = run_reporting(
            &models,
            &history,
            &GymSchedule::default(),
            now,
            &Settings::default(),
            &progress,
        );
        // Every complete day is backtested, beyond the nightly 28.
        assert_eq!(progress.get(Stage::Backtest), Some((40, 40, true)));
        let out = rows(&past, 6, DateTime::<Utc>::MIN_UTC);
        let mut seen = HashSet::new();
        for entry in &out.history {
            assert!(
                seen.insert(entry.target),
                "two forecasts for {}",
                entry.target
            );
            let hour = (entry.target - TimeDelta::hours(1)).duration_trunc(TimeDelta::hours(1))?;
            assert_eq!(entry.made_at, hour);
        }
        let tz = GymSchedule::default().timezone();
        let days: HashSet<_> = out
            .history
            .iter()
            .map(|e| e.target.with_timezone(&tz).date_naive())
            .collect();
        // The first day has nothing to fit on.
        assert_eq!(days.len(), 39);
        assert!(out.log.iter().all(|e| e.model.is_some()));
        assert_ne!(out.log.len(), 0);
        Ok(())
    }
}
