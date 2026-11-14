//! The forecast performance page's figures: the period it covers, and
//! everything it shows, computed once when its data arrives.

mod live;
mod model;

use chrono::{DateTime, NaiveDate, TimeDelta, TimeZone, Utc};
use hardy_core::{
    Tz,
    accuracy::{self, ErrorSums, Measures},
    analytics::midnight_local_as_utc,
    db::{BenchmarkNight, ModelErrors, ScoredForecast},
};
use hardy_ml::live::SELECTED;

pub use self::{
    live::{LiveScore, ScatterPoint, backtest_vs_live, live_board, live_scores},
    model::{ModelDetail, detail},
};

/// Days in the error-over-time chart's smoothed line.
const WEEK: i64 = 7;

/// The period the page covers, ending today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Period {
    Week,
    #[default]
    Month,
    Quarter,
    All,
}

impl Period {
    /// Every period, shortest first.
    pub const ALL: [Period; 4] = [Period::Week, Period::Month, Period::Quarter, Period::All];

    /// Label on the period selector.
    pub fn label(self) -> &'static str {
        match self {
            Period::Week => "7 d",
            Period::Month => "28 d",
            Period::Quarter => "90 d",
            Period::All => "All",
        }
    }

    fn days(self) -> Option<i64> {
        match self {
            Period::Week => Some(7),
            Period::Month => Some(28),
            Period::Quarter => Some(90),
            Period::All => None,
        }
    }

    /// Whole gym-local days: the period ends with today, the one before
    /// has the same length (none for [`Period::All`], which starts at
    /// `first_reading`).
    pub fn window(
        self,
        now: DateTime<Utc>,
        tz: Tz,
        first_reading: Option<DateTime<Utc>>,
    ) -> Window {
        let today = now.with_timezone(&tz).date_naive();
        match self.days() {
            Some(days) => {
                let first = today - TimeDelta::days(days - 1);
                Window {
                    since: midnight_local_as_utc(first, tz),
                    previous: Some(midnight_local_as_utc(first - TimeDelta::days(days), tz)),
                }
            }
            None => Window {
                since: first_reading.unwrap_or_else(|| {
                    Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0)
                        .single()
                        .unwrap_or_default()
                }),
                previous: None,
            },
        }
    }
}

/// The instants a period covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub since: DateTime<Utc>,
    /// Start of the period before, for comparison.
    pub previous: Option<DateTime<Utc>>,
}

impl Window {
    /// Where loading starts: the period before, if any.
    pub fn load_from(&self) -> DateTime<Utc> {
        self.previous.unwrap_or(self.since)
    }
}

/// The sub-pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tab {
    #[default]
    Overview,
    Models,
}

impl Tab {
    /// Every sub-page, in order.
    pub const ALL: [Tab; 2] = [Tab::Overview, Tab::Models];

    pub fn label(self) -> &'static str {
        match self {
            Tab::Overview => "Overview",
            Tab::Models => "Models",
        }
    }
}

/// Which numbers the leaderboard shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Source {
    /// The newest nightly backtest.
    #[default]
    Backtest,
    /// The logged live forecasts in the period.
    Live,
}

impl Source {
    pub const ALL: [Source; 2] = [Source::Backtest, Source::Live];

    pub fn label(self) -> &'static str {
        match self {
            Source::Backtest => "Backtest",
            Source::Live => "Live",
        }
    }
}

/// What the page loads for a period.
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceData {
    pub window: Window,
    /// The default forecast an hour ahead, from the period before on.
    pub history: Vec<ScoredForecast>,
    /// Every model's logged live forecasts in the period.
    pub model_errors: ModelErrors,
    /// The nightly benchmarks in the period.
    pub nights: Vec<BenchmarkNight>,
}

/// One day of the error-over-time chart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DayPoint {
    pub day: NaiveDate,
    /// Mean absolute error of the default forecast an hour ahead.
    pub mae: f64,
    /// Plain averages' error on the same targets.
    pub baseline: Option<f64>,
    /// The forecast's error over the week ending this day.
    pub weekly: f64,
}

/// The page's figures for one period.
#[derive(Debug, Clone, PartialEq)]
pub struct Performance {
    pub window: Window,
    /// The default forecast an hour ahead, this period and the one before.
    pub current: Option<Measures>,
    pub previous: Option<Measures>,
    /// The default forecast's logged live forecasts, every lead time.
    pub live_default: Option<Measures>,
    pub daily: Vec<DayPoint>,
    pub live: std::collections::HashMap<String, LiveScore>,
    pub nights: Vec<BenchmarkNight>,
    pub model_errors: ModelErrors,
}

/// The mean of the error behind the plain-averages comparison.
#[allow(clippy::cast_precision_loss)]
fn baseline_mae(sums: &ErrorSums) -> Option<f64> {
    (sums.baseline_count > 0).then_some(sums.baseline_abs / sums.baseline_count as f64)
}

impl Performance {
    /// Computes the page from its data.
    pub fn build(data: PerformanceData, tz: Tz) -> Self {
        let since = data.window.since;
        let split = data.history.partition_point(|r| r.target < since);
        let (before, during) = data.history.split_at(split);
        let days = accuracy::by_day(during, tz);
        let weeks = accuracy::trailing(&days, WEEK);
        let daily = days
            .iter()
            .zip(&weeks)
            .filter_map(|((day, sums), (_, week))| {
                Some(DayPoint {
                    day: *day,
                    mae: sums.measures()?.mae,
                    baseline: baseline_mae(sums),
                    weekly: week.measures()?.mae,
                })
            })
            .collect();
        let mut live_default = ErrorSums::default();
        for (_, _, sums) in data
            .model_errors
            .by_step
            .iter()
            .filter(|(model, _, _)| model == SELECTED)
        {
            live_default.merge(sums);
        }
        Self {
            window: data.window,
            current: accuracy::total(during).measures(),
            previous: data
                .window
                .previous
                .and_then(|_| accuracy::total(before).measures()),
            live_default: live_default.measures(),
            daily,
            live: live_scores(&data.model_errors),
            nights: data.nights,
            model_errors: data.model_errors,
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use hardy_core::accuracy::Pair;

    use super::*;

    fn tz() -> Tz {
        "Europe/Berlin".parse().unwrap_or(Tz::UTC)
    }

    fn utc(d: u32, h: u32) -> Result<DateTime<Utc>> {
        Utc.with_ymd_and_hms(2024, 6, d, h, 0, 0)
            .single()
            .context("time")
    }

    fn scored(target: DateTime<Utc>, error: f64) -> ScoredForecast {
        ScoredForecast {
            target,
            made_at: target - TimeDelta::hours(1),
            model: "naive".to_string(),
            pair: Pair {
                actual: 30.0,
                forecast: 30.0 + error,
                band: None,
                baseline: Some(30.0 + 2.0 * error),
                last_week: None,
            },
        }
    }

    #[test]
    fn test_window_covers_whole_days_and_the_period_before() -> Result<()> {
        // Monday 17 June 10:00 UTC; Berlin is UTC+2.
        let window = Period::Week.window(utc(17, 10)?, tz(), None);
        assert_eq!(window.since, utc(10, 22)?, "Tuesday 11 June 00:00 local");
        assert_eq!(window.previous, Some(utc(3, 22)?));
        assert_eq!(window.load_from(), utc(3, 22)?);

        let first = utc(1, 8)?;
        let all = Period::All.window(utc(17, 10)?, tz(), Some(first));
        assert_eq!((all.since, all.previous), (first, None));
        Ok(())
    }

    #[test]
    fn test_build_splits_the_periods_and_charts_each_day() -> Result<()> {
        let window = Period::Week.window(utc(17, 10)?, tz(), None);
        let data = PerformanceData {
            window,
            history: vec![
                scored(utc(5, 10)?, 8.0),
                scored(utc(16, 10)?, 2.0),
                scored(utc(16, 12)?, 4.0),
                scored(utc(17, 9)?, -6.0),
            ],
            model_errors: ModelErrors::default(),
            nights: Vec::new(),
        };
        let page = Performance::build(data, tz());

        let current = page.current.context("current")?;
        assert_eq!(current.count, 3);
        assert_relative_eq!(current.mae, 4.0);
        assert_relative_eq!(page.previous.context("previous")?.mae, 8.0);
        assert_eq!(page.daily.len(), 2);
        assert_relative_eq!(page.daily[0].mae, 3.0);
        assert_relative_eq!(page.daily[0].baseline.context("baseline")?, 6.0);
        // The week ending the 17th holds all three forecasts.
        assert_relative_eq!(page.daily[1].weekly, 4.0);
        assert_eq!(page.live_default, None);
        Ok(())
    }

    #[test]
    fn test_labels() {
        assert_eq!(
            Period::ALL.map(Period::label),
            ["7 d", "28 d", "90 d", "All"]
        );
        assert_eq!(Tab::ALL.map(Tab::label), ["Overview", "Models"]);
        assert_eq!(Source::ALL.map(Source::label), ["Backtest", "Live"]);
    }
}
