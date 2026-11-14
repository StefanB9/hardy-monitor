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
/// Occupancy bands of the error-by-level chart, in points.
const LEVEL_WIDTH: f64 = 10.0;
/// Bins of the error histogram and the errors it spans, in points.
const HISTOGRAM_BIN: f64 = 2.0;
const HISTOGRAM_LIMIT: f64 = 20.0;

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
    /// When and at what occupancy the forecast is off.
    WhenWhere,
    /// Bias and how well the band works.
    IntervalsBias,
}

impl Tab {
    /// Every sub-page, in order.
    pub const ALL: [Tab; 4] = [
        Tab::Overview,
        Tab::Models,
        Tab::WhenWhere,
        Tab::IntervalsBias,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Tab::Overview => "Overview",
            Tab::Models => "Models",
            Tab::WhenWhere => "When & where",
            Tab::IntervalsBias => "Intervals & bias",
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
    /// Mean of forecast − reading: positive when it ran high.
    pub bias: f64,
    /// Share of readings inside the band, and its mean width.
    pub coverage: Option<f64>,
    pub width: Option<f64>,
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
    /// The period's scored forecasts, oldest first.
    pub scored: Vec<ScoredForecast>,
    /// Error sums per weekday (0 = Monday) and hour.
    pub grid: Vec<[ErrorSums; 24]>,
    /// Error sums per occupancy band of the reading, by lower edge.
    pub levels: Vec<(f64, ErrorSums)>,
    /// Forecasts per error bin (forecast − reading), by lower edge.
    pub histogram: Vec<(f64, i64)>,
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
                let measures = sums.measures()?;
                Some(DayPoint {
                    day: *day,
                    mae: measures.mae,
                    baseline: baseline_mae(sums),
                    weekly: week.measures()?.mae,
                    bias: measures.bias,
                    coverage: measures.coverage,
                    width: measures.width,
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
            grid: accuracy::by_weekday_hour(during, tz),
            levels: accuracy::by_level(during, LEVEL_WIDTH),
            histogram: accuracy::error_histogram(during, HISTOGRAM_BIN, HISTOGRAM_LIMIT),
            scored: during.to_vec(),
            live: live_scores(&data.model_errors),
            nights: data.nights,
            model_errors: data.model_errors,
        }
    }
}

/// Error sums per hour of the day, every weekday together.
pub fn by_hour(grid: &[[ErrorSums; 24]]) -> [ErrorSums; 24] {
    let mut hours = [ErrorSums::default(); 24];
    for day in grid {
        for (hour, sums) in hours.iter_mut().zip(day) {
            hour.merge(sums);
        }
    }
    hours
}

/// Error sums per weekday (0 = Monday), every hour together.
pub fn by_weekday(grid: &[[ErrorSums; 24]]) -> Vec<ErrorSums> {
    grid.iter()
        .map(|day| {
            let mut total = ErrorSums::default();
            for sums in day {
                total.merge(sums);
            }
            total
        })
        .collect()
}

/// One model's live measures per step (1 = 15 min ahead).
pub fn lead_curve(errors: &ModelErrors, model: &str) -> Vec<(u32, Measures)> {
    errors
        .by_step
        .iter()
        .filter(|(m, _, _)| m == model)
        .filter_map(|(_, step, sums)| Some((*step, sums.measures()?)))
        .collect()
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
        // The period's forecasts only: by hour, by level, and their errors.
        assert_eq!(page.scored.len(), 3);
        assert_relative_eq!(page.daily[1].bias, -6.0);
        assert_eq!(page.daily[1].coverage, None);
        assert_eq!(by_hour(&page.grid).iter().map(|s| s.count).sum::<i64>(), 3);
        // 10:00 and 12:00 UTC on Sunday are 12:00 and 14:00 in Berlin.
        assert_eq!(by_weekday(&page.grid)[6].count, 2);
        assert_eq!(page.levels.len(), 1);
        assert_eq!(page.histogram.iter().map(|(_, n)| n).sum::<i64>(), 3);
        Ok(())
    }

    #[test]
    fn test_lead_curve_follows_one_model_by_step() {
        let sums = |error: f64| {
            let mut s = ErrorSums::default();
            s.add(&Pair {
                actual: 30.0,
                forecast: 30.0 + error,
                band: None,
                baseline: None,
                last_week: None,
            });
            s
        };
        let errors = ModelErrors {
            by_step: vec![
                ("naive".to_string(), 1, sums(1.0)),
                ("naive".to_string(), 2, sums(3.0)),
                ("theta".to_string(), 1, sums(9.0)),
            ],
            by_day: Vec::new(),
        };
        let curve = lead_curve(&errors, "naive");
        assert_eq!(curve.len(), 2);
        assert_eq!(curve[1].0, 2);
        assert_relative_eq!(curve[1].1.mae, 3.0);
        assert_eq!(lead_curve(&errors, "ridge"), Vec::new());
    }

    #[test]
    fn test_labels() {
        assert_eq!(
            Period::ALL.map(Period::label),
            ["7 d", "28 d", "90 d", "All"]
        );
        assert_eq!(
            Tab::ALL.map(Tab::label),
            ["Overview", "Models", "When & where", "Intervals & bias"]
        );
        assert_eq!(Source::ALL.map(Source::label), ["Backtest", "Live"]);
    }
}
