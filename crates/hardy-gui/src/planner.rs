//! The Plan tab's state: the next seven days, their estimates and the
//! quietest windows for the remembered choices.

use std::path::PathBuf;

use chrono::{DateTime, Datelike, NaiveDate, TimeDelta, Utc};
use hardy_core::{
    GymSchedule, QuarterHourAverage,
    analytics::visit::{
        Expected, SlotEstimate, VisitConstraints, VisitPlan, plan_day, recent_weeks_estimates,
        with_forecast,
    },
    is_bavarian_holiday,
};
use hardy_ml::PredictionWithConfidence;

use crate::{
    plan_settings::{self, PlanSettings, VisitLength},
    time_range::Step,
};

/// Days offered: today and the next six.
pub const DAYS: usize = 7;

/// One day of the plan.
#[derive(Debug, Clone, PartialEq)]
pub struct DayPlan {
    pub date: NaiveDate,
    pub holiday: bool,
    /// One per quarter hour the gym is open.
    pub estimates: Vec<SlotEstimate>,
    pub plan: VisitPlan,
}

/// Which time range a control changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeKind {
    Workday,
    Weekend,
}

/// Which end of a time range a control changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeEnd {
    Earliest,
    Latest,
}

/// Whether `date` uses the weekend time range (holidays do).
pub fn is_weekend_like(date: NaiveDate) -> bool {
    date.weekday().number_from_monday() > 5 || is_bavarian_holiday(date)
}

/// The plan of today and the next six days: recent weeks per quarter hour,
/// `forecast` laid over where it reaches, windows for `settings` not before
/// `now`.
pub fn build_days(
    now: DateTime<Utc>,
    schedule: &GymSchedule,
    averages: &[QuarterHourAverage],
    forecast: &[PredictionWithConfidence],
    settings: &PlanSettings,
) -> Vec<DayPlan> {
    let tz = schedule.timezone();
    let today = now.with_timezone(&tz).date_naive();
    let points: Vec<(DateTime<Utc>, Expected)> = forecast
        .iter()
        .map(|p| {
            (
                p.timestamp,
                Expected {
                    mean: p.predicted_value,
                    low: p.confidence_low,
                    high: p.confidence_high,
                },
            )
        })
        .collect();
    today
        .iter_days()
        .take(DAYS)
        .map(|date| {
            let mut estimates = recent_weeks_estimates(date, schedule, averages);
            with_forecast(&mut estimates, points.iter().copied());
            let range = if is_weekend_like(date) {
                settings.weekend
            } else {
                settings.workday
            };
            let constraints = VisitConstraints {
                length: settings.length.duration(),
                earliest: TimeDelta::hours(i64::from(range.earliest)),
                latest: TimeDelta::hours(i64::from(range.latest)),
                not_before: now,
            };
            let plan = plan_day(&estimates, &constraints, tz);
            DayPlan {
                date,
                holiday: is_bavarian_holiday(date),
                estimates,
                plan,
            }
        })
        .collect()
}

/// The Plan tab's choices and computed days.
#[derive(Debug)]
pub struct Planner {
    settings: PlanSettings,
    /// Where the settings are kept; `None` keeps them for this session only.
    path: Option<PathBuf>,
    selected: usize,
    averages: Vec<QuarterHourAverage>,
    days: Vec<DayPlan>,
}

impl Planner {
    /// The planner with the settings stored at `path`.
    pub fn new(path: Option<PathBuf>) -> Self {
        let settings = path
            .as_deref()
            .map_or_else(PlanSettings::default, plan_settings::load);
        Self {
            settings,
            path,
            selected: 0,
            averages: Vec::new(),
            days: Vec::new(),
        }
    }

    pub fn settings(&self) -> &PlanSettings {
        &self.settings
    }

    pub fn days(&self) -> &[DayPlan] {
        &self.days
    }

    /// Index of the chosen day (0: today).
    pub fn selected(&self) -> usize {
        self.selected
    }

    pub fn selected_day(&self) -> Option<&DayPlan> {
        self.days.get(self.selected)
    }

    pub fn has_averages(&self) -> bool {
        !self.averages.is_empty()
    }

    pub fn set_averages(&mut self, averages: Vec<QuarterHourAverage>) {
        self.averages = averages;
    }

    pub fn select_day(&mut self, index: usize) {
        self.selected = index.min(DAYS - 1);
    }

    /// The previous or next day, within the seven.
    pub fn step_day(&mut self, step: Step) {
        let index = match step {
            Step::Back => self.selected.saturating_sub(1),
            Step::Forward => self.selected + 1,
        };
        self.select_day(index);
    }

    /// Changes the visit length and stores the settings.
    pub fn set_length(&mut self, length: VisitLength) {
        self.settings.length = length;
        self.store();
    }

    /// Moves one end of a time range and stores the settings.
    pub fn set_range(&mut self, kind: RangeKind, end: RangeEnd, hour: u32) {
        let range = match kind {
            RangeKind::Workday => &mut self.settings.workday,
            RangeKind::Weekend => &mut self.settings.weekend,
        };
        *range = match end {
            RangeEnd::Earliest => range.with_earliest(hour),
            RangeEnd::Latest => range.with_latest(hour),
        };
        self.store();
    }

    /// Recomputes the days (call when time, data, forecast or settings
    /// change).
    pub fn refresh(
        &mut self,
        now: DateTime<Utc>,
        schedule: &GymSchedule,
        forecast: &[PredictionWithConfidence],
    ) {
        self.days = build_days(now, schedule, &self.averages, forecast, &self.settings);
    }

    /// A small file: written right away, a failure only logged (the choice
    /// still applies for this session).
    fn store(&self) {
        if let Some(path) = &self.path
            && let Err(e) = plan_settings::save(path, &self.settings)
        {
            tracing::warn!("Plan settings not saved: {e:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::{TimeZone, Timelike};
    use hardy_core::analytics::visit::EstimateSource;
    use hardy_ml::PredictionMethod;

    use super::*;
    use crate::plan_settings::HourRange;

    #[allow(non_upper_case_globals)]
    const Berlin: hardy_core::Tz = hardy_core::Tz::Europe__Berlin;

    fn berlin(y: i32, m: u32, d: u32, h: u32, mi: u32) -> Result<DateTime<Utc>> {
        Berlin
            .with_ymd_and_hms(y, m, d, h, mi, 0)
            .single()
            .map(|t| t.with_timezone(&Utc))
            .context("valid Berlin time")
    }

    /// Every quarter hour of every weekday at `mean`, seen on four days.
    fn flat(mean: f64) -> Vec<QuarterHourAverage> {
        (0..7)
            .flat_map(|weekday| {
                (0..96).map(move |quarter| QuarterHourAverage {
                    weekday,
                    quarter,
                    avg_percentage: mean,
                    std_dev: 2.0,
                    days: 4,
                })
            })
            .collect()
    }

    fn point(t: DateTime<Utc>, value: f64) -> PredictionWithConfidence {
        PredictionWithConfidence {
            timestamp: t,
            predicted_value: value,
            confidence_low: value - 3.0,
            confidence_high: value + 3.0,
            confidence_score: 0.5,
            method: PredictionMethod::HistoricalAverage,
        }
    }

    #[test]
    fn test_weekend_like_days() -> Result<()> {
        let day = |d| NaiveDate::from_ymd_opt(2024, 10, d).context("date");
        assert!(!is_weekend_like(day(2)?)); // Wednesday
        assert!(is_weekend_like(day(3)?)); // German Unity Day
        assert!(is_weekend_like(day(5)?)); // Saturday
        assert!(is_weekend_like(day(6)?)); // Sunday
        Ok(())
    }

    #[test]
    fn test_seven_days_from_today() -> Result<()> {
        // Monday 17 June 2024, 10:00.
        let now = berlin(2024, 6, 17, 10, 0)?;
        let days = build_days(
            now,
            &GymSchedule::default(),
            &flat(30.0),
            &[],
            &PlanSettings::default(),
        );
        let dates: Vec<_> = days.iter().map(|d| d.date.day()).collect();
        assert_eq!(dates, vec![17, 18, 19, 20, 21, 22, 23]);
        assert!(days.iter().all(|d| !d.holiday));
        // Weekday 06:00–23:00, weekend 09:00–21:00.
        assert_eq!(days[0].estimates.len(), 68);
        assert_eq!(days[5].estimates.len(), 48);
        Ok(())
    }

    #[test]
    fn test_today_plans_from_now() -> Result<()> {
        let now = berlin(2024, 6, 17, 10, 5)?;
        let days = build_days(
            now,
            &GymSchedule::default(),
            &flat(30.0),
            &[],
            &PlanSettings::default(),
        );
        let best = days[0].plan.best.context("a window today")?;
        assert_eq!(best.start, berlin(2024, 6, 17, 10, 15)?);
        // Tomorrow starts at opening.
        let tomorrow = days[1].plan.best.context("a window tomorrow")?;
        assert_eq!(tomorrow.start, berlin(2024, 6, 18, 6, 0)?);
        Ok(())
    }

    #[test]
    fn test_forecast_is_laid_over_and_can_win() -> Result<()> {
        let now = berlin(2024, 6, 17, 10, 0)?;
        let quiet: Vec<_> = (0..6)
            .map(|i| berlin(2024, 6, 17, 14, 0).map(|t| point(t + TimeDelta::minutes(15 * i), 5.0)))
            .collect::<Result<_>>()?;
        let settings = PlanSettings {
            length: VisitLength::OneHour,
            ..PlanSettings::default()
        };
        let days = build_days(now, &GymSchedule::default(), &flat(30.0), &quiet, &settings);
        let best = days[0].plan.best.context("a window")?;
        assert_eq!(best.start, berlin(2024, 6, 17, 14, 0)?);
        assert_eq!(best.source, EstimateSource::Forecast);
        assert_relative_eq!(best.expected.mean, 5.0);
        assert_relative_eq!(best.expected.low, 2.0);
        Ok(())
    }

    #[test]
    fn test_workday_and_weekend_ranges_apply() -> Result<()> {
        let now = berlin(2024, 6, 17, 5, 0)?;
        let settings = PlanSettings {
            length: VisitLength::OneHour,
            workday: HourRange {
                earliest: 17,
                latest: 24,
            },
            weekend: HourRange {
                earliest: 12,
                latest: 24,
            },
        };
        let days = build_days(now, &GymSchedule::default(), &flat(30.0), &[], &settings);
        let start_hour = |i: usize| {
            days[i]
                .plan
                .best
                .map(|w| w.start.with_timezone(&Berlin).hour())
        };
        assert_eq!(start_hour(0), Some(17)); // Monday
        assert_eq!(start_hour(5), Some(12)); // Saturday
        Ok(())
    }

    #[test]
    fn test_holiday_is_marked() -> Result<()> {
        // Wednesday 2 October 2024; the 3rd is German Unity Day.
        let now = berlin(2024, 10, 2, 8, 0)?;
        let days = build_days(
            now,
            &GymSchedule::default(),
            &flat(30.0),
            &[],
            &PlanSettings::default(),
        );
        let holidays: Vec<_> = days.iter().map(|d| d.holiday).collect();
        assert_eq!(
            holidays,
            vec![false, true, false, false, false, false, false]
        );
        Ok(())
    }

    #[test]
    fn test_planner_remembers_changes() {
        let dir = std::env::temp_dir().join(format!("hardy-planner-{}", std::process::id()));
        let path = dir.join("plan.json");
        let _ = std::fs::remove_file(&path);
        let mut planner = Planner::new(Some(path.clone()));
        planner.set_length(VisitLength::TwoHours);
        planner.set_range(RangeKind::Workday, RangeEnd::Earliest, 18);
        let reopened = Planner::new(Some(path));
        assert_eq!(reopened.settings().length, VisitLength::TwoHours);
        assert_eq!(reopened.settings().workday.earliest, 18);
        assert_eq!(reopened.settings().weekend, HourRange::default());
    }

    #[test]
    fn test_planner_selects_and_refreshes() -> Result<()> {
        let mut planner = Planner::new(None);
        assert!(!planner.has_averages());
        planner.set_averages(flat(30.0));
        planner.refresh(berlin(2024, 6, 17, 10, 0)?, &GymSchedule::default(), &[]);
        assert_eq!(planner.days().len(), DAYS);
        planner.select_day(9);
        assert_eq!(planner.selected(), DAYS - 1);
        assert_eq!(planner.selected_day().map(|d| d.date.day()), Some(23));
        Ok(())
    }

    #[test]
    fn test_arrow_steps_stay_within_the_week() {
        let mut planner = Planner::new(None);
        planner.select_day(DAYS - 1);
        planner.step_day(Step::Forward);
        assert_eq!(planner.selected(), DAYS - 1);
        planner.step_day(Step::Back);
        assert_eq!(planner.selected(), DAYS - 2);
        planner.select_day(0);
        planner.step_day(Step::Back);
        assert_eq!(planner.selected(), 0);
        planner.step_day(Step::Forward);
        assert_eq!(planner.selected(), 1);
    }
}
