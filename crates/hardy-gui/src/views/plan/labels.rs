//! Labels and explanations of the Plan tab.

use chrono::{DateTime, NaiveDate, Utc};
use hardy_core::{
    Tz,
    analytics::{
        is_unpredictable,
        visit::{DayPart, EstimateSource, Expected, QUARTER, VisitWindow},
    },
};

use crate::{plan_settings::VisitLength, planner::DayPlan};

/// "Today", "Tomorrow" or e.g. "Wed 19".
pub(super) fn chip_title(date: NaiveDate, today: NaiveDate) -> String {
    if date == today {
        "Today".to_string()
    } else if today.succ_opt() == Some(date) {
        "Tomorrow".to_string()
    } else {
        date.format("%a %-d").to_string()
    }
}

/// A whole hour of the time range, e.g. "17:00"; 24 is midnight.
pub(super) fn hour_label(hour: u32) -> String {
    format!("{hour:02}:00")
}

/// "18:00 – 19:30" in the gym's time.
pub(super) fn window_text(window: &VisitWindow, tz: Tz) -> String {
    format!(
        "{} – {}",
        window.start.with_timezone(&tz).format("%H:%M"),
        window.end.with_timezone(&tz).format("%H:%M")
    )
}

/// "~8% expected (5–11%)", plus "· unpredictable" when the range is wider
/// than ±[`hardy_core::analytics::UNPREDICTABLE_SPREAD`] points.
pub(super) fn expected_text(expected: Expected) -> String {
    let text = format!(
        "~{:.0}% expected ({:.0}–{:.0}%)",
        expected.mean, expected.low, expected.high
    );
    if is_unpredictable((expected.high - expected.low) / 2.0) {
        format!("{text} · unpredictable")
    } else {
        text
    }
}

/// Where the day's estimate comes from, e.g. "Typical Monday, last 4 weeks".
pub(super) fn source_text(source: EstimateSource, date: NaiveDate) -> String {
    match source {
        EstimateSource::Forecast => "Forecast".to_string(),
        EstimateSource::RecentWeeks => format!("Typical {}, last 4 weeks", date.format("%A")),
        EstimateSource::RecentSundays => "Holiday – based on recent Sundays".to_string(),
    }
}

pub(super) fn part_label(part: DayPart) -> &'static str {
    match part {
        DayPart::Morning => "Morning",
        DayPart::Midday => "Midday",
        DayPart::Afternoon => "Afternoon",
        DayPart::Evening => "Evening",
    }
}

/// Why a day has no suggested window.
pub(super) fn no_window_reason(
    day: &DayPlan,
    now: DateTime<Utc>,
    length: VisitLength,
    has_averages: bool,
) -> String {
    let opens = day.estimates.first().map(|e| e.start);
    let closes = day.estimates.last().map(|e| e.start + QUARTER);
    let reason = if !has_averages {
        "Collecting data…"
    } else if closes.is_some_and(|end| now >= end) {
        "Closed for the rest of today"
    } else if opens.is_some_and(|start| now > start)
        && closes.is_some_and(|end| now + length.duration() > end)
    {
        return format!("Not enough time left today for {}", length.label());
    } else if day.estimates.iter().all(|e| e.expected.is_none()) {
        "Not enough data for this day yet"
    } else {
        "No window fits – try a shorter visit or a wider time range"
    };
    reason.to_string()
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use chrono::{TimeDelta, TimeZone};
    use hardy_core::analytics::visit::{SlotEstimate, VisitPlan};

    use super::*;

    const BERLIN: Tz = Tz::Europe__Berlin;

    fn date(d: u32) -> Result<NaiveDate> {
        NaiveDate::from_ymd_opt(2024, 6, d).context("date")
    }

    fn berlin(d: u32, h: u32, mi: u32) -> Result<DateTime<Utc>> {
        BERLIN
            .with_ymd_and_hms(2024, 6, d, h, mi, 0)
            .single()
            .map(|t| t.with_timezone(&Utc))
            .context("valid time")
    }

    /// Monday 17 June from 06:00, quarter hours known or not.
    fn day(known: &[bool]) -> Result<DayPlan> {
        let start = berlin(17, 6, 0)?;
        Ok(DayPlan {
            date: date(17)?,
            holiday: false,
            estimates: known
                .iter()
                .enumerate()
                .map(|(i, k)| SlotEstimate {
                    start: start + QUARTER * i32::try_from(i).unwrap_or(0),
                    expected: k.then_some(Expected {
                        mean: 10.0,
                        low: 5.0,
                        high: 15.0,
                    }),
                    source: EstimateSource::RecentWeeks,
                })
                .collect(),
            plan: VisitPlan::default(),
        })
    }

    #[test]
    fn test_chip_titles() -> Result<()> {
        assert_eq!(chip_title(date(17)?, date(17)?), "Today");
        assert_eq!(chip_title(date(18)?, date(17)?), "Tomorrow");
        assert_eq!(chip_title(date(19)?, date(17)?), "Wed 19");
        Ok(())
    }

    #[test]
    fn test_hour_labels() {
        assert_eq!(hour_label(0), "00:00");
        assert_eq!(hour_label(17), "17:00");
        assert_eq!(hour_label(24), "24:00");
    }

    #[test]
    fn test_window_and_expected_text() -> Result<()> {
        let window = VisitWindow {
            start: berlin(17, 18, 0)?,
            end: berlin(17, 19, 30)?,
            expected: Expected {
                mean: 8.2,
                low: 4.6,
                high: 11.4,
            },
            source: EstimateSource::Forecast,
        };
        assert_eq!(window_text(&window, BERLIN), "18:00 – 19:30");
        assert_eq!(expected_text(window.expected), "~8% expected (5–11%)");
        let wide = Expected {
            mean: 19.0,
            low: 0.0,
            high: 45.0,
        };
        assert_eq!(expected_text(wide), "~19% expected (0–45%) · unpredictable");
        Ok(())
    }

    #[test]
    fn test_source_texts() -> Result<()> {
        assert_eq!(source_text(EstimateSource::Forecast, date(17)?), "Forecast");
        assert_eq!(
            source_text(EstimateSource::RecentWeeks, date(17)?),
            "Typical Monday, last 4 weeks"
        );
        assert_eq!(
            source_text(EstimateSource::RecentSundays, date(17)?),
            "Holiday – based on recent Sundays"
        );
        Ok(())
    }

    #[test]
    fn test_part_labels() {
        let labels: Vec<_> = DayPart::ALL.iter().map(|p| part_label(*p)).collect();
        assert_eq!(labels, vec!["Morning", "Midday", "Afternoon", "Evening"]);
    }

    #[test]
    fn test_no_window_reasons() -> Result<()> {
        let morning = berlin(17, 7, 0)?;
        let length = VisitLength::OneHour;
        assert_eq!(
            no_window_reason(&day(&[true; 8])?, morning, length, false),
            "Collecting data…"
        );
        assert_eq!(
            no_window_reason(&day(&[false; 8])?, morning, length, true),
            "Not enough data for this day yet"
        );
        assert_eq!(
            no_window_reason(&day(&[true; 8])?, morning, length, true),
            "No window fits – try a shorter visit or a wider time range"
        );
        // Past the day's last quarter hour.
        assert_eq!(
            no_window_reason(
                &day(&[true; 8])?,
                morning + TimeDelta::hours(1),
                length,
                true
            ),
            "Closed for the rest of today"
        );
        // Open until 08:00, but 07:45 leaves no time for an hour.
        assert_eq!(
            no_window_reason(
                &day(&[true; 8])?,
                morning + TimeDelta::minutes(45),
                length,
                true
            ),
            "Not enough time left today for 1 h"
        );
        Ok(())
    }
}
