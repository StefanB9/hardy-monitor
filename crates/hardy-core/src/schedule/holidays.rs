//! Bavarian public holidays (the gym keeps weekend hours on them).

use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use chrono::{Datelike, NaiveDate};

/// Whether `date` is a public holiday in Bavaria (fixed dates and
/// the\nEaster-dependent ones).
pub fn is_bavarian_holiday(date: NaiveDate) -> bool {
    let (d, m) = (date.day(), date.month());
    let year = date.year();

    match (m, d) {
        (1 | 5 | 11, 1) | (1, 6) | (8, 15) | (10, 3) | (12, 25 | 26) => return true,
        _ => {}
    }

    if let Some(easter) = easter_date_cached(year) {
        let ordinal = date.ordinal();
        let easter_ordinal = easter.ordinal();

        if ordinal == easter_ordinal - 2 {
            return true;
        }
        if ordinal == easter_ordinal + 1 {
            return true;
        }
        if ordinal == easter_ordinal + 39 {
            return true;
        }
        if ordinal == easter_ordinal + 50 {
            return true;
        }
        if ordinal == easter_ordinal + 60 {
            return true;
        }
    }

    false
}

/// Thread-safe memoized wrapper around `easter_date`.
///
/// Easter computation is pure and deterministic for a given year. During ML
/// training (~1000 calls with the same year), this avoids redundant
/// recomputation.
fn easter_date_cached(year: i32) -> Option<NaiveDate> {
    static CACHE: LazyLock<Mutex<HashMap<i32, Option<NaiveDate>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let mut guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard.entry(year).or_insert_with(|| easter_date(year))
}

#[allow(clippy::many_single_char_names)]
fn easter_date(year: i32) -> Option<NaiveDate> {
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;

    NaiveDate::from_ymd_opt(year, month.cast_unsigned(), day.cast_unsigned())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use chrono::NaiveDate;

    use super::*;

    #[test]
    fn test_easter_2024() -> Result<()> {
        let easter = easter_date(2024).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(2024, 3, 31).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_2025() -> Result<()> {
        let easter = easter_date(2025).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(2025, 4, 20).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_2026() -> Result<()> {
        let easter = easter_date(2026).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(2026, 4, 5).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_historical_1999() -> Result<()> {
        let easter = easter_date(1999).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(1999, 4, 4).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_edge_early_march() -> Result<()> {
        let easter = easter_date(2008).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(2008, 3, 23).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_easter_edge_late_april() -> Result<()> {
        let easter = easter_date(2038).ok_or_else(|| anyhow::anyhow!("Easter date not found"))?;
        let expected =
            NaiveDate::from_ymd_opt(2038, 4, 25).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert_eq!(easter, expected);
        Ok(())
    }

    #[test]
    fn test_fixed_holidays() -> Result<()> {
        let holidays = [
            (2024, 1, 1),
            (2024, 1, 6),
            (2024, 5, 1),
            (2024, 8, 15),
            (2024, 10, 3),
            (2024, 11, 1),
            (2024, 12, 25),
            (2024, 12, 26),
        ];

        for (y, m, d) in holidays {
            let date = NaiveDate::from_ymd_opt(y, m, d)
                .ok_or_else(|| anyhow::anyhow!("Invalid date: {y}-{m}-{d}"))?;
            assert!(is_bavarian_holiday(date));
        }
        Ok(())
    }

    #[test]
    fn test_variable_holidays_2024() -> Result<()> {
        let holidays = [
            (2024, 3, 29),
            (2024, 4, 1),
            (2024, 5, 9),
            (2024, 5, 20),
            (2024, 5, 30),
        ];

        for (y, m, d) in holidays {
            let date = NaiveDate::from_ymd_opt(y, m, d)
                .ok_or_else(|| anyhow::anyhow!("Invalid date: {y}-{m}-{d}"))?;
            assert!(is_bavarian_holiday(date));
        }
        Ok(())
    }

    #[test]
    fn test_regular_weekday_not_holiday() -> Result<()> {
        let date1 =
            NaiveDate::from_ymd_opt(2024, 2, 13).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert!(!is_bavarian_holiday(date1));

        let date2 =
            NaiveDate::from_ymd_opt(2024, 7, 17).ok_or_else(|| anyhow::anyhow!("Invalid date"))?;
        assert!(!is_bavarian_holiday(date2));

        Ok(())
    }

    #[cfg(test)]
    mod proptest_tests {
        use proptest::prelude::*;

        use super::*;

        proptest! {
            #[test]
            fn easter_always_in_march_or_april(year in 1900i32..2100) {
                if let Some(easter) = easter_date(year) {
                    let month = easter.month();
                    prop_assert!(month == 3 || month == 4,
                        "Easter should be in March or April, got month {} for year {}",
                        month, year);
                }
            }

            #[test]
            fn easter_always_on_sunday(year in 1900i32..2100) {
                if let Some(easter) = easter_date(year) {
                    prop_assert_eq!(easter.weekday().num_days_from_monday(), 6,
                        "Easter should always be on Sunday for year {}", year);
                }
            }

            #[test]
            fn easter_date_is_valid(year in 1583i32..4099) {
                let result = easter_date(year);
                prop_assert!(result.is_some(),
                    "easter_date should return Some for year {}", year);
            }
        }
    }
}
