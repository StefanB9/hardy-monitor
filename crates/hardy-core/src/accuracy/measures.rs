//! Accuracy measures of scored forecasts: running sums that can be added
//! up from single forecasts or from the database, and the measures
//! computed from them.

/// Weight of the distance outside the band in the interval score: 2 / α
/// for the 80 % band (α = 0.2).
const OUTSIDE_WEIGHT: f64 = 10.0;

/// One forecast and the reading it is scored against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pair {
    pub actual: f64,
    pub forecast: f64,
    /// The forecast's 80 % band, if it had one.
    pub band: Option<(f64, f64)>,
    /// Plain averages for the same target: the slot's mean reading on the
    /// same weekday over the four weeks before.
    pub baseline: Option<f64>,
    /// The reading one week before the target.
    pub last_week: Option<f64>,
}

/// The interval score of an 80 % band `[low, high]` for `actual`: its
/// width plus a penalty for the distance outside it.
pub fn interval_score(low: f64, high: f64, actual: f64) -> f64 {
    let outside = (low - actual).max(0.0) + (actual - high).max(0.0);
    (high - low) + OUTSIDE_WEIGHT * outside
}

/// Running sums of scored forecasts. Counts are `i64` like the database's.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ErrorSums {
    pub count: i64,
    /// Sum of absolute errors.
    pub abs: f64,
    pub squared: f64,
    /// Sum of forecast − actual.
    pub signed: f64,
    /// Sum of the readings (for WAPE).
    pub actual: f64,
    /// Forecasts above the reading.
    pub over: i64,
    pub within_5: i64,
    pub within_10: i64,
    /// Targets with a baseline, the forecast's and the baseline's absolute
    /// errors on them.
    pub baseline_count: i64,
    pub abs_on_baseline: f64,
    pub baseline_abs: f64,
    /// Targets with a reading a week before, the forecast's and that
    /// reading's absolute errors on them.
    pub last_week_count: i64,
    pub abs_on_last_week: f64,
    pub last_week_abs: f64,
    /// Forecasts with a band, those whose reading was inside it, and the
    /// sums of the bands' widths and interval scores.
    pub banded: i64,
    pub covered: i64,
    pub width: f64,
    pub interval_score: f64,
}

/// The accuracy measures of a group of forecasts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measures {
    pub count: i64,
    /// Mean absolute error in points.
    pub mae: f64,
    pub rmse: f64,
    /// Mean of forecast − actual: positive when forecasts run high.
    pub bias: f64,
    /// Absolute errors relative to the readings' total.
    pub wape: Option<f64>,
    /// MAE relative to the reading a week before on the same targets.
    pub mase: Option<f64>,
    /// How much lower the error is than plain averages' (0.25 = 25 %).
    pub skill: Option<f64>,
    /// Share of readings inside the band.
    pub coverage: Option<f64>,
    /// Mean band width in points.
    pub width: Option<f64>,
    /// Mean interval score of the band.
    pub interval_score: Option<f64>,
    /// Shares of forecasts within 5 and 10 points of the reading.
    pub within_5: f64,
    pub within_10: f64,
    /// Share of forecasts above the reading.
    pub over_share: f64,
}

impl ErrorSums {
    /// Adds one scored forecast.
    pub fn add(&mut self, pair: &Pair) {
        let error = pair.forecast - pair.actual;
        let abs = error.abs();
        self.count += 1;
        self.abs += abs;
        self.squared += error * error;
        self.signed += error;
        self.actual += pair.actual;
        self.over += i64::from(error > 0.0);
        self.within_5 += i64::from(abs <= 5.0);
        self.within_10 += i64::from(abs <= 10.0);
        if let Some(baseline) = pair.baseline {
            self.baseline_count += 1;
            self.abs_on_baseline += abs;
            self.baseline_abs += (baseline - pair.actual).abs();
        }
        if let Some(last_week) = pair.last_week {
            self.last_week_count += 1;
            self.abs_on_last_week += abs;
            self.last_week_abs += (last_week - pair.actual).abs();
        }
        if let Some((low, high)) = pair.band {
            self.banded += 1;
            self.covered += i64::from((low..=high).contains(&pair.actual));
            self.width += high - low;
            self.interval_score += interval_score(low, high, pair.actual);
        }
    }

    /// Adds another group's sums.
    pub fn merge(&mut self, other: &ErrorSums) {
        self.count += other.count;
        self.abs += other.abs;
        self.squared += other.squared;
        self.signed += other.signed;
        self.actual += other.actual;
        self.over += other.over;
        self.within_5 += other.within_5;
        self.within_10 += other.within_10;
        self.baseline_count += other.baseline_count;
        self.abs_on_baseline += other.abs_on_baseline;
        self.baseline_abs += other.baseline_abs;
        self.last_week_count += other.last_week_count;
        self.abs_on_last_week += other.abs_on_last_week;
        self.last_week_abs += other.last_week_abs;
        self.banded += other.banded;
        self.covered += other.covered;
        self.width += other.width;
        self.interval_score += other.interval_score;
    }

    /// The measures; `None` when nothing was scored.
    #[allow(clippy::cast_precision_loss)]
    pub fn measures(&self) -> Option<Measures> {
        let n = (self.count > 0).then_some(self.count as f64)?;
        let ratio = |num: f64, den: f64| (den > 0.0).then_some(num / den);
        let banded = (self.banded > 0).then_some(self.banded as f64);
        Some(Measures {
            count: self.count,
            mae: self.abs / n,
            rmse: (self.squared / n).sqrt(),
            bias: self.signed / n,
            wape: ratio(self.abs, self.actual),
            mase: ratio(self.abs_on_last_week, self.last_week_abs),
            skill: ratio(self.abs_on_baseline, self.baseline_abs).map(|r| 1.0 - r),
            coverage: banded.map(|b| self.covered as f64 / b),
            width: banded.map(|b| self.width / b),
            interval_score: banded.map(|b| self.interval_score / b),
            within_5: self.within_5 as f64 / n,
            within_10: self.within_10 as f64 / n,
            over_share: self.over as f64 / n,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    fn pair(actual: f64, forecast: f64) -> Pair {
        Pair {
            actual,
            forecast,
            band: None,
            baseline: None,
            last_week: None,
        }
    }

    fn sums(pairs: &[Pair]) -> ErrorSums {
        let mut sums = ErrorSums::default();
        for p in pairs {
            sums.add(p);
        }
        sums
    }

    #[test]
    fn test_measures_of_nothing_is_none() {
        assert_eq!(ErrorSums::default().measures(), None);
    }

    #[test]
    fn test_measures_point_errors() -> Result<()> {
        // Errors (forecast - actual): +2, -4, +6.
        let m = sums(&[pair(10.0, 12.0), pair(20.0, 16.0), pair(30.0, 36.0)])
            .measures()
            .context("measures")?;
        assert_eq!(m.count, 3);
        assert_relative_eq!(m.mae, 4.0);
        assert_relative_eq!(m.rmse, (56.0_f64 / 3.0).sqrt());
        assert_relative_eq!(m.bias, 4.0 / 3.0);
        assert_relative_eq!(m.wape.context("wape")?, 12.0 / 60.0);
        assert_relative_eq!(m.over_share, 2.0 / 3.0);
        assert_relative_eq!(m.within_5, 2.0 / 3.0);
        assert_relative_eq!(m.within_10, 1.0);
        assert_eq!(m.skill, None);
        assert_eq!(m.mase, None);
        assert_eq!(m.coverage, None);
        Ok(())
    }

    #[test]
    fn test_skill_and_mase_use_only_targets_with_a_reference() -> Result<()> {
        let m = sums(&[
            Pair {
                baseline: Some(14.0),
                last_week: Some(18.0),
                ..pair(10.0, 12.0)
            },
            // No reference: counts for the MAE only.
            pair(20.0, 40.0),
        ])
        .measures()
        .context("measures")?;
        // Forecast error 2 against baseline error 4 and last-week error 8.
        assert_relative_eq!(m.skill.context("skill")?, 0.5);
        assert_relative_eq!(m.mase.context("mase")?, 0.25);
        assert_relative_eq!(m.mae, 11.0);
        Ok(())
    }

    #[test]
    fn test_band_measures() -> Result<()> {
        let m = sums(&[
            Pair {
                band: Some((5.0, 15.0)),
                ..pair(10.0, 10.0)
            },
            // 3 above the band: score = 10 + 10 × 3.
            Pair {
                band: Some((5.0, 15.0)),
                ..pair(18.0, 10.0)
            },
            pair(10.0, 10.0),
        ])
        .measures()
        .context("measures")?;
        assert_relative_eq!(m.coverage.context("coverage")?, 0.5);
        assert_relative_eq!(m.width.context("width")?, 10.0);
        assert_relative_eq!(m.interval_score.context("score")?, 25.0);
        Ok(())
    }

    #[test]
    fn test_interval_score_penalises_below_and_above() {
        assert_relative_eq!(interval_score(10.0, 20.0, 15.0), 10.0);
        assert_relative_eq!(interval_score(10.0, 20.0, 8.0), 30.0);
        assert_relative_eq!(interval_score(10.0, 20.0, 21.0), 20.0);
    }

    #[test]
    fn test_skill_without_baseline_error_is_none() -> Result<()> {
        let m = sums(&[Pair {
            baseline: Some(10.0),
            ..pair(10.0, 12.0)
        }])
        .measures()
        .context("measures")?;
        assert_eq!(m.skill, None);
        Ok(())
    }

    fn arb_pair() -> impl Strategy<Value = Pair> {
        (
            0.0f64..100.0,
            0.0f64..100.0,
            proptest::option::of((0.0f64..20.0, 0.0f64..20.0)),
            proptest::option::of(0.0f64..100.0),
            proptest::option::of(0.0f64..100.0),
        )
            .prop_map(|(actual, forecast, band, baseline, last_week)| Pair {
                actual,
                forecast,
                band: band.map(|(below, above)| (forecast - below, forecast + above)),
                baseline,
                last_week,
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn merging_sums_equals_adding_every_pair(
            left in proptest::collection::vec(arb_pair(), 0..30),
            right in proptest::collection::vec(arb_pair(), 0..30),
        ) {
            let mut merged = sums(&left);
            merged.merge(&sums(&right));
            let all: Vec<Pair> = left.iter().chain(&right).copied().collect();
            let whole = sums(&all);
            prop_assert_eq!(merged.count, whole.count);
            prop_assert!((merged.abs - whole.abs).abs() < 1e-6);
            prop_assert!((merged.interval_score - whole.interval_score).abs() < 1e-6);
            prop_assert_eq!(merged.covered, whole.covered);
        }

        #[test]
        fn measures_stay_in_their_ranges(pairs in proptest::collection::vec(arb_pair(), 1..60)) {
            let m = sums(&pairs).measures();
            prop_assert!(m.is_some());
            if let Some(m) = m {
                prop_assert!(m.mae >= 0.0 && m.rmse + 1e-9 >= m.mae);
                prop_assert!(m.bias.abs() <= m.mae + 1e-9);
                prop_assert!((0.0..=1.0).contains(&m.within_5));
                prop_assert!(m.within_5 <= m.within_10);
                prop_assert!((0.0..=1.0).contains(&m.over_share));
                let coverage_ok = m.coverage.is_none_or(|c| (0.0..=1.0).contains(&c));
                prop_assert!(coverage_ok);
                let score_ok = match (m.interval_score, m.width) {
                    (Some(s), Some(w)) => s + 1e-9 >= w,
                    (None, None) => true,
                    _ => false,
                };
                prop_assert!(score_ok);
            }
        }
    }
}
