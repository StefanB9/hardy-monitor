//! Occupancy history as a sorted time series with fast window statistics.
//!
//! Feature extraction reads history only through a [`HistoryView`] cut off at
//! the forecast anchor, so a feature can never see data from after the moment
//! the forecast is made.

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::db::{DataSource, OccupancyLog};

/// Measured occupancy readings, sorted by time, unique timestamps.
#[derive(Debug, Clone, Default)]
pub struct History {
    times: Vec<DateTime<Utc>>,
    values: Vec<f64>,
    /// `prefix[i]` = sum of `values[..i]`.
    prefix: Vec<f64>,
}

impl History {
    /// Builds a history from arbitrary points: sorts, keeps the last value
    /// per timestamp and drops non-finite values.
    pub fn new(mut points: Vec<(DateTime<Utc>, f64)>) -> Self {
        points.retain(|(_, v)| v.is_finite());
        points.sort_by_key(|(t, _)| *t);
        points.dedup_by(|later, earlier| {
            if later.0 == earlier.0 {
                earlier.1 = later.1;
                true
            } else {
                false
            }
        });
        let mut history = Self {
            times: Vec::with_capacity(points.len()),
            values: Vec::with_capacity(points.len()),
            prefix: Vec::with_capacity(points.len() + 1),
        };
        history.prefix.push(0.0);
        history.append_sorted(points);
        history
    }

    /// History from stored rows, keeping observed readings: measured ones
    /// and smoothed ones (a reading with a spike corrected). Interpolated
    /// gaps and boundary anchors were never observed.
    pub fn from_logs(logs: &[OccupancyLog]) -> Self {
        Self::new(
            logs.iter()
                .filter(|log| matches!(log.source, DataSource::Measured | DataSource::Smoothed))
                .map(|log| (log.timestamp, log.percentage))
                .collect(),
        )
    }

    /// Appends points newer than the current last reading; older or
    /// duplicate points are ignored.
    pub fn extend(&mut self, points: impl IntoIterator<Item = (DateTime<Utc>, f64)>) {
        let last = self.last_time();
        let mut newer: Vec<_> = points
            .into_iter()
            .filter(|(t, v)| v.is_finite() && last.is_none_or(|last| *t > last))
            .collect();
        newer.sort_by_key(|(t, _)| *t);
        newer.dedup_by_key(|(t, _)| *t);
        self.append_sorted(newer);
    }

    fn append_sorted(&mut self, points: Vec<(DateTime<Utc>, f64)>) {
        let mut running = self.prefix.last().copied().unwrap_or(0.0);
        for (t, v) in points {
            self.times.push(t);
            self.values.push(v);
            running += v;
            self.prefix.push(running);
        }
    }

    /// Number of readings.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether there are no readings.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Time of the oldest reading.
    pub fn first_time(&self) -> Option<DateTime<Utc>> {
        self.times.first().copied()
    }

    /// Time of the newest reading.
    pub fn last_time(&self) -> Option<DateTime<Utc>> {
        self.times.last().copied()
    }

    /// All readings.
    pub fn view(&self) -> HistoryView<'_> {
        HistoryView {
            times: &self.times,
            values: &self.values,
            prefix: &self.prefix,
        }
    }

    /// Readings at or before `t` — everything a forecast made at `t` may
    /// use.
    pub fn view_until(&self, t: DateTime<Utc>) -> HistoryView<'_> {
        let end = self.times.partition_point(|x| *x <= t);
        HistoryView {
            times: &self.times[..end],
            values: &self.values[..end],
            prefix: &self.prefix[..=end],
        }
    }

    /// A new history with only the readings strictly before `cutoff`.
    #[must_use]
    pub fn before(&self, cutoff: DateTime<Utc>) -> History {
        let end = self.times.partition_point(|x| *x < cutoff);
        History {
            times: self.times[..end].to_vec(),
            values: self.values[..end].to_vec(),
            prefix: self.prefix[..=end].to_vec(),
        }
    }
}

/// A borrowed, time-truncated slice of a [`History`].
#[derive(Debug, Clone, Copy)]
pub struct HistoryView<'a> {
    times: &'a [DateTime<Utc>],
    values: &'a [f64],
    prefix: &'a [f64],
}

impl<'a> HistoryView<'a> {
    /// The latest reading in the view.
    pub fn latest(&self) -> Option<(DateTime<Utc>, f64)> {
        Some((*self.times.last()?, *self.values.last()?))
    }

    /// The reading closest to `t`, if one lies within `tolerance`.
    pub fn value_near(&self, t: DateTime<Utc>, tolerance: TimeDelta) -> Option<f64> {
        let idx = self.times.partition_point(|x| *x < t);
        [idx.checked_sub(1), Some(idx)]
            .into_iter()
            .flatten()
            .filter(|&i| i < self.times.len())
            .map(|i| ((self.times[i] - t).abs(), self.values[i]))
            .filter(|(distance, _)| *distance <= tolerance)
            .min_by_key(|(distance, _)| *distance)
            .map(|(_, v)| v)
    }

    /// Mean of readings in `[start, end]`.
    pub fn mean_between(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Option<f64> {
        let (from, to) = self.range(start, end);
        let count = to.checked_sub(from).filter(|&n| n > 0)?;
        #[allow(clippy::cast_precision_loss)]
        let count = count as f64;
        Some((self.prefix[to] - self.prefix[from]) / count)
    }

    /// Least-squares slope of readings in `[start, end]`, in percentage
    /// points per hour. Needs at least two readings at distinct times.
    pub fn slope_per_hour(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Option<f64> {
        let (from, to) = self.range(start, end);
        if to.saturating_sub(from) < 2 {
            return None;
        }
        let origin = self.times[from];
        let (mut n, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in from..to {
            #[allow(clippy::cast_precision_loss)]
            let x = (self.times[i] - origin).num_seconds() as f64 / 3600.0;
            let y = self.values[i];
            n += 1.0;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let denominator = n * sxx - sx * sx;
        (denominator.abs() > f64::EPSILON).then(|| (n * sxy - sx * sy) / denominator)
    }

    /// Readings in `[start, end]`.
    pub fn between(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> impl Iterator<Item = (DateTime<Utc>, f64)> + 'a {
        let (from, to) = self.range(start, end);
        self.times[from..to]
            .iter()
            .copied()
            .zip(self.values[from..to].iter().copied())
    }

    /// All readings in the view.
    pub fn iter(&self) -> impl Iterator<Item = (DateTime<Utc>, f64)> + 'a {
        self.times.iter().copied().zip(self.values.iter().copied())
    }

    /// Number of readings in the view.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether the view holds no readings.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    fn range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> (usize, usize) {
        let from = self.times.partition_point(|x| *x < start);
        let to = self.times.partition_point(|x| *x <= end).max(from);
        (from, to)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use approx::assert_relative_eq;
    use chrono::TimeZone;
    use proptest::prelude::*;

    use super::*;

    fn at(minute: i64) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2024, 6, 17, 8, 0, 0)
            .single()
            .unwrap_or_default()
            + TimeDelta::minutes(minute)
    }

    fn series(values: &[f64]) -> History {
        History::new(
            values
                .iter()
                .enumerate()
                .map(|(i, &v)| (at(i64::try_from(i).unwrap_or(0)), v))
                .collect(),
        )
    }

    #[test]
    fn test_history_new_sorts_dedups_and_drops_non_finite() {
        let h = History::new(vec![
            (at(2), 30.0),
            (at(0), 10.0),
            (at(1), f64::NAN),
            (at(2), 35.0),
        ]);
        let points: Vec<_> = h.view().iter().collect();
        assert_eq!(points, vec![(at(0), 10.0), (at(2), 35.0)]);
    }

    #[test]
    fn test_history_extend_appends_only_newer() {
        let mut h = series(&[10.0, 20.0]);
        h.extend([(at(0), 99.0), (at(3), 40.0), (at(2), 30.0)]);
        let values: Vec<f64> = h.view().iter().map(|(_, v)| v).collect();
        assert_eq!(values, vec![10.0, 20.0, 30.0, 40.0]);
        assert_relative_eq!(
            h.view().mean_between(at(0), at(3)).unwrap_or_default(),
            25.0
        );
    }

    #[test]
    fn test_history_view_until_excludes_later_readings() -> Result<()> {
        let h = series(&[10.0, 20.0, 30.0]);
        let v = h.view_until(at(1));
        assert_eq!(v.latest(), Some((at(1), 20.0)));
        assert_eq!(v.value_near(at(2), TimeDelta::minutes(5)), Some(20.0));
        assert_relative_eq!(v.mean_between(at(0), at(10)).context("has data")?, 15.0);
        Ok(())
    }

    #[test]
    fn test_value_near_respects_tolerance() {
        let h = series(&[10.0, 20.0]);
        assert_eq!(h.view().value_near(at(5), TimeDelta::minutes(3)), None);
        assert_eq!(
            h.view().value_near(at(3), TimeDelta::minutes(3)),
            Some(20.0)
        );
    }

    #[test]
    fn test_slope_per_hour_of_linear_ramp() -> Result<()> {
        // +1 point per minute = +60 per hour.
        let h = series(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let slope = h
            .view()
            .slope_per_hour(at(0), at(4))
            .context("enough points")?;
        assert_relative_eq!(slope, 60.0, epsilon = 1e-9);
        assert_eq!(h.view().slope_per_hour(at(0), at(0)), None);
        Ok(())
    }

    #[test]
    fn test_before_cuts_strictly() {
        let h = series(&[10.0, 20.0, 30.0]);
        let b = h.before(at(2));
        assert_eq!(b.len(), 2);
        assert_eq!(b.last_time(), Some(at(1)));
    }

    #[test]
    fn test_from_logs_keeps_observations_only() {
        let log = |id, minute, source| OccupancyLog {
            id,
            timestamp: at(minute),
            percentage: 10.0,
            source,
        };
        let h = History::from_logs(&[
            log(1, 0, DataSource::Measured),
            log(2, 1, DataSource::Interpolated),
            log(3, 2, DataSource::Smoothed),
            log(4, 3, DataSource::Measured),
            log(5, 4, DataSource::Boundary),
        ]);
        // Smoothed rows are readings with a spike corrected; interpolated
        // gaps and boundary anchors were never observed.
        assert_eq!(h.len(), 3);
        assert_eq!(h.last_time(), Some(at(3)));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn mean_between_matches_naive(
            values in prop::collection::vec(0.0f64..100.0, 1..200),
            a in 0i64..220,
            b in 0i64..220,
        ) {
            let h = series(&values);
            let (start, end) = (at(a.min(b)), at(a.max(b)));
            let naive: Vec<f64> = h.view().iter()
                .filter(|(t, _)| *t >= start && *t <= end)
                .map(|(_, v)| v)
                .collect();
            let fast = h.view().mean_between(start, end);
            if naive.is_empty() {
                prop_assert!(fast.is_none());
            } else {
                #[allow(clippy::cast_precision_loss)]
                let expected = naive.iter().sum::<f64>() / naive.len() as f64;
                prop_assert!((fast.unwrap_or(f64::NAN) - expected).abs() < 1e-9);
            }
        }
    }
}
