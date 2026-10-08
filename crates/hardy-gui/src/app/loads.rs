//! Bookkeeping for database loads: which result is current, how many loads
//! are pending, and the errors they raised.

use std::time::{Duration, Instant};

use hardy_core::error::AppError;

/// Identifies one load of a kind that may be superseded by a newer one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RequestId(u64);

/// Hands out request numbers so only the newest load's result is applied;
/// a slow load for an old range must not overwrite a newer one.
#[derive(Debug, Default)]
pub(crate) struct LatestOnly {
    issued: u64,
}

impl LatestOnly {
    /// Numbers a new load, superseding all earlier ones.
    pub(crate) fn issue(&mut self) -> RequestId {
        self.issued += 1;
        RequestId(self.issued)
    }

    /// Whether `id` is the newest load issued.
    pub(crate) fn is_current(&self, id: RequestId) -> bool {
        id.0 == self.issued
    }
}

/// Database loads still in flight, for the "Updating…" status.
#[derive(Debug, Default)]
pub(crate) struct PendingLoads {
    count: u32,
    /// When the count last rose from zero.
    since: Option<Instant>,
}

impl PendingLoads {
    /// Loads quicker than this never show the status, so it does not
    /// flicker.
    pub(crate) const DELAY: Duration = Duration::from_millis(200);

    pub(crate) fn begin(&mut self, now: Instant) {
        if self.count == 0 {
            self.since = Some(now);
        }
        self.count += 1;
    }

    pub(crate) fn finish(&mut self) {
        self.count = self.count.saturating_sub(1);
        if self.count == 0 {
            self.since = None;
        }
    }

    /// Whether loads have been pending for at least [`Self::DELAY`].
    pub(crate) fn is_visible(&self, now: Instant) -> bool {
        self.since
            .is_some_and(|since| now.saturating_duration_since(since) >= Self::DELAY)
    }
}

/// Where an error came from; a success from the same source clears it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorSource {
    LatestReading,
    Chart,
    Week,
    Insights,
    Accuracy,
    ForecastHistory,
    ModelStatus,
    Model,
    AlertSettings,
    Export,
    /// An invalid custom chart range.
    ChartRangeInput,
    /// Invalid repair dates.
    RepairDatesInput,
}

/// The active error of each source, oldest raised first.
#[derive(Debug, Default)]
pub(crate) struct Errors {
    active: Vec<(ErrorSource, AppError)>,
}

impl Errors {
    /// Sets `source`'s error, making it the newest.
    pub(crate) fn raise(&mut self, source: ErrorSource, error: AppError) {
        self.clear(source);
        self.active.push((source, error));
    }

    pub(crate) fn clear(&mut self, source: ErrorSource) {
        self.active.retain(|(s, _)| *s != source);
    }

    pub(crate) fn clear_all(&mut self) {
        self.active.clear();
    }

    /// The value of a successful result (clearing `source`'s error), or
    /// `None` after raising the error.
    pub(crate) fn record<T>(
        &mut self,
        source: ErrorSource,
        result: Result<T, AppError>,
    ) -> Option<T> {
        match result {
            Ok(value) => {
                self.clear(source);
                Some(value)
            }
            Err(error) => {
                self.raise(source, error);
                None
            }
        }
    }

    /// The most recently raised error still active.
    pub(crate) fn latest(&self) -> Option<&AppError> {
        self.active.last().map(|(_, error)| error)
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use anyhow::{Context, Result};
    use hardy_core::error::AppError;

    use super::*;

    #[test]
    fn test_latest_only_accepts_only_the_newest_request() {
        let mut requests = LatestOnly::default();
        let old = requests.issue();
        let new = requests.issue();
        // The newer load answers first, then the older one.
        assert!(requests.is_current(new));
        assert!(!requests.is_current(old));
    }

    #[test]
    fn test_pending_loads_stays_pending_until_the_last_finishes() {
        let start = Instant::now();
        let later = start + PendingLoads::DELAY;
        let mut pending = PendingLoads::default();
        pending.begin(start);
        pending.begin(start + Duration::from_millis(50));
        pending.finish();
        assert!(pending.is_visible(later));
        pending.finish();
        assert!(!pending.is_visible(later));
    }

    #[test]
    fn test_pending_loads_hidden_before_the_delay() {
        let start = Instant::now();
        let mut pending = PendingLoads::default();
        pending.begin(start);
        assert!(!pending.is_visible(start + Duration::from_millis(10)));
        assert!(pending.is_visible(start + PendingLoads::DELAY));
    }

    #[test]
    fn test_pending_loads_delay_restarts_after_idle() {
        let start = Instant::now();
        let mut pending = PendingLoads::default();
        pending.begin(start);
        pending.finish();
        let again = start + Duration::from_secs(60);
        pending.begin(again);
        assert!(!pending.is_visible(again + Duration::from_millis(10)));
    }

    #[test]
    fn test_pending_loads_finish_never_underflows() {
        let now = Instant::now();
        let mut pending = PendingLoads::default();
        pending.finish();
        pending.begin(now);
        assert!(pending.is_visible(now + PendingLoads::DELAY));
    }

    #[test]
    fn test_errors_success_clears_only_its_source() -> Result<()> {
        let mut errors = Errors::default();
        errors.raise(ErrorSource::Chart, AppError::validation("chart broke"));
        errors.clear(ErrorSource::LatestReading);
        let shown = errors.latest().context("chart error still shown")?;
        assert!(shown.to_string().contains("chart broke"), "{shown}");
        errors.clear(ErrorSource::Chart);
        assert!(errors.latest().is_none());
        Ok(())
    }

    #[test]
    fn test_errors_show_the_most_recently_raised() -> Result<()> {
        let mut errors = Errors::default();
        errors.raise(ErrorSource::Chart, AppError::validation("chart"));
        errors.raise(ErrorSource::Week, AppError::validation("week"));
        assert!(
            errors
                .latest()
                .context("shown")?
                .to_string()
                .contains("week")
        );
        // Raising again moves the chart error to the front.
        errors.raise(ErrorSource::Chart, AppError::validation("chart again"));
        assert!(
            errors
                .latest()
                .context("shown")?
                .to_string()
                .contains("chart again")
        );
        errors.clear(ErrorSource::Chart);
        assert!(
            errors
                .latest()
                .context("shown")?
                .to_string()
                .contains("week")
        );
        Ok(())
    }

    #[test]
    fn test_errors_clear_all() {
        let mut errors = Errors::default();
        errors.raise(ErrorSource::Chart, AppError::validation("chart"));
        errors.raise(ErrorSource::Export, AppError::validation("export"));
        errors.clear_all();
        assert!(errors.latest().is_none());
    }

    #[test]
    fn test_errors_record_takes_the_value_or_raises() {
        let mut errors = Errors::default();
        let failed: Result<u8, AppError> = Err(AppError::validation("no"));
        assert_eq!(errors.record(ErrorSource::Accuracy, failed), None);
        assert!(errors.latest().is_some());
        assert_eq!(errors.record(ErrorSource::Accuracy, Ok(7)), Some(7));
        assert!(errors.latest().is_none());
    }
}
