//! Retrying transient failures with capped exponential backoff.

use std::{future::Future, time::Duration};

use crate::error::AppError;

/// Exponential backoff schedule for retrying transient errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    max_attempts: u32,
    initial_delay: Duration,
    max_delay: Duration,
}

impl RetryPolicy {
    /// Creates a policy making at most `max_attempts` attempts in total,
    /// waiting `initial_delay` before the first retry and doubling up to
    /// `max_delay`.
    pub fn new(
        max_attempts: u32,
        initial_delay: Duration,
        max_delay: Duration,
    ) -> Result<Self, AppError> {
        if max_attempts == 0 {
            return Err(AppError::Config(
                "retry max_attempts must be at least 1".to_string(),
            ));
        }
        if initial_delay.is_zero() {
            return Err(AppError::Config(
                "retry initial_delay must be > 0".to_string(),
            ));
        }
        if initial_delay > max_delay {
            return Err(AppError::Config(format!(
                "retry initial_delay ({initial_delay:?}) must be <= max_delay ({max_delay:?})"
            )));
        }
        Ok(Self {
            max_attempts,
            initial_delay,
            max_delay,
        })
    }

    /// Total number of attempts, including the first.
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    /// Delay before retry number `retry` (1-based):
    /// `initial_delay * 2^(retry - 1)`, capped at `max_delay`.
    pub fn delay_before_retry(&self, retry: u32) -> Duration {
        2_u32
            .checked_pow(retry.saturating_sub(1))
            .and_then(|factor| self.initial_delay.checked_mul(factor))
            .map_or(self.max_delay, |delay| delay.min(self.max_delay))
    }
}

/// Runs `op` until it succeeds, fails with a non-retryable error, or the
/// policy's attempts are used up. Only errors for which
/// [`AppError::is_retryable`] holds are retried.
#[tracing::instrument(skip(policy, op))]
pub async fn retry<T, F, Fut>(
    policy: &RetryPolicy,
    operation: &str,
    mut op: F,
) -> Result<T, AppError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, AppError>>,
{
    let mut attempt = 1;
    loop {
        match op().await {
            Ok(value) => return Ok(value),
            Err(e) if e.is_retryable() && attempt < policy.max_attempts => {
                let delay = policy.delay_before_retry(attempt);
                tracing::warn!(
                    attempt,
                    max_attempts = policy.max_attempts,
                    delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                    error = %e,
                    "transient failure, retrying"
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use anyhow::{Context, Result};
    use proptest::prelude::*;

    use super::*;
    use crate::error::NetworkErrorKind;

    const TEST_TIMEOUT: Duration = Duration::from_secs(600);

    fn policy(max_attempts: u32) -> Result<RetryPolicy> {
        RetryPolicy::new(
            max_attempts,
            Duration::from_secs(1),
            Duration::from_secs(10),
        )
        .context("valid policy")
    }

    fn timeout_error() -> AppError {
        AppError::Network {
            message: "timed out".to_string(),
            kind: NetworkErrorKind::Timeout,
        }
    }

    #[test]
    fn test_retry_policy_rejects_zero_attempts() {
        let result = RetryPolicy::new(0, Duration::from_secs(1), Duration::from_secs(2));
        assert!(matches!(result, Err(AppError::Config(_))));
    }

    #[test]
    fn test_retry_policy_rejects_zero_initial_delay() {
        let result = RetryPolicy::new(3, Duration::ZERO, Duration::from_secs(2));
        assert!(matches!(result, Err(AppError::Config(_))));
    }

    #[test]
    fn test_retry_policy_rejects_initial_above_max() {
        let result = RetryPolicy::new(3, Duration::from_secs(5), Duration::from_secs(2));
        assert!(matches!(result, Err(AppError::Config(_))));
    }

    #[test]
    fn test_retry_policy_delay_doubles_then_caps() -> Result<()> {
        let p = RetryPolicy::new(10, Duration::from_secs(2), Duration::from_secs(10))?;
        let delays: Vec<u64> = (1..=5).map(|r| p.delay_before_retry(r).as_secs()).collect();
        assert_eq!(delays, vec![2, 4, 8, 10, 10]);
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn test_retry_succeeds_after_transient_failures() -> Result<()> {
        let calls = AtomicU32::new(0);
        let started = tokio::time::Instant::now();

        let result = tokio::time::timeout(
            TEST_TIMEOUT,
            retry(&policy(3)?, "test", || async {
                if calls.fetch_add(1, Ordering::SeqCst) < 2 {
                    Err(timeout_error())
                } else {
                    Ok(42)
                }
            }),
        )
        .await
        .context("retry should finish")?;

        assert_eq!(result.context("third attempt succeeds")?, 42);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(started.elapsed(), Duration::from_secs(1 + 2));
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn test_retry_does_not_retry_non_retryable_errors() -> Result<()> {
        let calls = AtomicU32::new(0);

        let result: Result<(), AppError> = tokio::time::timeout(
            TEST_TIMEOUT,
            retry(&policy(5)?, "test", || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(AppError::validation("bad value"))
            }),
        )
        .await
        .context("retry should finish")?;

        assert!(matches!(result, Err(AppError::Validation(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn test_retry_gives_up_after_max_attempts() -> Result<()> {
        let calls = AtomicU32::new(0);

        let result: Result<(), AppError> = tokio::time::timeout(
            TEST_TIMEOUT,
            retry(&policy(3)?, "test", || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(timeout_error())
            }),
        )
        .await
        .context("retry should finish")?;

        assert!(result.is_err_and(|e| e.is_retryable()));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn retry_delay_is_monotonic_and_capped(
            initial_ms in 1u64..10_000,
            extra_ms in 0u64..100_000,
            retry_n in 1u32..64,
        ) {
            let initial = Duration::from_millis(initial_ms);
            let max = Duration::from_millis(initial_ms + extra_ms);
            let p = RetryPolicy::new(64, initial, max)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let this = p.delay_before_retry(retry_n);
            let next = p.delay_before_retry(retry_n + 1);
            prop_assert!(this >= initial);
            prop_assert!(this <= max);
            prop_assert!(next >= this);
        }
    }
}
