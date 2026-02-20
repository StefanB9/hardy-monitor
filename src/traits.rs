//! Abstractions for time and side effects to enable testing.
//!
//! This module provides traits for:
//! - `Clock`: Abstracting time access for deterministic testing
//! - `Notifier`: Abstracting system notifications for testing

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::Result;
use chrono::{DateTime, Local, Utc};
use futures::future::BoxFuture;
use tracing::warn;


/// Trait for abstracting time access.
///
/// This allows injecting mock clocks during testing to create
/// deterministic, reproducible tests for time-dependent logic.
pub trait Clock: Send + Sync {
    /// Get the current time in UTC.
    fn now_utc(&self) -> DateTime<Utc>;

    /// Get the current time in the local timezone.
    fn now_local(&self) -> DateTime<Local>;
}

/// System clock implementation using real time.
#[derive(Debug, Clone, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_utc(&self) -> DateTime<Utc> {
        Utc::now()
    }

    fn now_local(&self) -> DateTime<Local> {
        Local::now()
    }
}

/// Mock clock for testing with controllable time.
#[derive(Debug, Clone)]
pub struct MockClock {
    utc_time: Arc<Mutex<DateTime<Utc>>>,
}

impl MockClock {
    /// Create a new mock clock set to the given UTC time.
    pub fn new(time: DateTime<Utc>) -> Self {
        Self {
            utc_time: Arc::new(Mutex::new(time)),
        }
    }

    /// Set the mock clock to a new time.
    pub fn set_time(&self, time: DateTime<Utc>) {
        *self
            .utc_time
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = time;
    }

    /// Advance the clock by a duration.
    pub fn advance(&self, duration: chrono::Duration) {
        let mut time = self
            .utc_time
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *time += duration;
    }
}

impl Clock for MockClock {
    fn now_utc(&self) -> DateTime<Utc> {
        *self
            .utc_time
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn now_local(&self) -> DateTime<Local> {
        self.now_utc().with_timezone(&Local)
    }
}


/// Trait for abstracting system notifications.
///
/// This allows testing notification logic without actually
/// sending system notifications.
pub trait Notifier: Send + Sync {
    /// Send a notification with the given title and body.
    ///
    /// The returned future is tied to the lifetime of `self` but **not** to
    /// the string arguments — implementors must clone any strings they need
    /// before returning the future.
    fn notify<'s>(&'s self, title: &str, body: &str) -> BoxFuture<'s, Result<()>>;
}

/// System notifier implementation using notify-rust.
#[cfg(feature = "gui")]
#[derive(Debug, Clone, Default)]
pub struct SystemNotifier;

#[cfg(feature = "gui")]
impl Notifier for SystemNotifier {
    fn notify<'s>(&'s self, title: &str, body: &str) -> BoxFuture<'s, Result<()>> {
        let title = title.to_string();
        let body = body.to_string();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                notify_rust::Notification::new()
                    .summary(&title)
                    .body(&body)
                    .appname("Hardy Monitor")
                    .show()
            })
            .await
            .map_err(|e| anyhow::anyhow!("desktop notification task panicked: {e}"))??;
            Ok(())
        })
    }
}

/// Combined notifier that sends to both desktop and ntfy.sh.
#[cfg(feature = "gui")]
#[derive(Debug, Clone)]
pub struct CombinedNotifier {
    ntfy_topic: Option<String>,
    ntfy_server: String,
    /// Pre-built HTTP client shared across all notification calls.
    /// `reqwest::Client` is backed by an `Arc`; cloning it is a pointer copy.
    http_client: reqwest::Client,
}

#[cfg(feature = "gui")]
impl CombinedNotifier {
    /// Create a new combined notifier.
    ///
    /// # Arguments
    /// * `ntfy_topic` - Optional ntfy.sh topic name for phone notifications
    /// * `ntfy_server` - Base URL of the ntfy server (e.g. `"https://ntfy.sh"`)
    pub fn new(ntfy_topic: Option<String>, ntfy_server: String) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|e| {
                warn!(error = %e, "failed to build HTTP client with custom timeout; using default");
                reqwest::Client::new()
            });
        Self {
            ntfy_topic,
            ntfy_server,
            http_client,
        }
    }
}

#[cfg(feature = "gui")]
impl Notifier for CombinedNotifier {
    fn notify<'s>(&'s self, title: &str, body: &str) -> BoxFuture<'s, Result<()>> {
        let title = title.to_string();
        let body = body.to_string();
        Box::pin(async move {
            let t = title.clone();
            let b = body.clone();
            match tokio::task::spawn_blocking(move || {
                notify_rust::Notification::new()
                    .summary(&t)
                    .body(&b)
                    .appname("Hardy Monitor")
                    .show()
            })
            .await
            {
                Err(e) => warn!(error = %e, "desktop notification task panicked"),
                Ok(Err(e)) => warn!(error = %e, "desktop notification failed"),
                Ok(Ok(())) => {}
            }

            if let Some(ref topic) = self.ntfy_topic {
                let url = format!("{}/{topic}", self.ntfy_server);
                let message = format!("{title}\n{body}");
                match self.http_client.post(&url).body(message).send().await {
                    Err(e) => warn!(error = %e, %url, "ntfy send failed"),
                    Ok(resp) if !resp.status().is_success() => {
                        warn!(status = %resp.status(), %url, "ntfy returned error status");
                    }
                    Ok(_) => {}
                }
            }

            Ok(())
        })
    }
}

/// Mock notifier for testing that records all notifications.
#[derive(Debug, Clone, Default)]
pub struct MockNotifier {
    notifications: Arc<Mutex<Vec<(String, String)>>>,
}

impl MockNotifier {
    /// Create a new mock notifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get all notifications that have been sent.
    pub fn get_notifications(&self) -> Vec<(String, String)> {
        self.notifications
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Get the count of notifications sent.
    pub fn notification_count(&self) -> usize {
        self.notifications
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Clear all recorded notifications.
    pub fn clear(&self) {
        self.notifications
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    /// Check if any notification was sent.
    pub fn was_called(&self) -> bool {
        !self
            .notifications
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }
}

impl Notifier for MockNotifier {
    fn notify<'s>(&'s self, title: &str, body: &str) -> BoxFuture<'s, Result<()>> {
        let title = title.to_string();
        let body = body.to_string();
        let notifications = self.notifications.clone();
        Box::pin(async move {
            notifications
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((title, body));
            Ok(())
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] 
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn test_system_clock_returns_current_time() {
        let clock = SystemClock;
        let before = Utc::now();
        let clock_time = clock.now_utc();
        let after = Utc::now();

        assert!(clock_time >= before);
        assert!(clock_time <= after);
    }

    #[test]
    fn test_mock_clock_returns_set_time() {
        let fixed_time = Utc.with_ymd_and_hms(2024, 6, 15, 14, 30, 0).unwrap();
        let clock = MockClock::new(fixed_time);

        assert_eq!(clock.now_utc(), fixed_time);
    }

    #[test]
    fn test_mock_clock_can_be_updated() {
        let time1 = Utc.with_ymd_and_hms(2024, 6, 15, 10, 0, 0).unwrap();
        let time2 = Utc.with_ymd_and_hms(2024, 6, 15, 14, 0, 0).unwrap();

        let clock = MockClock::new(time1);
        assert_eq!(clock.now_utc(), time1);

        clock.set_time(time2);
        assert_eq!(clock.now_utc(), time2);
    }

    #[test]
    fn test_mock_clock_advance() {
        let start = Utc.with_ymd_and_hms(2024, 6, 15, 10, 0, 0).unwrap();
        let clock = MockClock::new(start);

        clock.advance(chrono::Duration::hours(2));

        let expected = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
        assert_eq!(clock.now_utc(), expected);
    }

    #[tokio::test]
    async fn test_mock_notifier_records_notifications() {
        let notifier = MockNotifier::new();

        assert!(!notifier.was_called());
        assert_eq!(notifier.notification_count(), 0);

        notifier.notify("Title 1", "Body 1").await.unwrap();
        assert!(notifier.was_called());
        assert_eq!(notifier.notification_count(), 1);

        notifier.notify("Title 2", "Body 2").await.unwrap();
        assert_eq!(notifier.notification_count(), 2);

        let notifications = notifier.get_notifications();
        assert_eq!(
            notifications[0],
            ("Title 1".to_string(), "Body 1".to_string())
        );
        assert_eq!(
            notifications[1],
            ("Title 2".to_string(), "Body 2".to_string())
        );
    }

    #[tokio::test]
    async fn test_mock_notifier_clear() {
        let notifier = MockNotifier::new();

        notifier.notify("Test", "Test").await.unwrap();
        assert!(notifier.was_called());

        notifier.clear();
        assert!(!notifier.was_called());
        assert_eq!(notifier.notification_count(), 0);
    }
}
