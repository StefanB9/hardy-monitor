//! Abstractions for time and side effects to enable testing.
//!
//! This module provides traits for:
//! - `Clock`: Abstracting time access for deterministic testing
//! - `Notifier`: Abstracting system notifications for testing

use std::sync::{Arc, Mutex};

use anyhow::Result;
use chrono::{DateTime, Local, Utc};

// ==================== Clock Trait ====================

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
        *self.utc_time.lock().unwrap() = time;
    }

    /// Advance the clock by a duration.
    pub fn advance(&self, duration: chrono::Duration) {
        let mut time = self.utc_time.lock().unwrap();
        *time = *time + duration;
    }
}

impl Clock for MockClock {
    fn now_utc(&self) -> DateTime<Utc> {
        *self.utc_time.lock().unwrap()
    }

    fn now_local(&self) -> DateTime<Local> {
        self.now_utc().with_timezone(&Local)
    }
}

// ==================== Notifier Trait ====================

/// Trait for abstracting system notifications.
///
/// This allows testing notification logic without actually
/// sending system notifications.
pub trait Notifier: Send + Sync {
    /// Send a notification with the given title and body.
    fn notify(&self, title: &str, body: &str) -> Result<()>;
}

/// System notifier implementation using notify-rust.
#[cfg(feature = "gui")]
#[derive(Debug, Clone, Default)]
pub struct SystemNotifier;

#[cfg(feature = "gui")]
impl Notifier for SystemNotifier {
    fn notify(&self, title: &str, body: &str) -> Result<()> {
        notify_rust::Notification::new()
            .summary(title)
            .body(body)
            .appname("Hardy Monitor")
            .show()?;
        Ok(())
    }
}

/// Combined notifier that sends to both desktop and ntfy.sh.
#[cfg(feature = "gui")]
#[derive(Debug, Clone)]
pub struct CombinedNotifier {
    ntfy_topic: Option<String>,
}

#[cfg(feature = "gui")]
impl CombinedNotifier {
    /// Create a new combined notifier.
    ///
    /// # Arguments
    /// * `ntfy_topic` - Optional ntfy.sh topic name for phone notifications
    pub fn new(ntfy_topic: Option<String>) -> Self {
        Self { ntfy_topic }
    }
}

#[cfg(feature = "gui")]
impl Notifier for CombinedNotifier {
    fn notify(&self, title: &str, body: &str) -> Result<()> {
        // Send desktop notification
        notify_rust::Notification::new()
            .summary(title)
            .body(body)
            .appname("Hardy Monitor")
            .show()?;

        // Send ntfy.sh notification if configured
        if let Some(ref topic) = self.ntfy_topic {
            let url = format!("https://ntfy.sh/{}", topic);
            let message = format!("{}\n{}", title, body);

            // Spawn async task to send ntfy notification (fire and forget)
            let url_clone = url.clone();
            let message_clone = message.clone();
            std::thread::spawn(move || {
                // Use blocking reqwest to avoid async complexity
                if let Ok(client) = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(10))
                    .build()
                {
                    let _ = client.post(&url_clone).body(message_clone).send();
                }
            });
        }

        Ok(())
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
        self.notifications.lock().unwrap().clone()
    }

    /// Get the count of notifications sent.
    pub fn notification_count(&self) -> usize {
        self.notifications.lock().unwrap().len()
    }

    /// Clear all recorded notifications.
    pub fn clear(&self) {
        self.notifications.lock().unwrap().clear();
    }

    /// Check if any notification was sent.
    pub fn was_called(&self) -> bool {
        !self.notifications.lock().unwrap().is_empty()
    }
}

impl Notifier for MockNotifier {
    fn notify(&self, title: &str, body: &str) -> Result<()> {
        self.notifications
            .lock()
            .unwrap()
            .push((title.to_string(), body.to_string()));
        Ok(())
    }
}

#[cfg(test)]
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

    #[test]
    fn test_mock_notifier_records_notifications() {
        let notifier = MockNotifier::new();

        assert!(!notifier.was_called());
        assert_eq!(notifier.notification_count(), 0);

        notifier.notify("Title 1", "Body 1").unwrap();
        assert!(notifier.was_called());
        assert_eq!(notifier.notification_count(), 1);

        notifier.notify("Title 2", "Body 2").unwrap();
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

    #[test]
    fn test_mock_notifier_clear() {
        let notifier = MockNotifier::new();

        notifier.notify("Test", "Test").unwrap();
        assert!(notifier.was_called());

        notifier.clear();
        assert!(!notifier.was_called());
        assert_eq!(notifier.notification_count(), 0);
    }
}
