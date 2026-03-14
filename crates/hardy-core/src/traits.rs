//! Abstractions for time and side effects to enable testing.
//!
//! This module provides traits for:
//! - `Clock`: Abstracting time access for deterministic testing
//! - `Notifier`: Abstracting system notifications for testing

use std::sync::{Arc, Mutex};

use anyhow::Result;
use chrono::{DateTime, Local, Utc};
use futures::future::BoxFuture;

pub trait Clock: Send + Sync {
    fn now_utc(&self) -> DateTime<Utc>;

    fn now_local(&self) -> DateTime<Local>;
}

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

#[derive(Debug, Clone)]
pub struct MockClock {
    utc_time: Arc<Mutex<DateTime<Utc>>>,
}

impl MockClock {
    pub fn new(time: DateTime<Utc>) -> Self {
        Self {
            utc_time: Arc::new(Mutex::new(time)),
        }
    }

    pub fn set_time(&self, time: DateTime<Utc>) {
        *self
            .utc_time
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = time;
    }

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

pub trait Notifier: Send + Sync {
    fn notify<'s>(&'s self, title: &str, body: &str) -> BoxFuture<'s, Result<()>>;
}

#[derive(Debug, Clone, Default)]
pub struct MockNotifier {
    notifications: Arc<Mutex<Vec<(String, String)>>>,
}

impl MockNotifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_notifications(&self) -> Vec<(String, String)> {
        self.notifications
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn notification_count(&self) -> usize {
        self.notifications
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    pub fn clear(&self) {
        self.notifications
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

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
    async fn test_mock_notifier_records_notifications() -> Result<()> {
        let notifier = MockNotifier::new();

        assert!(!notifier.was_called());
        assert_eq!(notifier.notification_count(), 0);

        notifier.notify("Title 1", "Body 1").await?;
        assert!(notifier.was_called());
        assert_eq!(notifier.notification_count(), 1);

        notifier.notify("Title 2", "Body 2").await?;
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

        Ok(())
    }

    #[tokio::test]
    async fn test_mock_notifier_clear() -> Result<()> {
        let notifier = MockNotifier::new();

        notifier.notify("Test", "Test").await?;
        assert!(notifier.was_called());

        notifier.clear();
        assert!(!notifier.was_called());
        assert_eq!(notifier.notification_count(), 0);
        Ok(())
    }
}
