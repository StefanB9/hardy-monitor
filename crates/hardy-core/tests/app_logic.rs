//! Integration tests for application logic using mock dependencies.
//!
//! These tests verify time-dependent behavior and notification logic
//! using `MockClock` and `MockNotifier` for deterministic, reproducible tests.
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::manual_string_new)]

use chrono::{Duration as ChronoDuration, TimeZone, Timelike, Utc};
use chrono_tz::Europe::Berlin;
use hardy_core::{
    Clock, MockClock, MockNotifier, Notifier, calculate_predictions_with_clock,
    config::{ScheduleConfig, ScheduleHours},
    db::HourlyAverage,
    find_best_time_today_with_clock,
    schedule::GymSchedule,
};

fn create_test_schedule(
    weekday_open: u32,
    weekday_close: u32,
    weekend_open: u32,
    weekend_close: u32,
) -> GymSchedule {
    let config = ScheduleConfig {
        timezone: chrono_tz::Europe::Berlin,
        weekday: ScheduleHours {
            open_hour: weekday_open,
            close_hour: weekday_close,
        },
        weekend: ScheduleHours {
            open_hour: weekend_open,
            close_hour: weekend_close,
        },
    };
    GymSchedule::new(&config)
}

#[tokio::test]
async fn test_notification_debounce_only_fires_once() {
    let notifier = MockNotifier::new();

    let threshold = 30.0;
    let mut was_below_threshold = false;
    let notifications_enabled = true;

    let percentage1 = 25.0;
    let is_below1 = percentage1 < threshold;
    if notifications_enabled && is_below1 && !was_below_threshold {
        notifier
            .notify("Test", &format!("Gym at {percentage1:.0}%"))
            .await
            .expect("notification should succeed");
    }
    was_below_threshold = is_below1;

    assert_eq!(notifier.notification_count(), 1, "First drop should notify");

    let percentage2 = 20.0;
    let is_below2 = percentage2 < threshold;
    if notifications_enabled && is_below2 && !was_below_threshold {
        notifier
            .notify("Test", &format!("Gym at {percentage2:.0}%"))
            .await
            .expect("notification should succeed");
    }
    was_below_threshold = is_below2;

    assert_eq!(
        notifier.notification_count(),
        1,
        "Second reading below should not notify again"
    );

    let percentage3 = 40.0;
    let is_below3 = percentage3 < threshold;
    if notifications_enabled && is_below3 && !was_below_threshold {
        notifier
            .notify("Test", &format!("Gym at {percentage3:.0}%"))
            .await
            .expect("notification should succeed");
    }
    was_below_threshold = is_below3;

    assert_eq!(
        notifier.notification_count(),
        1,
        "Above threshold should not notify"
    );
    assert!(
        !was_below_threshold,
        "State should reset when above threshold"
    );

    let percentage4 = 28.0;
    let is_below4 = percentage4 < threshold;
    if notifications_enabled && is_below4 && !was_below_threshold {
        notifier
            .notify("Test", &format!("Gym at {percentage4:.0}%"))
            .await
            .expect("notification should succeed");
    }

    assert_eq!(
        notifier.notification_count(),
        2,
        "New drop after recovery should notify again"
    );
}

#[tokio::test]
async fn test_notification_disabled_no_notification() {
    let notifier = MockNotifier::new();

    let threshold = 30.0;
    let mut was_below_threshold = false;
    let notifications_enabled = false;

    let percentage = 10.0;
    let is_below = percentage < threshold;
    if notifications_enabled && is_below && !was_below_threshold {
        notifier
            .notify("Test", &format!("Gym at {percentage:.0}%"))
            .await
            .expect("notification should succeed");
    }
    was_below_threshold = is_below;

    assert_eq!(
        notifier.notification_count(),
        0,
        "Disabled notifications should not fire"
    );
    assert!(was_below_threshold, "State should still update");
}

#[tokio::test]
async fn test_notification_at_exact_threshold() {
    let notifier = MockNotifier::new();

    let threshold = 30.0;
    let mut was_below_threshold = false;
    let notifications_enabled = true;

    let percentage = 30.0;
    let is_below = percentage < threshold;
    if notifications_enabled && is_below && !was_below_threshold {
        notifier
            .notify("Test", "At threshold")
            .await
            .expect("notification should succeed");
    }
    was_below_threshold = is_below;

    assert_eq!(
        notifier.notification_count(),
        0,
        "Exactly at threshold should not notify"
    );
    assert!(
        !was_below_threshold,
        "30.0 is not below 30.0, state should be false"
    );
}

#[tokio::test]
async fn test_notification_message_format() {
    let notifier = MockNotifier::new();

    notifier
        .notify("Hardy's Gym Monitor", "Gym is empty! 25%")
        .await
        .expect("notification should succeed");

    let notifications = notifier.get_notifications();
    assert_eq!(notifications.len(), 1);
    assert_eq!(notifications[0].0, "Hardy's Gym Monitor");
    assert_eq!(notifications[0].1, "Gym is empty! 25%");
}

#[test]
fn test_predictions_use_mock_clock_time() {
    // Monday 08:00 UTC is 10:00 CEST, so the next slots are local 11 and 12,
    // plotted at 09:00 and 10:00 UTC.
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 8, 0, 0).unwrap());
    let schedule = create_test_schedule(0, 24, 0, 24);

    let baseline = vec![
        HourlyAverage {
            weekday: 0,
            hour: 11,
            avg_percentage: 30.0,
            sample_count: 10,
        },
        HourlyAverage {
            weekday: 0,
            hour: 12,
            avg_percentage: 50.0,
            sample_count: 10,
        },
    ];

    let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

    assert_eq!(predictions.len(), 2);
    assert_eq!(predictions[0].0.hour(), 9);
    assert_eq!(predictions[0].1, 30.0);
    assert_eq!(predictions[1].0.hour(), 10);
    assert_eq!(predictions[1].1, 50.0);
}

#[test]
fn test_predictions_update_as_time_advances() {
    // Monday 10:00 CEST.
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 8, 0, 0).unwrap());
    let schedule = create_test_schedule(0, 24, 0, 24);

    let baseline = vec![
        HourlyAverage {
            weekday: 0,
            hour: 11,
            avg_percentage: 25.0,
            sample_count: 5,
        },
        HourlyAverage {
            weekday: 0,
            hour: 12,
            avg_percentage: 45.0,
            sample_count: 5,
        },
        HourlyAverage {
            weekday: 0,
            hour: 13,
            avg_percentage: 65.0,
            sample_count: 5,
        },
    ];

    let predictions1 = calculate_predictions_with_clock(&baseline, &schedule, &clock);
    assert_eq!(predictions1.len(), 2);
    assert_eq!(predictions1[0].1, 25.0);
    assert_eq!(predictions1[1].1, 45.0);

    clock.advance(ChronoDuration::hours(1));

    let predictions2 = calculate_predictions_with_clock(&baseline, &schedule, &clock);
    assert_eq!(predictions2.len(), 2);
    assert_eq!(predictions2[0].1, 45.0);
    assert_eq!(predictions2[1].1, 65.0);
}

#[test]
fn test_predictions_skip_closed_hours() {
    // Monday 21:00 CEST: 22:00 is the closing minute (still open), 23:00 is
    // closed.
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 19, 0, 0).unwrap());
    let schedule = create_test_schedule(6, 22, 8, 20);

    let baseline = vec![
        HourlyAverage {
            weekday: 0,
            hour: 22,
            avg_percentage: 40.0,
            sample_count: 5,
        },
        HourlyAverage {
            weekday: 0,
            hour: 23,
            avg_percentage: 30.0,
            sample_count: 5,
        },
    ];

    let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

    assert_eq!(
        predictions.len(),
        1,
        "Predictions should be filtered by schedule"
    );
    assert_eq!(predictions[0].1, 40.0);
}

#[test]
fn test_find_best_time_uses_mock_clock_day() {
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap());

    let data = vec![
        HourlyAverage {
            weekday: 0,
            hour: 8,
            avg_percentage: 60.0,
            sample_count: 5,
        },
        HourlyAverage {
            weekday: 0,
            hour: 14,
            avg_percentage: 15.0,
            sample_count: 5,
        },
        HourlyAverage {
            weekday: 1,
            hour: 10,
            avg_percentage: 5.0,
            sample_count: 5,
        },
    ];

    let result = find_best_time_today_with_clock(&data, Berlin, &clock);
    assert!(result.is_some());
    let (_, avg) = result.expect("should find best time for Monday");
    assert_eq!(avg, 15.0, "Should find best time for Monday only");
}

#[test]
fn test_analytics_at_day_boundary() {
    // Sunday 23:59 CEST; two minutes later it is Monday in the gym.
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 16, 21, 59, 0).unwrap());

    let data = vec![HourlyAverage {
        weekday: 6,
        hour: 23,
        avg_percentage: 20.0,
        sample_count: 5,
    }];

    let result = find_best_time_today_with_clock(&data, Berlin, &clock);

    clock.advance(ChronoDuration::minutes(2));

    let monday_data = vec![HourlyAverage {
        weekday: 0,
        hour: 0,
        avg_percentage: 25.0,
        sample_count: 5,
    }];

    let result2 = find_best_time_today_with_clock(&monday_data, Berlin, &clock);
    assert!(result.is_some(), "Sunday slot should match before midnight");
    assert!(result2.is_some(), "Monday slot should match after midnight");
}

#[test]
fn test_schedule_with_mock_clock() {
    // 10:00 UTC on a Monday in June is 12:00 CEST in the gym.
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap());
    let schedule = create_test_schedule(6, 22, 8, 20);

    assert!(
        schedule.is_open(&clock.now_utc()),
        "Gym should be open at 12:00 local on Monday"
    );
}

#[test]
fn test_schedule_open_close_transitions() {
    // 05:00 UTC = 07:00 CEST (open), 12:00 UTC = 14:00 (open),
    // 00:00 UTC = 02:00 (closed) — independent of the host timezone.
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 5, 0, 0).unwrap());
    let schedule = create_test_schedule(6, 22, 8, 20);

    assert!(
        schedule.is_open(&clock.now_utc()),
        "Should be open at 07:00"
    );

    clock.advance(ChronoDuration::hours(7));
    assert!(
        schedule.is_open(&clock.now_utc()),
        "Should be open at 14:00"
    );

    clock.advance(ChronoDuration::hours(12));
    assert!(
        !schedule.is_open(&clock.now_utc()),
        "Should be closed at 02:00"
    );
}

#[tokio::test]
async fn test_notifier_clear_and_reuse() {
    let notifier = MockNotifier::new();

    notifier
        .notify("Title1", "Body1")
        .await
        .expect("notification should succeed");
    notifier
        .notify("Title2", "Body2")
        .await
        .expect("notification should succeed");
    assert_eq!(notifier.notification_count(), 2);

    notifier.clear();
    assert_eq!(notifier.notification_count(), 0);
    assert!(!notifier.was_called());

    notifier
        .notify("Title3", "Body3")
        .await
        .expect("notification should succeed");
    assert_eq!(notifier.notification_count(), 1);

    let notifications = notifier.get_notifications();
    assert_eq!(notifications[0].0, "Title3");
}

#[tokio::test]
async fn test_notifier_empty_messages() {
    let notifier = MockNotifier::new();

    notifier
        .notify("", "")
        .await
        .expect("notification should succeed");
    assert!(notifier.was_called());

    let notifications = notifier.get_notifications();
    assert_eq!(notifications[0], ("".to_string(), "".to_string()));
}

#[tokio::test]
async fn test_notifier_unicode_content() {
    let notifier = MockNotifier::new();

    notifier
        .notify("🏋️ Gym Alert", "空いています！ (Empty!)")
        .await
        .expect("notification should succeed");

    let notifications = notifier.get_notifications();
    assert_eq!(notifications[0].0, "🏋️ Gym Alert");
    assert_eq!(notifications[0].1, "空いています！ (Empty!)");
}
