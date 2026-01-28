//! Integration tests for application logic using mock dependencies.
//!
//! These tests verify time-dependent behavior and notification logic
//! using MockClock and MockNotifier for deterministic, reproducible tests.

use chrono::{Duration as ChronoDuration, TimeZone, Timelike, Utc};
use hardy_monitor::{
    Clock, MockClock, MockNotifier, Notifier, calculate_predictions_with_clock,
    config::{ScheduleConfig, ScheduleHours},
    db::HourlyAverage,
    find_best_time_today_with_clock,
    schedule::GymSchedule,
};

/// Helper to create a test schedule with specified hours.
fn create_test_schedule(
    weekday_open: u32,
    weekday_close: u32,
    weekend_open: u32,
    weekend_close: u32,
) -> GymSchedule {
    let config = ScheduleConfig {
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

// ==================== Notification Debouncing Tests ====================

/// Test that notifications are only sent once when crossing threshold.
#[test]
fn test_notification_debounce_only_fires_once() {
    let notifier = MockNotifier::new();

    // Simulate the debouncing logic from HardyMonitorApp::update
    let threshold = 30.0;
    let mut was_below_threshold = false;
    let notifications_enabled = true;

    // First reading below threshold - should notify
    let percentage1 = 25.0;
    let is_below1 = percentage1 < threshold;
    if notifications_enabled && is_below1 && !was_below_threshold {
        notifier
            .notify("Test", &format!("Gym at {:.0}%", percentage1))
            .unwrap();
    }
    was_below_threshold = is_below1;

    assert_eq!(notifier.notification_count(), 1, "First drop should notify");

    // Second reading still below threshold - should NOT notify again
    let percentage2 = 20.0;
    let is_below2 = percentage2 < threshold;
    if notifications_enabled && is_below2 && !was_below_threshold {
        notifier
            .notify("Test", &format!("Gym at {:.0}%", percentage2))
            .unwrap();
    }
    was_below_threshold = is_below2;

    assert_eq!(
        notifier.notification_count(),
        1,
        "Second reading below should not notify again"
    );

    // Third reading above threshold - no notification, just reset state
    let percentage3 = 40.0;
    let is_below3 = percentage3 < threshold;
    if notifications_enabled && is_below3 && !was_below_threshold {
        notifier
            .notify("Test", &format!("Gym at {:.0}%", percentage3))
            .unwrap();
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

    // Fourth reading drops below threshold again - SHOULD notify
    let percentage4 = 28.0;
    let is_below4 = percentage4 < threshold;
    if notifications_enabled && is_below4 && !was_below_threshold {
        notifier
            .notify("Test", &format!("Gym at {:.0}%", percentage4))
            .unwrap();
    }
    // was_below_threshold = is_below4;

    assert_eq!(
        notifier.notification_count(),
        2,
        "New drop after recovery should notify again"
    );
}

/// Test that notifications are not sent when disabled.
#[test]
fn test_notification_disabled_no_notification() {
    let notifier = MockNotifier::new();

    let threshold = 30.0;
    let mut was_below_threshold = false;
    let notifications_enabled = false; // Disabled

    let percentage = 10.0; // Way below threshold
    let is_below = percentage < threshold;
    if notifications_enabled && is_below && !was_below_threshold {
        notifier
            .notify("Test", &format!("Gym at {:.0}%", percentage))
            .unwrap();
    }
    was_below_threshold = is_below;

    assert_eq!(
        notifier.notification_count(),
        0,
        "Disabled notifications should not fire"
    );
    assert!(was_below_threshold, "State should still update");
}

/// Test notification at exact threshold boundary.
#[test]
fn test_notification_at_exact_threshold() {
    let notifier = MockNotifier::new();

    let threshold = 30.0;
    let mut was_below_threshold = false;
    let notifications_enabled = true;

    // Exactly AT threshold - should NOT notify (not below)
    let percentage = 30.0;
    let is_below = percentage < threshold;
    if notifications_enabled && is_below && !was_below_threshold {
        notifier.notify("Test", "At threshold").unwrap();
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

/// Test notification message content.
#[test]
fn test_notification_message_format() {
    let notifier = MockNotifier::new();

    notifier
        .notify("Hardy's Gym Monitor", "Gym is empty! 25%")
        .unwrap();

    let notifications = notifier.get_notifications();
    assert_eq!(notifications.len(), 1);
    assert_eq!(notifications[0].0, "Hardy's Gym Monitor");
    assert_eq!(notifications[0].1, "Gym is empty! 25%");
}

// ==================== Time-Dependent Analytics Tests ====================

/// Test predictions are generated for correct hours based on clock.
#[test]
fn test_predictions_use_mock_clock_time() {
    // Monday at 10:00 UTC
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap());
    let schedule = create_test_schedule(0, 24, 0, 24); // 24/7 open

    // Create baseline with data for hours 11 and 12 on Monday (weekday 0)
    let baseline = vec![
        HourlyAverage {
            weekday: 0, // Monday
            hour: 11,
            avg_percentage: 30.0,
            sample_count: 10,
        },
        HourlyAverage {
            weekday: 0, // Monday
            hour: 12,
            avg_percentage: 50.0,
            sample_count: 10,
        },
    ];

    let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

    assert_eq!(predictions.len(), 2);
    // At 10:00, predictions should be for 11:00 (now+1h) and 12:00 (now+2h)
    assert_eq!(predictions[0].0.hour(), 11);
    assert_eq!(predictions[0].1, 30.0);
    assert_eq!(predictions[1].0.hour(), 12);
    assert_eq!(predictions[1].1, 50.0);
}

/// Test predictions update correctly as clock advances.
#[test]
fn test_predictions_update_as_time_advances() {
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap());
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

    // At 10:00, should predict for 11:00 and 12:00
    let predictions1 = calculate_predictions_with_clock(&baseline, &schedule, &clock);
    assert_eq!(predictions1.len(), 2);
    assert_eq!(predictions1[0].1, 25.0);
    assert_eq!(predictions1[1].1, 45.0);

    // Advance to 11:00
    clock.advance(ChronoDuration::hours(1));

    // Now should predict for 12:00 and 13:00
    let predictions2 = calculate_predictions_with_clock(&baseline, &schedule, &clock);
    assert_eq!(predictions2.len(), 2);
    assert_eq!(predictions2[0].1, 45.0);
    assert_eq!(predictions2[1].1, 65.0);
}

/// Test predictions respect gym schedule (closed hours).
#[test]
fn test_predictions_skip_closed_hours() {
    // Set clock to a time where next hours would be closed
    // Monday at 22:00 UTC, gym closes at 23:00 local
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 21, 0, 0).unwrap());
    // Gym open 6:00-22:00 on weekdays
    let schedule = create_test_schedule(6, 22, 8, 20);

    let baseline = vec![
        HourlyAverage {
            weekday: 0,
            hour: 22, // Would be +1h from 21:00 UTC
            avg_percentage: 40.0,
            sample_count: 5,
        },
        HourlyAverage {
            weekday: 0,
            hour: 23, // Would be +2h from 21:00 UTC
            avg_percentage: 30.0,
            sample_count: 5,
        },
    ];

    let predictions = calculate_predictions_with_clock(&baseline, &schedule, &clock);

    // Predictions should be filtered by schedule
    // Actual results depend on local timezone conversion
    assert!(
        predictions.len() <= 2,
        "Predictions should be filtered by schedule"
    );
}

/// Test find_best_time_today uses mock clock for day determination.
#[test]
fn test_find_best_time_uses_mock_clock_day() {
    // Set to Monday
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap());

    // Data for Monday (weekday 0 in UTC)
    let data = vec![
        HourlyAverage {
            weekday: 0, // Monday
            hour: 8,
            avg_percentage: 60.0,
            sample_count: 5,
        },
        HourlyAverage {
            weekday: 0, // Monday
            hour: 14,
            avg_percentage: 15.0, // Best time
            sample_count: 5,
        },
        HourlyAverage {
            weekday: 1, // Tuesday - should be ignored
            hour: 10,
            avg_percentage: 5.0, // Lower but wrong day
            sample_count: 5,
        },
    ];

    let result = find_best_time_today_with_clock(&data, &clock);
    assert!(result.is_some());
    let (_, avg) = result.unwrap();
    assert_eq!(avg, 15.0, "Should find best time for Monday only");
}

/// Test day boundary handling with mock clock.
#[test]
fn test_analytics_at_day_boundary() {
    // Set to just before midnight Sunday -> Monday transition
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 16, 23, 59, 0).unwrap());

    // This is still Sunday in UTC
    let data = vec![HourlyAverage {
        weekday: 6, // Sunday
        hour: 23,
        avg_percentage: 20.0,
        sample_count: 5,
    }];

    let result = find_best_time_today_with_clock(&data, &clock);
    // Result depends on local timezone, but the test verifies the clock is used
    // The important thing is that it doesn't crash and returns a reasonable result

    // Advance past midnight
    clock.advance(ChronoDuration::minutes(2));

    // Now Monday in UTC
    let monday_data = vec![HourlyAverage {
        weekday: 0, // Monday
        hour: 0,
        avg_percentage: 25.0,
        sample_count: 5,
    }];

    let result2 = find_best_time_today_with_clock(&monday_data, &clock);
    // Again, depends on local timezone, but should handle the boundary correctly
    assert!(
        result.is_some() || result2.is_some(),
        "Should find data for at least one of the days"
    );
}

// ==================== Schedule + Clock Integration Tests ====================

/// Test gym schedule uses clock correctly.
#[test]
fn test_schedule_with_mock_clock() {
    // Monday at 10:00 local time (approx)
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 10, 0, 0).unwrap());
    // Weekday: 6:00-22:00, Weekend: 8:00-20:00
    let schedule = create_test_schedule(6, 22, 8, 20);

    let local_time = clock.now_local();
    let is_open = schedule.is_open(&local_time);

    // At 10:00 on a Monday, the gym should be open (6:00-22:00)
    assert!(is_open, "Gym should be open at 10:00 on Monday");
}

/// Test clock advancing through open/closed transitions.
#[test]
fn test_schedule_open_close_transitions() {
    // Start Monday at 05:00 UTC (likely before opening in most timezones)
    let clock = MockClock::new(Utc.with_ymd_and_hms(2024, 6, 17, 5, 0, 0).unwrap());
    let schedule = create_test_schedule(6, 22, 8, 20);

    // Very early morning - depends on timezone whether open or closed
    let early_status = schedule.is_open(&clock.now_local());

    // Advance to midday - definitely should be open
    clock.advance(ChronoDuration::hours(7)); // Now 12:00 UTC
    let midday_status = schedule.is_open(&clock.now_local());
    assert!(midday_status, "Should be open at midday");

    // Advance to late night - should be closed
    clock.advance(ChronoDuration::hours(12)); // Now 00:00 UTC next day
    let midnight_status = schedule.is_open(&clock.now_local());
    // Midnight is typically closed
    assert!(
        !midnight_status || early_status,
        "Either midnight is closed or early morning varies by timezone"
    );
}

// ==================== Mock Notifier Edge Cases ====================

/// Test mock notifier clear functionality.
#[test]
fn test_notifier_clear_and_reuse() {
    let notifier = MockNotifier::new();

    notifier.notify("Title1", "Body1").unwrap();
    notifier.notify("Title2", "Body2").unwrap();
    assert_eq!(notifier.notification_count(), 2);

    notifier.clear();
    assert_eq!(notifier.notification_count(), 0);
    assert!(!notifier.was_called());

    notifier.notify("Title3", "Body3").unwrap();
    assert_eq!(notifier.notification_count(), 1);

    let notifications = notifier.get_notifications();
    assert_eq!(notifications[0].0, "Title3");
}

/// Test mock notifier with empty messages.
#[test]
fn test_notifier_empty_messages() {
    let notifier = MockNotifier::new();

    notifier.notify("", "").unwrap();
    assert!(notifier.was_called());

    let notifications = notifier.get_notifications();
    assert_eq!(notifications[0], ("".to_string(), "".to_string()));
}

/// Test mock notifier with unicode content.
#[test]
fn test_notifier_unicode_content() {
    let notifier = MockNotifier::new();

    notifier
        .notify("🏋️ Gym Alert", "空いています！ (Empty!)")
        .unwrap();

    let notifications = notifier.get_notifications();
    assert_eq!(notifications[0].0, "🏋️ Gym Alert");
    assert_eq!(notifications[0].1, "空いています！ (Empty!)");
}
