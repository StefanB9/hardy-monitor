//! Integration tests for notification logic using `MockNotifier`.
#![allow(clippy::float_cmp)]
#![allow(clippy::manual_string_new)]

use anyhow::{Context, Result};
use hardy_core::{MockNotifier, Notifier};

#[tokio::test]
async fn test_notification_debounce_only_fires_once() -> Result<()> {
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
            .context("notification should succeed")?;
    }
    was_below_threshold = is_below1;

    assert_eq!(notifier.notification_count(), 1, "First drop should notify");

    let percentage2 = 20.0;
    let is_below2 = percentage2 < threshold;
    if notifications_enabled && is_below2 && !was_below_threshold {
        notifier
            .notify("Test", &format!("Gym at {percentage2:.0}%"))
            .await
            .context("notification should succeed")?;
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
            .context("notification should succeed")?;
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
            .context("notification should succeed")?;
    }

    assert_eq!(
        notifier.notification_count(),
        2,
        "New drop after recovery should notify again"
    );
    Ok(())
}

#[tokio::test]
async fn test_notification_disabled_no_notification() -> Result<()> {
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
            .context("notification should succeed")?;
    }
    was_below_threshold = is_below;

    assert_eq!(
        notifier.notification_count(),
        0,
        "Disabled notifications should not fire"
    );
    assert!(was_below_threshold, "State should still update");
    Ok(())
}

#[tokio::test]
async fn test_notification_at_exact_threshold() -> Result<()> {
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
            .context("notification should succeed")?;
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
    Ok(())
}

#[tokio::test]
async fn test_notification_message_format() -> Result<()> {
    let notifier = MockNotifier::new();

    notifier
        .notify("Hardy's Gym Monitor", "Gym is empty! 25%")
        .await
        .context("notification should succeed")?;

    let notifications = notifier.get_notifications();
    assert_eq!(notifications.len(), 1);
    assert_eq!(notifications[0].0, "Hardy's Gym Monitor");
    assert_eq!(notifications[0].1, "Gym is empty! 25%");
    Ok(())
}

#[tokio::test]
async fn test_notifier_clear_and_reuse() -> Result<()> {
    let notifier = MockNotifier::new();

    notifier
        .notify("Title1", "Body1")
        .await
        .context("notification should succeed")?;
    notifier
        .notify("Title2", "Body2")
        .await
        .context("notification should succeed")?;
    assert_eq!(notifier.notification_count(), 2);

    notifier.clear();
    assert_eq!(notifier.notification_count(), 0);
    assert!(!notifier.was_called());

    notifier
        .notify("Title3", "Body3")
        .await
        .context("notification should succeed")?;
    assert_eq!(notifier.notification_count(), 1);

    let notifications = notifier.get_notifications();
    assert_eq!(notifications[0].0, "Title3");
    Ok(())
}

#[tokio::test]
async fn test_notifier_empty_messages() -> Result<()> {
    let notifier = MockNotifier::new();

    notifier
        .notify("", "")
        .await
        .context("notification should succeed")?;
    assert!(notifier.was_called());

    let notifications = notifier.get_notifications();
    assert_eq!(notifications[0], ("".to_string(), "".to_string()));
    Ok(())
}

#[tokio::test]
async fn test_notifier_unicode_content() -> Result<()> {
    let notifier = MockNotifier::new();

    notifier
        .notify("🏋️ Gym Alert", "空いています！ (Empty!)")
        .await
        .context("notification should succeed")?;

    let notifications = notifier.get_notifications();
    assert_eq!(notifications[0].0, "🏋️ Gym Alert");
    assert_eq!(notifications[0].1, "空いています！ (Empty!)");
    Ok(())
}
