use std::collections::VecDeque;

use chrono::{DateTime, Local, Utc};

/// Extract 1h rolling average, 3h rolling average, and linear trend from recent
/// observations.
///
/// Returns `(avg_1h, avg_3h, trend)`. Defaults to `(50.0, 50.0, 0.0)` when no
/// data is available.
pub(super) fn extract_momentum(recent_data: &VecDeque<(DateTime<Utc>, f64)>) -> (f64, f64, f64) {
    if recent_data.is_empty() {
        return (50.0, 50.0, 0.0);
    }

    let now = recent_data.back().map(|(t, _)| *t).unwrap_or_else(Utc::now);
    let one_hour_ago = now - chrono::Duration::hours(1);
    let three_hours_ago = now - chrono::Duration::hours(3);

    let recent_1h: Vec<f64> = recent_data
        .iter()
        .filter(|(t, _)| *t >= one_hour_ago)
        .map(|(_, v)| *v)
        .collect();
    let recent_avg_1h = if recent_1h.is_empty() {
        50.0
    } else {
        recent_1h.iter().sum::<f64>() / recent_1h.len() as f64
    };

    let recent_3h: Vec<f64> = recent_data
        .iter()
        .filter(|(t, _)| *t >= three_hours_ago)
        .map(|(_, v)| *v)
        .collect();
    let recent_avg_3h = if recent_3h.is_empty() {
        50.0
    } else {
        recent_3h.iter().sum::<f64>() / recent_3h.len() as f64
    };

    let recent_trend = calculate_trend(&recent_3h);

    (recent_avg_1h, recent_avg_3h, recent_trend)
}

/// Least-squares linear trend over a value series.
///
/// Returns the slope scaled by 60 (units per hour at 1-minute resolution).
/// Returns `0.0` for series with fewer than 2 values or constant x-variance.
pub(super) fn calculate_trend(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    if denominator.abs() < f64::EPSILON {
        0.0
    } else {
        (numerator / denominator) * 60.0
    }
}

/// Today's average-so-far and yesterday's average from recent observations.
///
/// Returns `(day_avg_so_far, prev_day_avg)`. Defaults to `50.0` for each if
/// no data is available for that day.
pub(super) fn extract_day_features(
    recent_data: &VecDeque<(DateTime<Utc>, f64)>,
    local_time: &DateTime<Local>,
) -> (f64, f64) {
    let today = local_time.date_naive();
    let yesterday = today - chrono::Duration::days(1);

    let mut today_values = Vec::new();
    let mut yesterday_values = Vec::new();

    for (timestamp, value) in recent_data {
        let local_ts = timestamp.with_timezone(&Local);
        let date = local_ts.date_naive();

        if date == today {
            today_values.push(*value);
        } else if date == yesterday {
            yesterday_values.push(*value);
        }
    }

    let day_avg_so_far = if today_values.is_empty() {
        50.0
    } else {
        today_values.iter().sum::<f64>() / today_values.len() as f64
    };

    let prev_day_avg = if yesterday_values.is_empty() {
        50.0
    } else {
        yesterday_values.iter().sum::<f64>() / yesterday_values.len() as f64
    };

    (day_avg_so_far, prev_day_avg)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_calculate_trend_increasing() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let trend = calculate_trend(&values);

        assert!(
            trend > 0.0,
            "Trend should be positive for increasing values"
        );
    }

    #[test]
    fn test_calculate_trend_decreasing() {
        let values = vec![50.0, 40.0, 30.0, 20.0, 10.0];

        let trend = calculate_trend(&values);

        assert!(
            trend < 0.0,
            "Trend should be negative for decreasing values"
        );
    }

    #[test]
    fn test_calculate_trend_flat() {
        let values = vec![30.0, 30.0, 30.0, 30.0, 30.0];

        let trend = calculate_trend(&values);

        assert_relative_eq!(trend, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_momentum_empty() {
        let recent: VecDeque<(DateTime<Utc>, f64)> = VecDeque::new();

        let (avg_1h, avg_3h, trend) = extract_momentum(&recent);

        assert_relative_eq!(avg_1h, 50.0);
        assert_relative_eq!(avg_3h, 50.0);
        assert_relative_eq!(trend, 0.0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_extract_momentum_defaults_on_empty(
            _dummy in 0..1_i32,
        ) {
            let empty: VecDeque<(DateTime<Utc>, f64)> = VecDeque::new();
            let (a1, a3, t) = extract_momentum(&empty);
            prop_assert_eq!(a1.to_bits(), 50.0_f64.to_bits());
            prop_assert_eq!(a3.to_bits(), 50.0_f64.to_bits());
            prop_assert_eq!(t.to_bits(), 0.0_f64.to_bits());
        }

        #[test]
        fn prop_calculate_trend_zero_for_constant(
            value in -1000.0_f64..1000.0,
            len in 2_usize..50,
        ) {
            let values = vec![value; len];
            let trend = calculate_trend(&values);
            prop_assert!(
                trend.abs() < 1e-10,
                "Trend of constant series should be 0, got {trend}"
            );
        }

        #[test]
        fn prop_calculate_trend_finite(
            values in proptest::collection::vec(-1000.0_f64..1000.0, 2..50),
        ) {
            let trend = calculate_trend(&values);
            prop_assert!(trend.is_finite(), "Trend should be finite, got {trend}");
        }

        #[test]
        fn prop_calculate_trend_sign(
            base in 0.0_f64..100.0,
            step in 0.1_f64..10.0,
            len in 3_usize..20,
        ) {
            // Strictly increasing
            let increasing: Vec<f64> = (0..len).map(|i| base + step * i as f64).collect();
            let trend_inc = calculate_trend(&increasing);
            prop_assert!(trend_inc > 0.0, "Increasing series should have positive trend, got {trend_inc}");

            // Strictly decreasing
            let decreasing: Vec<f64> = (0..len).map(|i| base - step * i as f64).collect();
            let trend_dec = calculate_trend(&decreasing);
            prop_assert!(trend_dec < 0.0, "Decreasing series should have negative trend, got {trend_dec}");
        }
    }
}
