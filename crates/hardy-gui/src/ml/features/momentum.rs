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

    let now = recent_data.back().map_or_else(Utc::now, |(t, _)| *t);
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
        #[allow(clippy::cast_precision_loss)]
        let recent_1h_count_f64 = recent_1h.len() as f64;

        recent_1h.iter().sum::<f64>() / recent_1h_count_f64
    };

    let recent_3h: Vec<f64> = recent_data
        .iter()
        .filter(|(t, _)| *t >= three_hours_ago)
        .map(|(_, v)| *v)
        .collect();
    let recent_avg_3h = if recent_3h.is_empty() {
        50.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let recent_3h_count_f64 = recent_3h.len() as f64;

        recent_3h.iter().sum::<f64>() / recent_3h_count_f64
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

    #[allow(clippy::cast_precision_loss)]
    let n = values.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    #[allow(clippy::cast_precision_loss)]
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

/// 6-hour rolling average of occupancy values from recent observations.
///
/// Returns `50.0` (neutral default) if no data is available in the 6-hour
/// window.
pub(super) fn extract_avg_6h(recent_data: &VecDeque<(DateTime<Utc>, f64)>) -> f64 {
    if recent_data.is_empty() {
        return 50.0;
    }

    let now = recent_data.back().map_or_else(Utc::now, |(t, _)| *t);
    let six_hours_ago = now - chrono::Duration::hours(6);

    let recent_6h: Vec<f64> = recent_data
        .iter()
        .filter(|(t, _)| *t >= six_hours_ago)
        .map(|(_, v)| *v)
        .collect();

    if recent_6h.is_empty() {
        return 50.0;
    }

    #[allow(clippy::cast_precision_loss)]
    let n = recent_6h.len() as f64;
    recent_6h.iter().sum::<f64>() / n
}

/// Population standard deviation of occupancy values within the last 1 hour.
///
/// Returns `0.0` if fewer than 2 data points are available in the window.
pub(super) fn extract_volatility(recent_data: &VecDeque<(DateTime<Utc>, f64)>) -> f64 {
    if recent_data.is_empty() {
        return 0.0;
    }

    let now = recent_data.back().map_or_else(Utc::now, |(t, _)| *t);
    let one_hour_ago = now - chrono::Duration::hours(1);

    let recent_1h: Vec<f64> = recent_data
        .iter()
        .filter(|(t, _)| *t >= one_hour_ago)
        .map(|(_, v)| *v)
        .collect();

    if recent_1h.len() < 2 {
        return 0.0;
    }

    #[allow(clippy::cast_precision_loss)]
    let n = recent_1h.len() as f64;
    let mean = recent_1h.iter().sum::<f64>() / n;
    let variance = recent_1h.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

    variance.sqrt()
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
        #[allow(clippy::cast_precision_loss)]
        let today_count_f64 = today_values.len() as f64;
        today_values.iter().sum::<f64>() / today_count_f64
    };

    let prev_day_avg = if yesterday_values.is_empty() {
        50.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let yesterday_count_f64 = yesterday_values.len() as f64;
        yesterday_values.iter().sum::<f64>() / yesterday_count_f64
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

    // ── extract_avg_6h tests ────────────────────────────────────────

    #[test]
    fn test_extract_avg_6h_empty() {
        let recent: VecDeque<(DateTime<Utc>, f64)> = VecDeque::new();
        let avg = extract_avg_6h(&recent);
        assert_relative_eq!(avg, 50.0);
    }

    #[test]
    fn test_extract_avg_6h_with_data() {
        let now = Utc::now();
        // 4 values within the last 6 hours: 20, 40, 60, 80 → mean = 50
        let recent: VecDeque<(DateTime<Utc>, f64)> = vec![
            (now - chrono::Duration::hours(5), 20.0),
            (now - chrono::Duration::hours(3), 40.0),
            (now - chrono::Duration::hours(1), 60.0),
            (now, 80.0),
        ]
        .into();

        let avg = extract_avg_6h(&recent);
        assert_relative_eq!(avg, 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_avg_6h_excludes_old_data() {
        let now = Utc::now();
        // One value outside 6h window (10h ago), two within (3h, 0h)
        let recent: VecDeque<(DateTime<Utc>, f64)> = vec![
            (now - chrono::Duration::hours(10), 0.0),
            (now - chrono::Duration::hours(3), 60.0),
            (now, 80.0),
        ]
        .into();

        let avg = extract_avg_6h(&recent);
        // Only 60 and 80 are within 6h → mean = 70
        assert_relative_eq!(avg, 70.0, epsilon = 1e-10);
    }

    // ── extract_volatility tests ─────────────────────────────────────

    #[test]
    fn test_extract_volatility_empty() {
        let recent: VecDeque<(DateTime<Utc>, f64)> = VecDeque::new();
        let vol = extract_volatility(&recent);
        assert_relative_eq!(vol, 0.0);
    }

    #[test]
    fn test_extract_volatility_constant() {
        let now = Utc::now();
        let recent: VecDeque<(DateTime<Utc>, f64)> = (0..60)
            .map(|i| (now - chrono::Duration::minutes(i), 42.0))
            .collect();

        let vol = extract_volatility(&recent);
        assert_relative_eq!(vol, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_volatility_varying() {
        let now = Utc::now();
        // Values: 10, 20, 30, 40, 50 — mean=30, pop variance=200, pop std=~14.14
        let recent: VecDeque<(DateTime<Utc>, f64)> = vec![
            (now - chrono::Duration::minutes(4), 10.0),
            (now - chrono::Duration::minutes(3), 20.0),
            (now - chrono::Duration::minutes(2), 30.0),
            (now - chrono::Duration::minutes(1), 40.0),
            (now, 50.0),
        ]
        .into();

        let vol = extract_volatility(&recent);
        assert!(vol > 0.0, "Volatility should be positive for varying data");
        assert_relative_eq!(vol, 200.0_f64.sqrt(), epsilon = 1e-10);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_extract_avg_6h_finite(
            values in proptest::collection::vec(0.0_f64..100.0, 0..120),
        ) {
            let now = Utc::now();
            let recent: VecDeque<(DateTime<Utc>, f64)> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| (now - chrono::Duration::minutes(i64::try_from(i * 3).unwrap_or_default()), v))
                .collect();

            let avg = extract_avg_6h(&recent);
            prop_assert!(avg.is_finite(), "6h average must be finite, got {avg}");
        }

        #[test]
        fn prop_extract_volatility_non_negative(
            values in proptest::collection::vec(0.0_f64..100.0, 0..60),
        ) {
            let now = Utc::now();
            let recent: VecDeque<(DateTime<Utc>, f64)> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| (now - chrono::Duration::minutes(i64::try_from(i).unwrap_or_default()), v))
                .collect();

            let vol = extract_volatility(&recent);
            prop_assert!(vol >= 0.0, "Volatility must be non-negative, got {vol}");
        }

        #[test]
        fn prop_extract_volatility_finite(
            values in proptest::collection::vec(0.0_f64..100.0, 0..60),
        ) {
            let now = Utc::now();
            let recent: VecDeque<(DateTime<Utc>, f64)> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| (now - chrono::Duration::minutes(i64::try_from(i).unwrap_or_default()), v))
                .collect();

            let vol = extract_volatility(&recent);
            prop_assert!(vol.is_finite(), "Volatility must be finite, got {vol}");
        }

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
        len in 3_u32..20_u32, // Changed to u32
    ) {
        // Strictly increasing
        let increasing: Vec<f64> = (0..len)
            .map(|i| base + step * f64::from(i)) // Clean conversion
            .collect();
        let trend_inc = calculate_trend(&increasing);
        prop_assert!(trend_inc > 0.0, "Increasing series should have positive trend, got {trend_inc}");

        // Strictly decreasing
        let decreasing: Vec<f64> = (0..len)
            .map(|i| base - step * f64::from(i)) // Clean conversion
            .collect();
        let trend_dec = calculate_trend(&decreasing);
        prop_assert!(trend_dec < 0.0, "Decreasing series should have negative trend, got {trend_dec}");
    }
    }
}
