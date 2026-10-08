//! Design tokens: colours, the occupancy colour scale, type and spacing
//! scales. Every view and widget takes its look from here.

use iced::Color;

const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::from_rgb8(r, g, b)
}

// ── Surfaces ──────────────────────────────────────────────────────────

/// Window background.
pub const BG_APP: Color = rgb(0x0F, 0x11, 0x15);
/// Sidebar background.
pub const BG_SIDEBAR: Color = rgb(0x13, 0x16, 0x1C);
/// Card background.
pub const BG_CARD: Color = rgb(0x18, 0x1C, 0x24);
/// Inputs, secondary buttons, inactive segments.
pub const BG_ELEVATED: Color = rgb(0x22, 0x27, 0x31);
/// Card and control borders.
pub const BORDER: Color = rgb(0x2B, 0x31, 0x3D);
/// Chart grid lines.
pub const GRID: Color = rgb(0x24, 0x29, 0x33);

// ── Text ──────────────────────────────────────────────────────────────

/// Main text.
pub const TEXT_PRIMARY: Color = rgb(0xE8, 0xEB, 0xF0);
/// Supporting text.
pub const TEXT_SECONDARY: Color = rgb(0xA3, 0xAB, 0xB9);
/// Captions and muted labels.
pub const TEXT_TERTIARY: Color = rgb(0x6E, 0x77, 0x88);

// ── Accents ───────────────────────────────────────────────────────────

/// Interactive elements and the actual-occupancy line.
pub const ACCENT: Color = rgb(0x5B, 0x9D, 0xFF);
/// Forecasts.
pub const FORECAST: Color = rgb(0xA7, 0x8B, 0xFA);
/// Healthy or fresh state.
pub const SUCCESS: Color = rgb(0x3D, 0xD6, 0x8C);
/// Stale data and other degraded states.
pub const WARNING: Color = rgb(0xF5, 0xB8, 0x41);
/// Errors.
pub const DANGER: Color = rgb(0xEF, 0x5A, 0x5A);

// ── Occupancy scale ───────────────────────────────────────────────────

/// Below the low threshold.
pub const OCC_QUIET: Color = rgb(0x3D, 0xD6, 0x8C);
/// Between the thresholds.
pub const OCC_MODERATE: Color = rgb(0xF5, 0xB8, 0x41);
/// At or above the high threshold.
pub const OCC_BUSY: Color = rgb(0xF2, 0x70, 0x4E);
/// 100 % on the continuous scale.
pub const OCC_PACKED: Color = rgb(0xD9, 0x3B, 0x5B);
/// Cells or slots without any readings.
pub const NO_DATA: Color = rgb(0x2A, 0x2F, 0x3A);

/// Occupancy category, using the configured low/high thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLevel {
    /// Below the low threshold.
    Quiet,
    /// Between the thresholds.
    Moderate,
    /// At or above the high threshold.
    Busy,
}

impl OccupancyLevel {
    /// The level for `percentage` given the low and high thresholds.
    pub fn from_percentage(percentage: f64, low: f64, high: f64) -> Self {
        if percentage < low {
            OccupancyLevel::Quiet
        } else if percentage < high {
            OccupancyLevel::Moderate
        } else {
            OccupancyLevel::Busy
        }
    }

    /// Display name.
    pub fn label(self) -> &'static str {
        match self {
            OccupancyLevel::Quiet => "Quiet",
            OccupancyLevel::Moderate => "Moderate",
            OccupancyLevel::Busy => "Busy",
        }
    }

    /// Colour on the occupancy scale.
    pub fn color(self) -> Color {
        match self {
            OccupancyLevel::Quiet => OCC_QUIET,
            OccupancyLevel::Moderate => OCC_MODERATE,
            OccupancyLevel::Busy => OCC_BUSY,
        }
    }
}

/// Continuous colour for a percentage: quiet at 0, moderate at `low`, busy
/// at `high`, packed at 100. Heatmap cells and legends use the same stops,
/// so a colour always means the same occupancy.
pub fn occupancy_color(percentage: f64, low: f64, high: f64) -> Color {
    let p = percentage.clamp(0.0, 100.0);
    let stops = [
        (0.0, OCC_QUIET),
        (low.clamp(1.0, 98.0), OCC_MODERATE),
        (high.clamp(low.clamp(1.0, 98.0) + 1.0, 99.0), OCC_BUSY),
        (100.0, OCC_PACKED),
    ];
    for pair in stops.windows(2) {
        let ((p0, c0), (p1, c1)) = (pair[0], pair[1]);
        if p <= p1 {
            #[allow(clippy::cast_possible_truncation)]
            let t = ((p - p0) / (p1 - p0)).clamp(0.0, 1.0) as f32;
            return mix(c0, c1, t);
        }
    }
    OCC_PACKED
}

/// Linear blend from `a` (t = 0) to `b` (t = 1).
pub fn mix(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

/// `color` at reduced opacity, for tinted backgrounds.
pub fn tint(color: Color, alpha: f32) -> Color {
    Color { a: alpha, ..color }
}

// ── Type scale (logical pixels) ───────────────────────────────────────

/// The big occupancy number.
pub const TEXT_DISPLAY: f32 = 44.0;
/// Page titles.
pub const TEXT_TITLE: f32 = 24.0;
/// Card values.
pub const TEXT_LARGE: f32 = 20.0;
/// Card headings.
pub const TEXT_HEADING: f32 = 16.0;
/// Body text.
pub const TEXT_BODY: f32 = 14.0;
/// Captions and axis labels.
pub const TEXT_CAPTION: f32 = 12.0;

// ── Spacing and shape ─────────────────────────────────────────────────

/// Spacing step.
pub const SPACE_XS: f32 = 4.0;
/// Spacing step.
pub const SPACE_S: f32 = 8.0;
/// Spacing step.
pub const SPACE_M: f32 = 12.0;
/// Spacing step.
pub const SPACE_L: f32 = 16.0;
/// Spacing step.
pub const SPACE_XL: f32 = 24.0;
/// Spacing step.
pub const SPACE_XXL: f32 = 32.0;

/// Card corner radius.
pub const RADIUS_CARD: f32 = 14.0;
/// Button and input corner radius.
pub const RADIUS_CONTROL: f32 = 8.0;
/// Inner padding of cards.
pub const CARD_PADDING: f32 = 20.0;

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    const LOW: f64 = 30.0;
    const HIGH: f64 = 60.0;

    fn close(a: Color, b: Color) -> bool {
        (a.r - b.r).abs() < 1e-3 && (a.g - b.g).abs() < 1e-3 && (a.b - b.b).abs() < 1e-3
    }

    #[test]
    fn test_occupancy_level_uses_thresholds() {
        assert_eq!(
            OccupancyLevel::from_percentage(29.9, LOW, HIGH),
            OccupancyLevel::Quiet
        );
        assert_eq!(
            OccupancyLevel::from_percentage(30.0, LOW, HIGH),
            OccupancyLevel::Moderate
        );
        assert_eq!(
            OccupancyLevel::from_percentage(60.0, LOW, HIGH),
            OccupancyLevel::Busy
        );
        assert_eq!(OccupancyLevel::Busy.label(), "Busy");
        assert!(close(OccupancyLevel::Quiet.color(), OCC_QUIET));
    }

    #[test]
    fn test_occupancy_color_hits_stops() {
        assert!(close(occupancy_color(0.0, LOW, HIGH), OCC_QUIET));
        assert!(close(occupancy_color(LOW, LOW, HIGH), OCC_MODERATE));
        assert!(close(occupancy_color(HIGH, LOW, HIGH), OCC_BUSY));
        assert!(close(occupancy_color(100.0, LOW, HIGH), OCC_PACKED));
        assert!(close(occupancy_color(250.0, LOW, HIGH), OCC_PACKED));
    }

    #[test]
    fn test_level_color_matches_scale_at_category_start() {
        for (p, level) in [
            (0.0, OccupancyLevel::Quiet),
            (LOW, OccupancyLevel::Moderate),
            (HIGH, OccupancyLevel::Busy),
        ] {
            assert!(close(occupancy_color(p, LOW, HIGH), level.color()));
        }
    }

    #[test]
    fn test_mix_and_tint() {
        assert!(close(mix(OCC_QUIET, OCC_BUSY, 0.0), OCC_QUIET));
        assert!(close(mix(OCC_QUIET, OCC_BUSY, 1.0), OCC_BUSY));
        assert!((tint(ACCENT, 0.2).a - 0.2).abs() < f32::EPSILON);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn occupancy_color_is_always_valid(
            p in -50.0f64..200.0,
            low in 0.0f64..100.0,
            high in 0.0f64..100.0,
        ) {
            let c = occupancy_color(p, low, high);
            for v in [c.r, c.g, c.b, c.a] {
                prop_assert!((0.0..=1.0).contains(&v), "{c:?}");
            }
        }
    }
}
