//! System tray status: a tooltip summary and the icon with a status dot.

use chrono::{DateTime, Utc};
use hardy_core::Tz;
use iced::Color;
use tray_icon::{Icon, TrayIcon};

/// The tray icon's original pixels, kept to draw the status dot onto.
#[derive(Debug, Clone)]
pub struct TrayBase {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// What the tray summarises.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrayStatus<'a> {
    /// Live occupancy and its level label; `None` while closed or stale.
    pub occupancy: Option<(f64, &'a str)>,
    /// e.g. "14:00" or "tomorrow 14:00".
    pub quiet_hour: Option<&'a str>,
    /// Shown instead of the occupancy when readings stopped.
    pub warning: Option<&'a str>,
    /// e.g. "Closed · opens 06:00".
    pub opening: &'a str,
}

/// Tooltip text, most important first.
pub(crate) fn tooltip_text(status: &TrayStatus<'_>) -> String {
    if let Some(warning) = status.warning {
        return warning.to_string();
    }
    let mut text = match status.occupancy {
        Some((percentage, level)) => format!("{percentage:.0}% · {level}"),
        None => status.opening.to_string(),
    };
    if let Some(quiet) = status.quiet_hour {
        text.push_str(" · next quiet hour ");
        text.push_str(quiet);
    }
    text
}

/// "14:00" today, "tomorrow 14:00", else "Thu 14:00" (gym-local).
pub(crate) fn quiet_label(start: DateTime<Utc>, now: DateTime<Utc>, tz: Tz) -> String {
    let start = start.with_timezone(&tz);
    let today = now.with_timezone(&tz).date_naive();
    let time = start.format("%H:%M");
    if start.date_naive() == today {
        time.to_string()
    } else if today.succ_opt() == Some(start.date_naive()) {
        format!("tomorrow {time}")
    } else {
        start.format("%a %H:%M").to_string()
    }
}

/// `base` with a filled dot of `color` in the bottom-right corner, ringed
/// in dark so it stands out on light and dark taskbars.
pub(crate) fn badge_icon(base: &TrayBase, color: [u8; 4]) -> Vec<u8> {
    const RING: [u8; 4] = [15, 17, 21, 255];
    let mut rgba = base.rgba.clone();
    #[allow(clippy::cast_precision_loss)]
    let (w, h) = (base.width as f32, base.height as f32);
    let radius = (w.min(h) * 0.22).max(2.0);
    let ring = radius + (radius * 0.25).max(1.0);
    let (cx, cy) = (w - ring, h - ring);
    for y in 0..base.height {
        for x in 0..base.width {
            #[allow(clippy::cast_precision_loss)]
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let distance = (px - cx).hypot(py - cy);
            let paint = if distance <= radius {
                color
            } else if distance <= ring {
                RING
            } else {
                continue;
            };
            let i = ((y * base.width + x) * 4) as usize;
            if let Some(pixel) = rgba.get_mut(i..i + 4) {
                pixel.copy_from_slice(&paint);
            }
        }
    }
    rgba
}

/// RGBA bytes of an iced colour.
pub(crate) fn rgba8(color: Color) -> [u8; 4] {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let channel = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    [
        channel(color.r),
        channel(color.g),
        channel(color.b),
        channel(color.a),
    ]
}

/// The tray icon, updated only when its text or colour changes.
pub struct Tray {
    icon: TrayIcon,
    base: TrayBase,
    tooltip: String,
    color: Option<[u8; 4]>,
}

impl Tray {
    /// Wraps the tray icon; `base` is the icon the status dot is drawn on.
    pub fn new(icon: TrayIcon, base: TrayBase) -> Self {
        Self {
            icon,
            base,
            tooltip: String::new(),
            color: None,
        }
    }

    pub(crate) fn update(&mut self, tooltip: String, color: [u8; 4]) {
        if tooltip != self.tooltip {
            if let Err(e) = self.icon.set_tooltip(Some(&tooltip)) {
                tracing::debug!(error = %e, "tray tooltip not updated");
            }
            self.tooltip = tooltip;
        }
        if self.color != Some(color) {
            let rgba = badge_icon(&self.base, color);
            match Icon::from_rgba(rgba, self.base.width, self.base.height) {
                Ok(icon) => {
                    if let Err(e) = self.icon.set_icon(Some(icon)) {
                        tracing::debug!(error = %e, "tray icon not updated");
                    }
                }
                Err(e) => tracing::warn!(error = %e, "could not build tray icon"),
            }
            self.color = Some(color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status<'a>() -> TrayStatus<'a> {
        TrayStatus {
            occupancy: Some((23.4, "Quiet")),
            quiet_hour: Some("14:00"),
            warning: None,
            opening: "Open · closes 23:00",
        }
    }

    #[test]
    fn test_tooltip_open_with_quiet_hour() {
        assert_eq!(
            tooltip_text(&status()),
            "23% · Quiet · next quiet hour 14:00"
        );
        let no_quiet = TrayStatus {
            quiet_hour: None,
            ..status()
        };
        assert_eq!(tooltip_text(&no_quiet), "23% · Quiet");
    }

    #[test]
    fn test_tooltip_closed_and_stale() {
        let closed = TrayStatus {
            occupancy: None,
            quiet_hour: Some("tomorrow 14:00"),
            opening: "Closed · opens 06:00",
            ..status()
        };
        assert_eq!(
            tooltip_text(&closed),
            "Closed · opens 06:00 · next quiet hour tomorrow 14:00"
        );
        let stale = TrayStatus {
            occupancy: None,
            warning: Some("No new readings since 20:39"),
            ..status()
        };
        assert_eq!(tooltip_text(&stale), "No new readings since 20:39");
    }

    fn base(size: u32) -> TrayBase {
        TrayBase {
            rgba: vec![255; (size * size * 4) as usize],
            width: size,
            height: size,
        }
    }

    fn pixel(rgba: &[u8], size: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * size + x) * 4) as usize;
        [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
    }

    #[test]
    fn test_badge_icon_draws_dot_bottom_right_only() {
        let size = 32;
        let green = [61, 214, 140, 255];
        let out = badge_icon(&base(size), green);
        assert_eq!(out.len(), (size * size * 4) as usize);
        // Near the bottom-right corner: the dot.
        assert_eq!(pixel(&out, size, size - 7, size - 7), green);
        // Top-left untouched.
        assert_eq!(pixel(&out, size, 2, 2), [255; 4]);
    }

    #[test]
    fn test_badge_icon_keeps_size_for_tiny_icons() {
        let out = badge_icon(&base(4), [0, 0, 0, 255]);
        assert_eq!(out.len(), 4 * 4 * 4);
    }

    #[test]
    fn test_quiet_label_relative_day() -> anyhow::Result<()> {
        use anyhow::Context;
        use chrono::{TimeDelta, TimeZone};
        let tz = hardy_core::GymSchedule::default().timezone();
        // Monday 2024-06-17 20:00 local.
        let now = tz
            .with_ymd_and_hms(2024, 6, 17, 20, 0, 0)
            .single()
            .context("time")?
            .with_timezone(&Utc);
        assert_eq!(quiet_label(now + TimeDelta::hours(1), now, tz), "21:00");
        assert_eq!(
            quiet_label(now + TimeDelta::hours(18), now, tz),
            "tomorrow 14:00"
        );
        assert_eq!(
            quiet_label(now + TimeDelta::hours(42), now, tz),
            "Wed 14:00"
        );
        Ok(())
    }

    #[test]
    fn test_rgba8_converts_channels() {
        assert_eq!(rgba8(Color::from_rgb8(61, 214, 140)), [61, 214, 140, 255]);
    }
}
