//! Maps times and values to canvas coordinates.

use chrono::{DateTime, TimeDelta, Utc};
use iced::{Point, Size};

/// Maps times and values to canvas coordinates.
#[derive(Clone, Copy)]
pub(super) struct Plot {
    pub(super) start: DateTime<Utc>,
    pub(super) span_secs: f32,
    pub(super) left: f32,
    pub(super) top: f32,
    pub(super) width: f32,
    pub(super) height: f32,
    pub(super) max_value: f64,
}

use super::{HistoryChart, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT, PAD_TOP};

impl Plot {
    pub(super) fn new(chart: &HistoryChart<'_>, size: Size, max_value: f64) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let span_secs = (chart.range_end - chart.range_start).num_seconds().max(1) as f32;
        Self {
            start: chart.range_start,
            span_secs,
            left: PAD_LEFT,
            top: PAD_TOP,
            width: (size.width - PAD_LEFT - PAD_RIGHT).max(1.0),
            height: (size.height - PAD_TOP - PAD_BOTTOM).max(1.0),
            max_value,
        }
    }

    pub(super) fn x(&self, t: DateTime<Utc>) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let offset = (t - self.start).num_seconds() as f32;
        self.left + offset / self.span_secs * self.width
    }

    pub(super) fn y(&self, value: f64) -> f32 {
        #[allow(clippy::cast_possible_truncation)]
        let ratio = (value.clamp(0.0, self.max_value) / self.max_value) as f32;
        self.top + self.height - ratio * self.height
    }

    pub(super) fn point(&self, t: DateTime<Utc>, value: f64) -> Point {
        Point::new(self.x(t), self.y(value))
    }

    pub(super) fn bottom(&self) -> f32 {
        self.top + self.height
    }

    pub(super) fn time_at(&self, x: f32) -> Option<DateTime<Utc>> {
        let ratio = (x - self.left) / self.width;
        #[allow(clippy::cast_possible_truncation)]
        (0.0..=1.0)
            .contains(&ratio)
            .then(|| self.start + TimeDelta::seconds((ratio * self.span_secs) as i64))
    }
}
