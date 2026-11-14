//! "Forecasts": how good the forecasts are, over a chosen period. Sub-pages
//! share the period selector.

mod accuracy;
mod charts;
mod detail;
mod leaderboard;
mod models;
mod overview;

pub(crate) use accuracy::DAYS as ACCURACY_DAYS;
use hardy_core::{Tz, accuracy::AccuracySummary};
use hardy_ml::suite::{Family, ModelId};
use iced::{
    Alignment, Element, Length,
    widget::{Space, canvas::Cache, column, row},
};

use crate::{
    app::Message,
    leaderboard::{Leaderboard, Sort},
    performance::{ModelDetail, Performance, Period, Source, Tab},
    style,
    views::components::{empty_state, scroll, segmented},
};

/// The charts of the Forecasts tab, redrawn when their data changes.
#[derive(Default)]
pub struct Caches {
    pub daily: Cache,
    pub scatter: Cache,
    pub backtest: Cache,
    pub live_daily: Cache,
    pub lead_time: Cache,
}

impl Caches {
    /// Clears the open model's charts.
    pub fn clear_detail(&self) {
        self.backtest.clear();
        self.live_daily.clear();
        self.lead_time.clear();
    }

    /// Clears every chart.
    pub fn clear_all(&self) {
        self.daily.clear();
        self.scatter.clear();
        self.clear_detail();
    }
}

/// Everything the Forecasts view shows.
#[derive(Clone, Copy)]
pub struct ForecastsProps<'a> {
    pub timezone: Tz,
    pub tab: Tab,
    pub period: Period,
    /// E.g. "last 28 days".
    pub period_label: &'a str,
    pub source: Source,
    pub picked: Option<ModelId>,
    /// `None` until the period's data has loaded.
    pub performance: Option<&'a Performance>,
    pub detail: Option<&'a ModelDetail>,
    pub caches: &'a Caches,
    /// The newest backtest per model.
    pub leaderboard: &'a Leaderboard,
    pub leaderboard_sort: Sort,
    /// The family the table is narrowed to; `None` for every model.
    pub leaderboard_family: Option<Family>,
    /// A run was requested and the forecaster has not started it yet.
    pub run_pending: bool,
    /// Why the forecaster's last run failed.
    pub last_run_error: Option<&'a str>,
    /// The hourly log's scores over the last days.
    pub accuracy: Option<&'a AccuracySummary>,
}

/// Green for a clear improvement over the baseline, amber for a small one.
fn improvement_color(improvement: f64) -> iced::Color {
    if improvement >= 0.15 {
        style::SUCCESS
    } else if improvement > 0.0 {
        style::WARNING
    } else {
        style::DANGER
    }
}

/// A bias in points with its sign, e.g. "+0.4 pts"; one that rounds to
/// zero shows no sign.
pub(super) fn signed_points(value: f64) -> String {
    if value.abs() < 0.05 {
        "0.0 pts".to_string()
    } else {
        format!("{value:+.1} pts")
    }
}

/// How the period selector names a period in captions.
pub fn period_label(period: Period) -> &'static str {
    match period {
        Period::Week => "last 7 days",
        Period::Month => "last 28 days",
        Period::Quarter => "last 90 days",
        Period::All => "all history",
    }
}

/// The Forecasts page.
pub fn view(props: ForecastsProps<'_>) -> Element<'_, Message> {
    let tabs: Vec<(&str, Tab)> = Tab::ALL.iter().map(|t| (t.label(), *t)).collect();
    let periods: Vec<(&str, Period)> = Period::ALL.iter().map(|p| (p.label(), *p)).collect();
    let controls = row![
        segmented(&tabs, props.tab, Message::ForecastsTabSelected),
        Space::new().width(Length::Fill),
        segmented(&periods, props.period, Message::ForecastsPeriodSelected),
    ]
    .align_y(Alignment::Center);

    let page: Element<'_, Message> = match props.tab {
        Tab::Overview => match props.performance {
            Some(performance) => overview::view(&props, performance),
            None => empty_state("Loading the forecasts' performance…"),
        },
        Tab::Models => models::view(&props),
    };
    column![controls, scroll(page)]
        .spacing(style::SPACE_L)
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_improvement_color_bands() {
        assert_eq!(improvement_color(0.25), style::SUCCESS);
        assert_eq!(improvement_color(0.05), style::WARNING);
        assert_eq!(improvement_color(-0.1), style::DANGER);
    }

    #[test]
    fn test_signed_points_drop_the_sign_of_zero() {
        assert_eq!(signed_points(0.42), "+0.4 pts");
        assert_eq!(signed_points(-1.25), "-1.2 pts");
        assert_eq!(signed_points(-0.01), "0.0 pts");
    }

    #[test]
    fn test_period_labels() {
        assert_eq!(period_label(Period::Month), "last 28 days");
        assert_eq!(period_label(Period::All), "all history");
    }
}
