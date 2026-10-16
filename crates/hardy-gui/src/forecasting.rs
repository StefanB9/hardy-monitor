//! The GUI's forecasting state: recent history (for slot averages and the
//! trend arrow), the default forecast the daemon publishes, and the state of
//! its models and benchmark.

use chrono::{DateTime, TimeDelta, Utc};
use hardy_core::{
    GymSchedule, Tz,
    db::{BenchmarkRun, LiveForecast, MlState, ModelRecord, OccupancyLog},
};
use hardy_ml::{History, PredictionMethod, PredictionWithConfidence, SlotProfile, suite::ModelId};
use iced::widget::combo_box;

use crate::{
    leaderboard::{self, Leaderboard},
    quiet_window::{QuietWindow, next_quiet_window},
};

/// History kept for the slot averages the quiet window falls back to.
pub(crate) const HISTORY_DAYS: i64 = 28;

/// What the daemon reports about its forecasting models.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SuiteStatus {
    /// Each model's fit state.
    pub records: Vec<ModelRecord>,
    /// The newest benchmark.
    pub benchmark: Option<BenchmarkRun>,
    /// Model key selected per horizon bucket code.
    pub selection: Vec<(i16, String)>,
    pub state: MlState,
}

/// A model's forecast as (slot end, value) points.
pub type ForecastLine = (ModelId, Vec<(DateTime<Utc>, f64)>);

/// A published forecast older than this is labelled: the forecaster seems
/// to be offline (it publishes every quarter hour while the gym is open).
const STALE_AFTER: TimeDelta = TimeDelta::minutes(30);
/// Older than this, the quiet hour no longer relies on the forecast.
const IGNORED_AFTER: TimeDelta = TimeDelta::hours(1);

/// Most models compared on the chart at once.
pub(crate) const MAX_OVERLAYS: usize = 5;

/// Which forecast the GUI shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ForecastChoice {
    /// The model selected for each horizon by the benchmark.
    #[default]
    Best,
    Model(ModelId),
}

impl ForecastChoice {
    /// Stored key of the forecast rows.
    fn key(self) -> &'static str {
        match self {
            ForecastChoice::Best => hardy_ml::live::SELECTED,
            ForecastChoice::Model(id) => id.key(),
        }
    }
}

impl std::fmt::Display for ForecastChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForecastChoice::Best => f.write_str("Best per horizon"),
            ForecastChoice::Model(id) => ModelOption(*id).fmt(f),
        }
    }
}

/// A model as a choice in a list, labelled with its family so typing the
/// family finds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelOption(pub ModelId);

impl std::fmt::Display for ModelOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} · {}", self.0.name(), self.0.family().name())
    }
}

/// Forecasting state for the GUI.
#[derive(Debug)]
pub(crate) struct Forecasting {
    history: History,
    status: SuiteStatus,
    leaderboard: Leaderboard,
    /// Every published forecast (all models and the default).
    published: Vec<LiveForecast>,
    choice: ForecastChoice,
    overlays: Vec<ModelId>,
    /// Overlay forecasts within the chart's horizon.
    overlay_lines: Vec<ForecastLine>,
    max_hours_ahead: u32,
    /// No quiet window is suggested this soon after opening.
    opening_grace: TimeDelta,
    /// Published points after now (up to 24 hours).
    upcoming: Vec<PredictionWithConfidence>,
    /// How many of `upcoming` lie within the chart's horizon.
    charted: usize,
    quiet_window: Option<QuietWindow>,
    /// The forecast picker's options: the default, then every published
    /// model.
    forecast_picker: combo_box::State<ForecastChoice>,
    /// Models that can be added for comparison.
    compare_picker: combo_box::State<ModelOption>,
}

impl Forecasting {
    pub(crate) fn new(max_hours_ahead: u32, opening_grace: TimeDelta) -> Self {
        Self {
            history: History::default(),
            leaderboard: leaderboard::build(&SuiteStatus::default()),
            status: SuiteStatus::default(),
            published: Vec::new(),
            choice: ForecastChoice::Best,
            overlays: Vec::new(),
            overlay_lines: Vec::new(),
            max_hours_ahead,
            opening_grace,
            upcoming: Vec::new(),
            charted: 0,
            quiet_window: None,
            forecast_picker: combo_box::State::new(vec![ForecastChoice::Best]),
            compare_picker: combo_box::State::new(Vec::new()),
        }
    }

    /// Start of the range to fetch next: everything after the newest reading
    /// held, or the full window on first load.
    pub(crate) fn history_fetch_start(&self, now: DateTime<Utc>) -> DateTime<Utc> {
        self.history
            .last_time()
            .map_or(now - TimeDelta::days(HISTORY_DAYS), |t| {
                t + TimeDelta::seconds(1)
            })
    }

    /// Adds newly fetched rows (only observed, only newer ones).
    pub(crate) fn add_history(&mut self, logs: &[OccupancyLog]) {
        if self.history.is_empty() {
            self.history = History::from_logs(logs);
        } else {
            self.history
                .extend(History::from_logs(logs).view().iter().collect::<Vec<_>>());
        }
    }

    pub(crate) fn set_status(&mut self, status: SuiteStatus) {
        self.leaderboard = leaderboard::build(&status);
        self.status = status;
        self.rebuild_pickers();
    }

    pub(crate) fn set_published(&mut self, forecast: Vec<LiveForecast>) {
        self.published = forecast;
        if let ForecastChoice::Model(id) = self.choice
            && !self.available_models().contains(&id)
        {
            self.choice = ForecastChoice::Best;
        }
        self.rebuild_pickers();
    }

    /// Rebuilds the pickers' options after the models, their order or the
    /// shown ones changed.
    fn rebuild_pickers(&mut self) {
        let available = self.available_models();
        let mut choices = Vec::with_capacity(available.len() + 1);
        choices.push(ForecastChoice::Best);
        choices.extend(available.iter().map(|id| ForecastChoice::Model(*id)));
        self.forecast_picker = combo_box::State::new(choices);
        let addable = available
            .into_iter()
            .filter(|id| !self.overlays.contains(id) && self.choice != ForecastChoice::Model(*id))
            .map(ModelOption)
            .collect();
        self.compare_picker = combo_box::State::new(addable);
    }

    pub(crate) fn forecast_picker(&self) -> &combo_box::State<ForecastChoice> {
        &self.forecast_picker
    }

    pub(crate) fn compare_picker(&self) -> &combo_box::State<ModelOption> {
        &self.compare_picker
    }

    /// Models with a published forecast, family by family, best first.
    pub(crate) fn available_models(&self) -> Vec<ModelId> {
        self.leaderboard
            .model_order()
            .into_iter()
            .filter(|id| self.published.iter().any(|f| f.model == id.key()))
            .collect()
    }

    pub(crate) fn choice(&self) -> ForecastChoice {
        self.choice
    }

    /// Shows `choice` (call [`Forecasting::refresh`] afterwards).
    pub(crate) fn choose(&mut self, choice: ForecastChoice) {
        self.choice = choice;
        self.rebuild_pickers();
    }

    /// Models drawn for comparison.
    #[cfg(test)]
    pub(crate) fn overlays(&self) -> &[ModelId] {
        &self.overlays
    }

    /// Adds a comparison line, up to [`MAX_OVERLAYS`].
    pub(crate) fn add_overlay(&mut self, id: ModelId) {
        if !self.overlays.contains(&id) && self.overlays.len() < MAX_OVERLAYS {
            self.overlays.push(id);
        }
        self.rebuild_pickers();
    }

    pub(crate) fn remove_overlay(&mut self, id: ModelId) {
        self.overlays.retain(|o| *o != id);
        self.rebuild_pickers();
    }

    /// The comparison forecasts within the chart's horizon, in overlay
    /// order.
    pub(crate) fn overlay_lines(&self) -> &[ForecastLine] {
        &self.overlay_lines
    }

    /// Recomputes the upcoming forecast, the overlays and the next quiet
    /// window.
    pub(crate) fn refresh(&mut self, schedule: &GymSchedule, now: DateTime<Utc>) {
        let key = self.choice.key();
        let horizon = now + TimeDelta::hours(i64::from(self.max_hours_ahead));
        self.overlay_lines = self
            .overlays
            .iter()
            .map(|id| {
                let points = self
                    .published
                    .iter()
                    .filter(|f| f.model == id.key() && f.target > now && f.target <= horizon)
                    .map(|f| (f.target, f.value))
                    .collect();
                (*id, points)
            })
            .collect();
        self.upcoming = self
            .published
            .iter()
            .filter(|f| f.model == key && f.target > now)
            .map(|f| {
                let method = match self.choice {
                    ForecastChoice::Model(id) => PredictionMethod::Model(id),
                    ForecastChoice::Best => f
                        .source
                        .as_deref()
                        .and_then(ModelId::from_key)
                        .map_or(PredictionMethod::HistoricalAverage, PredictionMethod::Model),
                };
                PredictionWithConfidence::new(
                    f.target,
                    f.value,
                    f.low.unwrap_or(f.value),
                    f.high.unwrap_or(f.value),
                    0.5,
                    method,
                )
            })
            .collect();
        self.charted = self.upcoming.partition_point(|p| p.timestamp <= horizon);
        let profile = SlotProfile::from_history(self.history.view(), schedule.timezone());
        let usable = self
            .made_at()
            .is_some_and(|made| now - made <= IGNORED_AFTER);
        let forecasts: &[PredictionWithConfidence] = if usable { &self.upcoming } else { &[] };
        self.quiet_window =
            next_quiet_window(now, schedule, forecasts, &profile, self.opening_grace);
    }

    /// When the default forecast was made.
    fn made_at(&self) -> Option<DateTime<Utc>> {
        self.published
            .iter()
            .filter(|f| f.model == hardy_ml::live::SELECTED)
            .map(|f| f.made_at)
            .max()
    }

    /// A note when the forecast is old, i.e. the forecaster seems offline.
    pub(crate) fn forecast_notice(&self, now: DateTime<Utc>, tz: Tz) -> Option<String> {
        let made = self.made_at().filter(|made| now - *made > STALE_AFTER)?;
        Some(format!(
            "Forecast from {} – forecaster offline",
            made.with_timezone(&tz).format("%H:%M")
        ))
    }

    /// The forecast within the chart's horizon.
    pub(crate) fn forecasts(&self) -> &[PredictionWithConfidence] {
        &self.upcoming[..self.charted]
    }

    pub(crate) fn quiet_window(&self) -> Option<&QuietWindow> {
        self.quiet_window.as_ref()
    }

    /// The measured reading closest to `t`, within five minutes.
    pub(crate) fn reading_near(&self, t: DateTime<Utc>) -> Option<f64> {
        self.history.view().value_near(t, TimeDelta::minutes(5))
    }

    /// Name of the model behind the next forecast value.
    pub(crate) fn forecast_source(&self) -> Option<&'static str> {
        self.upcoming.first().and_then(|p| match p.method {
            PredictionMethod::Model(id) => Some(id.name()),
            PredictionMethod::HistoricalAverage => None,
        })
    }

    /// The models and their benchmark, best first.
    pub(crate) fn leaderboard(&self) -> &Leaderboard {
        &self.leaderboard
    }

    /// Why the daemon's last run failed, if it did.
    pub(crate) fn last_error(&self) -> Option<&str> {
        self.status.state.last_error.as_deref()
    }

    /// A run was requested and the daemon has not started it yet.
    pub(crate) fn run_pending(&self) -> bool {
        let state = &self.status.state;
        state
            .retrain_requested_at
            .is_some_and(|requested| state.last_attempt_at.is_none_or(|a| a < requested))
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use hardy_core::db::DataSource;

    use super::*;

    fn at(minute: i64) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2024, 6, 17, 8, 0, 0)
            .single()
            .unwrap_or_default()
            + TimeDelta::minutes(minute)
    }

    fn log(id: i64, minute: i64, source: DataSource) -> OccupancyLog {
        OccupancyLog {
            id,
            timestamp: at(minute),
            percentage: 30.0,
            source,
        }
    }

    fn published(minute: i64, value: f64) -> LiveForecast {
        LiveForecast {
            model: "selected".to_string(),
            target: at(minute),
            made_at: at(0),
            value,
            low: Some(value - 4.0),
            high: None,
            source: Some("naive".to_string()),
        }
    }

    #[test]
    fn test_an_old_forecast_is_flagged_then_left_out_of_the_quiet_hour() {
        let schedule = GymSchedule::default();
        let tz = schedule.timezone();
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        // Made at 10:00 Berlin (08:00 UTC), every quarter hour for 15 hours.
        f.set_published((1..60).map(|k| published(15 * k, 10.0)).collect());
        f.refresh(&schedule, at(20));
        assert_eq!(f.forecast_notice(at(20), tz), None);
        assert!(
            f.quiet_window()
                .is_some_and(|w| w.source == crate::quiet_window::Source::Forecast)
        );

        f.refresh(&schedule, at(45));
        assert_eq!(
            f.forecast_notice(at(45), tz).as_deref(),
            Some("Forecast from 10:00 – forecaster offline")
        );
        assert!(
            f.quiet_window()
                .is_some_and(|w| w.source == crate::quiet_window::Source::Forecast)
        );

        f.refresh(&schedule, at(61));
        assert!(
            f.quiet_window()
                .is_none_or(|w| w.source == crate::quiet_window::Source::Averages)
        );
        assert!(!f.forecasts().is_empty(), "the line stays on the chart");
    }

    #[test]
    fn test_history_fetch_is_incremental() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        let now = at(100);
        assert_eq!(
            f.history_fetch_start(now),
            now - TimeDelta::days(HISTORY_DAYS)
        );

        f.add_history(&[
            log(1, 0, DataSource::Measured),
            log(2, 1, DataSource::Interpolated),
            log(3, 2, DataSource::Measured),
        ]);
        assert_eq!(f.history_fetch_start(now), at(2) + TimeDelta::seconds(1));

        f.add_history(&[log(4, 3, DataSource::Measured)]);
        assert_eq!(f.history_fetch_start(now), at(3) + TimeDelta::seconds(1));
    }

    #[test]
    fn test_run_pending_until_attempted() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        assert!(!f.run_pending());
        let status = |requested, attempted| SuiteStatus {
            state: MlState {
                retrain_requested_at: Some(at(requested)),
                last_attempt_at: Some(at(attempted)),
                last_error: None,
            },
            ..SuiteStatus::default()
        };
        f.set_status(status(10, 5));
        assert!(f.run_pending());
        f.set_status(status(10, 11));
        assert!(!f.run_pending());
    }

    #[test]
    fn test_refresh_keeps_future_points_and_caps_the_chart() {
        let mut f = Forecasting::new(1, TimeDelta::hours(1));
        f.set_published(vec![
            published(0, 10.0),
            published(30, 20.0),
            published(60, 30.0),
            published(90, 40.0),
        ]);
        f.refresh(&GymSchedule::default(), at(15));
        let charted: Vec<f64> = f.forecasts().iter().map(|p| p.predicted_value).collect();
        assert_eq!(charted, [20.0, 30.0], "within one hour of now");
        assert_eq!(f.forecasts()[0].confidence_low, 16.0);
        assert_eq!(f.forecasts()[0].confidence_high, 20.0, "no upper bound");
        assert_eq!(f.forecast_source(), Some("Persistence"));
        assert!(f.quiet_window().is_some());
    }

    fn row(model: &str, minute: i64, value: f64) -> LiveForecast {
        LiveForecast {
            model: model.to_string(),
            source: None,
            ..published(minute, value)
        }
    }

    #[test]
    fn test_choosing_a_model_shows_its_forecast() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        f.set_published(vec![
            published(30, 20.0),
            row("naive", 30, 25.0),
            row("similar_days", 30, 35.0),
        ]);
        f.refresh(&GymSchedule::default(), at(0));
        assert_eq!(f.choice(), ForecastChoice::Best);
        assert_eq!(
            f.available_models(),
            vec![ModelId::Naive, ModelId::SimilarDays]
        );

        f.choose(ForecastChoice::Model(ModelId::SimilarDays));
        f.refresh(&GymSchedule::default(), at(0));
        assert_eq!(f.forecasts()[0].predicted_value, 35.0);
        assert_eq!(f.forecast_source(), Some("Similar days"));
    }

    #[test]
    fn test_a_model_without_a_forecast_falls_back_to_the_best() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        f.set_published(vec![published(30, 20.0), row("naive", 30, 25.0)]);
        f.choose(ForecastChoice::Model(ModelId::Naive));
        // The next publish has no forecast from it any more.
        f.set_published(vec![published(30, 21.0)]);
        f.refresh(&GymSchedule::default(), at(0));
        assert_eq!(f.choice(), ForecastChoice::Best);
        assert_eq!(f.forecasts()[0].predicted_value, 21.0);
    }

    #[test]
    fn test_overlays_are_unique_capped_and_drawn_within_the_horizon() {
        let mut f = Forecasting::new(1, TimeDelta::hours(1));
        let mut rows = vec![published(30, 20.0)];
        for id in ModelId::ALL {
            rows.push(row(id.key(), 30, 40.0));
            rows.push(row(id.key(), 120, 50.0));
        }
        f.set_published(rows);
        for id in ModelId::ALL {
            f.add_overlay(*id);
        }
        f.add_overlay(ModelId::Naive);
        assert_eq!(f.overlays().len(), MAX_OVERLAYS);
        assert_eq!(f.overlays()[0], ModelId::ALL[0]);

        f.refresh(&GymSchedule::default(), at(0));
        let lines = f.overlay_lines();
        assert_eq!(lines.len(), MAX_OVERLAYS);
        assert_eq!(lines[0].1, vec![(at(30), 40.0)], "beyond one hour is cut");

        f.remove_overlay(ModelId::ALL[0]);
        assert_eq!(f.overlays().len(), MAX_OVERLAYS - 1);
        assert!(!f.overlays().contains(&ModelId::ALL[0]));
    }

    #[test]
    fn test_pickers_offer_published_models_by_family_without_the_shown_ones() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        let mut rows = vec![published(30, 20.0)];
        for id in [
            ModelId::Mlp,
            ModelId::Ridge,
            ModelId::WeeklySeasonalNaive,
            ModelId::Naive,
        ] {
            rows.push(row(id.key(), 30, 30.0));
        }
        f.set_published(rows);
        f.choose(ForecastChoice::Model(ModelId::Mlp));
        f.add_overlay(ModelId::Ridge);
        assert_eq!(
            f.available_models(),
            [
                ModelId::Naive,
                ModelId::WeeklySeasonalNaive,
                ModelId::Ridge,
                ModelId::Mlp
            ]
        );
        let choices = f.forecast_picker().options();
        assert_eq!(choices[0], ForecastChoice::Best);
        assert_eq!(choices.len(), 5);
        let addable: Vec<ModelId> = f.compare_picker().options().iter().map(|o| o.0).collect();
        assert_eq!(addable, [ModelId::Naive, ModelId::WeeklySeasonalNaive]);
        f.remove_overlay(ModelId::Ridge);
        assert_eq!(f.compare_picker().options().len(), 3);
    }

    #[test]
    fn test_model_labels_name_the_family_for_searching() {
        assert_eq!(
            ModelOption(ModelId::PatchTst).to_string(),
            "PatchTST · Neural networks"
        );
        assert_eq!(
            ForecastChoice::Model(ModelId::Naive).to_string(),
            "Persistence · Baselines"
        );
        assert_eq!(ForecastChoice::Best.to_string(), "Best per horizon");
    }

    #[test]
    fn test_nothing_published_means_no_forecast() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        f.refresh(&GymSchedule::default(), at(0));
        assert_eq!(f.forecasts().len(), 0);
        assert_eq!(f.forecast_source(), None);
    }

    #[test]
    fn test_reading_near_finds_close_measurement_only() {
        let mut f = Forecasting::new(6, TimeDelta::hours(1));
        f.add_history(&[
            log(1, 0, DataSource::Measured),
            log(2, 30, DataSource::Measured),
        ]);
        assert_eq!(f.reading_near(at(2)), Some(30.0));
        assert_eq!(f.reading_near(at(15)), None);
    }
}
