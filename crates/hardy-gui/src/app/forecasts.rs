//! The Forecasts tab: loading a period's performance data, the sub-page,
//! which numbers the leaderboard shows, the model picked for its detail,
//! and the charts' caches.

use hardy_core::error::AppError;
use hardy_ml::suite::ModelId;
use iced::Task;

use super::{
    HardyMonitorApp, Message, RequestId,
    loads::{ErrorSource, LatestOnly},
    tasks::db_err,
};
use crate::{
    performance::{ModelDetail, Performance, PerformanceData, Period, Source, Tab, detail},
    views::forecasts::Caches,
};

/// The Forecasts tab's state (this session only).
#[derive(Default)]
pub(crate) struct ForecastsPage {
    pub tab: Tab,
    pub period: Period,
    pub source: Source,
    /// The model whose detail is open.
    pub picked: Option<ModelId>,
    requests: LatestOnly,
    pub performance: Option<Performance>,
    pub detail: Option<ModelDetail>,
    pub caches: Caches,
}

impl ForecastsPage {
    fn refresh_detail(&mut self) {
        self.detail = self
            .picked
            .zip(self.performance.as_ref())
            .map(|(id, p)| detail(id, &p.nights, &p.model_errors));
        self.caches.clear_detail();
    }

    /// Picks `id`, or closes its detail if it is open.
    fn pick(&mut self, id: ModelId) {
        self.picked = (self.picked != Some(id)).then_some(id);
        self.refresh_detail();
    }

    fn apply(&mut self, performance: Performance) {
        self.performance = Some(performance);
        self.caches.clear_all();
        self.refresh_detail();
    }
}

impl HardyMonitorApp {
    /// Loads the performance data of the chosen period.
    pub(super) fn load_performance(&mut self) -> Task<Message> {
        let now = self.clock.now_utc();
        let tz = self.schedule.timezone();
        let window = self
            .forecasts
            .period
            .window(now, tz, self.data.first_reading);
        let id = self.forecasts.requests.issue();
        let db = self.db.clone();
        self.load(
            async move {
                let (history, model_errors, nights) = tokio::join!(
                    db.scored_forecast_history(window.load_from(), now, tz),
                    db.model_forecast_errors(window.since, now, tz),
                    db.benchmark_history(window.since),
                );
                let data = PerformanceData {
                    window,
                    history: history?,
                    model_errors: model_errors?,
                    nights: nights?,
                };
                // Off the UI thread: "All" can be a year of forecasts.
                anyhow::Ok(Box::new(Performance::build(data, tz)))
            },
            move |r| Message::PerformanceLoaded(id, r.map_err(db_err("load_performance"))),
        )
    }

    /// Sub-page, period, leaderboard numbers, model picked and loaded data.
    pub(super) fn update_forecasts(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ForecastsTabSelected(tab) => self.forecasts.tab = tab,
            Message::ForecastsPeriodSelected(period) => {
                self.forecasts.period = period;
                return self.load_performance();
            }
            Message::LeaderboardSourceSelected(source) => self.forecasts.source = source,
            Message::ModelPicked(id) => self.forecasts.pick(id),
            Message::PerformanceLoaded(id, result) => self.apply_performance(id, result),
            _ => {}
        }
        Task::none()
    }

    fn apply_performance(&mut self, id: RequestId, result: Result<Box<Performance>, AppError>) {
        // A newer load for another period is on its way.
        if self.forecasts.requests.is_current(id)
            && let Some(performance) = self.errors.record(ErrorSource::Performance, result)
        {
            self.forecasts.apply(*performance);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_picking_a_model_again_closes_its_detail() {
        let mut page = ForecastsPage::default();
        page.pick(ModelId::Naive);
        assert_eq!(page.picked, Some(ModelId::Naive));
        // No data yet: nothing to show in the detail.
        assert_eq!(page.detail, None);
        page.pick(ModelId::Theta);
        assert_eq!(page.picked, Some(ModelId::Theta));
        page.pick(ModelId::Theta);
        assert_eq!(page.picked, None);
    }
}
