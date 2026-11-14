//! The views and the components they share.

pub mod components;
pub mod data;
pub mod forecasts;
pub mod insights;
pub mod maintenance;
pub mod now;
pub mod opening;
pub mod plan;
pub mod schema_notice;
pub mod week;

pub use data::DataProps;
pub use forecasts::ForecastsProps;
pub use insights::InsightsProps;
pub use maintenance::MaintenanceProps;
pub use now::NowProps;
pub use plan::PlanProps;
pub use week::WeekProps;
