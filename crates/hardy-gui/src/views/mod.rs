//! The four views and the components they share.

pub mod components;
pub mod insights;
pub mod model_data;
pub mod now;
pub mod opening;
pub mod schema_notice;
pub mod week;

pub use insights::InsightsProps;
pub use model_data::ModelDataProps;
pub use now::NowProps;
pub use week::WeekProps;
