//! The five views and the components they share.

pub mod components;
pub mod data;
pub mod insights;
pub mod model_data;
pub mod now;
pub mod opening;
pub mod schema_notice;
pub mod week;

pub use data::DataProps;
pub use insights::InsightsProps;
pub use model_data::ModelDataProps;
pub use now::NowProps;
pub use week::WeekProps;
