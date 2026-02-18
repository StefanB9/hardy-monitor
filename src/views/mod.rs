//! View Modules for Hardy Monitor
//!
//! This module contains the extracted view components, each accepting
//! only the specific data references they need rather than the entire app
//! state.

pub mod components;
pub mod dashboard;
pub mod data_repair;
pub mod insights;
pub mod weekly_pattern;

pub use dashboard::DashboardProps;
pub use data_repair::DataRepairProps;
pub use insights::InsightsProps;
pub use weekly_pattern::WeeklyPatternProps;
