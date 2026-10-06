//! Shared building blocks for all views, styled from [`crate::style`].

mod controls;
mod surfaces;

pub use controls::{
    date_input, ghost_button, primary_button, secondary_button, segmented, small_button, switch,
};
pub use surfaces::{badge, card, card_with_actions, empty_state, legend_item, scroll, stat};
