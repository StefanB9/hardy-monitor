//! The network designs.

mod basis;
mod conv;
mod dense;
mod patch;
mod recurrent;

use burn::{
    nn::{Initializer, Linear, LinearConfig},
    prelude::*,
};

pub use self::{
    basis::{NBeatsDesign, NHitsDesign},
    conv::TcnDesign,
    dense::MlpDesign,
    patch::PatchTstDesign,
    recurrent::{GruDesign, LstmDesign},
};
use super::Neural;

/// Multi-layer perceptron on the last day of slots and the covariates.
pub type Mlp = Neural<MlpDesign>;
/// Temporal convolutional network on the last week of slots.
pub type Tcn = Neural<TcnDesign>;
/// Gated recurrent unit network on the last day of slots.
pub type Gru = Neural<GruDesign>;
/// Long short-term memory network on the last day of slots.
pub type Lstm = Neural<LstmDesign>;
/// N-BEATS (generic blocks) on the last week of deviations.
pub type NBeats = Neural<NBeatsDesign>;
/// N-HiTS on the last week of deviations.
pub type NHits = Neural<NHitsDesign>;
/// `PatchTST` (patches of 16 slots, transformer encoder) on the last week of
/// deviations.
pub type PatchTst = Neural<PatchTstDesign>;

/// Slots in a day and in a week.
const DAY: usize = 96;
const WEEK: usize = 672;

/// A linear output layer starting at zero: an untrained network forecasts
/// no deviation, i.e. the weekly profile.
fn zero_head(input: usize, output: usize, device: &Device) -> Linear {
    LinearConfig::new(input, output)
        .with_initializer(Initializer::Zeros)
        .init(device)
}

/// The deviation channel of `series` as `[batch, input]`.
fn deviations(series: Tensor<3>) -> Tensor<2> {
    let [batch, input, _] = series.dims();
    series.slice(s![.., .., 1..2]).reshape([batch, input])
}

/// The last time step of `[batch, time, width]` as `[batch, width]`.
fn last_step(sequence: Tensor<3>) -> Tensor<2> {
    let [batch, time, width] = sequence.dims();
    sequence
        .slice(s![.., time - 1..time, ..])
        .reshape([batch, width])
}

#[cfg(test)]
pub(super) mod test_shapes {
    use burn::prelude::*;

    use crate::{
        grid::HORIZON_STEPS,
        suite::neural::{
            Architecture,
            train::Net,
            windows::{CHANNELS, COVARIATES},
        },
    };

    /// The design's network maps a batch of 3 windows to 3 × 96 finite
    /// values.
    pub(crate) fn check<A: Architecture>() -> anyhow::Result<()> {
        let device = Device::flex();
        let net = A::build(&device);
        let series = Tensor::<3>::ones([3, A::INPUT, CHANNELS], &device) * 0.1;
        let covariates = Tensor::<2>::ones([3, COVARIATES], &device) * 0.1;
        let out = net.forward(series, covariates);
        assert_eq!(out.dims(), [3, HORIZON_STEPS]);
        let values = out
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        assert!(values.iter().all(|v| v.is_finite()));
        Ok(())
    }
}
