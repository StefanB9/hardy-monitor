//! `PatchTST` (Nie et al. 2023): the last week of deviations cut into
//! patches of 16 slots, each embedded as a token with a learned position;
//! a transformer encoder relates the patches and a linear head maps all
//! tokens to the forecast.

use burn::{
    module::Param,
    nn::{
        Initializer, Linear, LinearConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
};

use super::{WEEK, deviations, zero_head};
use crate::{
    grid::HORIZON_STEPS,
    suite::neural::{Architecture, train::Net},
};

const PATCH: usize = 16;
const PATCHES: usize = WEEK / PATCH;
const MODEL: usize = 32;
const HEADS: usize = 4;
const FEED_FORWARD: usize = 64;
const LAYERS: usize = 2;

/// See [`super::PatchTst`].
#[derive(Debug)]
pub struct PatchTstDesign;

/// See the module docs.
#[derive(Module, Debug)]
pub struct PatchNet {
    embed: Linear,
    position: Param<Tensor<3>>,
    encoder: TransformerEncoder,
    head: Linear,
}

impl Net for PatchNet {
    fn forward(&self, series: Tensor<3>, _: Tensor<2>) -> Tensor<2> {
        let x = deviations(series);
        let [batch, _] = x.dims();
        let tokens = self.embed.forward(x.reshape([batch, PATCHES, PATCH])) + self.position.val();
        let encoded = self.encoder.forward(TransformerEncoderInput::new(tokens));
        self.head.forward(encoded.reshape([batch, PATCHES * MODEL]))
    }
}

impl Architecture for PatchTstDesign {
    type Net = PatchNet;

    const INPUT: usize = WEEK;
    const MIN_DAYS: i64 = 8;

    fn build(device: &Device) -> PatchNet {
        PatchNet {
            embed: LinearConfig::new(PATCH, MODEL).init(device),
            position: Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init([1, PATCHES, MODEL], device),
            // No dropout: a fit must be reproducible.
            encoder: TransformerEncoderConfig::new(MODEL, FEED_FORWARD, HEADS, LAYERS)
                .with_dropout(0.0)
                .with_norm_first(true)
                .init(device),
            head: zero_head(PATCHES * MODEL, HORIZON_STEPS, device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::neural::nets::test_shapes::check;

    #[test]
    fn test_patchtst_shapes() -> anyhow::Result<()> {
        check::<PatchTstDesign>()
    }
}
