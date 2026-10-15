//! A temporal convolutional network: a strided convolution turns the week
//! of slots into hours, then dilated convolutions, each doubling the
//! reach, with residual connections; the newest hour feeds the head
//! together with the covariates.

use burn::{
    nn::{
        Linear, LinearConfig, Relu,
        conv::{Conv1d, Conv1dConfig},
    },
    prelude::*,
};

use super::{WEEK, last_step, zero_head};
use crate::{
    grid::HORIZON_STEPS,
    suite::neural::{
        Architecture,
        train::Net,
        windows::{CHANNELS, COVARIATES},
    },
};

const FILTERS: usize = 16;
/// Slots per hour: the first convolution's width and stride.
const HOUR: usize = 4;
const KERNEL: usize = 3;
/// Dilations 1, 2, …, 32: the newest hour sees 127 hours back.
const LAYERS: u32 = 6;
const HIDDEN: usize = 128;

/// See [`super::Tcn`].
#[derive(Debug)]
pub struct TcnDesign;

/// See the module docs.
#[derive(Module, Debug)]
pub struct TcnNet {
    input: Conv1d,
    layers: Vec<Conv1d>,
    head: Linear,
    output: Linear,
    activation: Relu,
}

impl Net for TcnNet {
    fn forward(&self, series: Tensor<3>, covariates: Tensor<2>) -> Tensor<2> {
        let mut x = self.input.forward(series.swap_dims(1, 2));
        for (layer, dilation) in self.layers.iter().zip((0..LAYERS).map(|i| 1_usize << i)) {
            let y = self.activation.forward(layer.forward(x.clone()));
            // Unpadded: the output is shorter by the reach of the kernel.
            let [_, _, length] = x.dims();
            x = x.slice(s![.., .., (KERNEL - 1) * dilation..length]) + y;
        }
        let newest = last_step(x.swap_dims(1, 2));
        let x = Tensor::cat(vec![newest, covariates], 1);
        self.output
            .forward(self.activation.forward(self.head.forward(x)))
    }
}

impl Architecture for TcnDesign {
    type Net = TcnNet;

    const INPUT: usize = WEEK;
    const MIN_DAYS: i64 = 8;

    fn build(device: &Device) -> TcnNet {
        TcnNet {
            input: Conv1dConfig::new(CHANNELS, FILTERS, HOUR)
                .with_stride(HOUR)
                .init(device),
            layers: (0..LAYERS)
                .map(|i| {
                    Conv1dConfig::new(FILTERS, FILTERS, KERNEL)
                        .with_dilation(1 << i)
                        .init(device)
                })
                .collect(),
            head: LinearConfig::new(FILTERS + COVARIATES, HIDDEN).init(device),
            output: zero_head(HIDDEN, HORIZON_STEPS, device),
            activation: Relu::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::neural::nets::test_shapes::check;

    #[test]
    fn test_tcn_shapes() -> anyhow::Result<()> {
        check::<TcnDesign>()
    }
}
