//! A multi-layer perceptron.

use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

use super::{DAY, zero_head};
use crate::{
    grid::HORIZON_STEPS,
    suite::neural::{
        Architecture,
        train::Net,
        windows::{CHANNELS, COVARIATES},
    },
};

const HIDDEN: usize = 256;

/// See [`super::Mlp`].
#[derive(Debug)]
pub struct MlpDesign;

/// Two hidden layers on the flattened window and the covariates.
#[derive(Module, Debug)]
pub struct MlpNet {
    first: Linear,
    second: Linear,
    output: Linear,
    activation: Relu,
}

impl Net for MlpNet {
    fn forward(&self, series: Tensor<3>, covariates: Tensor<2>) -> Tensor<2> {
        let [batch, input, channels] = series.dims();
        let x = Tensor::cat(
            vec![series.reshape([batch, input * channels]), covariates],
            1,
        );
        let x = self.activation.forward(self.first.forward(x));
        let x = self.activation.forward(self.second.forward(x));
        self.output.forward(x)
    }
}

impl Architecture for MlpDesign {
    type Net = MlpNet;

    const INPUT: usize = DAY;
    const MIN_DAYS: i64 = 1;

    fn build(device: &Device) -> MlpNet {
        MlpNet {
            first: LinearConfig::new(DAY * CHANNELS + COVARIATES, HIDDEN).init(device),
            second: LinearConfig::new(HIDDEN, HIDDEN).init(device),
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
    fn test_mlp_shapes() -> anyhow::Result<()> {
        check::<MlpDesign>()
    }
}
