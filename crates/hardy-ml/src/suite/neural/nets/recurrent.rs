//! Recurrent networks: a GRU or an LSTM reads the last day slot by slot;
//! its final state and the covariates feed the head.

use burn::{
    nn::{
        Linear, LinearConfig, Relu,
        gru::{Gru, GruConfig},
        lstm::{Lstm, LstmConfig},
    },
    prelude::*,
};

use super::{DAY, last_step, zero_head};
use crate::{
    grid::HORIZON_STEPS,
    suite::neural::{
        Architecture,
        train::Net,
        windows::{CHANNELS, COVARIATES},
    },
};

const STATE: usize = 64;
const HIDDEN: usize = 128;

/// See [`super::Gru`].
#[derive(Debug)]
pub struct GruDesign;

/// See [`super::Lstm`].
#[derive(Debug)]
pub struct LstmDesign;

/// See the module docs.
#[derive(Module, Debug)]
pub struct GruNet {
    cell: Gru,
    head: Linear,
    output: Linear,
    activation: Relu,
}

/// See the module docs.
#[derive(Module, Debug)]
pub struct LstmNet {
    cell: Lstm,
    head: Linear,
    output: Linear,
    activation: Relu,
}

impl Net for GruNet {
    fn forward(&self, series: Tensor<3>, covariates: Tensor<2>) -> Tensor<2> {
        let state = last_step(self.cell.forward(series, None));
        let x = Tensor::cat(vec![state, covariates], 1);
        self.output
            .forward(self.activation.forward(self.head.forward(x)))
    }
}

impl Net for LstmNet {
    fn forward(&self, series: Tensor<3>, covariates: Tensor<2>) -> Tensor<2> {
        let (outputs, _) = self.cell.forward(series, None);
        let x = Tensor::cat(vec![last_step(outputs), covariates], 1);
        self.output
            .forward(self.activation.forward(self.head.forward(x)))
    }
}

impl Architecture for GruDesign {
    type Net = GruNet;

    const INPUT: usize = DAY;
    const MIN_DAYS: i64 = 2;

    fn build(device: &Device) -> GruNet {
        GruNet {
            cell: GruConfig::new(CHANNELS, STATE, true).init(device),
            head: LinearConfig::new(STATE + COVARIATES, HIDDEN).init(device),
            output: zero_head(HIDDEN, HORIZON_STEPS, device),
            activation: Relu::new(),
        }
    }
}

impl Architecture for LstmDesign {
    type Net = LstmNet;

    const INPUT: usize = DAY;
    const MIN_DAYS: i64 = 2;

    fn build(device: &Device) -> LstmNet {
        LstmNet {
            cell: LstmConfig::new(CHANNELS, STATE, true).init(device),
            head: LinearConfig::new(STATE + COVARIATES, HIDDEN).init(device),
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
    fn test_recurrent_shapes() -> anyhow::Result<()> {
        check::<GruDesign>()?;
        check::<LstmDesign>()
    }
}
