//! N-BEATS (Oreshkin et al. 2020) and N-HiTS (Challu et al. 2023): stacks
//! of fully connected blocks on the last week of deviations. Each block
//! explains part of its input (backcast), passes the rest on and adds its
//! part of the forecast. N-HiTS blocks first pool the input and forecast a
//! few points, interpolated to every slot: coarse blocks learn the slow
//! shape, fine ones the detail.

use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

use super::{WEEK, deviations, zero_head};
use crate::{
    grid::HORIZON_STEPS,
    suite::neural::{Architecture, train::Net},
};

const WIDTH: usize = 64;
/// Fully connected layers per block.
const DEPTH: usize = 3;
const BLOCKS: usize = 3;
/// N-HiTS: input pooling and forecast points per block.
const POOLS: [usize; BLOCKS] = [8, 4, 1];
const POINTS: [usize; BLOCKS] = [8, 24, HORIZON_STEPS];

/// See [`super::NBeats`].
#[derive(Debug)]
pub struct NBeatsDesign;

/// See [`super::NHits`].
#[derive(Debug)]
pub struct NHitsDesign;

/// One block: a fully connected stack with a backcast and a forecast head.
#[derive(Module, Debug)]
pub struct Block {
    layers: Vec<Linear>,
    backcast: Linear,
    forecast: Linear,
    activation: Relu,
}

impl Block {
    fn new(input: usize, points: usize, device: &Device) -> Self {
        Self {
            layers: (0..DEPTH)
                .map(|i| LinearConfig::new(if i == 0 { input } else { WIDTH }, WIDTH).init(device))
                .collect(),
            backcast: LinearConfig::new(WIDTH, WEEK).init(device),
            forecast: zero_head(WIDTH, points, device),
            activation: Relu::new(),
        }
    }

    /// (backcast, forecast points) of `x`.
    fn forward(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let hidden = self
            .layers
            .iter()
            .fold(x, |x, layer| self.activation.forward(layer.forward(x)));
        (
            self.backcast.forward(hidden.clone()),
            self.forecast.forward(hidden),
        )
    }
}

/// See the module docs.
#[derive(Module, Debug)]
pub struct BasisNet {
    blocks: Vec<Block>,
    /// Whether the blocks pool and interpolate (N-HiTS).
    hierarchical: bool,
}

/// Max-pools `[batch, length]` over windows of `size`.
fn pool(x: Tensor<2>, size: usize) -> Tensor<2> {
    if size == 1 {
        return x;
    }
    let [batch, length] = x.dims();
    x.reshape([batch, length / size, size])
        .max_dim(2)
        .reshape([batch, length / size])
}

/// The `[points, HORIZON_STEPS]` matrix interpolating `points` evenly
/// spaced values linearly to every step.
fn interpolation(points: usize) -> Vec<f32> {
    let mut matrix = vec![0.0_f32; points * HORIZON_STEPS];
    #[allow(clippy::cast_precision_loss)]
    let spacing = (HORIZON_STEPS - 1) as f32 / (points.max(2) - 1) as f32;
    for step in 0..HORIZON_STEPS {
        #[allow(clippy::cast_precision_loss)]
        let position = step as f32 / spacing;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let below = (position.floor() as usize).min(points - 1);
        let above = (below + 1).min(points - 1);
        #[allow(clippy::cast_precision_loss)]
        let weight = position - below as f32;
        matrix[below * HORIZON_STEPS + step] += 1.0 - weight;
        if above != below {
            matrix[above * HORIZON_STEPS + step] += weight;
        }
    }
    matrix
}

impl Net for BasisNet {
    fn forward(&self, series: Tensor<3>, _: Tensor<2>) -> Tensor<2> {
        let mut residual = deviations(series);
        let [batch, _] = residual.dims();
        let device = residual.device();
        let mut forecast = Tensor::<2>::zeros([batch, HORIZON_STEPS], &device);
        for (i, block) in self.blocks.iter().enumerate() {
            let input = if self.hierarchical {
                pool(residual.clone(), POOLS[i])
            } else {
                residual.clone()
            };
            let (backcast, points) = block.forward(input);
            residual = residual - backcast;
            forecast = forecast
                + if self.hierarchical && POINTS[i] != HORIZON_STEPS {
                    let matrix = Tensor::<2>::from_data(
                        TensorData::new(interpolation(POINTS[i]), [POINTS[i], HORIZON_STEPS]),
                        &device,
                    );
                    points.matmul(matrix)
                } else {
                    points
                };
        }
        forecast
    }
}

impl Architecture for NBeatsDesign {
    type Net = BasisNet;

    const INPUT: usize = WEEK;
    const MIN_DAYS: i64 = 8;

    fn build(device: &Device) -> BasisNet {
        BasisNet {
            blocks: (0..BLOCKS)
                .map(|_| Block::new(WEEK, HORIZON_STEPS, device))
                .collect(),
            hierarchical: false,
        }
    }
}

impl Architecture for NHitsDesign {
    type Net = BasisNet;

    const INPUT: usize = WEEK;
    const MIN_DAYS: i64 = 8;

    fn build(device: &Device) -> BasisNet {
        BasisNet {
            blocks: POOLS
                .iter()
                .zip(POINTS)
                .map(|(&size, points)| Block::new(WEEK / size, points, device))
                .collect(),
            hierarchical: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::suite::neural::nets::test_shapes::check;

    #[test]
    fn test_basis_shapes() -> anyhow::Result<()> {
        check::<NBeatsDesign>()?;
        check::<NHitsDesign>()
    }

    #[test]
    fn test_interpolation_hits_the_points_and_is_linear_between() {
        let points = 4;
        let matrix = interpolation(points);
        let values = [0.0_f32, 3.0, 6.0, 30.0];
        let at = |step: usize| -> f32 {
            (0..points)
                .map(|j| values[j] * matrix[j * HORIZON_STEPS + step])
                .sum()
        };
        // Points at steps 0, 31.67, 63.33 and 95.
        assert_relative_eq!(at(0), 0.0);
        assert_relative_eq!(at(95), 30.0, epsilon = 1e-4);
        assert_relative_eq!(at(19), 3.0 * 19.0 / (95.0 / 3.0), epsilon = 1e-4);
        for step in 0..HORIZON_STEPS {
            let column: f32 = (0..points).map(|j| matrix[j * HORIZON_STEPS + step]).sum();
            assert_relative_eq!(column, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_pool_takes_the_maximum_of_each_window() -> anyhow::Result<()> {
        let device = Device::flex();
        let x = Tensor::<2>::from_data(
            TensorData::new(vec![1.0_f32, 5.0, 2.0, 0.0, 3.0, 4.0, 9.0, 8.0], [1, 8]),
            &device,
        );
        let pooled = pool(x, 4)
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        assert_eq!(pooled, vec![5.0, 9.0]);
        Ok(())
    }
}
