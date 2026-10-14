//! Training: Adam on the masked squared error of the target deviations, in
//! mini-batches of a seeded shuffle, with early stopping on the validation
//! windows (the best epoch's weights are kept).

use std::sync::Mutex;

use burn::{
    grad_clipping::GradientClippingConfig,
    optim::{AdamConfig, GradientsParams},
    prelude::*,
};

use super::windows::{CHANNELS, COVARIATES, Windows};
use crate::{error::MlError, grid::HORIZON_STEPS};

/// A network: input slots `[batch, input, CHANNELS]` and covariates
/// `[batch, COVARIATES]` to deviations `[batch, HORIZON_STEPS]`.
pub trait Net: Module {
    fn forward(&self, series: Tensor<3>, covariates: Tensor<2>) -> Tensor<2>;
}

/// How a network is trained.
#[derive(Debug, Clone, Copy)]
pub(super) struct Training {
    pub(super) max_epochs: usize,
    /// Epochs without a better validation loss before stopping.
    pub(super) patience: usize,
    pub(super) batch: usize,
    pub(super) learning_rate: f64,
}

impl Default for Training {
    fn default() -> Self {
        Self {
            max_epochs: 60,
            patience: 8,
            batch: 64,
            learning_rate: 1e-2,
        }
    }
}

/// Fixed seed of weight initialisation and shuffling.
const SEED: u64 = 42;
/// Largest gradient norm per step (keeps the recurrent nets stable).
const MAX_GRADIENT_NORM: f32 = 1.0;

/// Burn seeds a device globally: initialising one network at a time keeps
/// parallel fits reproducible.
static INITIALISING: Mutex<()> = Mutex::new(());

/// A network built by `build` with seeded weights.
pub(super) fn seeded<N>(build: impl FnOnce(&Device) -> N, device: &Device) -> N {
    // SAFETY (poisoning): the lock guards no data, only the order of seeding
    // and building, so a panic while holding it leaves nothing to repair.
    let _guard = INITIALISING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    device.seed(SEED);
    build(device)
}

/// `SplitMix64`: a tiny, fixed pseudo-random sequence for shuffling.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn shuffle(&mut self, values: &mut [usize]) {
        for i in (1..values.len()).rev() {
            let j = usize::try_from(self.next() % (i as u64 + 1)).unwrap_or(0);
            values.swap(i, j);
        }
    }
}

/// The windows as tensors on `device`: series, covariates, targets, mask.
pub(super) fn tensors(
    windows: &Windows,
    device: &Device,
) -> (Tensor<3>, Tensor<2>, Tensor<2>, Tensor<2>) {
    let n = windows.len();
    let flat = |values: &[f32], width: usize| {
        Tensor::<2>::from_data(TensorData::new(values.to_vec(), [n, width]), device)
    };
    let series = Tensor::<3>::from_data(
        TensorData::new(windows.series.clone(), [n, windows.input(), CHANNELS]),
        device,
    );
    (
        series,
        flat(&windows.covariates, COVARIATES),
        flat(&windows.targets, HORIZON_STEPS),
        flat(&windows.mask, HORIZON_STEPS),
    )
}

/// Mean squared error over the targets that count.
fn masked_loss(prediction: Tensor<2>, targets: Tensor<2>, mask: Tensor<2>) -> Tensor<1> {
    let count = mask.clone().sum().clamp_min(1.0);
    ((prediction - targets).powf_scalar(2.0) * mask).sum() / count
}

/// The masked loss of `net` on `windows` (no gradients).
pub(super) fn evaluate<N: Net>(net: &N, windows: &Windows, batch: usize) -> f64 {
    let device = Device::flex();
    let (mut total, mut count) = (0.0, 0.0);
    let all: Vec<usize> = (0..windows.len()).collect();
    for chunk in all.chunks(batch.max(1)) {
        let part = windows.select(chunk);
        let weight: f64 = part.mask.iter().map(|&m| f64::from(m)).sum();
        let (series, covariates, targets, mask) = tensors(&part, &device);
        let loss = masked_loss(net.forward(series, covariates), targets, mask);
        total += f64::from(loss.into_scalar::<f32>()) * weight;
        count += weight;
    }
    if count > 0.0 {
        total / count
    } else {
        f64::INFINITY
    }
}

/// Trains a network built by `build` on `train`, stopping early on
/// `validation` (or after `settings.max_epochs` without one). Returns it
/// without gradient tracking.
pub(super) fn train<N: Net>(
    build: impl FnOnce(&Device) -> N,
    train: &Windows,
    validation: &Windows,
    settings: &Training,
) -> Result<N, MlError> {
    if train.len() == 0 {
        return Err(MlError::Fit("no training windows".to_string()));
    }
    let device = Device::flex().autodiff();
    let mut net = seeded(build, &device);
    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(MAX_GRADIENT_NORM)))
        .init();
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut random = SplitMix(SEED);
    // The untrained network is the bar every epoch has to beat.
    let mut best: Option<(f64, N)> = (validation.len() > 0).then(|| {
        let start = net.valid();
        (evaluate(&start, validation, settings.batch), start)
    });
    let mut stale = 0;
    for epoch in 0..settings.max_epochs {
        random.shuffle(&mut order);
        for chunk in order.chunks(settings.batch.max(1)) {
            let (series, covariates, targets, mask) = tensors(&train.select(chunk), &device);
            let loss = masked_loss(net.forward(series, covariates), targets, mask);
            let gradients = GradientsParams::from_grads(loss.backward(), &net);
            net = optimizer.step(settings.learning_rate, net, gradients);
        }
        if validation.len() == 0 {
            continue;
        }
        let candidate = net.valid();
        let loss = evaluate(&candidate, validation, settings.batch);
        tracing::debug!(epoch, validation = loss, "network epoch");
        if !loss.is_finite() {
            return Err(MlError::Fit(format!("validation loss {loss}")));
        }
        if best.as_ref().is_none_or(|(b, _)| loss < *b) {
            best = Some((loss, candidate));
            stale = 0;
        } else {
            stale += 1;
            if stale >= settings.patience {
                break;
            }
        }
    }
    Ok(best.map_or_else(|| net.valid(), |(_, net)| net))
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use burn::nn::{Linear, LinearConfig};

    use super::*;

    /// One linear layer on the covariates only.
    #[derive(Module, Debug)]
    struct Linearly {
        layer: Linear,
    }

    impl Net for Linearly {
        fn forward(&self, _: Tensor<3>, covariates: Tensor<2>) -> Tensor<2> {
            self.layer.forward(covariates)
        }
    }

    fn build(device: &Device) -> Linearly {
        Linearly {
            layer: LinearConfig::new(COVARIATES, HORIZON_STEPS).init(device),
        }
    }

    /// Windows whose every target is `0.3 × covariate[0]`, with the
    /// second half of the targets masked out and set to nonsense.
    fn windows(n: usize, seed: u64) -> Windows {
        let mut random = SplitMix(seed);
        let mut out = Windows::empty(4);
        for _ in 0..n {
            #[allow(clippy::cast_precision_loss)]
            let x = (random.next() % 1000) as f32 / 1000.0;
            out.series.extend(std::iter::repeat_n(0.0, 4 * CHANNELS));
            out.covariates.push(x);
            out.covariates
                .extend(std::iter::repeat_n(0.0, COVARIATES - 1));
            for step in 0..HORIZON_STEPS {
                let counts = step < HORIZON_STEPS / 2;
                out.targets.push(if counts { 0.3 * x } else { 9.0 });
                out.mask.push(f32::from(u8::from(counts)));
            }
        }
        out
    }

    #[test]
    fn test_training_learns_the_counted_targets_only() -> Result<()> {
        let settings = Training {
            max_epochs: 100,
            learning_rate: 1e-2,
            ..Training::default()
        };
        let net = train(build, &windows(400, 1), &windows(100, 2), &settings)?;
        let loss = evaluate(&net, &windows(100, 3), 64);
        assert!(loss < 1e-3, "loss {loss}");
        let (series, covariates, _, _) = tensors(&windows(1, 4), &Device::flex());
        let x = covariates.clone().slice([0..1, 0..1]).into_scalar::<f32>();
        let out = net
            .forward(series, covariates)
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        assert!((out[0] - 0.3 * x).abs() < 0.05, "{} vs {}", out[0], 0.3 * x);
        Ok(())
    }

    #[test]
    fn test_training_is_reproducible() -> Result<()> {
        let settings = Training {
            max_epochs: 3,
            ..Training::default()
        };
        let a = train(build, &windows(200, 1), &windows(50, 2), &settings)?;
        let b = train(build, &windows(200, 1), &windows(50, 2), &settings)?;
        let (series, covariates, _, _) = tensors(&windows(5, 9), &Device::flex());
        let out = |n: &Linearly| n.forward(series.clone(), covariates.clone()).into_data();
        assert_eq!(out(&a), out(&b));
        Ok(())
    }

    #[test]
    fn test_training_needs_windows() {
        assert!(
            train(
                build,
                &Windows::empty(4),
                &Windows::empty(4),
                &Training::default()
            )
            .is_err()
        );
    }

    #[test]
    fn test_shuffle_is_a_fixed_permutation() {
        let mut values: Vec<usize> = (0..50).collect();
        SplitMix(SEED).shuffle(&mut values);
        let mut again: Vec<usize> = (0..50).collect();
        SplitMix(SEED).shuffle(&mut again);
        assert_eq!(values, again);
        assert_ne!(values, (0..50).collect::<Vec<_>>());
        values.sort_unstable();
        assert_eq!(values, (0..50).collect::<Vec<_>>());
    }
}
