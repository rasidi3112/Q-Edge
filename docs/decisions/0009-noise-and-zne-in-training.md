# 0009. Noise on the device, ZNE in every forward pass

## Context

Research question 1 compares a VQC head with a classical head with and
without noise and with and without zero-noise extrapolation. The old code
ran "ZNE" on a noiseless simulator, folded circuits by hand, and returned
NaN at scale factors above 1 (Q3, Q6). Noisy simulation must amplify noise
when a circuit is folded and must stay fast enough for FL rounds.

## Decision

- Gate noise is a `qml.NoiseModel` attached to `default.mixed` with
  `qml.add_noise(device, ...)`. Device transforms run after QNode transforms,
  so `qml.noise.fold_global` folds gates first and every folded gate gets
  its own channel. Readout noise is a device transform that appends bit
  flips after the last gate, so folding does not amplify it.
- ZNE uses PennyLane's `fold_global` and its Richardson, polynomial and
  exponential extrapolation. The "adaptive" selection is removed (Q4).
- In models, ZNE wraps the QNode (`mitigate_with_zne`), so training and
  evaluation both see mitigated outputs. PennyLane extrapolates over the
  requested factors, so that path accepts only odd integer factors, which
  folding reaches exactly. `zne.estimate` extrapolates over the achieved
  factors for any input, and run records store them.
- Gradients come from backpropagation through the simulator when there are
  no shots, on `default.mixed` as well. They are exact gradients of the
  noisy, mitigated model. With `--shots`, the parameter-shift rule is used,
  as on hardware.

## Consequences

- Noisy runs are about 2x slower than noiseless ones, and ZNE with factors
  1, 3, 5 is about 8x slower still (261 s vs 5 s for 10 clients and 30
  rounds on breast_cancer).
- Backprop gradients are not available on hardware. Results with shots
  are the ones comparable to a device, and they are much slower.
- The simulated noise model (one channel per gate and wire) is simpler than
  real devices: there is no crosstalk, no idle noise and no coherent errors.
