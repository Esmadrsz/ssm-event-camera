# Linear State-Space Model for Event-Camera-Style Tracking

A from-scratch PyTorch implementation of a genuine **linear** State-Space
Model (SSM) — in the S4/S5/LRU family — applied to a simulated event-camera
tracking task. This project replaces an earlier version that used a
standard nonlinear RNN cell (`tanh` inside the recurrence) mislabeled as
an "SSM" — see [What Changed](#what-changed-from-the-first-version) below
for why that distinction actually matters.

## The Core Idea

A linear SSM has **no nonlinearity inside its recurrence**:

```
x_k = Ad ⊙ x_{k-1} + Bd u_k      (diagonal state transition, no tanh/relu)
y_k = C x_k
```

Because the recurrence is linear and time-invariant, it can be computed
**two mathematically equivalent ways**:

1. **Recurrent** — sequential, one step at a time. `O(1)` memory per step,
   ideal for streaming/deployment.
2. **Convolutional** — the whole sequence output is a single FFT
   convolution of the input with a fixed kernel. Fully parallel, `O(L log L)`.

This project implements both, and **proves numerically that they agree**
— that's the actual mathematical claim of the S4/S5 architecture family
(Gu et al. 2021; Smith et al. 2022) and the stable log-space diagonal
parameterization here follows Orvieto et al. 2023 ("Resurrecting Recurrent
Neural Networks for Long Sequences" / LRU).

## The Task: Tracking a Moving Spot from Event Data

`src/event_data.py` simulates a simplified event camera: a bright spot
sweeps across a row of pixels, and the signal fed to the model is derived
from **brightness changes only** (ON/OFF event counts + event centroid per
time step) — not the raw frame. This is a toy simulation, not a substitute
for a real dataset like DVS-Gesture or N-MNIST, but it has the structural
properties that matter: sparsity, an ON/OFF polarity signal, and a genuine
downstream task (recovering the spot's true position from the event stream).

## Results

All numbers below are real output from `train.py`, not illustrative.

**1. Correctness — recurrent vs. convolutional equivalence:**

| | Value |
|---|---|
| Max \|recurrent − convolutional\| output difference | 6.71 × 10⁻⁸ |

This is floating-point precision, i.e. the two computation paths are
mathematically identical, as they must be.

**2. Practical benefit — train-fast / deploy-cheap:**

The model is trained using the parallel convolutional form, then evaluated
using the recurrent form (the way it would actually run on a stream of
incoming events). Training time and final tracking accuracy vs. sequence
length, compared against a standard tanh-RNN trained on the identical
task (reproducible via `python3 benchmark.py`):

| Sequence length | LinearSSM train time | TanhRNN train time | Speedup | LinearSSM MAE | TanhRNN MAE |
|---:|---:|---:|---:|---:|---:|
| 200 steps  | 1.62s | 7.05s  | 4.4x  | 0.388 | 0.104 |
| 500 steps  | 0.80s | 18.94s | 23.7x | 0.337 | 0.143 |
| 1000 steps | 1.12s | 39.27s | 35.0x | 0.341 | 0.218 |
| 2000 steps | 1.34s | 81.28s | 60.4x | 1.220 | 1.445 |

The training-speed gap grows with sequence length, exactly as expected:
the RNN's training cost scales with the number of sequential
backprop-through-time steps, while the SSM's convolutional training cost
is dominated by the FFT and stays roughly flat. This is the real,
practically-important advantage of the linear/convolutional formulation.

**3. Tracking accuracy — an honest comparison:**

**The tanh-RNN is actually more accurate at short-to-medium sequence
lengths** (200–1000 steps) — I'm reporting that directly rather than
hiding it. On a sequence with mostly local (non-long-range) dependencies,
the extra nonlinear expressiveness of a plain RNN can help it fit the
task slightly better. Only at the longest tested length (2000 steps) does
the LinearSSM edge ahead on accuracy too — consistent with the
well-documented advantage of linear SSMs on long-range dependencies, but
the effect here is modest and not perfectly monotonic, so I'm not
overclaiming it from a single toy task. The training-speed advantage, on
the other hand, is unambiguous and grows consistently with length.

## What Changed From the First Version

The original version of this repo used this recurrence:
```python
h = torch.tanh(self.A(h) + self.B(x[:, t, :]))   # nonlinear — a standard RNN cell
```
Despite the "SSM" naming, a `tanh` inside the recurrence breaks the
property that actually defines this model family: linearity, which is
what makes the convolutional/parallel computation valid in the first
place. That version also used a plain sine wave as "event camera data,"
with no polarity, sparsity, or underlying task. Both are fixed here:
`src/linear_ssm.py` has no nonlinearity in the state update, and
`src/event_data.py` generates an actual sparse ON/OFF event signal tied
to a ground-truth tracking task.

## Tech Stack

`Python` · `PyTorch` (core linear algebra + autograd) · `NumPy` (data simulation) · `Matplotlib` (visualization)

## Repository Structure

```
.
├── src/
│   ├── linear_ssm.py     # the LinearSSM layer (recurrent + convolutional forms)
│   └── event_data.py      # simulated event-camera tracking data
├── tests/
│   └── test_linear_ssm.py # equivalence test + gradient/stability sanity checks
├── train.py                 # end-to-end training + comparison script
├── benchmark.py              # reproduces the sequence-length comparison table
├── output/
│   └── ssm_tracking_results.png
└── README.md
```

## How to Run

```bash
pip install torch numpy matplotlib pytest

python3 train.py          # runs the full comparison, saves output/ssm_tracking_results.png
python3 benchmark.py      # reproduces the sequence-length comparison table above
pytest tests/ -v            # runs the equivalence + sanity tests
```

## Future Work

- Test on a genuinely long-range task (e.g. recall a signal seen 500+
  steps ago) where linear SSMs are expected to actually beat nonlinear
  RNNs on accuracy, not just speed.
- Swap the real-diagonal parameterization for a complex-diagonal one
  (closer to the actual S4D/S5 formulation) to capture oscillatory dynamics.
- Try a real neuromorphic dataset (DVS-Gesture or N-MNIST) instead of the
  simulated moving-spot task.
- Stack multiple LinearSSM layers with pointwise nonlinearities *between*
  layers (not inside the recurrence) — this is how real S4/S5 models
  compose depth without sacrificing the linear-recurrence property.

