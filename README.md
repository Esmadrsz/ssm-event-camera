# A Minimal State-Space Model for Event-Camera-Style Signals

This repository contains a small, from-scratch PyTorch implementation of a discrete-time linear state-space model (SSM). It is not an attempt to reproduce S4; rather, it is a deliberately minimal exercise meant to make the recurrence explicit enough to reason about by hand, as a first step before working with structured SSMs and event-based camera data.

## Background

My training is in PDEs and dynamical systems (my master's thesis applied the Modified Kudryashov method to the MEW equation), so the state equation underlying S4,

```
x'(t) = A x(t) + B u(t)
y(t)  = C x(t)
```

was a natural entry point. The question I wanted to answer for myself was how much of the continuous-time intuition survives once the system is discretized and A, B, C become learned parameters. The implementation here is the simplest version of that question: a recurrence `h_t = tanh(A h_{t-1} + B x_t)`, `y_t = C h_t`, with A, B, C as unconstrained dense linear layers trained via a standard sequential loop. It is fit to a synthetic proxy signal, a noisy sine wave, in place of genuine event-stream data, which I have not yet integrated.

## Relation to structured SSMs (S4, S5, Mamba)

The distance between this implementation and the architectures it is inspired by is worth stating precisely, since it is the more informative part of the exercise.

The matrix A is free and unconstrained, whereas S4 initializes A according to HiPPO theory, giving the hidden state a provable ability to compress input history under a specific polynomial basis; without this, the recurrence has no principled mechanism for long-range memory and is closer to a standard RNN. There is also no genuine discretization step: S4 derives its discrete parameters (A_bar, B_bar) from a continuous system via a zero-order hold with a learnable step size Δ, whereas here A and B are learned directly with no continuous-time correspondence behind them. The forward pass is a sequential Python loop rather than the convolutional or parallel-scan formulation that makes S4/S5/Mamba tractable on long sequences. A and B are also fixed after training, in contrast to Mamba's input-dependent selection mechanism. Finally, the training signal itself is a regularly sampled sine wave rather than the sparse, asynchronous event stream this model is ultimately intended to process.

## Next steps

- Replace the dense A with a structured, diagonal or HiPPO-initialized parameterization, and evaluate the effect on longer sequences.
- Implement the convolutional or parallel-scan formulation and compare runtime against the sequential loop.
- Move from synthetic signals to real event-camera data (N-MNIST, then DVS-Gesture), with attention to the irregular timing of events rather than treating them as uniformly sampled.

## Files

- `ssm_demo.py`: model definition, synthetic data generation, training loop, and visualization of input against model output.

## References

- Gu, Goel, and Ré (2021), *Efficiently Modeling Long Sequences with Structured State Spaces* (S4)
- Smith, Warrington, and Linderman (2022), *Simplified State Space Layers for Sequence Modeling* (S5)
- Gu and Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
