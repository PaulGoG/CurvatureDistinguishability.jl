# API Reference & Module Documentation

This page details the core functionalities exported by the `TwoWaveformDistinguishability.jl` pipeline. The mathematical engine is meticulously decoupled into four primary domains: **Physics**, **Detector**, **Geometry**, and **Inference**.

## Physics Module
The `Physics` module manages the generation of frequency-domain waveforms and the analytic Power Spectral Density (PSD) models for space-based interferometry.

```@docs
TwoWaveformDistinguishability.Physics.analytic_noise_psd
TwoWaveformDistinguishability.Physics.scaled_waveform_model
```

## Detector Module
The `Detector` module is responsible for Time Delay Interferometry (TDI). It translates the raw astrophysical strain into the noise-orthogonal A, E, and T channels, applying the necessary orbital Doppler phase shifts and antenna pattern amplitude modulations.

```@docs
TwoWaveformDistinguishability.Detector.tdi_modulation
TwoWaveformDistinguishability.Detector.project_to_tdi
```

## Geometry Module
The `Geometry` module is the differential geometry heart of the pipeline. It uses high-precision Automatic Differentiation (`ForwardDiff.jl`) to compute exact analytical gradients, Jacobians, and Extrinsic Curvatures on the signal manifold. It features a decoupled architecture to prevent memory (VRAM/RAM) overflow.

```@docs
TwoWaveformDistinguishability.Geometry.inner_product
TwoWaveformDistinguishability.Geometry.multi_channel_inner_product
TwoWaveformDistinguishability.Geometry.compute_tangent_basis
TwoWaveformDistinguishability.Geometry.compute_extrinsic_curvature_from_basis
TwoWaveformDistinguishability.Geometry.compute_extrinsic_curvature
```

## Inference Module
The `Inference` module leverages `Optim.jl` to numerically search the single-source parameter space to find the minimum distance ($D^2$) to a composite two-source signal.

```@docs
TwoWaveformDistinguishability.Inference.calculate_numerical_distance
```

## Hardware Abstraction Layer
The `Hardware` module provides dynamic detection of available computational backends (CPU threads, CUDA, AMDGPU, Metal, oneAPI) to automatically route array broadcasts.

```@docs
TwoWaveformDistinguishability.Hardware.get_best_backend
TwoWaveformDistinguishability.Hardware.to_backend
```

## Orchestrator
The `Orchestrator` manages the high-level Execution loops and the **Dynamic Memory Manager**.

```@docs
TwoWaveformDistinguishability.Orchestrator.run_pipeline
```
