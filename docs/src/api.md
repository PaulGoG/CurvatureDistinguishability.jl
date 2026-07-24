# API Reference & Module Documentation

Core functionality exported by `TwoWaveformDistinguishability.jl`, decoupled
into physics, detector response, bounds, geometry, inference, hardware,
configuration, provenance, plotting and orchestration.

## Physics

```@docs
TwoWaveformDistinguishability.Physics.NoiseParams
TwoWaveformDistinguishability.Physics.robson_confusion_params
TwoWaveformDistinguishability.Physics.analytic_noise_psd
TwoWaveformDistinguishability.Physics.WaveformParams
TwoWaveformDistinguishability.Physics.waveform_params
TwoWaveformDistinguishability.Physics.spin_beta
TwoWaveformDistinguishability.Physics.strain_bin
TwoWaveformDistinguishability.Physics.scaled_waveform_model
```

## Detector

```@docs
TwoWaveformDistinguishability.Detector.tdi_modulation_bin
TwoWaveformDistinguishability.Detector.project_to_tdi
TwoWaveformDistinguishability.Detector.n_channels
```

## Bounds

```@docs
TwoWaveformDistinguishability.Bounds.ParameterBounds
TwoWaveformDistinguishability.Bounds.default_bounds
TwoWaveformDistinguishability.Bounds.bounds_from_config
TwoWaveformDistinguishability.Bounds.deviation_box
TwoWaveformDistinguishability.Bounds.ray_box_crossing
TwoWaveformDistinguishability.Bounds.clamp_interior
```

## Geometry

```@docs
TwoWaveformDistinguishability.Geometry.inner_product
TwoWaveformDistinguishability.Geometry.multi_channel_inner_product
TwoWaveformDistinguishability.Geometry.flat_response
TwoWaveformDistinguishability.Geometry.compute_tangent_basis
TwoWaveformDistinguishability.Geometry.value_and_directional_derivs
TwoWaveformDistinguishability.Geometry.compute_extrinsic_curvature_from_basis
TwoWaveformDistinguishability.Geometry.compute_extrinsic_curvature
```

## Inference

```@docs
TwoWaveformDistinguishability.Inference.loss_function
TwoWaveformDistinguishability.Inference.calculate_numerical_distance
TwoWaveformDistinguishability.Inference.optimization_diagnostics
```

## Backends

```@docs
TwoWaveformDistinguishability.Backends.get_best_backend
TwoWaveformDistinguishability.Backends.to_backend
TwoWaveformDistinguishability.Backends.backend_name
TwoWaveformDistinguishability.Backends.register_backend!
```

## Configuration

```@docs
TwoWaveformDistinguishability.Config.PipelineSettings
TwoWaveformDistinguishability.Config.load_and_validate_config
```

## Provenance

```@docs
TwoWaveformDistinguishability.Provenance.run_id_from_config
TwoWaveformDistinguishability.Provenance.unique_run_dir
TwoWaveformDistinguishability.Provenance.snapshot_config
TwoWaveformDistinguishability.Provenance.backup_existing!
TwoWaveformDistinguishability.Provenance.write_run_metadata
TwoWaveformDistinguishability.Provenance.git_state
```

## Plotting

```@docs
TwoWaveformDistinguishability.Plotting.twd_theme
TwoWaveformDistinguishability.Plotting.save_figure
TwoWaveformDistinguishability.Plotting.decade_ticks
TwoWaveformDistinguishability.Plotting.pi_ticks
TwoWaveformDistinguishability.Plotting.scaling_figure
TwoWaveformDistinguishability.Plotting.residual_figure
TwoWaveformDistinguishability.Plotting.zone_figure
```

## Orchestrator

```@docs
TwoWaveformDistinguishability.Orchestrator.run_pipeline
```
