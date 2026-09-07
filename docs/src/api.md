# API Reference & Module Documentation

Core functionality exported by `CurvatureDistinguishability.jl`, decoupled
into physics, detector response, residual diagnostics, bounds, geometry,
inference, backends, configuration, provenance, plotting and orchestration.

```@docs
CurvatureDistinguishability
```

## Module inventory

```@docs
CurvatureDistinguishability.Physics
CurvatureDistinguishability.Detector
CurvatureDistinguishability.Residuals
CurvatureDistinguishability.Bounds
CurvatureDistinguishability.Geometry
CurvatureDistinguishability.Fitting
CurvatureDistinguishability.Inference
CurvatureDistinguishability.Backends
CurvatureDistinguishability.Config
CurvatureDistinguishability.Provenance
CurvatureDistinguishability.Plotting
CurvatureDistinguishability.Orchestrator
CurvatureDistinguishability.RunFigures
```

## Physics

```@docs
CurvatureDistinguishability.Physics.NoiseParams
CurvatureDistinguishability.Physics.robson_confusion_params
CurvatureDistinguishability.Physics.analytic_noise_psd
CurvatureDistinguishability.Physics.WaveformParams
CurvatureDistinguishability.Physics.waveform_params
CurvatureDistinguishability.Physics.spin_beta
CurvatureDistinguishability.Physics.strain_bin
CurvatureDistinguishability.Physics.scaled_waveform_model
CurvatureDistinguishability.Physics.SECONDS_PER_YEAR
```

## Detector

```@docs
CurvatureDistinguishability.Detector.tdi_modulation_bin
CurvatureDistinguishability.Detector.project_to_tdi
CurvatureDistinguishability.Detector.n_channels
```

## Residuals

```@docs
CurvatureDistinguishability.Residuals.residual_spectrum
```

## Bounds

```@docs
CurvatureDistinguishability.Bounds.ParameterBounds
CurvatureDistinguishability.Bounds.default_bounds
CurvatureDistinguishability.Bounds.bounds_from_config
CurvatureDistinguishability.Bounds.deviation_box
CurvatureDistinguishability.Bounds.ray_box_crossing
CurvatureDistinguishability.Bounds.clamp_interior
CurvatureDistinguishability.Bounds.PARAM_KEYS
```

## Geometry

```@docs
CurvatureDistinguishability.Geometry.inner_product
CurvatureDistinguishability.Geometry.multi_channel_inner_product
CurvatureDistinguishability.Geometry.flat_response
CurvatureDistinguishability.Geometry.compute_tangent_basis
CurvatureDistinguishability.Geometry.value_and_directional_derivs
CurvatureDistinguishability.Geometry.compute_extrinsic_curvature_from_basis
CurvatureDistinguishability.Geometry.compute_extrinsic_curvature
CurvatureDistinguishability.Geometry.GS_NORM_TOL
CurvatureDistinguishability.Geometry.K_UNDERFLOW
```

## Inference

```@docs
CurvatureDistinguishability.Inference.loss_function
CurvatureDistinguishability.Inference.calculate_numerical_distance
CurvatureDistinguishability.Inference.optimization_diagnostics
CurvatureDistinguishability.Inference.clear_device_buffers!
CurvatureDistinguishability.Inference.column_sums_via!
```

## Backends

```@docs
CurvatureDistinguishability.Backends.get_best_backend
CurvatureDistinguishability.Backends.to_backend
CurvatureDistinguishability.Backends.backend_name
CurvatureDistinguishability.Backends.register_backend!
CurvatureDistinguishability.Backends.reclaim_device_memory!
CurvatureDistinguishability.Backends.cpu_model
CurvatureDistinguishability.Backends.device_fingerprint
```

## Configuration

```@docs
CurvatureDistinguishability.Config.PipelineSettings
CurvatureDistinguishability.Config.SweepSpec
CurvatureDistinguishability.Config.MapSpec
CurvatureDistinguishability.Config.load_and_validate_config
CurvatureDistinguishability.Config.settings_from_config
```

## Provenance

```@docs
CurvatureDistinguishability.Provenance.run_id_from_config
CurvatureDistinguishability.Provenance.effective_config
CurvatureDistinguishability.Provenance.identity_config
CurvatureDistinguishability.Provenance.EXECUTION_SECTIONS
CurvatureDistinguishability.Provenance.unique_run_dir
CurvatureDistinguishability.Provenance.snapshot_config
CurvatureDistinguishability.Provenance.backup_existing!
CurvatureDistinguishability.Provenance.write_run_metadata
CurvatureDistinguishability.Provenance.write_hardware_fingerprint
CurvatureDistinguishability.Provenance.git_state
```

## Plotting

```@docs
CurvatureDistinguishability.Plotting.publication_theme
CurvatureDistinguishability.Plotting.save_figure
CurvatureDistinguishability.Plotting.decade_ticks
CurvatureDistinguishability.Plotting.pi_ticks
CurvatureDistinguishability.Plotting.scaling_figure
CurvatureDistinguishability.Plotting.residual_figure
CurvatureDistinguishability.Plotting.ResidualFigureMeta
CurvatureDistinguishability.Plotting.zone_figure
```

## Fitting

```@docs
CurvatureDistinguishability.Fitting.loglog_slope
CurvatureDistinguishability.Fitting.ratio_correction_fit
CurvatureDistinguishability.Fitting.optimizer_floor
CurvatureDistinguishability.Fitting.above_floor_mask
```

## Orchestrator

```@docs
CurvatureDistinguishability.Orchestrator.run_pipeline
```

## Run figure regeneration

```@docs
CurvatureDistinguishability.RunFigures.run_cases
CurvatureDistinguishability.RunFigures.sweep_figures
CurvatureDistinguishability.RunFigures.zone_map_figure
```
