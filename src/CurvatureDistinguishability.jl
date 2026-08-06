"""
    CurvatureDistinguishability

Computational proof of the quartic distinguishability law for space-based
gravitational-wave interferometry: the squared distance between a two-source
signal and the best-fit single-source template scales as
`D² ≈ (1/16) K(u) δ⁴`, governed by the extrinsic curvature `K(u)` of the
signal manifold. The package provides the waveform/detector model
(`Physics`, `Detector`), the differential-geometry engine (`Geometry`),
box-constrained inference (`Inference`), physical parameter bounds
(`Bounds`), backend dispatch with GPU package extensions (`Backends`),
validated configuration (`Config`), provenance utilities (`Provenance`),
publication plotting (`Plotting`), the pipeline driver (`Orchestrator`) and
figure regeneration from persisted run artifacts (`RunFigures`).
"""
module CurvatureDistinguishability

include("Backends.jl")
using .Backends
export get_best_backend, to_backend, backend_name

include("Physics.jl")
using .Physics
export NoiseParams, robson_confusion_params, analytic_noise_psd,
    WaveformParams, waveform_params, spin_beta, strain_bin,
    scaled_waveform_model, SECONDS_PER_YEAR

include("Detector.jl")
using .Detector
export tdi_modulation_bin, project_to_tdi, n_channels

include("Bounds.jl")
using .Bounds
export ParameterBounds,
    default_bounds, bounds_from_config, deviation_box,
    ray_box_crossing, clamp_interior

include("Geometry.jl")
using .Geometry
export inner_product, multi_channel_inner_product, compute_tangent_basis,
    compute_extrinsic_curvature_from_basis, compute_extrinsic_curvature

include("Inference.jl")
using .Inference
export calculate_numerical_distance, optimization_diagnostics, loss_function

include("Provenance.jl")
using .Provenance
export run_id_from_config, backup_existing!

include("Config.jl")
using .Config
export PipelineSettings, load_and_validate_config

include("Plotting.jl")
using .Plotting
export publication_theme, save_figure, scaling_figure, residual_figure, zone_figure

include("Orchestrator.jl")
using .Orchestrator
export run_pipeline

include("RunFigures.jl")
using .RunFigures
export run_cases, sweep_figures, zone_map_figure

end
