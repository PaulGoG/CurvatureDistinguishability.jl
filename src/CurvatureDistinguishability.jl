"""
    CurvatureDistinguishability

Computational proof of the quartic distinguishability law for space-based
gravitational-wave interferometry: the squared distance between a two-source
signal and the best-fit single-source template scales as
`D² ≈ (1/16) K(u) δ⁴`, governed by the extrinsic curvature `K(u)` of the
signal manifold. The package provides the waveform/detector model
(`Physics`, `Detector`), residual-spectrum diagnostics (`Residuals`), the
differential-geometry engine (`Geometry`),
box-constrained inference (`Inference`), physical parameter bounds
(`Bounds`), backend dispatch with GPU package extensions (`Backends`),
validated configuration (`Config`), provenance utilities (`Provenance`),
shared fit statistics (`Fitting`), publication plotting (`Plotting`), the
pipeline driver (`Orchestrator`), work-item checkpoints (`Checkpoint`), worker
liveness telemetry (`Heartbeat`), process supervision with a retry budget
(`Supervision`, wired to the pipeline by `Campaign`) and figure regeneration
from persisted run artifacts (`RunFigures`).
"""
module CurvatureDistinguishability

include("Heartbeat.jl")
using .Heartbeat

include("Supervision.jl")
using .Supervision
export SupervisionSettings, WorkerSpec, supervise

include("Checkpoint.jl")
using .Checkpoint

include("Backends.jl")
using .Backends
export get_best_backend, to_backend, backend_name

include("Physics.jl")
using .Physics
export NoiseParams, robson_confusion_params, analytic_noise_psd,
    WaveformParams, waveform_params, spin_beta, pn_phase, harmonic_phase,
    harmonic_amplitudes, total_mass, isco_frequency, second_source,
    SECONDS_PER_YEAR

include("Detector.jl")
using .Detector
export channel_strain, channel_strain_bin, n_channels

include("Residuals.jl")
using .Residuals
export residual_spectrum

include("Bounds.jl")
using .Bounds
export ParameterBounds,
    default_bounds, bounds_from_config, deviation_box,
    ray_box_crossing, clamp_interior

include("Geometry.jl")
using .Geometry
export inner_product, multi_channel_inner_product, compute_tangent_basis,
    compute_extrinsic_curvature_from_basis, compute_extrinsic_curvature,
    boundary_radius, cap_unbounded_radii!, cap_at_prior

include("Fitting.jl")
using .Fitting

include("Inference.jl")
using .Inference
export calculate_numerical_distance, optimization_diagnostics, loss_function

include("Provenance.jl")
using .Provenance
export run_id_from_config,
    effective_config, identity_config, backup_existing!,
    snapshot_manifest

include("Config.jl")
using .Config
export PipelineSettings, SweepSpec, MapSpec, load_and_validate_config

include("Plotting.jl")
using .Plotting
export publication_theme,
    save_figure, canvas_width, scaling_figure, residual_figure,
    ResidualFigureMeta, zone_figure,
    scaling_panel!, residual_panel!, zone_panel!,
    composite_scaling_figure, composite_zone_figure, composite_residual_figure

include("Orchestrator.jl")
using .Orchestrator
using .Orchestrator: DEFAULT_CONFIG, PipelineStageError, ResourceBudgetError
export run_pipeline
public DEFAULT_CONFIG, PipelineStageError, ResourceBudgetError

include("Campaign.jl")
using .Campaign
export supervise_pipeline

include("RunFigures.jl")
using .RunFigures
export run_cases, sweep_figures, zone_map_figure, composite_figures,
    sweep_panel_data, zone_panel_data, residual_panel_data

end
