module TwoWaveformDistinguishability

include("Hardware.jl")
using .Hardware
export get_best_backend, to_backend

include("Physics.jl")
using .Physics
export analytic_noise_psd, scaled_waveform_model

include("Detector.jl")
using .Detector
export project_to_tdi

include("Geometry.jl")
using .Geometry
export inner_product, compute_extrinsic_curvature, multi_channel_inner_product, compute_tangent_basis, compute_extrinsic_curvature_from_basis

include("Inference.jl")
using .Inference
export calculate_numerical_distance

include("Orchestrator.jl")
using .Orchestrator
export run_pipeline

end
