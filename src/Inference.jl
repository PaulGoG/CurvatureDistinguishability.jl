module Inference

using Optim
using ForwardDiff
using KernelAbstractions
using ..Physics
using ..Geometry
using ..Hardware
using ..Detector

export calculate_numerical_distance

"""
    calculate_numerical_distance(data_stream::Tuple, theta_guess::AbstractVector, freqs::AbstractVector, Sn_vals::AbstractVector, df::Real; kwargs...) -> Tuple{Float64, Vector{Float64}, Optim.OptimizationResults}

Performs a high-precision numerical optimization (`LBFGS` from `Optim.jl`) to find the minimum distance \$D^2\$ between a composite two-source `data_stream` and the theoretical single-source manifold.

It utilizes `ForwardDiff.jl` dual-number analytical Hessians. The function dynamically branches into an allocation-free generator loop (`sum(1:N) do i`) for CPU execution to prevent Garbage Collection throttling, or a fully vectorized array broadcast path for GPU acceleration.

# Arguments
- `data_stream`: A tuple of `(A, E, T)` data arrays containing the superposition of two sources.
- `theta_guess`: The initial parameter guess (length 6 vector) for the optimizer.
- `freqs`: Frequency grid.
- `Sn_vals`: Noise PSD vector.
- `df`: Frequency resolution.

# Returns
- `D2`: The squared scalar distance \$D^2\$ of the unabsorbable residual.
- `best_fit`: The parameter vector of the single source that best absorbed the two-source signal.
- `opt_res`: The raw `Optim.jl` convergence results struct.
"""
function calculate_numerical_distance(data_stream::Tuple, theta_guess::AbstractVector, 
                                      freqs::AbstractVector, Sn_vals::AbstractVector, df::Real;
                                      g_tol::Real=1e-12, iterations::Int=1000, 
                                      backend=get_best_backend(), kwargs...)
    
    # Extract configurable parameters with fallback to defaults
    mass_scale = get(kwargs, :mass_scale, 10.0)
    time_scale = get(kwargs, :time_scale, 1000.0)
    amp_scale = get(kwargs, :amp_scale, 1e-21)
    eta = get(kwargs, :eta, 0.25)
    amp_33_factor = get(kwargs, :amp_33_factor, 0.1)
    
    data_A, data_E, data_T = data_stream
    
    function loss_func(p)
        A_scale, Mc_scale, tc_scale, phic, chi1, chi2 = p[1], p[2], p[3], p[4], p[5], p[6]
        A = A_scale * amp_scale
        Mc = Mc_scale * mass_scale
        tc = tc_scale * time_scale
        chi_eff = 0.5 * (chi1 + chi2)
        beta = (113.0/3.0 - 76.0*eta/3.0) * chi_eff / 4.0
        
        if backend isa KernelAbstractions.CPU
            # Allocation-free loop for CPU multi-threading (avoids GC lock contention)
            dist_sq = sum(1:length(freqs)) do i
                f = freqs[i]
                v_param = (π * Mc * f)^(1/3)
                
                # 22 mode
                amp_22 = A * (f ^ (-7/6))
                phase_22 = 2 * π * f * tc - phic - (3/128) * (v_param^(-5)) * (1.0 - 4.0 * beta * (v_param^3))
                h_22 = amp_22 * exp(1im * phase_22)
                
                # 33 mode
                amp_33 = (amp_33_factor * A) * (f ^ (-7/6)) * v_param
                phase_33 = 1.5 * phase_22
                h_33 = amp_33 * exp(1im * phase_33)
                
                h_strain = h_22 + h_33
                
                # TDI Projection inline for allocation-free CPU loop
                mod_A, mod_E, mod_T = tdi_modulation(f, p; kwargs...)
                model_A = mod_A * h_strain
                model_E = mod_E * h_strain
                model_T = mod_T * h_strain
                
                diff_A = data_A[i] - model_A
                diff_E = data_E[i] - model_E
                diff_T = data_T[i] - model_T
                
                (real(conj(diff_A) * diff_A) + real(conj(diff_E) * diff_E) + real(conj(diff_T) * diff_T)) / Sn_vals[i]
            end
            return 4.0 * df * dist_sq
        else
            # Hardware-agnostic broadcasting for GPUs
            h_model = scaled_waveform_model(p, freqs; kwargs...)
            mod_A, mod_E, mod_T = project_to_tdi(h_model, freqs, p; kwargs...)
            
            diff_A = @. data_A - mod_A
            diff_E = @. data_E - mod_E
            diff_T = @. data_T - mod_T
            
            dist_sq_A = mapreduce((d, s) -> real(conj(d) * d) / s, +, diff_A, Sn_vals)
            dist_sq_E = mapreduce((d, s) -> real(conj(d) * d) / s, +, diff_E, Sn_vals)
            dist_sq_T = mapreduce((d, s) -> real(conj(d) * d) / s, +, diff_T, Sn_vals)
            
            return 4.0 * df * (dist_sq_A + dist_sq_E + dist_sq_T)
        end
    end
    
    # Utilize ForwardDiff to provide exact analytical gradients and hessians to the optimizer
    function g!(G, x)
        ForwardDiff.gradient!(G, loss_func, x)
    end
    function h!(H, x)
        ForwardDiff.hessian!(H, loss_func, x)
    end
    
    obj = TwiceDifferentiable(loss_func, g!, h!, theta_guess)
    
    # Run the optimizer
    opt_res = optimize(obj, theta_guess, LBFGS(), 
                       Optim.Options(g_tol=g_tol, iterations=iterations, show_trace=false))
    
    d2_num = Optim.minimum(opt_res)
    best_fit = Optim.minimizer(opt_res)
    
    return d2_num, best_fit, opt_res
end

end # module