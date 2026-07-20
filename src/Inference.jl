module Inference

using Optim
using ForwardDiff
using KernelAbstractions
using ..Physics
using ..Detector
using ..Hardware
using ..Bounds

export calculate_numerical_distance, optimization_diagnostics, loss_function

# --- loss -------------------------------------------------------------------

"""
KernelAbstractions kernel: per-bin contribution to the squared residual
between the 2-channel (A, E) data and the single-source model at parameters
`θ` (an isbits `NTuple`, so `ForwardDiff.Dual` elements compile to GPU code).
"""
@kernel function loss_bins!(out, @Const(freqs), @Const(Sn), @Const(data_A), @Const(data_E),
                            θ::NTuple{6}, wp::WaveformParams)
    i = @index(Global, Linear)
    @inbounds begin
        f = freqs[i]
        A = θ[1] * wp.amp_scale
        Mc = θ[2] * wp.mass_scale
        tc = θ[3] * wp.time_scale
        beta = spin_beta(θ[5], θ[6], wp.eta)
        h = strain_bin(f, A, Mc, tc, θ[4], beta, wp.amp_33_factor)
        mod_A, mod_E = tdi_modulation_bin(f, Mc, tc, wp)
        diff_A = data_A[i] - mod_A * h
        diff_E = data_E[i] - mod_E * h
        out[i] = (real(conj(diff_A) * diff_A) + real(conj(diff_E) * diff_E)) / Sn[i]
    end
end

function device_loss(p::AbstractVector, freqs, Sn_vals, data_A, data_E, df::Real,
                     wp::WaveformParams, backend)
    T = eltype(p)
    θ = ntuple(i -> p[i], Val(6))
    out = KernelAbstractions.zeros(backend, T, length(freqs))
    kern = loss_bins!(backend)
    kern(out, freqs, Sn_vals, data_A, data_E, θ, wp; ndrange = length(freqs))
    KernelAbstractions.synchronize(backend)
    return 4.0 * df * sum(out)
end

function cpu_loss(p::AbstractVector, freqs, Sn_vals, data_stream::Tuple, df::Real,
                  wp::WaveformParams)
    A = p[1] * wp.amp_scale
    Mc = p[2] * wp.mass_scale
    tc = p[3] * wp.time_scale
    phic = p[4]
    beta = spin_beta(p[5], p[6], wp.eta)

    # Allocation-free loop (avoids GC lock contention under outer threading).
    dist_sq = sum(1:length(freqs)) do i
        @inbounds begin
            f = freqs[i]
            h = strain_bin(f, A, Mc, tc, phic, beta, wp.amp_33_factor)
            mod_A, mod_E = tdi_modulation_bin(f, Mc, tc, wp)
            diff_A = data_stream[1][i] - mod_A * h
            diff_E = data_stream[2][i] - mod_E * h
            c = real(conj(diff_A) * diff_A) + real(conj(diff_E) * diff_E)
            if length(data_stream) == 3
                diff_T = data_stream[3][i] # model T channel is identically zero
                c += real(conj(diff_T) * diff_T)
            end
            c / Sn_vals[i]
        end
    end
    return 4.0 * df * dist_sq
end

"""
    loss_function(data_stream, freqs, Sn_vals, df, wp, backend) -> p -> D²(p)

Squared noise-weighted distance between `data_stream` and the single-source
model, as a closure over the scaled parameter vector. On the CPU backend an
allocation-free scalar loop is used; on GPU backends a single fused
KernelAbstractions kernel (2-channel only — disable the T channel, which is
identically zero, for GPU runs). Differentiable with `ForwardDiff`.
"""
function loss_function(data_stream::Tuple, freqs::AbstractVector, Sn_vals::AbstractVector,
                       df::Real, wp::WaveformParams, backend)
    if backend isa KernelAbstractions.CPU
        return p -> cpu_loss(p, freqs, Sn_vals, data_stream, df, wp)
    end
    length(data_stream) == 2 ||
        error("GPU path supports the 2-channel (A, E) configuration only; " *
              "set include_t_channel = false (the T channel is identically zero).")
    return p -> device_loss(p, freqs, Sn_vals, data_stream[1], data_stream[2], df, wp, backend)
end

# --- optimization -----------------------------------------------------------

"""
    calculate_numerical_distance(data_stream, theta_guess, freqs, Sn_vals, df;
                                 g_tol = 1e-12, iterations = 1000,
                                 backend = get_best_backend(),
                                 optimizer = :ipnewton,
                                 bounds = default_bounds(), kwargs...)

Minimum squared distance `D²` between the composite `data_stream` and the
single-source manifold, found by box-constrained optimization within the
physical `bounds`:

- `:ipnewton` (default): interior-point Newton using the exact ForwardDiff
  Hessian — fast convergence and a much lower convergence floor than L-BFGS.
- `:lbfgs_box`: `Fminbox(LBFGS())`, gradient-only.
- `:lbfgs`: legacy unconstrained L-BFGS (regression comparisons).

Physics keywords (`mass_scale`, `sky_theta`, …) are accepted via `kwargs`.
Returns `(D², best_fit, optim_result)`.
"""
function calculate_numerical_distance(data_stream::Tuple, theta_guess::AbstractVector,
                                      freqs::AbstractVector, Sn_vals::AbstractVector, df::Real;
                                      g_tol::Real = 1e-12, iterations::Int = 1000,
                                      backend = get_best_backend(),
                                      optimizer::Symbol = :ipnewton,
                                      bounds::Union{Nothing,ParameterBounds} = default_bounds(),
                                      kwargs...)
    wp = waveform_params(; kwargs...)
    loss = loss_function(data_stream, freqs, Sn_vals, df, wp, backend)

    g!(G, x) = ForwardDiff.gradient!(G, loss, x)
    h!(H, x) = ForwardDiff.hessian!(H, loss, x)

    opts = Optim.Options(g_tol = g_tol, iterations = iterations, show_trace = false)

    opt_res = if optimizer === :ipnewton
        bounds === nothing && error("optimizer = :ipnewton requires bounds")
        x0 = clamp_interior(theta_guess, bounds)
        obj = TwiceDifferentiable(loss, g!, h!, x0)
        cons = TwiceDifferentiableConstraints(collect(bounds.lower), collect(bounds.upper))
        optimize(obj, cons, x0, IPNewton(), opts)
    elseif optimizer === :lbfgs_box
        bounds === nothing && error("optimizer = :lbfgs_box requires bounds")
        x0 = clamp_interior(theta_guess, bounds)
        obj = OnceDifferentiable(loss, g!, x0)
        optimize(obj, collect(bounds.lower), collect(bounds.upper), x0, Fminbox(LBFGS()), opts)
    elseif optimizer === :lbfgs
        obj = TwiceDifferentiable(loss, g!, h!, collect(float.(theta_guess)))
        optimize(obj, collect(float.(theta_guess)), LBFGS(), opts)
    else
        error("Unknown optimizer :$optimizer (expected :ipnewton, :lbfgs_box or :lbfgs)")
    end

    return Optim.minimum(opt_res), Optim.minimizer(opt_res), opt_res
end

"""
    optimization_diagnostics(opt_res, best_fit, bounds) -> NamedTuple

Convergence diagnostics persisted per optimization: convergence flag,
iteration count, final gradient norm, and whether the best fit sits on an
active physical bound (within a relative tolerance of the bound width).
"""
function optimization_diagnostics(opt_res, best_fit::AbstractVector,
                                  bounds::Union{Nothing,ParameterBounds})
    at_bound = false
    if bounds !== nothing
        for i in eachindex(best_fit)
            bounds.periodic[i] && continue
            lo, hi = bounds.lower[i], bounds.upper[i]
            tol = isfinite(lo) && isfinite(hi) ? 1e-6 * (hi - lo) : 1e-6 * max(1.0, abs(best_fit[i]))
            (isfinite(lo) && abs(best_fit[i] - lo) <= tol) && (at_bound = true)
            (isfinite(hi) && abs(best_fit[i] - hi) <= tol) && (at_bound = true)
        end
    end
    return (converged = Optim.converged(opt_res),
            iterations = Optim.iterations(opt_res),
            g_norm = Optim.g_residual(opt_res),
            at_bound = at_bound)
end

end # module
