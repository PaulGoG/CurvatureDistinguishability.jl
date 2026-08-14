"""
Box-constrained minimization of the squared noise-weighted distance to the
single-source manifold, with a KernelAbstractions loss kernel and the
Float64-lanes dual-number path for GPU backends.
"""
module Inference

using DocStringExtensions: TYPEDSIGNATURES
using ForwardDiff: ForwardDiff
using KernelAbstractions: KernelAbstractions, @Const, @index, @kernel
using Optim: Optim, Fminbox, IPNewton, LBFGS, OnceDifferentiable,
    TwiceDifferentiable, TwiceDifferentiableConstraints, optimize
using ..Physics
using ..Physics: N_PARAMS
using ..Detector
using ..Backends
using ..Bounds

export calculate_numerical_distance, optimization_diagnostics, loss_function
public clear_device_buffers!

# --- loss -------------------------------------------------------------------

"""
KernelAbstractions kernel: per-bin contribution to the squared residual
between the 2-channel (A, E) data and the single-source model at parameters
`θ` (an isbits `NTuple`, so `ForwardDiff.Dual` elements compile to GPU code).
"""
@kernel function loss_bins!(out, @Const(freqs), @Const(Sn), @Const(data_A), @Const(data_E),
    θ::NTuple{N_PARAMS}, wp::WaveformParams)
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

# GPU execution is serialized library-wide and device buffers are cached:
# GPU drivers are not reliably safe under concurrent multi-task access
# (observed Level Zero segmentation fault), and per-evaluation device
# allocation at ~10³–10⁴ calls per sweep destabilizes long sessions
# (observed oneAPI freed-reference failure). The lock is never taken on CPU
# paths and adds no GPU-side cost, since the device serializes kernels; the
# cache reduces device allocations to one buffer per (backend, eltype, shape).
const GPU_LOCK = ReentrantLock()
const DEVICE_BUFFER_CACHE = Dict{Tuple{UInt,DataType,Dims},Any}()

function device_buffer(backend, ::Type{T}, dims::Dims) where {T}
    key = (objectid(backend), T, dims)
    buf = get(DEVICE_BUFFER_CACHE, key, nothing)
    if buf === nothing
        buf = KernelAbstractions.zeros(backend, T, dims)
        DEVICE_BUFFER_CACHE[key] = buf
    end
    return buf
end

"""
$(TYPEDSIGNATURES)

Empty the device-buffer cache, releasing every cached device array (the
backend frees them on garbage collection). Call between campaigns on
memory-constrained devices; the cache refills on demand.
"""
function clear_device_buffers!()
    lock(GPU_LOCK) do
        empty!(DEVICE_BUFFER_CACHE)
    end
    return nothing
end

# --- allocation-free device reductions ---------------------------------------
# GPU `sum`/`sum(…; dims)` allocates its result and internal partial buffers on
# the device on every call; at 10³–10⁴ loss evaluations per sweep this churn
# destabilizes long campaigns (observed: CUDA pool exhaustion following
# "Failed to query free GPU memory", oneAPI freed-reference failure). The
# fixed two-stage reduction below touches only cached device buffers: a
# strided partial-sums kernel writes a (P × M) buffer, and the host finishes
# the P-row sum. The CPU lanes path keeps `Base.sum` (pairwise, deterministic;
# pinned by the CPU-equivalence regression tests).

const REDUCTION_PARTIAL_ROWS = 256

reduction_rows(nrows::Int) = min(REDUCTION_PARTIAL_ROWS, nrows)

@kernel function partial_column_sums!(partial, @Const(out), nrows::Int, P::Int)
    p, m = @index(Global, NTuple)
    acc = 0.0
    i = p
    @inbounds while i <= nrows
        acc += out[i, m]
        i += P
    end
    @inbounds partial[p, m] = acc
end

"""
Column sums of the `(nrows × M)` device matrix `out` through the caller-owned
`(P × M)` partial buffer; returns a host `Vector{Float64}` of length `M`.
No device memory is allocated.
"""
function column_sums_via!(partial, backend, out, nrows::Int)
    P, M = size(partial)
    partial_column_sums!(backend)(partial, out, nrows, P; ndrange = (P, M))
    KernelAbstractions.synchronize(backend)
    host = Array(partial)
    return vec(sum(host; dims = 1))
end

# Function barrier below `device_buffer`'s untyped cache: `out` is concretely
# typed here, so the kernel launch and reduction pay one dynamic dispatch per
# evaluation instead of one per operation. `partial === nothing` selects the
# generic allocating reduction (CPU path, or non-Float64 eltypes).
function launch_loss!(out, partial, backend, freqs, Sn_vals, data_A, data_E, θ, wp, df)
    loss_bins!(backend)(out, freqs, Sn_vals, data_A, data_E, θ, wp; ndrange = length(freqs))
    KernelAbstractions.synchronize(backend)
    s =
        partial === nothing ? sum(out) :
        column_sums_via!(partial, backend, reshape(out, length(out), 1), length(out))[1]
    return 4.0 * df * s
end

function device_loss(p::AbstractVector, freqs, Sn_vals, data_A, data_E, df::Real,
    wp::WaveformParams, backend)
    T = eltype(p)
    θ = ntuple(i -> p[i], Val(N_PARAMS))
    if backend isa KernelAbstractions.CPU
        out = KernelAbstractions.zeros(backend, T, length(freqs))
        return launch_loss!(out, nothing, backend, freqs, Sn_vals, data_A, data_E, θ,
            wp, df)
    end
    lock(GPU_LOCK) do
        out = device_buffer(backend, T, (length(freqs),))
        partial =
            T === Float64 ?
            device_buffer(backend, Float64, (reduction_rows(length(freqs)), 1)) : nothing
        return launch_loss!(out, partial, backend, freqs, Sn_vals, data_A, data_E, θ,
            wp, df)
    end
end

# --- dual-number lanes path --------------------------------------------------
# Device arrays with Dual eltypes break some GPU runtimes (observed: Level Zero
# rejects the 56-byte-eltype reduction with ZE_RESULT_ERROR_INVALID_SIZE), so
# under ForwardDiff the kernel computes with Duals internally but stores the
# contribution as scalar Float64 "lanes" (value + partials, recursively) in a
# plain n × M matrix; the M standard reductions then run on every backend and
# the scalar Dual is reassembled on the host.

rebuild_dual(::Type{Float64}, s, i::Int) = (s[i], i + 1)
function rebuild_dual(::Type{ForwardDiff.Dual{T,V,N}}, s, i::Int) where {T,V,N}
    v, j = rebuild_dual(V, s, i)
    parts, j2 = _rebuild_partials(V, s, j, Val(N))
    return ForwardDiff.Dual{T,V,N}(v, ForwardDiff.Partials{N,V}(parts)), j2
end
_rebuild_partials(::Type{V}, s, i::Int, ::Val{0}) where {V} = ((), i)
function _rebuild_partials(::Type{V}, s, i::Int, ::Val{K}) where {V,K}
    p1, j = rebuild_dual(V, s, i)
    rest, j2 = _rebuild_partials(V, s, j, Val(K - 1))
    return (p1, rest...), j2
end

# Direct recursive per-scalar stores (no intermediate NTuple{M}): materializing
# the 49-lane tuple of a nested Hessian dual makes some GPU compilers fall back
# to heap allocation (`gpu_malloc` InvalidIRError on IGC); writing each lane as
# it is produced keeps the kernel allocation-free. Lane order matches
# `rebuild_dual`: value first, then partials 1..N, recursively.
@inline function _store_dual!(out, i, k::Int, x::Float64)
    @inbounds out[i, k] = x
    return k + 1
end
@inline function _store_dual!(out, i, k::Int, d::ForwardDiff.Dual{T,V,N}) where {T,V,N}
    k2 = _store_dual!(out, i, k, ForwardDiff.value(d))
    return _store_parts!(out, i, k2, ForwardDiff.partials(d), Val(N))
end
@inline _store_parts!(out, i, k::Int, ps, ::Val{0}) = k
@inline function _store_parts!(out, i, k::Int, ps, ::Val{J}) where {J}
    k2 = _store_parts!(out, i, k, ps, Val(J - 1))
    return _store_dual!(out, i, k2, ps[J])
end

@kernel function loss_bins_lanes!(out, @Const(freqs), @Const(Sn), @Const(data_A),
    @Const(data_E),
    θ::NTuple{N_PARAMS}, wp::WaveformParams)
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
        c = (real(conj(diff_A) * diff_A) + real(conj(diff_E) * diff_E)) / Sn[i]
        _store_dual!(out, i, 1, c)
    end
end

# Function barrier (see launch_loss!): concretely typed lanes launch.
# `partial === nothing` keeps the deterministic `Base.sum` reduction (CPU).
function launch_loss_lanes!(out, partial, backend, freqs, Sn_vals, data_A, data_E, θ, wp,
    df, ::Type{D}) where {D}
    loss_bins_lanes!(backend)(
        out,
        freqs,
        Sn_vals,
        data_A,
        data_E,
        θ,
        wp;
        ndrange = length(freqs),
    )
    KernelAbstractions.synchronize(backend)
    s =
        partial === nothing ? Array(vec(sum(out; dims = 1))) :
        column_sums_via!(partial, backend, out, size(out, 1))
    dual, _ = rebuild_dual(D, s, 1)
    return 4.0 * df * dual
end

function device_loss(p::AbstractVector{D}, freqs, Sn_vals, data_A, data_E, df::Real,
    wp::WaveformParams, backend) where {D<:ForwardDiff.Dual}
    θ = ntuple(i -> p[i], Val(N_PARAMS))
    M = sizeof(D) ÷ sizeof(Float64)
    if backend isa KernelAbstractions.CPU
        out = KernelAbstractions.zeros(backend, Float64, (length(freqs), M))
        return launch_loss_lanes!(
            out,
            nothing,
            backend,
            freqs,
            Sn_vals,
            data_A,
            data_E,
            θ,
            wp,
            df,
            D,
        )
    end
    lock(GPU_LOCK) do
        out = device_buffer(backend, Float64, (length(freqs), M))
        partial = device_buffer(backend, Float64, (reduction_rows(length(freqs)), M))
        fill!(out, 0.0)
        return launch_loss_lanes!(
            out,
            partial,
            backend,
            freqs,
            Sn_vals,
            data_A,
            data_E,
            θ,
            wp,
            df,
            D,
        )
    end
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
$(TYPEDSIGNATURES)

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
        error(
            "GPU path supports the 2-channel (A, E) configuration only; " *
            "set include_t_channel = false (the T channel is identically zero).",
        )
    if freqs isa Array || Sn_vals isa Array || any(a -> a isa Array, data_stream)
        error(
            "Backend is $(typeof(backend)) but the frequency/PSD/data arrays are CPU " *
            "Arrays — move them with to_backend(x, backend), or pass backend = CPU(). " *
            "(get_best_backend() returns a GPU whenever one is functional, so pass the " *
            "backend explicitly when your arrays live on the host.)",
        )
    end
    return p ->
        device_loss(p, freqs, Sn_vals, data_stream[1], data_stream[2], df, wp, backend)
end

# --- optimization -----------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Minimum squared distance `D²` between the composite `data_stream` and the
single-source manifold, found by box-constrained optimization within the
physical `bounds`:

- `:ipnewton` (default): interior-point Newton using the exact ForwardDiff
  Hessian — fast convergence and a much lower convergence floor than L-BFGS.
- `:lbfgs_box`: `Fminbox(LBFGS())`, gradient-only.

Physics keywords (`mass_scale`, `sky_theta`, …) are accepted via `kwargs`.
Returns `(D², best_fit, optim_result)`.
"""
function calculate_numerical_distance(data_stream::Tuple, theta_guess::AbstractVector,
    freqs::AbstractVector, Sn_vals::AbstractVector, df::Real;
    g_tol::Real = 1e-10, iterations::Int = 100,
    backend = get_best_backend(),
    optimizer::Symbol = :ipnewton,
    bounds::Union{Nothing,ParameterBounds} = default_bounds(),
    hessian_chunk::Int = 0,
    kwargs...)
    wp = waveform_params(; kwargs...)
    loss = loss_function(data_stream, freqs, Sn_vals, df, wp, backend)

    # ForwardDiff configs are constructed once per solve (they depend only on
    # the parameter length), not on every optimizer callback — per-call
    # construction allocates fresh dual work arrays thousands of times per
    # sweep. hessian_chunk > 0 limits the outer dual width: (1+c)(1+6) lanes
    # per kernel launch instead of 49 — the fallback for GPU compilers whose
    # module build fails on the full nested-dual kernel (see docs/roadmap).
    x_proto = collect(Float64, theta_guess)
    grad_cfg = ForwardDiff.GradientConfig(loss, x_proto)
    g!(G, x) = ForwardDiff.gradient!(G, loss, x, grad_cfg)
    hess_cfg =
        hessian_chunk > 0 ?
        ForwardDiff.HessianConfig(loss, x_proto,
            ForwardDiff.Chunk(min(hessian_chunk, length(x_proto)))) :
        ForwardDiff.HessianConfig(loss, x_proto)
    h!(H, x) = ForwardDiff.hessian!(H, loss, x, hess_cfg)

    opts = Optim.Options(g_tol = g_tol, iterations = iterations, show_trace = false)

    opt_res = if optimizer === :ipnewton
        bounds === nothing && error("optimizer = :ipnewton requires bounds")
        x0 = clamp_interior(theta_guess, bounds)
        obj = TwiceDifferentiable(loss, g!, h!, x0)
        cons =
            TwiceDifferentiableConstraints(collect(bounds.lower), collect(bounds.upper))
        optimize(obj, cons, x0, IPNewton(), opts)
    elseif optimizer === :lbfgs_box
        bounds === nothing && error("optimizer = :lbfgs_box requires bounds")
        x0 = clamp_interior(theta_guess, bounds)
        obj = OnceDifferentiable(loss, g!, x0)
        optimize(
            obj,
            collect(bounds.lower),
            collect(bounds.upper),
            x0,
            Fminbox(LBFGS()),
            opts,
        )
    else
        error("Unknown optimizer :$optimizer (expected :ipnewton or :lbfgs_box)")
    end

    return Optim.minimum(opt_res), Optim.minimizer(opt_res), opt_res
end

# relative tolerance (of the bound width, or of the coordinate magnitude
# for one-sided bounds) within which a best-fit coordinate is reported as
# sitting on an active physical bound
const AT_BOUND_RTOL = 1e-6

"""
$(TYPEDSIGNATURES)

Convergence diagnostics persisted per optimization: convergence flag,
iteration count, final gradient norm, and whether the best fit sits on an
active physical bound (within a relative tolerance of the bound width).
"""
function optimization_diagnostics(opt_res::Optim.MultivariateOptimizationResults,
    best_fit::AbstractVector,
    bounds::Union{Nothing,ParameterBounds})
    at_bound = false
    if bounds !== nothing
        for i in eachindex(best_fit)
            bounds.periodic[i] && continue
            lo, hi = bounds.lower[i], bounds.upper[i]
            tol =
                isfinite(lo) && isfinite(hi) ? AT_BOUND_RTOL * (hi - lo) :
                AT_BOUND_RTOL * max(1.0, abs(best_fit[i]))
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
