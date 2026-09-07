"""
Fit statistics shared by the pipeline driver and display-time figure
regeneration: the quartic-law log-log slope, the `O(δ⁵)` ratio-correction
fit, optimizer-floor detection and the production clean-point rule. One
implementation consumed by both `Orchestrator` and `RunFigures`, so
run-time and display-time figures cannot diverge.
"""
module Fitting

using DocStringExtensions: TYPEDSIGNATURES
using Statistics: mean

public loglog_slope, ratio_correction_fit, optimizer_floor, above_floor_mask,
    MIN_FIT_POINTS

"""
Minimum number of clean points for the log-log slope and the
ratio-correction fits; with fewer points both return `NaN`.
"""
const MIN_FIT_POINTS = 3
# determinant underflow guard of the 2×2 ratio-correction normal equations
const DET_UNDERFLOW = 1e-300

"""
$(TYPEDSIGNATURES)

Least-squares slope of `log10(y)` against `log10(x)` with its standard
error (NaN with fewer than 3 points). Fits the quartic-law exponent of a
sweep's clean window; shared with the display-time refit in `RunFigures`.
"""
function loglog_slope(x::AbstractVector, y::AbstractVector)
    lx, ly = log10.(x), log10.(y)
    mx, my = mean(lx), mean(ly)
    sxx = sum(abs2, lx .- mx)
    slope = sum((lx .- mx) .* (ly .- my)) / sxx
    n = length(lx)
    se = n > 2 ? sqrt(sum(abs2, ly .- my .- slope .* (lx .- mx)) / ((n - 2) * sxx)) : NaN
    return slope, se
end

"""
$(TYPEDSIGNATURES)

Least-squares fit of `ratio − 1 ≈ c₁δ + c₂δ²` (2×2 normal equations solved in
closed form), quantifying the leading `O(δ⁵)` correction to the quartic law
relative to `D²_th`. Returns NaNs with fewer than 3 points.
"""
function ratio_correction_fit(deltas::AbstractVector, ratio::AbstractVector)
    n = length(deltas)
    n >= MIN_FIT_POINTS || return NaN, NaN, NaN
    y = ratio .- 1.0
    s2 = sum(d^2 for d in deltas)
    s3 = sum(d^3 for d in deltas)
    s4 = sum(d^4 for d in deltas)
    b1 = sum(deltas .* y)
    b2 = sum(deltas .^ 2 .* y)
    det = s2 * s4 - s3^2
    abs(det) < DET_UNDERFLOW && return NaN, NaN, NaN
    c1 = (s4 * b1 - s3 * b2) / det
    c2 = (s2 * b2 - s3 * b1) / det
    resid = y .- c1 .* deltas .- c2 .* deltas .^ 2
    c1_err = n > 2 ? sqrt(max(0.0, sum(abs2, resid) / (n - 2)) * s4 / det) : NaN
    return c1, c1_err, c2
end

"""
$(TYPEDSIGNATURES)

Bootstrap estimate of a sweep's optimizer floor: the floor-dominated points
are the contiguous run of separations, starting from the smallest, whose
`D²_num/D²_theo` ratio is at or above `ratio_threshold`
(`[pipeline.sweep_settings].floor_detection_ratio`) — a flat floor under a
`δ⁴` theory makes that ratio decrease monotonically with `δ`, so
super-threshold ratios at large `δ` mark the breakdown of the leading-order
law, not the floor. The floor level is the largest floor-dominated
`D²_num`; `NaN` when the smallest separation is already clean.
"""
function optimizer_floor(
    D2_num::AbstractVector, ratio::AbstractVector, ratio_threshold::Real)
    n_floor = 0
    for r in ratio
        r >= ratio_threshold || break
        n_floor += 1
    end
    return n_floor == 0 ? NaN : maximum(view(D2_num, 1:n_floor))
end

"""
$(TYPEDSIGNATURES)

Production clean-point rule, shared by the sweep stage and all display-time
figure regeneration (`RunFigures`): only points strictly above the optimizer
floor are clean (all positive points when no floor was detected). Borderline
points inside the floor band are excluded even when their ratio is close to
unity, and convergence flags never exclude a point — the iteration and
tolerance caps are strict enough that flagged points at ratio ≈ 1 are
genuine optima.
"""
above_floor_mask(D2_num::AbstractVector, floor_level::Real) =
    isnan(floor_level) ? (D2_num .> 0) : (D2_num .> floor_level)

end # module
