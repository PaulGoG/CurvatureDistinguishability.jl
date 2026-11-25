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

public loglog_slope, ratio_correction_fit, validity_radius, optimizer_floor,
    above_floor_mask, angular_fraction, perturbative_mask, MIN_FIT_POINTS

"""
Minimum number of clean points for the log-log slope and the
ratio-correction fits; with fewer points neither fit is reported (`NaN`).
"""
const MIN_FIT_POINTS = 3
# determinant underflow guard of the 2×2 ratio-correction normal equations
const DET_UNDERFLOW = 1e-300

"""
$(TYPEDSIGNATURES)

Unweighted least-squares slope of `log10(y)` against `log10(x)` with its
standard error (`NaN` with fewer than 3 points; the slope itself needs 2).
Fits the quartic-law exponent of a sweep's clean window. The points are
deterministic, so the standard error measures their scatter about a power law
over the fitted window, not a statistical uncertainty: the fitted value is an
effective slope over that window.
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
relative to `D²_theo`. Returns NaNs with fewer than 3 points.
`c₁_err` is the unweighted least-squares standard error of `c₁`, a measure of scatter in
the same sense as that of [`loglog_slope`](@ref).
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

Smallest separation `δ > 0` at which the fitted correction to the quartic law,
`ratio − 1 ≈ c₁δ + c₂δ²`, reaches `fraction` in magnitude: the smallest
positive root of `|c₁δ + c₂δ²| = fraction`. `c₁` enters only when it is
significant, `|c₁| ≥ 2 c₁_err`; otherwise, and when `c₁` or `c₁_err` is not
finite, the radius follows from `c₂` alone. A non-finite `c₂` counts as zero.
Returns `Inf` when neither term contributes.
"""
function validity_radius(c1::Real, c1_err::Real, c2::Real, fraction::Real)
    fraction > 0 ||
        throw(ArgumentError("validity_radius: fraction must be > 0, got $fraction"))
    a = (isfinite(c1) && isfinite(c1_err) && abs(c1) >= 2 * c1_err) ? Float64(c1) : 0.0
    b = isfinite(c2) ? Float64(c2) : 0.0
    a == 0 && b == 0 && return Inf
    b == 0 && return fraction / abs(a)
    # roots of b δ² + a δ + c = 0 with c = ∓fraction, in the form that avoids
    # cancellation when |b| fraction ≪ a² (a small c₂ beside a significant c₁)
    best = Inf
    for c in (-fraction, fraction)
        disc = a^2 - 4 * b * c
        disc < 0 && continue
        q = -(a + copysign(sqrt(disc), a)) / 2
        for r in (q / b, q == 0 ? NaN : c / q)
            r > 0 && (best = min(best, r))
        end
    end
    return best
end

"""
$(TYPEDSIGNATURES)

Points admitted to the `O(δ⁵)` ratio-correction fit: the `clean` (above
optimizer floor) separations whose `D²_num/D²_theo` ratio departs from
unity by at most `max_departure`
(`[pipeline.sweep_settings].correction_fit_max_departure`). The expansion
`1 + c₁δ + c₂δ²` is perturbative in the departure from the leading-order
law, so separations far off the law would bias `c₁`, and with it the
validity window derived from it, rather than measure it.
"""
function perturbative_mask(ratio::AbstractVector, clean::AbstractVector{Bool},
    max_departure::Real)
    return clean .& (abs.(ratio .- 1.0) .<= max_departure)
end

"""
$(TYPEDSIGNATURES)

Estimate of a sweep's optimizer floor: the floor-dominated points
are the contiguous run of separations, starting from the smallest, whose
`D²_num/D²_theo` ratio is at or above `ratio_threshold`
(`[pipeline.sweep_settings].floor_detection_ratio`). A flat floor under a
`δ⁴` theory makes that ratio decrease monotonically with `δ`, so
super-threshold ratios at large `δ` mark the breakdown of the leading-order
law, not the floor. A non-positive `D²_num` (a fit that returned zero) counts
as floor-dominated whatever its ratio. The floor level is the largest
floor-dominated `D²_num`; `NaN` when the smallest separation is already clean.
"""
function optimizer_floor(
    D2_num::AbstractVector, ratio::AbstractVector, ratio_threshold::Real)
    n_floor = 0
    for (d, r) in zip(D2_num, ratio)
        (r >= ratio_threshold || d <= 0) || break
        n_floor += 1
    end
    return n_floor == 0 ? NaN : maximum(view(D2_num, 1:n_floor))
end

"""
$(TYPEDSIGNATURES)

Production clean-point rule, shared by the sweep stage and `RunFigures`: a
point is clean when its `D²_num` lies strictly above the optimizer floor
(any positive point when no floor was detected), whatever its ratio.
Convergence flags are not consulted: at the iteration and tolerance caps,
flagged points at ratio ≈ 1 are optima.
"""
above_floor_mask(D2_num::AbstractVector, floor_level::Real) =
    isnan(floor_level) ? (D2_num .> 0) : (D2_num .> floor_level)

"""
$(TYPEDSIGNATURES)

Fraction of the full circle subtended by the directions flagged in `mask`.
Each direction of the cyclic grid `angle` [rad] (any order, spanning one
turn) carries half the gap to either neighbor, so the result is an angular
measure whatever the local sampling density; a plain count over-weights
refined regions. A gap between a flagged and an unflagged direction counts
half, so the fraction is resolved to about one sampling gap per transition.
Returns `0.0` for an empty grid and the flag itself for a single direction; a
grid that spans more than one turn is an `ArgumentError`.
"""
function angular_fraction(angle::AbstractVector{<:Real}, mask::AbstractVector{Bool})
    length(angle) == length(mask) || throw(
        DimensionMismatch(
            "angular_fraction: angle and mask must have equal lengths, got " *
            "$(length(angle)) and $(length(mask))",
        ),
    )
    n = length(angle)
    n == 0 && return 0.0
    n == 1 && return mask[1] ? 1.0 : 0.0
    lo, hi = extrema(angle)
    hi - lo <= 2π || throw(
        ArgumentError(
            "angular_fraction: the grid must span at most one turn, got angles " *
            "from $lo to $hi",
        ),
    )
    order = sortperm(angle)
    flagged = 0.0
    for k in 1:n
        mask[order[k]] || continue
        prev = angle[order[mod1(k - 1, n)]] - (k == 1 ? 2π : 0.0)
        next = angle[order[mod1(k + 1, n)]] + (k == n ? 2π : 0.0)
        flagged += (next - prev) / 2
    end
    return flagged / (2π)
end

end # module
