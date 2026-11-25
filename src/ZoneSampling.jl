"""
Sampling geometry of the confusion-zone maps: sampling directions in
Fisher-normalized planes, adaptive angular refinement, the vertices where the
zone boundary meets the prior box, and the mirroring of the half-circle onto
the full circle with prior capping.
"""
module ZoneSampling

using DocStringExtensions: TYPEDSIGNATURES
using ..Bounds: ray_box_crossing
using ..Geometry: cap_at_prior

# angular tolerance below which two map directions are the same vertex
const ANGLE_DEDUPE_TOL = 1e-10
# boundary-radius floor guarding the neighbor-ratio division of the
# angular refinement against exactly vanishing capped radii
const RADIUS_UNDERFLOW = 1e-300

"""
$(TYPEDSIGNATURES)

Unit direction in parameter units of the sampling angle `phi` of a plane
whose axes are measured in their Fisher scales `sigma_x`, `sigma_y`:
`(σ_x cos φ, σ_y sin φ)` normalized. Uniform sampling in `phi` resolves a
zone uniformly when its extent along each axis is proportional to that
axis' σ, which is what the Fisher metric predicts to leading order; the
map is odd in `phi ↦ phi + π`, so mirroring stays exact.
"""
@inline function unit_direction(phi::Real, sigma_x::Real, sigma_y::Real)
    vx = sigma_x * cos(phi)
    vy = sigma_y * sin(phi)
    inv_norm = 1 / hypot(vx, vy)
    return vx * inv_norm, vy * inv_norm
end

"""
$(TYPEDSIGNATURES)

Adaptive angular refinement of the half-circle `entries` (NamedTuples
`(phi, K, g, wall)`, `wall` the `WALL_SELF`/`WALL_ANTIPODE` mask of directions
known to end on a prior wall): up to `max_levels` passes insert the
midpoint direction wherever the capped radius of cyclically consecutive
directions jumps by more than the factor `ratio_tol` on either half of the
circle; `r_cap_pair(e)` returns the capped radii of a direction and of its
mirror image (in the Fisher-normalized sampling plane), whose prior-box crossing
differs when the box is asymmetric. Past the last direction of the half-circle
the circle continues with the mirror image of the first, so there the two
halves of the pair are exchanged. A pair with a non-finite radius is not
compared. Each pass evaluates all midpoints as one batch through
`eval_angles`. Leaves `entries` sorted by angle and returns the number of
directions added.
"""
function refine_directions!(entries::Vector, eval_angles, r_cap_pair;
    max_levels::Integer, ratio_tol::Real)
    added = 0
    for _ in 1:max_levels
        sort!(entries, by = e -> e.phi)
        rcaps = [r_cap_pair(e) for e in entries]
        mids = Float64[]
        for i in 1:length(entries)
            j = mod1(i + 1, length(entries))
            wraps = j == 1
            gap = (wraps ? π + entries[1].phi : entries[j].phi) - entries[i].phi
            jump = false
            for half in (1, 2)
                r1, r2 = rcaps[i][half], rcaps[j][wraps ? 3 - half : half]
                (isfinite(r1) && isfinite(r2)) || continue
                ratio = max(r1, r2) / max(min(r1, r2), RADIUS_UNDERFLOW)
                ratio > ratio_tol && (jump = true)
            end
            jump && push!(mids, entries[i].phi + gap / 2)
        end
        isempty(mids) && break
        curvature_new = eval_angles(mids)
        append!(
            entries,
            [
                (phi = mids[i], K = curvature_new[i][1], g = curvature_new[i][2],
                    wall = 0x00)
                for i in 1:length(mids)
            ],
        )
        added += length(mids)
    end
    sort!(entries, by = e -> e.phi)
    return added
end

"""
Wall mask bit of a half-circle direction entry: the direction itself ends
on a prior wall (it is a bisected crossover or a box corner).
"""
const WALL_SELF = 0x01

"""
Wall mask bit of a half-circle direction entry: its antipode ends on a
prior wall (the box need not be mirror-symmetric, so the two halves are
marked independently).
"""
const WALL_ANTIPODE = 0x02

"""
$(TYPEDSIGNATURES)

Wall mask bit of a full-circle angle: `WALL_SELF` on `[0, π)`,
`WALL_ANTIPODE` on `[π, 2π)`.
"""
wall_bit(alpha_full::Real) = mod(alpha_full, 2π) < π ? WALL_SELF : WALL_ANTIPODE

"""
$(TYPEDSIGNATURES)

Merge a wall-mask bit into the entry of `entries` whose direction lies
within `ANGLE_DEDUPE_TOL` of `phi`; returns `true` when such an
entry exists (the vertex is a duplicate).
"""
function mark_existing_wall!(entries::Vector, phi::Real, bit::UInt8)
    idx = findfirst(e -> abs(e.phi - phi) < ANGLE_DEDUPE_TOL, entries)
    idx === nothing && return false
    entries[idx] = merge(entries[idx], (wall = entries[idx].wall | bit,))
    return true
end

"""
$(TYPEDSIGNATURES)

Corner vertices at capping crossovers, located by bisection. The boundary polygon chords
over the direction where the mathematical contour pierces a prior wall
(`r_math = r_box`), chamfering the zone's corners, and the neighbor-ratio
refinement does not detect it, because the capped radius saturates at `r_box`
on the wall side. Scans the full circle (the box need not be
mirror-symmetric), brackets every capped/uncapped transition between
consecutive directions, bisects each bracket on the capping predicate
`capped_at(alpha, K)` for `iters` iterations, all active brackets in one
batch (the predicate stays defined where `r_math = Inf` on degenerate
directions, at which a sign-based bisection meets `Inf - Inf`), folds the
vertices onto the half-circle, removes duplicates and appends them to `entries`
(kept sorted) with the wall mask of the side that ends on the wall. The
capping stage flags these vertices prior-limited by construction, since
their `r_math = r_box` holds only to the bisection resolution. Returns
`(n_brackets, n_inserted)`.
"""
function insert_crossover_vertices!(entries::Vector, eval_angles, capped_at, iters::Int)
    n_entries = length(entries)
    alphas = vcat([e.phi for e in entries], [e.phi + π for e in entries])
    K_full_circle = vcat([e.K for e in entries], [e.K for e in entries]) # K is even
    caps = [capped_at(alphas[k], K_full_circle[k]) for k in 1:(2n_entries)]
    lo = Float64[]
    hi = Float64[]
    lo_capped = Bool[]
    for k in 1:(2n_entries)
        j = mod1(k + 1, 2n_entries)
        caps[k] == caps[j] && continue
        push!(lo, alphas[k])
        push!(hi, alphas[j] + (j == 1 ? 2π : 0.0))
        push!(lo_capped, caps[k])
    end
    isempty(lo) && return (0, 0)
    for _ in 1:iters
        mids = (lo .+ hi) ./ 2
        curvature_mid = eval_angles(mod.(mids, π)) # K is even
        for b in eachindex(mids)
            if capped_at(mids[b], curvature_mid[b][1]) == lo_capped[b]
                lo[b] = mids[b]
            else
                hi[b] = mids[b]
            end
        end
    end
    # fold onto the half-circle (the mirror adds the antipode);
    # remove duplicates, since a mirror-symmetric box yields the same φ twice
    phis_new = Float64[]
    walls_new = UInt8[]
    for b in eachindex(lo)
        alpha_vertex = (lo[b] + hi[b]) / 2
        phi_vertex = mod(alpha_vertex, Float64(π))
        bit = wall_bit(alpha_vertex)
        dup = findfirst(p -> abs(p - phi_vertex) < ANGLE_DEDUPE_TOL, phis_new)
        if dup !== nothing
            walls_new[dup] |= bit
            continue
        end
        mark_existing_wall!(entries, phi_vertex, bit) && continue
        push!(phis_new, phi_vertex)
        push!(walls_new, bit)
    end
    if !isempty(phis_new)
        curvature_new = eval_angles(phis_new)
        append!(
            entries,
            [
                (
                    phi = phis_new[i],
                    K = curvature_new[i][1],
                    g = curvature_new[i][2],
                    wall = walls_new[i],
                )
                for i in eachindex(phis_new)
            ],
        )
        sort!(entries, by = e -> e.phi)
    end
    return (length(lo), length(phis_new))
end

"""
$(TYPEDSIGNATURES)

Box-corner vertices: where two walls are simultaneously active,
consecutive samples sit on different walls and their chord cuts the box
corner. The corner direction is known analytically; if the ray through a
finite box corner is capped there (`capped_at`), the boundary passes
through that corner, which is inserted into `entries` (one K
evaluation per finite corner, at most four; kept sorted). `corner_angle`
maps a parameter-space corner `(cx, cy)` to the sampling angle of the
direction through it (the polar angle `atan(cy, cx)` by default; for
Fisher-normalized sampling the inverse of [`unit_direction`](@ref)). Returns
the number of corners inserted.
"""
function insert_box_corner_vertices!(entries::Vector, eval_angles, capped_at,
    box::NTuple{4,Float64}, corner_angle = (cx, cy) -> atan(cy, cx))
    corner_alphas = [
        corner_angle(cx, cy) for cx in (box[1], box[2]), cy in (box[3], box[4])
        if isfinite(cx) && isfinite(cy)
    ]
    isempty(corner_alphas) && return 0
    curvature_corner = eval_angles(mod.(corner_alphas, π))
    corner_new = Tuple{Float64,Float64,Float64,UInt8}[]
    for k in eachindex(corner_alphas)
        capped_at(corner_alphas[k], curvature_corner[k][1]) || continue
        phi_vertex = mod(corner_alphas[k], Float64(π))
        bit = wall_bit(corner_alphas[k])
        dup = findfirst(t -> abs(t[1] - phi_vertex) < ANGLE_DEDUPE_TOL, corner_new)
        if dup !== nothing
            t = corner_new[dup]
            corner_new[dup] = (t[1], t[2], t[3], t[4] | bit)
            continue
        end
        mark_existing_wall!(entries, phi_vertex, bit) && continue
        push!(
            corner_new,
            (phi_vertex, curvature_corner[k][1], curvature_corner[k][2], bit),
        )
    end
    if !isempty(corner_new)
        append!(
            entries,
            [(phi = t[1], K = t[2], g = t[3], wall = t[4]) for t in corner_new],
        )
        sort!(entries, by = e -> e.phi)
    end
    return length(corner_new)
end

"""
$(TYPEDSIGNATURES)

Mirror the half-circle `entries` to the full circle and cap every
direction at the prior box: K and g are exactly even in the direction, so
`r_math` is mirrored bitwise, while `r_box` is re-evaluated with the
exactly negated direction components (the prior box need not be
symmetric). `direction(phi)` maps a sampling angle to the unit direction
`(cos, sin)` in parameter units ([`unit_direction`](@ref) for
Fisher-normalized sampling; `(cos φ, sin φ)` by default). Vertices whose wall
mask marks the evaluated side are capped exactly at the wall and flagged
prior-limited; every other direction goes through [`cap_at_prior`](@ref).
Directions with `g < g_uu_degenerate` are flagged degenerate. Returns the
per-direction polar table as a NamedTuple of vectors (`angle`, `K_raw`,
`g_uu`, `r_math`, `r_box`, `r_cap`, `dir_cos`, `dir_sin`, `prior_limited`,
`degenerate`).
"""
function mirror_to_full_circle(entries::Vector, box::NTuple{4,Float64}, r_math_of,
    g_uu_degenerate::Real; direction = phi -> (cos(phi), sin(phi)))
    half = length(entries)
    angle = Vector{Float64}(undef, 2half)
    K_raw = similar(angle)
    g_uu = similar(angle)
    r_math = similar(angle)
    r_box = similar(angle)
    r_cap = similar(angle)
    dir_cos = similar(angle)
    dir_sin = similar(angle)
    prior_limited = falses(2half)
    degenerate = falses(2half)
    for (i, e) in enumerate(entries), half_idx in (0, 1)
        k = i + half_idx * half
        c, s = direction(e.phi)
        half_idx == 1 && ((c, s) = (-c, -s))
        angle[k] = e.phi + half_idx * π
        dir_cos[k] = c
        dir_sin[k] = s
        K_raw[k] = e.K
        g_uu[k] = e.g
        r_math[k] = r_math_of(e.K)
        r_box[k] = ray_box_crossing(c, s, box...)
        on_wall = (e.wall & (half_idx == 0 ? WALL_SELF : WALL_ANTIPODE)) != 0x00
        if on_wall && isfinite(r_box[k])
            # bisected crossover / box-corner vertex: on the wall by construction
            r_cap[k] = r_box[k]
            prior_limited[k] = true
        else
            r_cap[k], prior_limited[k] = cap_at_prior(r_math[k], r_box[k])
        end
        degenerate[k] = e.g < g_uu_degenerate
    end
    return (; angle, K_raw, g_uu, r_math, r_box, r_cap, dir_cos, dir_sin, prior_limited,
        degenerate)
end

end
