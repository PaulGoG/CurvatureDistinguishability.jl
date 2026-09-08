# Run Parameter Documentation

**Context:** LISA Two-Source Distinguishability Simulation

This document explains the physical rationale behind the production
configuration (`configs/production_cpu.toml`): the global grid, the seven 1D separation
sweeps, and the seven 2D confusion maps. It is the companion piece for
interpreting pipeline outputs.

## 0. Configuration Files

Two runnable base files ship: `configs/quickstart.toml` (minutes-scale
demonstration, the pipeline default) and `configs/production_cpu.toml`
(the reference campaign documented below). The GPU variants —
`quickstart_gpu.toml`, `production_gpu.toml`, `production_oneapi.toml` —
are thin overlays: each opens with `base_config = "<base>.toml"` (a path
relative to the overlay) followed by only the `[hardware]` keys that
differ. At load the overlay is deep-merged onto its base (sub-tables
recurse; scalars, arrays and `[[sweeps]]`/`[[maps]]` lists present in the
overlay replace the base's; one overlay level only), the merged table is
validated, and the run directory receives the merged, self-contained
`config.toml`; `metadata.toml` records `config_file` and `base_config`.
A fourth overlay, `quickstart_maps.toml`, replaces the quickstart work
items by nine maps on the quickstart grid — the seven production planes
at quickstart-scale base points plus two planes whose base point sits one
zone half-width from a spin wall — as a minutes-scale visual check of the
mapping stage (wall segments, corners, null-direction needles).

The run identifier hashes only the *identity* of a run — every section
except `[hardware]`, `[safety]` and `[monitoring]`, which describe how a
run executes rather than what it computes — so one physical case carries
one identifier on every machine; reruns land in `_r2`, `_r3`, … sibling
directories and `metadata.toml`/`hardware.txt` record backend, host and
timings. The per-host files under `configs/hosts/` are execution-only
overlays on `production_cpu.toml` (backend, thread count, memory budgets,
Hessian chunking) named by hardware model; they carry no host-specific
identifiers.

## 1. Global Simulation Grid

*   **Observation Time (``T_{\mathrm{obs}}``):** ``3.15576 \times 10^7`` s
    (1 Julian year — the same constant that drives the orbital motion).
*   **Frequency Range:** ``10^{-4}`` Hz to ``0.05`` Hz.
*   **Resolution:** at a spacing of ``1/T_{\mathrm{obs}}`` the grid spans
    ``\approx 1.57`` million frequency bins. The `[safety].max_ram_gb` budget
    is checked against this size before anything is allocated.
*   **Separations:** 30 log-spaced values of ``\delta/\delta_{\mathrm{min}}``
    over ``[10^{-2}, 10^{0.3}]``. ``\delta`` is measured in Fisher-metric
    units (``u`` is normalized to ``g(u,u)=1``) and the grid is anchored to
    the threshold separation ``\delta_{\mathrm{min}}`` of each direction, so
    every sweep covers the same ``D^2/\rho^2 \in [10^{-8}, 16]`` dynamic
    range: the leading-order law over six decades, the threshold crossing,
    and the onset of the ``\mathcal{O}(\delta^5)`` departure. The anchoring
    also keeps every fit far above the round-off floor of the parameters —
    for these loud sources (SNR of order ``10^3``–``10^4``) one Fisher σ is
    a relative change of ``10^{-5}``–``10^{-6}`` in the chirp mass, and a
    fixed absolute grid reaching ``10^{-5}`` σ would sit entirely below the
    double-precision resolution of the fit.
*   **Scenario constants:** ``\eta = 2/9`` (mass ratio 2:1), source at
    ecliptic longitude 180° and latitude 30°, inclination 30°, polarization
    45°, constellation phases zero at ``t = 0``. Masses are detector-frame
    masses, as in the LDC catalogues.
*   **Domain of validity:** the inspiral is windowed out at the innermost
    stable circular orbit — ``2.2`` mHz for the ``\mathcal{M} = 4`` s base
    points, ``35`` mHz for ``\mathcal{M} = 0.25`` s, above the band for
    ``\mathcal{M} = 0.1`` s — and the long-wavelength response is used up to
    ``2.6 f_\star``; see [Physics and Waveform Model](physics.md), §8. The
    maps assume equal source amplitudes.

The base points below are written in the scaled units of the configuration
(`distance_scale` = 1 Gpc, `mass_scale` = 1 s ``\approx 2.03\times
10^{5}\,M_\odot``, `time_scale` = ``10^{6}`` s):

| Base point | ``D_L`` [Gpc] | ``\mathcal{M}`` [``M_\odot``] | ``M`` [``M_\odot``] | ``t_c`` [d] | ``f_{\mathrm{ISCO}}`` [mHz] |
|---|---|---|---|---|---|
| `[5.0, 4.0, 20.0, …]` | 5 | ``8.1\times 10^{5}`` | ``2.0\times 10^{6}`` | 231 | 2.2 |
| `[8.0, 1.6, 15.0, …]` | 8 | ``3.2\times 10^{5}`` | ``8.0\times 10^{5}`` | 174 | 5.5 |
| `[5.0, 2.5, 10.0, …]` | 5 | ``5.1\times 10^{5}`` | ``1.3\times 10^{6}`` | 116 | 3.5 |
| `[3.0, 0.25, 25.0, …]` | 3 | ``5.1\times 10^{4}`` | ``1.3\times 10^{5}`` | 289 | 35 |
| `[2.0, 0.1, 20.0, …]` | 2 | ``2.0\times 10^{4}`` | ``5.0\times 10^{4}`` | 231 | 88 |

Distances of 2–8 Gpc correspond to redshifts ``z \approx 0.35``–``1.2``;
every coalescence lies inside the one-year observation.

## 2. 1D Parameter Separation Sweeps

The pipeline validates the geometric distance law ``D^2 \propto \delta^4``
across seven physical scenarios.

### `six_dimensional_diagonal_stress`
*   **Base:** `[8.0, 1.6, 15.0, 1.57, 0.6, -0.4]`; **Direction:** `[0.0, 0.707, -0.5, 0.3, 0.4, -0.1]`
*   A distant (8 Gpc) ``8\times 10^{5}\,M_\odot`` binary whose companion
    simultaneously becomes heavier, merges earlier, changes phase, and
    shifts both spins — no coordinate axis is special. This tests the
    tensorial character of the law: the residual unabsorbable power is
    governed by the extrinsic curvature regardless of how the displacement
    mixes the coordinates.

### `massive_binary_mass_time`
*   **Base:** `[5.0, 4.0, 20.0, 0.0, 0.8, 0.8]`; **Direction:** `[0.0, 1.0, 0.5, 0.0, 0.0, 0.0]`
*   The reference massive binary (``2\times 10^{6}\,M_\odot``, 5 Gpc,
    strongly aligned spins) along the chirp-rate/arrival-time direction
    that dominates realistic overlapping-source populations. With ISCO at
    2.2 mHz the signal occupies the low-frequency band where the galactic
    confusion noise is strongest.

### `unequal_amplitude_mass_time`
*   **Base/Direction:** as `massive_binary_mass_time`; **`amp_ratio`** ``q = 1/2``.
*   The second source carries half the amplitude (twice the luminosity
    distance at the same masses). The prediction acquires the harmonic-mean
    prefactor, ``D^2 = \tfrac{1}{16}K_{\rm num}\delta^4\,(2q/(1+q))^2``, and
    the optimizer is initialized at the weighted midpoint with effective
    amplitude ``(1+q)A``. With ``q=1/2``, using the equal-amplitude law instead
    would misplace the ratio panel by a factor ``9/4``.

### `extreme_spin_orbit_coupling`
*   **Base:** `[5.0, 2.5, 10.0, 1.57, 0.7, -0.7]`; **Direction:** `[0.0, 0.0, 0.0, 1.0, 0.5, 0.5]`
*   Strongly anti-aligned spins maximize the 1.5PN spin–orbit phase twist;
    the direction sweeps phase and both spins (base spins chosen so the
    second source stays inside ``|\chi|\le 1`` at ``\delta_{\max}``).

### `low_mass_time_shift`
*   **Base:** `[3.0, 0.25, 25.0, 0.0, 0.9, 0.0]`; **Direction:** `[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]`
*   A ``1.3\times 10^{5}\,M_\odot`` system whose inspiral spans the whole
    band, displaced purely in coalescence time: isolates the constellation
    mechanics — a time shift places the detector at a different point of
    its orbit, with different antenna patterns and Doppler phase, at every
    contributing frequency.

### `seed_binary_mass_spin_compensation`
*   **Base:** `[2.0, 0.1, 20.0, 0.0, 0.7, 0.5]`; **Direction:** `[0.0, 0.6, 0.0, 0.0, -0.55, -0.55]`
*   A light-seed binary (``5\times 10^{4}\,M_\odot``, ISCO above the band)
    probed along the classic 1.5PN chirp-mass/spin-orbit compensation
    direction: raising ``\mathcal{M}`` while lowering both aligned spins
    keeps the phasing nearly stationary, so ``K(u)`` is suppressed by the
    partial degeneracy. Empirically charts the validity boundary of the
    quartic law where curvature is weak.

### `near_degenerate_flat_spin`
*   **Base:** `[5.0, 4.0, 20.0, 0.0, 0.3, -0.3]`; **Direction:** `[0.0, 0.15, 0.0, 0.0, 0.4, -0.92]`
*   Dominantly the exactly flat spin combination of the 1.5PN phase,
    ``\mathrm{d}\beta = 0`` (``\mathrm{d}\chi_2/\mathrm{d}\chi_1 = -2.29`` at
    ``\eta = 2/9``; it reduces to ``\chi_a`` at equal mass), with a 15%
    chirp-mass admixture so the direction is measurable and the second
    source stays physical at ``\delta_{\max}``. The sharpest stress test of
    the law near a null direction.

## 3. 2D Zone of Confusion Mappings

These configurations evaluate the extrinsic curvature over a 2D plane to
draw the ``\delta_{\mathrm{min}}`` boundary. The angular resolution comes
from the global `[mapping].n_angles` (per-map override allowed); directions
are sampled uniformly in the Fisher-normalized plane (each axis measured in
its own ``\sigma`` at the base point, so that a plane like ``t_c`` against
``\phi_c``, whose axes differ by ``10^{4}``–``10^{5}`` in ``\sigma``, is
resolved uniformly rather than only along its axes), the solver computes
half the directions and mirrors (``K`` is exactly even), refines adaptively
near boundary spikes, and caps every direction at the physical prior box
(`Prior_Limited` flags in the CSV; the quoted prior-limited fraction is a
fraction of sampling angle). Wherever consecutive
directions straddle a capping transition, the solver additionally bisects
for the exact crossover direction ``r_{\mathrm{math}}(\varphi) =
r_{\mathrm{box}}(\varphi)`` and inserts it as a boundary vertex
(`[mapping].corner_bisect_iters`, default 25; 0 disables), so zone corners
at prior walls are exact rather than chamfered by a polygon chord — the
neighbor-ratio refinement alone cannot resolve them, because the capped
radius saturates at ``r_{\mathrm{box}}`` on the wall side. Note that
`replot.jl --rho` reuses the stored directions, so corner vertices are exact
only for the original threshold; rerun the map stage for publication-grade
corners at a very different ``\rho``.

1.  **Mass vs. Time (`mass_vs_time_degeneracy`, 2 vs 3)** — the classic chirp-rate / arrival-time degeneracy; base ``\theta_0 = [5.0, 4.0, 20.0, 0.0, 0.8, 0.8]``.
2.  **Spin 1 vs. Spin 2 (`spin1_vs_spin2_coupling`, 5 vs 6)** — the phase sees the spins only through ``\beta``, so the line ``\mathrm{d}\beta = 0`` is exactly flat: the mathematical zone diverges along it and the published zone is limited by the spin prior ``[-1, 1]`` there (a narrow wedge, since the loud source confines every measurable direction to a tiny radius).
3.  **Mass vs. Spin 1 (`mass_vs_spin1_twist`, 2 vs 5)** — mass/spin phasing trade-off; the zone crosses the ``\chi_1 \le 1`` bound and is capped there.
4.  **Time vs. Phase (`time_vs_phase_doppler`, 3 vs 4)** — absolute time translation vs orbital phase; the pure-phase direction is quasi-degenerate (``\partial^2_\varphi h \parallel`` tangent space), producing a near-singular boundary spike that the adaptive refinement resolves.
5.  **Mass vs. Phase, low-mass system (`mass_vs_phase_low_mass`, 2 vs 4)** — the chirp-rate/phase degeneracy at the band-filling base ``\theta_0 = [3.0, 0.25, 25.0, 0.0, 0.9, 0.0]``.
6.  **Time vs. Spin 1, high-spin system (`time_vs_spin1_high_spin`, 3 vs 5)** — the spin-orbit phase twist trades against arrival time at the strongly anti-aligned base ``\theta_0 = [5.0, 2.5, 10.0, 1.57, 0.7, -0.7]``.
7.  **Phase vs. Spin 1, seed binary (`phase_vs_spin1_seed`, 4 vs 5)** — a weakly chirping light-seed source leaves phase and spin loosely constrained: the zone is terminated by the physical walls (phase topology ``\pm\pi`` and the spin prior) over a substantial fraction of directions.
