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
    (1 Julian year — the same constant that drives the orbital Doppler
    modulation).
*   **Frequency Range:** ``10^{-4}`` Hz to ``0.05`` Hz.
*   **Resolution:** at a spacing of ``1/T_{\mathrm{obs}}`` the grid spans
    ``\approx 1.57`` million frequency bins. The `[safety].max_ram_gb` budget
    is checked against this size before anything is allocated.
*   **Separations:** 30 log-spaced values of ``\delta`` over
    ``[10^{-4.8}, 10^{-0.2}]``. ``\delta`` is measured in Fisher-metric units
    (``u`` is normalized to ``g(u,u)=1``), so the physical excursion at
    ``\delta_{\max}`` depends on how well the direction is measured; every
    base/direction pair below keeps the second source inside the physical
    priors at ``\delta_{\max}`` on the production grid.
*   **Domain of validity:** the inspiral model carries no merger cutoff
    and the band reaches ``2.6 f_\star``; for the massive base points
    (``\mathcal{M} = 15`` s) the innermost stable circular orbit lies at
    ``0.63`` mHz, far below ``f_{\max}`` — see
    [Physics and Waveform Model](physics.md), §7. The maps assume equal
    source amplitudes.

## 2. 1D Parameter Separation Sweeps

The pipeline validates the geometric distance law ``D^2 \propto \delta^4``
across seven physical scenarios.

### `six_dimensional_diagonal_stress`
*   **Base:** `[1.0, 1.25, 300.0, 1.57, 0.6, -0.4]`; **Direction:** `[0.0, 0.707, -0.5, 0.3, 0.4, -0.1]`
*   The second source simultaneously becomes heavier, merges earlier,
    changes phase, and shifts both spins — no coordinate axis is special.
    This tests the tensorial character of the law: the residual
    unabsorbable power is governed by the extrinsic curvature regardless of
    how the displacement mixes the coordinates.

### `massive_binary_mass_time`
*   **Base:** `[1.0, 1.5, 200.0, 0.0, 0.8, 0.8]`; **Direction:** `[0.0, 1.0, 0.5, 0.0, 0.0, 0.0]`
*   A representative massive (``\sim 3\times 10^6\,M_\odot`` via
    `mass_scale`) equal-mass binary with strongly aligned spins; the
    chirp-rate/arrival-time direction dominates realistic overlapping-source
    populations.

### `unequal_amplitude_mass_time`
*   **Base/Direction:** as `massive_binary_mass_time`; **`amp_ratio`** ``q = 1/2``.
*   The second source carries half the amplitude. The prediction acquires
    the harmonic-mean prefactor,
    ``D^2 = \tfrac{1}{16}K_{\rm num}\delta^4\,(2q/(1+q))^2``, and the
    optimizer is initialized at the weighted midpoint with effective
    amplitude ``(1+q)A``. With ``q=1/2``, using the equal-amplitude law instead
    would misplace the ratio panel by a factor ``9/4``.

### `extreme_spin_orbit_coupling`
*   **Base:** `[1.0, 2.0, 150.0, 1.57, 0.7, -0.7]`; **Direction:** `[0.0, 0.0, 0.0, 1.0, 0.5, 0.5]`
*   Strongly anti-aligned spins maximize the 1.5PN spin-orbit phase twist;
    the direction sweeps phase and both spins (base spins chosen so the
    second source stays inside ``|\chi|\le 1`` at ``\delta_{\max}``).

### `low_mass_time_shift`
*   **Base:** `[0.5, 0.1, 500.0, 0.0, 0.9, 0.0]`; **Direction:** `[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]`
*   A low-mass system (``\sim 2\times 10^5\,M_\odot``) displaced purely in
    coalescence time: isolates the TDI/Doppler mechanics — a time shift
    places the detector at a different point of its solar orbit at every
    contributing frequency.

### `seed_binary_mass_spin_compensation`
*   **Base:** `[0.8, 0.3, 400.0, 0.0, 0.7, 0.5]`; **Direction:** `[0.0, 0.6, 0.0, 0.0, -0.55, -0.55]`
*   A light-seed binary probed along the classic 1.5PN chirp-mass/spin-orbit
    compensation direction: raising ``\mathcal{M}`` while lowering both
    aligned spins keeps the phasing nearly stationary, so ``K(u)`` is
    suppressed by the partial degeneracy. Empirically charts the validity
    boundary of the quartic law where curvature is weak.

### `near_degenerate_antisymmetric_spin`
*   **Base:** `[1.0, 1.5, 200.0, 0.0, 0.3, -0.3]`; **Direction:** `[0.0, 0.15, 0.0, 0.0, 0.7, -0.7]`
*   Dominantly the exactly flat antisymmetric-spin combination (the
    waveform depends on spins only through
    ``\chi_{\rm eff}=(\chi_1+\chi_2)/2``) with a 15% chirp-mass admixture
    tuned so the second source stays physical at ``\delta_{\max}``
    (``g(u,u)\simeq 0.61``, nearly seven orders of magnitude below the
    well-measured directions). The sharpest stress test of the law near a
    null direction.

## 3. 2D Zone of Confusion Mappings

These configurations evaluate the extrinsic curvature over a 2D plane to
draw the ``\delta_{\mathrm{min}}`` boundary. The angular resolution comes
from the global `[mapping].n_angles` (per-map override allowed); the solver
computes half the directions and mirrors (``K`` is exactly even), refining
adaptively near boundary spikes, and caps every direction at the physical
prior box (`Prior_Limited` flags in the CSV). Wherever consecutive
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

1.  **Mass vs. Time (`mass_vs_time_degeneracy`, 2 vs 3)** — the classic chirp-rate / arrival-time degeneracy; base ``\theta_0 = [1.0, 1.5, 200.0, 0.0, 0.8, 0.8]``.
2.  **Spin 1 vs. Spin 2 (`spin1_vs_spin2_coupling`, 5 vs 6)** — the waveform sees only ``\chi_{\mathrm{eff}} = \frac{1}{2}(\chi_1 + \chi_2)``, so the anti-symmetric direction is exactly flat: the mathematical zone diverges there and the published zone is limited by the spin prior ``[-1, 1]`` (a wedge, not an ellipse).
3.  **Mass vs. Spin 1 (`mass_vs_spin1_twist`, 2 vs 5)** — mass/spin phasing trade-off; the zone crosses the ``\chi_1 \le 1`` bound and is capped there.
4.  **Time vs. Phase (`time_vs_phase_doppler`, 3 vs 4)** — absolute time translation vs orbital phase; the pure-phase direction is quasi-degenerate (``\partial^2_\varphi h \parallel`` tangent space), producing a near-singular boundary spike that the adaptive refinement resolves.
5.  **Mass vs. Phase, low-mass system (`mass_vs_phase_low_mass`, 2 vs 4)** — the chirp-rate/phase degeneracy at the low-mass base ``\theta_0 = [0.5, 0.1, 500.0, 0.0, 0.9, 0.0]``.
6.  **Time vs. Spin 1, high-spin system (`time_vs_spin1_high_spin`, 3 vs 5)** — the spin-orbit phase twist trades against arrival time at the strongly anti-aligned base ``\theta_0 = [1.0, 2.0, 150.0, 1.57, 0.7, -0.7]``.
7.  **Phase vs. Spin 1, seed binary (`phase_vs_spin1_seed`, 4 vs 5)** — a weakly chirping light-seed source leaves phase and spin loosely constrained: the zone is terminated by three distinct physical walls (phase topology ``\pm\pi`` and the spin prior), with ``\sim 38\%`` of directions prior-limited.
