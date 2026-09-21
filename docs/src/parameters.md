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
`quickstart_gpu.toml` and `production_gpu.toml` — are thin overlays: each
opens with `base_config = "<base>.toml"` (a path
relative to the overlay) followed by only the `[hardware]` keys that
differ. At load the overlay is deep-merged onto its base (sub-tables
recurse; scalars, arrays and `[[sweeps]]`/`[[maps]]` lists present in the
overlay replace the base's; one overlay level only), the merged table is
validated, and the run directory receives the merged, self-contained
`config.toml`; `metadata.toml` records `config_file` and `base_config`.
Both GPU overlays select the first functional backend and
`hessian_chunk = 2`. That is the portable baseline: the `[hardware]`
setting splits the ForwardDiff Hessian so each kernel launch carries
`(1+c)^2` lanes over `ceil(6/c)^2` launches, and nine lanes fit the
kernel-argument and per-launch time limits of every supported backend — the
full 49-lane kernel needs 2352 bytes of kernel arguments against the
2048-byte limit of the Intel integrated GPU, and spills registers heavily on
discrete cards.

A third overlay, `quickstart_maps.toml`, replaces the quickstart work
items by nine maps on the quickstart grid — the seven production planes
at quickstart-scale base points plus two planes whose base point sits one
zone half-width from a spin wall — as a minutes-scale visual check of the
mapping stage (wall segments, corners, null-direction needles).

The run identifier hashes only the *identity* of a run — every section
except `[hardware]`, `[safety]`, `[monitoring]` and `[supervision]`, which describe how a
run executes rather than what it computes — so one physical case carries
one identifier on every machine; reruns land in `_r2`, `_r3`, … sibling
directories and `metadata.toml`/`hardware.txt` record backend, host and
timings. No per-machine configuration files are needed: `max_threads` and
`[safety].max_ram_gb` are taken from the host when absent (Julia's thread
count and 80 % of system memory); the device budgets `max_vram_gb` /
`os_vram_overhead_gb` default to the fixed values 8 GB / 1 GB and are not
queried from the device.
`configs/campaigns/` holds the multi-host campaign plans, which vary the
Hessian chunk and, for the single-sweep overlays, narrow `[[sweeps]]` to
one case.

Execution keys left out of a configuration take host-adaptive defaults:
`[hardware].max_threads` the Julia thread count, `[safety].max_ram_gb`
80 % of system memory, `[safety].max_vram_gb` / `os_vram_overhead_gb` the
fixed defaults 8 GB / 1 GB. The production base
sets none of them, so it runs unmodified on any host. `[hardware]
.require_gpu = true` (default `false`) aborts a run before any output is
written when no functional GPU backend resolves for the requested
`gpu_backend`; every shipped GPU overlay sets it, so a missing vendor
package can never turn a GPU campaign into a silent CPU run.

### Supervised runs

`scripts/run_supervised.jl` runs one or more configurations under a second
Julia process that never loads a GPU package. The worker publishes a
heartbeat (one tick per loss evaluation, so the interval between ticks is
bounded by one kernel launch) and checkpoints every finished work item; the
supervisor samples the heartbeat and the worker's CPU time from
`/proc/<pid>/stat`. No progress for `idle_stall_s` with a CPU utilisation
below `cpu_idle_cores` is a hang: the worker is terminated (`SIGTERM`, then
`SIGKILL` after `term_grace_s`) and relaunched into the same run directory,
where finished stages and work items are skipped. A worker that is slow but
busy is left alone until `busy_stall_s`.

The retry budget has two parts. `max_retries` bounds the relaunches of a run.
`max_stalled_retries` bounds consecutive attempts that finish no new work
item: a deterministic failure therefore ends after a few attempts — the stage
is recorded as abandoned and the remaining stages run — while failures that
arrive after some progress only draw on `max_retries`. For a failure process
with an expected number μ of events per run, the smallest budget R with
P(N > R) < ε follows from the Poisson tail; μ ≈ 1 needs R = 5 for ε ≈ 4×10⁻⁴,
μ ≈ 10 needs R = 20 for ε ≈ 10⁻³.

| Key | Default | Bounds |
|---|---|---|
| `max_retries` | 5 | integer ≥ 0 |
| `max_stalled_retries` | 2 | integer ≥ 1 |
| `poll_interval_s` | 15 | 0.05–600 s |
| `startup_grace_s` | 1800 | > 0 s |
| `idle_stall_s` | 300 | ≥ 4 `poll_interval_s` |
| `busy_stall_s` | 7200 | ≥ `idle_stall_s` |
| `cpu_idle_cores` | 0.05 | (0, 1] cores |
| `term_grace_s` | 60 | > 0 s |
| `kill_wait_s` | 120 | > 0 s |
| `backoff_s` | 30 | ≥ 0 s |

Every attempt is recorded in `supervision.toml` in the run directory (verdict,
exit status or signal, CPU seconds, stall time, work items before and after),
and its console output in `logs/attempt_<k>.console.log`.

## 1. Global Simulation Grid

*   **Observation Time (``T_{\mathrm{obs}}``):** ``3.15576 \times 10^7`` s
    (1 Julian year — the same constant that drives the orbital motion). It
    sets the frequency resolution and the confusion-noise level; it limits the
    signal in time only with `[physics].observation_window = true`
    (`window_edge_time` [s], default ``10^6``, at most ``T_{\mathrm{obs}}/8``),
    which weights every harmonic by the window of an observation over
    ``[0, T_{\mathrm{obs}}]`` at its emission time
    ([Waveform Physics](physics.md), Finite observation). The key belongs to
    the run identity. The production configuration sets it; the quickstart
    configurations, whose coarse grids use ``T_{\mathrm{obs}}`` as a resolution
    only, leave it out. `configs/campaigns/` is the multi-host benchmark
    campaign as it was run, on its own frozen base without the window
    (`benchmark_base.toml`); `windowed_low_mass_maps.toml` there recomputes
    with the window the two maps whose base points radiate in band before
    ``t = 0``.
*   **Frequency Range:** ``10^{-4}`` Hz to ``0.05`` Hz.
*   **Resolution:** at a spacing of ``1/T_{\mathrm{obs}}`` the grid spans
    ``\approx 1.57`` million frequency bins. The `[safety].max_ram_gb` budget
    is checked against this size before anything is allocated.
*   **Separations:** 30 log-spaced values of ``\delta`` over
    ``[10^{-3}, 10^{2.5}]`` in Fisher-metric units (``u`` is normalized to
    ``g(u,u)=1``, so ``\delta = 1`` is one Fisher σ along the direction). The
    absolute grid runs from below the optimizer floor of the fits — the
    struck-through points inside the grey band of the scaling figure — through
    the leading-order law, the threshold crossing at ``\delta_{\mathrm{min}}``
    (tens to hundreds of σ for these loud sources) and the onset of the
    ``\mathcal{O}(\delta^5)`` departure. The floor is the double-precision
    resolution of the fitted parameters: for the quickstart source (SNR
    ``2\times 10^4``) it sits at ``D^2 \approx 3\times 10^{-13}``, reached at
    ``\delta \approx 0.08`` σ, and it moves to smaller ``\delta`` for quieter
    sources. A sweep whose grid does not bracket ``\delta_{\mathrm{min}}``
    logs a warning; the residual spectrum is evaluated at the clean
    separation nearest ``2\,\delta_{\mathrm{min}}``.
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
