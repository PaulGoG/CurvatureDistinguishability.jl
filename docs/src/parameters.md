# Run Parameters

This page gives the physical rationale of the production configuration
(`configs/production_cpu.toml`) and of the configuration layering. The
production configuration comprises the global grid, seven one-dimensional
separation sweeps and seven two-dimensional confusion maps.

## 0. Configuration files

Two base files ship for general use: `configs/quickstart.toml` (minutes-scale
demonstration, the pipeline default) and `configs/production_cpu.toml`
(the reference campaign documented below). The GPU variants,
`quickstart_gpu.toml` and `production_gpu.toml`, are thin overlays. Each
opens with `base_config = "<base>.toml"` (a path relative to the overlay)
followed by only the `[hardware]` keys that differ. At load the overlay is
deep-merged onto its base (sub-tables recurse; scalars, arrays and
`[[sweeps]]`/`[[maps]]` lists present in the overlay replace the base's;
one overlay level only). The merged table is validated, and the run
directory receives the merged, self-contained `config.toml`;
`metadata.toml` records `config_file` and `base_config`.
Both GPU overlays select the first functional backend and
`hessian_chunk = 2`. This is the portable baseline: the `[hardware]`
setting splits the ForwardDiff Hessian so each kernel launch carries
`(1+c)^2` lanes over `ceil(6/c)^2` launches, and nine lanes fit the
kernel-argument and per-launch time limits of every supported backend. The
full 49-lane kernel needs 2352 bytes of kernel arguments against the
2048-byte limit of the Intel integrated GPU, and spills registers heavily on
discrete cards.

A third overlay, `quickstart_maps.toml`, replaces the quickstart work
items by nine maps on the quickstart grid as a minutes-scale visual check of
the mapping stage (wall segments, corners, null-direction needles). The
nine maps are the seven production planes at quickstart-scale base points
plus two planes whose base point sits one zone half-width from a spin wall.

The run identifier hashes only the identity of a run: every section except
`[hardware]`, `[safety]`, `[monitoring]` and `[supervision]`, which describe
how a run executes rather than what it computes. One physical case
therefore carries one identifier on every machine; reruns land in `_r2`,
`_r3`, … sibling directories, and `metadata.toml`/`hardware.txt` record
backend, host and timings. The identifier hashes the values as written: `2`
against `2.0`, or a key set to its default against the key left out, give
two identifiers for the same computation. An interrupted run is continued
(`--resume`, or a supervised re-run) when the snapshot in its directory has
the same identity and the same `gpu_backend`, `hessian_chunk` and `require_gpu`; the other
execution keys may change between attempts. No per-machine configuration files are needed,
because execution keys left out of a configuration take host-adaptive
defaults: `[hardware].max_threads` the Julia thread count,
`[safety].max_ram_gb` 80 % of system memory, and `[safety].max_vram_gb` /
`os_vram_overhead_gb` the fixed defaults 8 GB / 1 GB, which are not queried
from the device. The production base sets none of them, so it runs
unmodified on any host. `[hardware].require_gpu = true` (default `false`)
stops a run before any result is written when no functional GPU backend
resolves for the requested `gpu_backend` (under `scripts/run_supervised.jl`
the run directory then holds only the supervisor's attempt log and ledger). Every shipped GPU overlay sets
it, so a missing vendor package can never turn a GPU campaign into a silent
CPU run. `configs/campaigns/` holds the multi-host campaign plans, which
vary the Hessian chunk and, for the single-sweep overlays, narrow
`[[sweeps]]` to one case.

`[pipeline].fit_coordinates` selects the coordinates of the fit:
`"parameters"` (default) or `"whitened"`, which requires
`optimizer = "ipnewton"`. The whitened fit is the one to use at phase orders
above 1.5 ([System Architecture](architecture.md)); a configuration that
sweeps at a higher order with the fit in the parameters loads with a
warning.

### Supervised runs

`scripts/run_supervised.jl` runs one or more configurations, each in a
worker process (`scripts/run_pipeline.jl`) watched by a supervisor that
never loads a GPU package. The worker publishes a heartbeat (one tick per
loss evaluation, so the interval between ticks is bounded by one kernel
launch) and checkpoints every finished work item; the supervisor samples
the heartbeat and the worker's CPU time from `/proc/<pid>/stat`. No
progress for `idle_stall_s` with a CPU utilization below `cpu_idle_cores`
is a hang: the worker is terminated (`SIGTERM`, then `SIGKILL` after
`term_grace_s`) and relaunched into the same run directory, where finished
stages and work items are skipped. A worker that is slow but busy is left
alone until `busy_stall_s`.

The retry budget has two parts. `max_retries` bounds the relaunches of a run.
`max_stalled_retries` bounds consecutive attempts that finish no new work
item. A deterministic failure therefore ends after a few attempts: the stage
is recorded as abandoned and the remaining stages run. Failures that arrive
after some progress only draw on `max_retries`. For a failure process
with an expected number μ of events per run, the smallest budget R with
P(N > R) < ε follows from the Poisson tail; μ = 1 gives P(N > 5) ≈ 6×10⁻⁴
and μ = 10 gives P(N > 20) ≈ 1.6×10⁻³.

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
and its console output in `logs/attempt_<k>.console.log`. The worker and the
supervisor each hold a lock in the run directory (`run.pid`,
`supervisor.pid`); a second process on the same directory stops with an
error, and the lock of a process of this host that no longer exists is
reclaimed. The lock of a process of another host (a job moved to another
node) is never reclaimed; the error names the file to remove once that
process has ended.

### Keys the shipped configurations leave at their defaults

These keys take the defaults below in every shipped configuration; the bounds
are those the parser enforces.

| Key | Default | Bounds | Meaning |
|---|---|---|---|
| `[pipeline.sweep_settings].floor_detection_ratio` | 2.0 | > 1 | ratio of optimized to predicted ``D^2`` at or above which the contiguous run of smallest separations counts as optimizer floor |
| `[pipeline.sweep_settings].secondary_minimum_gain` | 1.5 | > 1 | multi-start gain over the canonical start above which a sweep warns of secondary minima (`n_starts` > 1) |
| `[pipeline.sweep_settings].multi_start_parallel_scale` | 0.35 | > 0 | standard deviation of the extra-start perturbation along the sweep direction, in units of ``\delta`` |
| `[pipeline.sweep_settings].multi_start_transverse_scale` | ``10^{-3}`` | ≥ 0 | standard deviation of the isotropic extra-start perturbation, in scaled parameter units |
| `[pipeline.sweep_settings].correction_validity_fraction` | 0.1 | > 0 | relative departure from the leading-order law at which the validity radius ``\delta_{\mathrm{valid}}`` of the ``\mathcal{O}(\delta^5)`` fit is set: the smallest ``\delta`` with ``\lvert c_1\delta + c_2\delta^2\rvert`` equal to it, ``c_1`` counted only when it exceeds twice its standard error |
| `[pipeline.sweep_settings].residual_spectrum_windows` | 600 | integer ≥ 8 | number of log-uniform decimation windows of the residual spectrum |
| `[physics].spa_time` | `"newtonian"` | `"newtonian"`, `"phase"` | stationary-phase times of the response: Newtonian, or from the phase of the order `phase_pn_order`, which can pass `t_c` below the innermost stable orbit ([Waveform Physics](physics.md), §2 and §8) |
| `[mapping].unbounded_cap_factor` | 5.0 | > 1 | radius of unbounded map directions, in multiples of the largest finite capped radius |
| `[hardware].heap_size_hint_gb` | 0.0 | ≥ 0 GB | `--heap-size-hint` applied by the supervisor to its worker processes only (0 = none) |
| `[hardware].gc_between_stages` | `true` | none | device-buffer flush and full host garbage collection after every sweep and map |
| `[monitoring].progress_log_fraction` | 0.25 | [0, 1] | fraction of a stage's work items between `run.log` progress lines (0 disables them) |

## 1. Global simulation grid

- Observation time ``T_{\mathrm{obs}}``: ``3.15576 \times 10^7`` s (1 Julian
  year, the same constant that drives the orbital motion). It sets the
  frequency resolution and the confusion-noise level. It limits the signal
  in time only with `[physics].observation_window = true`, which weights
  every harmonic by the window of an observation over
  ``[0, T_{\mathrm{obs}}]`` at its emission time
  ([Waveform Physics](physics.md), Finite observation). The window edge is
  `window_edge_time` [s] (default ``10^6``, at most
  ``T_{\mathrm{obs}}/8``). The key belongs to the run identity. The
  production configuration sets it; the quickstart configurations, whose
  coarse grids use ``T_{\mathrm{obs}}`` as a resolution only, leave it out.
  `configs/campaigns/` holds the multi-host benchmark campaign on its own
  bases without the window (`benchmark_base.toml`, and
  `gpu_stress_benchmark.toml`, a two-year single-sweep timing benchmark);
  `windowed_low_mass_maps.toml` there recomputes with the window the two
  maps whose base points radiate in band before ``t = 0``.
- Frequency range: ``10^{-4}`` Hz to ``0.05`` Hz.
- Resolution: at a spacing of ``1/T_{\mathrm{obs}}`` the grid spans
  ``\approx 1.57`` million frequency bins. The `[safety].max_ram_gb` budget
  is checked against this size before anything is allocated.
- Separations: 30 log-spaced values of ``\delta`` over
  ``[10^{-3}, 10^{2.5}]`` in Fisher-metric units (``u`` is normalized to
  ``g(u,u)=1``, so ``\delta = 1`` is one Fisher σ along the direction). The
  absolute grid runs from below the optimizer floor of the fits (the
  struck-through points inside the grey band of the scaling figure) through
  the leading-order law, the threshold crossing at ``\delta_{\mathrm{min}}``
  (tens to hundreds of σ for these loud sources) and the onset of the
  ``\mathcal{O}(\delta^5)`` departure. The floor is the double-precision
  resolution of the fitted parameters: for the quickstart source (SNR
  ``2\times 10^4``) it sits at ``D^2 \approx 3\times 10^{-13}``, reached at
  ``\delta \approx 0.08`` σ, and it moves to smaller ``\delta`` for quieter
  sources. A sweep whose grid does not bracket ``\delta_{\mathrm{min}}``
  logs a warning; the residual spectrum is evaluated at the clean
  separation nearest ``2\,\delta_{\mathrm{min}}`` inside the validity radius
  of the ``\mathcal{O}(\delta^5)`` fit.
- Scenario constants: ``\eta = 0.2222`` (mass ratio 2:1), source at
  ecliptic longitude 180° and latitude 30°, inclination 30°, polarization
  45°, constellation phases zero at ``t = 0``. Masses are detector-frame
  masses, as in the LDC catalogues.
- Phase order: `[physics].phase_pn_order`, 1.5 (the default and the model
  of the production campaign) or any of 2, 2.5, 3, 3.5, 4, 4.5
  ([Waveform Physics](physics.md), §2). The key belongs to the run identity.
  From order 2 on, sweeps need `[pipeline].fit_coordinates = "whitened"`: the
  spin combination that the higher orders constrain is constrained weakly, and
  the fit in the parameters stalls on it. `configs/pn_orders/` holds overlays
  at every order: `quickstart_pn<order>.toml` (the quickstart sweeps and maps,
  whitened fit) and `production_maps_pn<order>.toml` (four maps on the
  production grid with `n_angles = 1024`, no sweeps: the spin plane and the
  chirp-mass–spin plane of the massive base point, and the spin plane at the
  low-mass and seed base points of the sweeps).
- Domain of validity: the inspiral is windowed out at the innermost stable
  circular orbit, at ``2.2`` mHz for the ``\mathcal{M} = 4`` s base points,
  ``35`` mHz for ``\mathcal{M} = 0.25`` s and above the band for
  ``\mathcal{M} = 0.1`` s. The long-wavelength response is used up to
  ``2.6 f_\star``; see [Waveform Physics](physics.md), §8. The
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

## 2. One-dimensional separation sweeps

The pipeline validates the geometric distance law ``D^2 \propto \delta^4``
across seven physical scenarios.

### `six_dimensional_diagonal_stress`
Base point `[8.0, 1.6, 15.0, 1.57, 0.6, -0.4]`,
direction `[0.0, 0.707, -0.5, 0.3, 0.4, -0.1]`.

A distant (8 Gpc) ``8\times 10^{5}\,M_\odot`` binary whose companion
simultaneously becomes heavier, merges earlier, changes phase, and shifts
both spins; no coordinate axis is special. This tests the tensorial
character of the law: the residual unabsorbable power is governed by the
extrinsic curvature regardless of how the displacement mixes the
coordinates.

### `massive_binary_mass_time`
Base point `[5.0, 4.0, 20.0, 0.0, 0.8, 0.8]`,
direction `[0.0, 1.0, 0.5, 0.0, 0.0, 0.0]`.

The reference massive binary (``2\times 10^{6}\,M_\odot``, 5 Gpc, strongly
aligned spins) along the chirp-rate/arrival-time direction that dominates
realistic overlapping-source populations. With ISCO at 2.2 mHz the signal
occupies the low-frequency band where the galactic confusion noise is
strongest.

### `unequal_amplitude_mass_time`
Base point and direction as `massive_binary_mass_time`, amplitude ratio
``q = 1/2`` (`amp_ratio`).

The second source carries half the amplitude (twice the luminosity
distance at the same masses). The prediction acquires the harmonic-mean
prefactor, ``D^2 = \tfrac{1}{16}K(u)\delta^4\,(2q/(1+q))^2``, and
the optimizer is initialized at the weighted midpoint with effective
amplitude ``(1+q)A``. With ``q=1/2``, using the equal-amplitude law instead
would misplace the ratio panel by a factor ``9/4``.

### `extreme_spin_orbit_coupling`
Base point `[5.0, 2.5, 10.0, 1.57, 0.7, -0.7]`,
direction `[0.0, 0.0, 0.0, 1.0, 0.5, 0.5]`.

Spins of opposite sign (``\chi_s = 0``, ``\chi_a = 0.7``) leave only the
mass-asymmetric part of the 1.5PN spin–orbit coefficient,
``\beta = 113\,\delta_m\chi_a/12``; the direction sweeps phase and both
spins (base spins chosen so the second source stays inside
``|\chi|\le 1`` at ``\delta_{\max}``).

### `low_mass_time_shift`
Base point `[3.0, 0.25, 25.0, 0.0, 0.9, 0.0]`,
direction `[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]`.

A ``1.3\times 10^{5}\,M_\odot`` system whose inspiral spans the band up to
its 35 mHz ISCO, displaced purely in coalescence time. The sweep isolates
the constellation mechanics: a time shift places the detector at a
different point of its orbit, with different antenna patterns and Doppler
phase, at every contributing frequency.

### `seed_binary_mass_spin_compensation`
Base point `[2.0, 0.1, 20.0, 0.0, 0.7, 0.5]`,
direction `[0.0, 0.6, 0.0, 0.0, -0.55, -0.55]`.

A light-seed binary (``5\times 10^{4}\,M_\odot``, ISCO above the band)
displaced in chirp mass and in both aligned spins at once: the chirp mass
increases while the spins decrease. The chirp-mass and spin–orbit
contributions to the phase have the same sign across the band and add; the
first-order variation is absorbed by the template as along every other
direction. The normalized curvature is the largest of the seven sweeps and
``\delta_{\mathrm{min}}`` the smallest.

### `near_degenerate_flat_spin`
Base point `[5.0, 4.0, 20.0, 0.0, 0.3, -0.3]`,
direction `[0.0, 0.15, 0.0, 0.0, 0.4, -0.92]`.

The direction is dominantly the exactly flat spin combination of the 1.5PN
phase, ``\mathrm{d}\beta = 0`` (``\mathrm{d}\chi_2/\mathrm{d}\chi_1 = -2.29``
at ``\eta = 2/9``; it reduces to ``\chi_a`` at equal mass), with a 15%
chirp-mass admixture so the direction is measurable and the second source
stays physical at ``\delta_{\max}``. Among the shipped sweeps it tests the
law closest to a null direction.

## 3. Two-dimensional zone-of-confusion maps

These configurations evaluate the extrinsic curvature over a
two-dimensional plane to draw the ``\delta_{\mathrm{min}}`` boundary. The
angular resolution comes from the global `[mapping].n_angles` (per-map
override allowed). Directions are sampled uniformly in the Fisher-normalized
plane, each axis measured in its own ``\sigma`` at the base point, so that a
plane like ``t_c`` against ``\phi_c``, whose axes differ by
``10^{4}``–``10^{5}`` in ``\sigma``, is resolved uniformly rather than only
along its axes. The solver computes half the directions and mirrors (``K``
is exactly even), refines adaptively near boundary spikes, and caps every
direction at the physical prior box. The CSV carries `Prior_Limited` flags.
The prior-limited fraction annotated in the figure is a fraction of sampling
angle; the percentages in the logs are fractions of the sampled directions.
Wherever consecutive directions straddle a capping transition, the solver
also bisects for the crossover direction
``r_{\mathrm{math}}(\varphi) = r_{\mathrm{box}}(\varphi)`` and inserts it as
a boundary vertex (`[mapping].corner_bisect_iters`, default 25; 0
disables). Zone corners at prior walls therefore lie on the walls instead of
being chamfered by a polygon chord. The neighbor-ratio refinement alone
cannot resolve them, because the capped radius saturates at
``r_{\mathrm{box}}`` on the wall side. `replot.jl --rho` reuses the stored
directions, so corner vertices lie on the walls only for the original
threshold; at a very different ``\rho`` the map stage has to be rerun to
place them.

1. Mass against time (`mass_vs_time_degeneracy`, axes 2 and 3): the
   chirp-rate/arrival-time degeneracy; base
   ``\theta_0 = [5.0, 4.0, 20.0, 0.0, 0.8, 0.8]``.
2. Spin 1 against spin 2 (`spin1_vs_spin2_coupling`, axes 5 and 6): at
   1.5PN order the phase sees the spins only through ``\beta``, so the line
   ``\mathrm{d}\beta = 0`` is exactly flat. The mathematical zone diverges
   along it, and the published zone is limited there by the spin prior
   ``[-1, 1]`` (a narrow wedge, since the loud source confines every
   measurable direction to a tiny radius).
3. Mass against spin 1 (`mass_vs_spin1_twist`, axes 2 and 5): mass/spin
   phasing trade-off; the zone stays inside the ``\chi_1 \le 1`` bound.
4. Time against phase (`time_vs_phase_doppler`, axes 3 and 4): absolute
   time translation against coalescence phase. The pure-phase direction is
   quasi-degenerate (``\partial^2_{\phi_c} h \parallel`` tangent space),
   producing a near-singular boundary spike that the adaptive refinement
   resolves.
5. Mass against phase, low-mass system (`mass_vs_phase_low_mass`, axes 2
   and 4): the chirp-rate/phase degeneracy at the low-mass base
   ``\theta_0 = [3.0, 0.25, 25.0, 0.0, 0.9, 0.0]``.
6. Time against spin 1, high-spin system (`time_vs_spin1_high_spin`, axes 3
   and 5): the spin–orbit phase term trades against arrival time at the
   opposite-spin base
   ``\theta_0 = [5.0, 2.5, 10.0, 1.57, 0.7, -0.7]``.
7. Phase against spin 1, seed binary (`phase_vs_spin1_seed`, axes 4 and 5):
   a weakly chirping light-seed source; the zone stays inside the phase cap
   ``\pm\pi`` and the spin prior in every direction.
