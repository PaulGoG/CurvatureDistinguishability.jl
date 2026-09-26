# Physics and Waveform Model

Equation-level description of the model implemented in `Physics.jl` and
`Detector.jl`: a frequency-domain aligned-spin inspiral of a massive
black-hole binary, complete to 1.5PN order in the phase and 0.5PN order in
the amplitude, observed through the long-wavelength response of the LISA
constellation on its analytic orbits. The Fourier-transform sign, the
ecliptic sky frame, the polarization basis and the A, E, T combinations
follow the LISA Rosetta Stone [LISARosettaStone2025](@cite); the masses are
detector-frame and the time and phase parameters are the TaylorF2
coalescence values ``t_c``, ``\varphi_c`` rather than the reference-frequency
parameters of that document. Arm length and single-link noise levels are those
of the mission definition [LISADefinitionStudy2023](@cite). §8 states the domain
of validity.

## 1. Parameters

The waveform depends on ``\vec{\theta} = (D_L, \mathcal{M}, t_c, \phi_c, \chi_1, \chi_2)``:
the luminosity distance, the (detector-frame) chirp mass in geometrised
seconds (``1\,\mathrm{s} \approx 2.03\times 10^{5}\,M_\odot``), the
coalescence time, the coalescence phase and the two dimensionless aligned
spins ``\chi_i \in [-1, 1]``, ``\chi_1`` on the heavier body. Internally
every component is ``\mathcal{O}(1)``, the physical values being restored
through the `[physics]` scales (`distance_scale`, `mass_scale`,
`time_scale`; 1 Gpc ``= 1.0293\times 10^{17}`` light-seconds), so the
Fisher matrix stays well conditioned. The amplitude of every harmonic is
the physical one, ``\propto \mathcal{M}^{5/6}/D_L`` at leading order, so
the tangent space carries the physical ``\partial/\partial\mathcal{M}``
coupling between amplitude and phasing.

The symmetric mass ratio ``\eta \in (0, 1/4]`` is a fixed property of the
scenario (`[physics].eta`), not a fitted parameter: the total mass follows
as ``M = \mathcal{M}\eta^{-3/5}`` and the mass asymmetry as
``\delta_m = (m_1 - m_2)/M = \sqrt{1 - 4\eta}``. The production campaign uses
``\eta = 2/9`` (mass ratio 2:1). Sky position (ecliptic longitude ``\lambda`` and latitude
``\beta``), inclination ``\iota`` and polarization angle ``\psi`` are
likewise scenario constants.

## 2. Intrinsic phasing (TaylorF2 to 1.5PN)

In the stationary-phase approximation the harmonic ``k`` of the orbital
phase radiates at ``f`` when the dominant ``(2,2)`` harmonic radiates at
``F = 2f/k``. With the total-mass velocity ``v = (\pi M F)^{1/3}`` the
post-Newtonian phase of the ``(2,2)`` harmonic is
```math
\psi_{\mathrm{PN}}(F) = \frac{3}{128\,\eta\, v^{5}}
\left[1 + \left(\frac{3715}{756} + \frac{55}{9}\eta\right) v^{2}
+ \left(4\beta - 16\pi\right) v^{3}\right],
```
the Newtonian, 1PN and 1.5PN (tail and spin–orbit) terms of TaylorF2, with
the Poisson–Will spin–orbit coefficient [Poisson1995](@cite)
```math
\beta = \frac{1}{12}\left[(113 - 76\eta)\,\chi_s + 113\,\delta_m\,\chi_a\right],
\qquad \chi_s = \tfrac{1}{2}(\chi_1 + \chi_2),\quad \chi_a = \tfrac{1}{2}(\chi_1 - \chi_2).
```
The spins enter the 1.5PN phase only through the single scalar
``\beta``, so at this order exactly one spin combination is a flat
direction of the signal manifold at every mass ratio (tangent rank 5):
at equal mass it is the antisymmetric combination ``\chi_a``, in general
it is the null line ``(113 - 76\eta + 113\delta_m)\,\mathrm{d}\chi_1 +
(113 - 76\eta - 113\delta_m)\,\mathrm{d}\chi_2 = 0`` — at ``\eta = 2/9``
``\mathrm{d}\chi_2/\mathrm{d}\chi_1 = -2.29``, the ``\chi_s`` and ``\chi_a``
coefficients being ``8.01`` and ``3.14``. Lifting it requires the 2PN
spin–spin term ([Roadmap](roadmap.md)).

The phase of harmonic ``k`` in the ``e^{-2\pi i f t}`` Fourier convention is
```math
\Psi_k(f) = 2\pi f t_c - \frac{k}{2}\phi_c + \frac{k}{2}\,\psi_{\mathrm{PN}}(2f/k) - \frac{\pi}{4},
```
so that ``\tilde h_k(f) \propto e^{-i\Psi_k(f)}``; the arrival-time term is
common to every harmonic, and the stationary-phase time of harmonic ``k``
is the Newtonian ``t_k = t(2f/k)`` with
``t(F) = t_c - 5\mathcal{M}/(256\, u^{8})``, ``u = (\pi\mathcal{M}F)^{1/3}``,
which is where the detector motion of §5 is evaluated.

## 3. Harmonic amplitudes (0.5PN)

The polarizations are the Blanchet–Iyer–Will–Wiseman amplitudes
[Blanchet1996](@cite) to 0.5PN order,
```math
h_{+,\times} = \frac{2 M \eta\, x}{D_L}\left[H^{(0)}_{+,\times} + x^{1/2} H^{(1/2)}_{+,\times}\right],
\qquad x = (\pi M F)^{2/3},
```
```math
H^{(0)}_+ = -(1 + c^2)\cos 2\varphi, \quad H^{(0)}_\times = -2c\sin 2\varphi,
```
```math
H^{(1/2)}_+ = -\frac{s}{8}\,\delta_m\left[(5 + c^2)\cos\varphi - 9(1 + c^2)\cos 3\varphi\right], \quad
H^{(1/2)}_\times = -\frac{3}{4}\, s\, c\, \delta_m\left[\sin\varphi - 3\sin 3\varphi\right],
```
with ``c = \cos\iota``, ``s = \sin\iota`` and ``\varphi`` the orbital
phase. The stationary-phase amplitude of harmonic ``k`` is
```math
a_k(f) = \frac{M\eta\,x}{D_L}\sqrt{\frac{2}{k\,\dot F}}, \qquad
\dot F = \frac{96}{5}\pi^{8/3}\mathcal{M}^{5/3}F^{11/3},
```
evaluated at ``F = 2f/k``, multiplied by the ``\cos k\varphi`` and
``\sin k\varphi`` coefficients above. For ``k = 2`` this reproduces the
standard ``\sqrt{5/24}\,\pi^{-2/3}\mathcal{M}^{5/6} f^{-7/6}(1 + c^2)/(2D_L)``;
the ``k = 1, 3`` amplitudes are proportional to ``\delta_m`` and vanish at
equal mass, where they are skipped. The (3,3) harmonic is therefore
physical, not phenomenological: its strength is fixed by the mass ratio.

## 4. Innermost stable orbit

The inspiral description ends at the Schwarzschild innermost stable
circular orbit, ``f_{\mathrm{ISCO}} = (6^{3/2}\pi M)^{-1}``. Every
harmonic is multiplied by the smooth window
```math
w_k(f) = \tfrac{1}{2}\left[1 - \tanh\frac{F/f_{\mathrm{ISCO}} - 1}{\sigma}\right],
\qquad F = 2f/k,
```
with relative width ``\sigma`` (`[physics].cutoff_width`, 0.1 in the
shipped configurations): unity below ISCO, one half at ISCO, and
``< 10^{-4}`` beyond ``1.5\,f_{\mathrm{ISCO}}``. The window keeps every
quantity a smooth function of ``\vec\theta`` and of ``f``, which the
differential-geometry engine requires, while removing the unphysical
extrapolation of the PN phase past the plunge. Merger and ringdown are not
modelled ([Roadmap](roadmap.md)).

## 5. Constellation orbits and Doppler phase

The spacecraft follow the analytic equal-arm orbits of Rubbo, Cornish and
Poujade [Rubbo2004](@cite) (their Eq. 1) to second order in the eccentricity ``e = L/(2\sqrt{3}R)``,
``R = 1`` AU ``= 499.005`` light-seconds: with the orbital phase
``\alpha(t) = 2\pi t/\mathrm{yr} + \kappa`` (`[physics].orbit_phase`) and the
spacecraft phases ``\beta_n = 2\pi(n-1)/3 + \lambda_c``
(`[physics].constellation_phase`; their spacecraft index ``n = 0, 1, 2`` is
``n = 1, 2, 3`` here, with the same orientation),
```math
\begin{aligned}
x_n &= R\cos\alpha + \tfrac{1}{2}eR\left[\cos(2\alpha - \beta_n) - 3\cos\beta_n\right]
 + \tfrac{1}{8}e^2R\left[3\cos(3\alpha - 2\beta_n) - 10\cos\alpha - 5\cos(\alpha - 2\beta_n)\right],\\
y_n &= R\sin\alpha + \tfrac{1}{2}eR\left[\sin(2\alpha - \beta_n) - 3\sin\beta_n\right]
 + \tfrac{1}{8}e^2R\left[3\sin(3\alpha - 2\beta_n) - 10\sin\alpha + 5\sin(\alpha - 2\beta_n)\right],\\
z_n &= -\sqrt{3}\,eR\cos(\alpha - \beta_n) + \sqrt{3}\,e^2R\left[\cos^2(\alpha - \beta_n) + 2\sin^2(\alpha - \beta_n)\right].
\end{aligned}
```
The constellation plane is inclined by 60° to the ecliptic and cartwheels
once per year in the retrograde sense; the arms stay equal to ``\mathcal{O}(e^2) \approx 0.3\,\%``.
A wavefront from the source direction ``\hat n = (\cos\beta\cos\lambda,
\cos\beta\sin\lambda, \sin\beta)`` reaches the constellation centre
``\vec R_c = R(\cos\alpha, \sin\alpha, 0)`` ``\hat n\cdot\vec R_c``
light-seconds before the solar-system barycentre, so the barycentric strain
acquires the Doppler phase
```math
\Delta_{\mathrm D}(f, t) = 2\pi f\,\hat n\cdot\vec R_c(t)
```
(a ``+`` sign in the ``e^{-2\pi i f t}`` convention), evaluated at the
stationary-phase time ``t_k`` of each harmonic.

## 6. Antenna patterns and the A, E channels

The wave-frame basis follows the Rosetta Stone: with
``\hat e_\lambda = (-\sin\lambda, \cos\lambda, 0)`` and
``\hat e_\beta = (-\sin\beta\cos\lambda, -\sin\beta\sin\lambda, \cos\beta)``,
``\hat u = -\hat e_\lambda``, ``\hat v = \hat e_\beta``, and the polarization
angle rotates them into ``\hat p = \hat u\cos\psi + \hat v\sin\psi``,
``\hat q = -\hat u\sin\psi + \hat v\cos\psi``; the polarization tensors are
``e^+ = \hat p\otimes\hat p - \hat q\otimes\hat q`` and
``e^\times = \hat p\otimes\hat q + \hat q\otimes\hat p``. In the
long-wavelength limit each spacecraft ``i`` synthesises a Michelson
interferometer with detector tensor
``D_i = \tfrac{1}{2}(\hat a_i\otimes\hat a_i - \hat b_i\otimes\hat b_i)``
from its two arm unit vectors, giving ``F^{+}_i = D_i : e^+`` and
``F^{\times}_i = D_i : e^\times`` (a 60° Michelson, hence the factor
``\sqrt{3}/2`` relative to a 90° detector is built in). The three
Michelson responses ``X, Y, Z`` are combined into the channels
``A = (Z - X)/\sqrt{2}``, ``E = (X - 2Y + Z)/\sqrt{6}`` of
[Prince2002](@cite) (their Eq. 18). For equal arms and equal, mutually
uncorrelated link noises the noise-correlation matrix of ``X, Y, Z`` has
equal diagonal entries ``S_X`` and equal off-diagonal entries ``S_{XY}``, and
``A``, ``E`` and ``T = (X + Y + Z)/\sqrt{3}`` diagonalise it with
``S_A = S_E = S_X - S_{XY}`` and ``S_T = S_X + 2S_{XY}`` (their Eq. 16). In
the long-wavelength limit every link and every test-mass displacement enters
two of the three Michelson responses with opposite signs, so ``X + Y + Z``
vanishes to leading order in ``f/f_\star`` for noise and signal alike:
``S_{XY} = -S_X/2`` and ``S_A = S_E = \tfrac{3}{2}S_X``. The channels are
therefore rescaled by ``\sqrt{2/3}``, so that each carries the
single-Michelson noise spectral density of §7:
```math
F_A = \frac{F_Z - F_X}{\sqrt{3}}, \qquad F_E = \frac{F_X - 2F_Y + F_Z}{3}.
```
Averaged over sky and polarization each channel has
``\langle F_+^2\rangle = \langle F_\times^2\rangle = 3/20`` (the rescaling gives
``\langle F_A^2\rangle = \langle F_X^2\rangle``, because
``\langle F_X F_Z\rangle = -\langle F_X^2\rangle/2``), hence
``\langle F_+^2 + F_\times^2\rangle = 3/10`` per channel. The response
``R = 3/10`` of [Robson2019](@cite) (their Eqs. 8–9) is the
single-polarization average summed over the two low-frequency channels,
``2 \times 3/20``: the same number for a different sum, and the sky-averaged
A + E signal-to-noise ratio of this model equals the one defined by their
sensitivity curve (their Eq. 13). The null combination ``X + Y + Z``
vanishes identically in this limit, so the T channel is zero and is
excluded by default (`[physics].include_t_channel`).

The observed strain of channel ``C \in \{A, E\}`` is
```math
\tilde h_C(f) = \sum_{k=1}^{3} w_k(f)\,\mathcal{T}(f)
\left[a^{+}_k(f) F^{+}_C(t_k) - i\, a^{\times}_k(f) F^{\times}_C(t_k)\right]
e^{-i\Psi_k(f)}\, e^{+i\Delta_{\mathrm D}(f, t_k)},
```
where ``\mathcal{T}(f) = [1 + 0.6 (f/f_\star)^2]^{-1/2}`` is the finite-arm
transfer roll-off of [Robson2019](@cite) (their Eq. 9) with
``f_\star = c/(2\pi L) = 19.09`` mHz for ``L = 2.5\times 10^{9}`` m; the arm
length is shared with the noise model and the orbit eccentricity through
`[noise].arm_length`. The ``-i`` between the polarizations is the
stationary-phase image of ``h_\times \propto \sin k\varphi`` against
``h_+ \propto \cos k\varphi``.

### Finite observation

The frequency-domain signal contains every frequency of the band, whenever
it was emitted: harmonic ``k`` at frequency ``f`` left the source at
``t_k(f) = t_c - \tau(2f/k)`` with
``\tau(F) = 5\mathcal{M}/[256\,(\pi\mathcal{M}F)^{8/3}]``, which for a light
system lies far before the observation. At the lower band edge
``\tau(10^{-4}\,\mathrm{Hz})`` is 0.13 yr for ``\mathcal{M} = 4`` s, 14 yr for
``\mathcal{M} = 0.25`` s and 63 yr for ``\mathcal{M} = 0.1`` s. With
`[physics].observation_window = true` each harmonic is weighted by the window
of an observation over ``[0, T_{\mathrm{obs}}]`` at its emission time,
```math
w_k(f) \;\to\; w_k(f)\, W\!\left(t_k(f)\right), \qquad
W(t) = \tfrac{1}{2}\left[\tanh\frac{t}{\Delta} - \tanh\frac{t - T_{\mathrm{obs}}}{\Delta}\right],
```
unity inside the observation, one half at either end and zero outside, with
edges of time scale ``\Delta`` = `window_edge_time`. In the stationary-phase
approximation a time-domain window that varies slowly against the local
chirp time ``1/\sqrt{\dot f}`` multiplies the harmonic at ``t_k(f)``;
``\Delta = 10^{6}`` s is two to three times ``1/\sqrt{\dot f}`` at ``t = 0``
for the lightest shipped systems (3.6 and 5.5 days). The window is smooth in
``t_k``, hence in ``\mathcal{M}`` and ``t_c``, through which its edges move in
frequency; a hard cut would not be differentiable. It removes the band below
``F_0`` with ``\tau(F_0) = t_c`` — 0.56 mHz for ``\mathcal{M} = 0.1`` s and
``t_c = 0.63`` yr — and leaves the signal above about ``2F_0`` unchanged to
``10^{-10}``. Without the key the window is absent and ``T_{\mathrm{obs}}``
only sets the frequency resolution and the confusion-noise level. The
distinction matters because the frequency grid is the analysis band of the
detector, not the span of the observation: the stationary-phase map
``t(f)`` assigns every grid frequency to an emission time, so a light
binary that fills the band has radiated at the low grid frequencies years
before ``t = 0``, and only the window restricts the signal to what a
``T_{\mathrm{obs}}``-long observation records.

## 7. Noise model

Inner products are weighted by the one-sided PSD ``S_n(f)`` of
[Robson2019](@cite): the Michelson-channel instrumental noise of their
Eq. 12,
```math
P_n(f) = \frac{P_{\mathrm{OMS}}}{L^2} + 2\left(1 + \cos^2(f/f_\star)\right)\frac{P_{\mathrm{acc}}}{(2\pi f)^4 L^2},
```
with ``P_{\mathrm{OMS}}`` (Eq. 10) the optical-metrology noise and
``P_{\mathrm{acc}}`` (Eq. 11) the test-mass acceleration noise, plus the
galactic confusion fit of Eq. 14,
```math
S_c(f) = A\, f^{-7/3}\, e^{-f^{\alpha} + \beta f \sin(\kappa f)} \left[ 1 + \tanh\!\big(\gamma (f_k - f)\big) \right], \qquad A = 9\times 10^{-45},
```
with ``(\alpha, \beta, \kappa, \gamma, f_k)`` from their Table 1, selected by
the observation time (1 yr: ``\alpha=0.171``, ``\beta=292``, ``\kappa=1020``,
``\gamma=1680``, ``f_k=2.15`` mHz) and overridable in `[noise]`. ``S_c`` depends on ``f`` and on
the observation time only: the confusion term is stationary, and the annual
modulation of the Galactic foreground by the constellation motion is not
represented.

The two terms are defined at different levels. ``P_n`` is the noise of one
channel and is used as such: the sky- and polarization-averaged response
``R(f)`` of their Eq. 9 is not applied to it, because the response is
modelled explicitly in §6, where its roll-off factor appears instead. The
confusion fit, by contrast, is calibrated on the two-channel sensitivity
curve, ``S_n = P_n/R + S_c`` (their Eq. 1; the amplitude ``A`` is quoted for
two channels), so it enters the per-channel PSD multiplied by the response:
```math
S_I(f) = P_n(f) + R(f)\, S_c(f), \qquad R(f) = \frac{3}{10}\,\frac{1}{1 + 0.6\,(f/f_\star)^2},
```
the same weight for ``I = A, E``
([`sky_averaged_response`](@ref CurvatureDistinguishability.Physics.sky_averaged_response)).
Dividing ``S_I`` by ``R`` recovers the published sensitivity curve (their
Eq. 13 + Eq. 14), and the sky- and polarization-averaged ``A + E``
signal-to-noise ratio of §6 equals the one computed from that curve. After
the conversion the Galactic term still exceeds ``P_n`` by an order of
magnitude at 1 mHz. The confusion fit is calibrated for observation times
between 0.5 and 4 yr; shorter grids use the nearest column.

## 8. Domain of validity

The model is an inspiral-only, aligned-spin, quasi-circular description in
the long-wavelength limit of the detector. Consequences:

- Above ISCO the signal is windowed out rather than continued into merger
  and ringdown. For the massive production base points
  (``\mathcal{M} = 4`` s, ``M \approx 2\times 10^{6}\,M_\odot``)
  ``f_{\mathrm{ISCO}} = 2.2`` mHz, so most of the band above a few mHz
  carries no signal and the merger SNR that a full inspiral–merger–ringdown
  approximant would add is absent. The lightest base point
  (``\mathcal{M} = 0.1`` s) has ``f_{\mathrm{ISCO}} = 88`` mHz, above the
  band edge.
- Without `observation_window` the signal fills the band from
  ``f_{\mathrm{min}}`` whatever its emission time. For base points with
  ``\tau(f_{\mathrm{min}}) > t_c`` (the two lightest of the production
  configuration) the Fisher norm and the zone along directions that involve
  the chirp mass or the spins then include decades of inspiral before
  ``t = 0``: along the mass–spin direction of the ``\mathcal{M} = 0.1`` s
  system the window lowers ``g(u,u)`` by a factor 9.6 and doubles
  ``\delta_{\mathrm{min}}``; coalescence-time and phase directions, and every
  heavier system, are unchanged.
- The long-wavelength response is used up to ``2.6 f_\star``; the finite-arm
  transfer function of the TDI observables is represented only by the
  Robson roll-off factor, not by the full arm-dependent transfer.
- The noise is the analytic Robson model, not an LDC noise realization; the
  pipeline validates the noiseless-baseline geometry ([Roadmap](roadmap.md)).
- Precession, eccentricity, higher PN orders in the phase (2PN spin–spin is
  the first planned extension) and higher harmonics beyond 0.5PN are not
  included.

The band and the mass scale are campaign choices
([Complex Run Parameters](parameters.md)); the geometric law itself makes
no assumption about them.
