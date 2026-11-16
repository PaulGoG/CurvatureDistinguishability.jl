# Waveform Physics

Equation-level description of the model implemented in `Physics.jl` and
`Detector.jl`: a frequency-domain aligned-spin inspiral of a massive
black-hole binary, to 1.5PN order in the phase by default (to 4.5PN on
request) and to 0.5PN order in the amplitude, observed through the long-wavelength
response of the LISA constellation on its analytic orbits. The
Fourier-transform sign, the
ecliptic sky frame, the polarization basis and the A, E, T combinations
follow the LISA science ground segment conventions
[Baghi2026SGSConventions](@cite); the masses are
detector-frame and the time and phase parameters are the TaylorF2
coalescence values ``t_c``, ``\phi_c`` rather than the reference-frequency
parameters of that document. Arm length and single-link noise levels are those
of the mission definition [LISADefinitionStudy2023](@cite). §8 states the domain
of validity.

## 1. Parameters

The waveform depends on ``\vec{\theta} = (D_L, \mathcal{M}, t_c, \phi_c, \chi_1, \chi_2)``:
the luminosity distance, the (detector-frame) chirp mass in geometrized
seconds (``1\,\mathrm{s} \approx 2.03\times 10^{5}\,M_\odot``), the
coalescence time, the coalescence phase and the two dimensionless aligned
spins ``\chi_i \in [-1, 1]``, ``\chi_1`` on the heavier body. Internally
every component is ``\mathcal{O}(1)``, the physical values being restored
through the `[physics]` scales (`distance_scale`, `mass_scale`,
`time_scale`; 1 Gpc ``= 1.0293\times 10^{17}`` light-seconds). The Fisher
matrix remains strongly anisotropic in these units, which is why the maps
sample directions in Fisher-normalized planes
([System Architecture](architecture.md)). The amplitude of every harmonic is
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

## 2. Intrinsic phasing (TaylorF2)

In the stationary-phase approximation the harmonic ``k`` of the orbital
phase radiates at ``f`` when the dominant ``(2,2)`` harmonic radiates at
``F = 2f/k``. With the total-mass velocity ``v = (\pi M F)^{1/3}`` the
post-Newtonian phase of the ``(2,2)`` harmonic is
```math
\psi_{\mathrm{PN}}(F) = \frac{3}{128\,\eta\, v^{5}}
\left[1 + \sum_{i=2}^{9}\left(\varphi_i + \varphi_{i,\ell}\ln v
+ \varphi_{i,\ell\ell}\ln^{2} v\right) v^{i}\right],
```
the TaylorF2 phase of two black holes with aligned spins, truncated at the
order `[physics].phase_pn_order`: order ``n`` keeps the terms with
``i \le 2n``. The coefficients are polynomials in ``\eta``, ``\delta_m``,
``\chi_s`` and ``\chi_a`` (`Physics.phase_coefficients`, which lists them with
their sources). ``\varphi_5`` alone is a constant phase, absorbed in
``\phi_c``, but its logarithm ``\varphi_{5,\ell} = 3\varphi_5`` is not; the
constant ``\varphi_8`` is a shift of ``t_c`` and is set to zero, and
``\varphi_{8,\ell\ell}`` is the first squared logarithm.

The default order 1.5 keeps the Newtonian, 1PN and 1.5PN terms,
```math
\varphi_2 = \frac{3715}{756} + \frac{55}{9}\eta, \qquad
\varphi_3 = 4\beta - 16\pi,
```
the tail ``-16\pi`` and the Poisson–Will spin–orbit coefficient [Poisson1995](@cite)
```math
\beta = \frac{1}{12}\left[(113 - 76\eta)\,\chi_s + 113\,\delta_m\,\chi_a\right],
\qquad \chi_s = \tfrac{1}{2}(\chi_1 + \chi_2),\quad \chi_a = \tfrac{1}{2}(\chi_1 - \chi_2).
```
Order 2 adds [Poisson1995, Arun2009](@cite)
```math
\varphi_4 = \frac{15293365}{508032} + \frac{27145}{504}\eta + \frac{3085}{72}\eta^2 - 10\,\sigma,
```
```math
\sigma = \left(\frac{81}{16} - \frac{\eta}{4}\right)\chi_s^2
+ \left(\frac{81}{16} - 20\eta\right)\chi_a^2 + \frac{81}{8}\,\delta_m\,\chi_s\chi_a ,
```
the spin–spin coefficient of two black holes with aligned spins
(``\varphi_4`` in this form is the coefficient of the same name in
[Khan2016](@cite), Appendix B). In terms of the bodies,
``\sigma = \tfrac{79}{8}\,\eta\,\chi_1\chi_2 + \tfrac{81}{16}\sum_i (m_i/M)^2\chi_i^2``:
the spin1–spin2 coupling and, for each body, the quadrupole–monopole and
self-spin terms, ``5 + 1/16``. The higher orders add:

| Order | Terms added | Sources |
|:------|:------------|:--------|
| 2.5 | non-spinning; next-to-leading-order spin–orbit; with ``\ln v`` | [Buonanno2009, Blanchet2024, Arun2009](@cite) |
| 3 | non-spinning, with ``\ln v``; spin–orbit tail; next-to-leading-order spin–spin | [Buonanno2009, Blanchet2024, Bohe2013SpinOrbitFlux, Bohe2015SpinSpin](@cite) |
| 3.5 | non-spinning; next-to-next-to-leading-order spin–orbit; spin–spin tail; leading cubic in spin | [Buonanno2009, Blanchet2024, Bohe2013SpinOrbitDynamics, Bohe2013SpinOrbitFlux, Nagar2019, Cho2022, Marsat2015CubicSpin](@cite) |
| 4 | non-spinning ``\ln v`` and ``\ln^2 v``; next-to-leading-order spin–orbit tail; next-to-next-to-leading-order spin–spin; leading quartic in spin | [Blanchet2023, Blanchet2024, Marsat2014SpinOrbitTail, Nagar2019, Cho2022, Siemonsen2018](@cite) |
| 4.5 | non-spinning only | [Blanchet2023, Blanchet2024](@cite) |

Through 3.5PN the phase of aligned-spin black holes is complete for point
particles; the absorption of energy and angular momentum by the horizons,
which enters at 2.5PN for spinning black holes, is not included. The 3.5PN
spin–spin tail [Nagar2019, Cho2022](@cite) is missing from several
frequency-domain forms in use, [Khan2016](@cite) among them. At 4PN the
spin–orbit tail has been given in frequency-domain form [Nagar2019](@cite),
while the next-to-next-to-leading-order spin–spin and quartic-in-spin terms
follow here from the stationary-phase reduction of the binding energy and
flux of the cited papers; the spin–spin flux at that order rests on a single
computation [Cho2022](@cite). At 4.5PN only the
non-spinning term is known, so with spins that order omits terms of
``\mathcal{O}(\chi v^9)``. The coefficients are tested against exact rational
evaluations of the literature forms at five points of ``(\eta, \chi_1, \chi_2)``,
two of them with ``\chi_s\chi_a \neq 0``.

The phase of harmonic ``k`` in the ``e^{-2\pi i f t}`` Fourier convention is
```math
\Psi_k(f) = 2\pi f t_c - \frac{k}{2}\phi_c + \frac{k}{2}\,\psi_{\mathrm{PN}}(2f/k) - \frac{\pi}{4},
```
so that ``\tilde h_k(f) \propto e^{-i\Psi_k(f)}``; the arrival-time term is
common to every harmonic, and the stationary-phase time of harmonic ``k``
is the Newtonian ``t_k = t(2f/k)`` with
``t(F) = t_c - 5\mathcal{M}/(256\, u^{8})``, ``u = (\pi\mathcal{M}F)^{1/3}``,
which is where the detector motion of §5 is evaluated. Since harmonic
``k`` carries ``-(k/2)\phi_c``, the waveform is ``4\pi``-periodic in
``\phi_c``: a ``2\pi`` shift reverses the sign of the odd harmonics, which
vanish only at equal mass. The confusion maps cap phase deviations at
``|\Delta\phi_c| \le \pi``, half the period of the dominant harmonic,
beyond which its contribution to ``D^2`` decreases again.

### Spin directions of the tangent space

At order 1.5 the spins enter the phase only through the single scalar
``\beta``, so exactly one spin combination is a flat
direction of the signal manifold at every mass ratio (tangent rank 5):
at equal mass it is the antisymmetric combination ``\chi_a``, in general
it is the null line ``(113 - 76\eta + 113\delta_m)\,\mathrm{d}\chi_1 +
(113 - 76\eta - 113\delta_m)\,\mathrm{d}\chi_2 = 0``. At ``\eta = 2/9`` the
slope is ``\mathrm{d}\chi_2/\mathrm{d}\chi_1 = -2.29``, the ``\chi_s`` and
``\chi_a`` coefficients being ``8.01`` and ``3.14``.

At order 2 the spins enter through ``\beta`` and ``\sigma``, and the tangent
space has rank 6 wherever the gradients of the two in the spin plane are not
parallel. They are parallel on one line through the origin of that plane,
```math
\frac{\chi_2}{\chi_1} = \frac{q\,(807 - 713\,q)}{807\,q - 713}, \qquad q = \frac{m_1}{m_2} \ge 1,
```
on which the rank stays 5, as it does at zero spins, where ``\sigma`` is of
second order. For unequal masses the line belongs to the truncation at 2PN
(``\chi_2 = -1.37\,\chi_1`` at ``\eta = 2/9``). Close to the line the second
spin combination is constrained only weakly: its independent part of the
tangent space shrinks in proportion to the distance of the base point from
the line, and the basis drops it below the Gram–Schmidt tolerance.

From order 2.5 on, ``\varphi_5`` is a second linear function of the spins,
independent of ``\beta``: the determinant of the gradients of ``\varphi_3``
and ``\varphi_5`` in ``(\chi_s, \chi_a)`` is
``-\delta_m\,\eta\,(2086535/567 + 9260\,\eta/9)``, nonzero for every
unequal-mass binary. The rank is then 6 everywhere in the spin plane, zero
spins and the 2PN line included. At equal mass, exchanging the two bodies
leaves the signal unchanged, so ``\chi_a`` enters quadratically at every
order and the rank is 5 on the line ``\chi_1 = \chi_2``. The second spin
combination stays the weakest direction of the tangent space at every order,
which is why sweeps from order 2 on need the fit in whitened coordinates
([System Architecture](architecture.md)).

### Convergence in post-Newtonian order

The overlays of `configs/pn_orders/` repeat the quickstart configuration
(sweeps and maps) and four spin maps of the production configuration at every
order, with the fit in whitened coordinates. The quickstart source
(``\mathcal{M} = 0.5`` s, ``\eta = 0.2222``, 1 to 16.5 mHz) reaches
``v = 0.40`` at the upper band edge, just below the innermost-stable-orbit
value ``6^{-1/2} = 0.41``. Its spin-plane zone at ``\chi_1 = \chi_2 = 0.8``
(principal half-axes) and its chirp-mass–coalescence-time zone (half-length
relative to order 1.5), with the log-log exponent of the two sweeps:

| Order | Spin zone, long half-axis | Spin zone, short half-axis | Long-axis slope ``\mathrm{d}\chi_2/\mathrm{d}\chi_1`` | ``\mathcal{M}``–``t_c`` zone | Exponent, ``\mathcal{M}``–``t_c`` sweep | Exponent, spin sweep |
|:--|--:|--:|--:|--:|--:|--:|
| 1.5 | prior-limited | ``1.57\times10^{-3}`` | ``-2.29`` | 1 | ``3.992 \pm 0.004`` | ``3.990 \pm 0.003`` |
| 2 | 0.154 | ``1.65\times10^{-2}`` | ``-3.60`` | 0.992 | ``3.992 \pm 0.004`` | ``3.984 \pm 0.004`` |
| 2.5 | 0.111 | ``1.07\times10^{-3}`` | ``-2.52`` | 1.007 | ``3.991 \pm 0.005`` | ``3.985 \pm 0.005`` |
| 3 | 0.192 | ``1.02\times10^{-3}`` | ``-2.47`` | 1.003 | ``3.991 \pm 0.005`` | ``3.985 \pm 0.005`` |
| 3.5 | 0.050 | ``0.97\times10^{-3}`` | ``-2.48`` | 1.008 | ``3.991 \pm 0.005`` | ``3.985 \pm 0.005`` |
| 4 | 0.101 | ``1.11\times10^{-3}`` | ``-2.46`` | 1.010 | ``3.989 \pm 0.006`` | ``3.985 \pm 0.005`` |
| 4.5 | 0.101 | ``1.11\times10^{-3}`` | ``-2.46`` | 1.010 | ``3.987 \pm 0.007`` | ``3.986 \pm 0.005`` |

The quartic law and the chirp-mass–coalescence-time zone do not depend on the
order beyond 1 %. In the spin plane the zone closes from order 2 on, and from
2.5PN its short half-axis, set by the spin–orbit term of that order, varies
by 14 %, between ``0.97`` and ``1.11\times10^{-3}``. The long half-axis, along the
combination that is flat at 1.5PN, does not converge: it alternates between
0.05 and 0.19 from 2.5PN to 4PN. That orders 4 and 4.5 agree is no sign of
convergence, since 4.5PN adds no spin term. For a source that reaches
``v \approx 0.4`` in the band the extent of the zone along that combination is
fixed by the truncation of the series, not by the signal; its closure, the
rank 6 of the tangent space, holds at every order from 2.

The production overlays (1 yr, 0.1 to 50 mHz, ``\eta = 0.2222``) compare
three sources in the spin plane: ``\mathcal{M} = 4`` s at
``\chi_1 = \chi_2 = 0.8`` and ``\mathcal{M} = 0.25`` s at
``(\chi_1, \chi_2) = (0.9, 0)``, both of which reach the innermost stable
orbit in the band, and ``\mathcal{M} = 0.1`` s at ``(0.7, 0.5)``, which stops
at ``v = 0.34`` at the band edge. Principal half-axes of the zones, the short
ones in units of ``10^{-3}``:

| Order | ``\mathcal{M} = 4`` s, long | short | ``\mathcal{M} = 0.25`` s, long | short | ``\mathcal{M} = 0.1`` s, long | short |
|:--|--:|--:|--:|--:|--:|--:|
| 1.5 | prior-limited | 2.48 | prior-limited | 0.87 | prior-limited | 0.67 |
| 2 | 0.191 | 5.57 | 0.298 | 3.46 | 0.265 | 2.62 |
| 2.5 | 0.220 | 1.85 | 0.141 | 0.90 | 0.127 | 2.04 |
| 3 | 0.236 | 1.30 | 0.128 | 0.74 | 0.158 | 1.46 |
| 3.5 | 0.055 | 1.65 | 0.054 | 1.19 | 0.148 | 0.70 |
| 4 | 0.118 | 2.65 | 0.080 | 0.71 | 0.148 | 0.61 |
| 4.5 | 0.118 | 2.65 | 0.080 | 0.71 | 0.148 | 0.61 |

For the source that stays below ``v = 0.34`` the long half-axis settles at
0.148 from 3.5PN on (0.158 at 3PN) and the short one changes by 13 % from
3.5PN to 4PN. For the two sources that reach the innermost stable orbit
neither axis converges: the long half-axis drops to 0.055 at 3.5PN and
returns to 0.12 and 0.08 at 4PN. Orders 4 and 4.5 coincide for every map, as
4.5PN adds no spin term. In the chirp-mass–spin plane of the
``\mathcal{M} = 4`` s source the short half-axis agrees within 3 % from 2.5PN
on, and the long one within 4 % from 3.5PN on. The size of the spin-plane
zone is therefore a converged property of the signal only for sources whose
signal in the band stays well below the innermost stable orbit.

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
equal mass, where they are skipped. The strength of the (3,3) harmonic is
therefore fixed by the mass ratio.

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
once per year in the retrograde sense; the arms are equal to first order in
``e``, and at second order their lengths vary by at most ``e/2 \approx 0.24\,\%``.
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

The wave-frame basis follows the science ground segment conventions
[Baghi2026SGSConventions](@cite): with
``\hat e_\lambda = (-\sin\lambda, \cos\lambda, 0)`` and
``\hat e_\beta = (-\sin\beta\cos\lambda, -\sin\beta\sin\lambda, \cos\beta)``,
``\hat u = -\hat e_\lambda``, ``\hat v = \hat e_\beta``, and the polarization
angle rotates them into ``\hat p = \hat u\cos\psi + \hat v\sin\psi``,
``\hat q = -\hat u\sin\psi + \hat v\cos\psi``; the polarization tensors are
``e^+ = \hat p\otimes\hat p - \hat q\otimes\hat q`` and
``e^\times = \hat p\otimes\hat q + \hat q\otimes\hat p``. The amplitudes of
§3 are defined with polarization vectors along and across the line of nodes
[Blanchet1996](@cite), a quarter turn from the wave-frame vectors of the
conventions document, so ``\psi`` (`[physics].polarization`) corresponds to
the polarization angle ``\psi + \pi/2`` of that document; the two
descriptions differ by more than a phase shift only through the odd
harmonics. In the long-wavelength limit each spacecraft ``i`` synthesizes a
Michelson interferometer with detector tensor
``D_i = \tfrac{1}{2}(\hat a_i\otimes\hat a_i - \hat b_i\otimes\hat b_i)``
from its two arm unit vectors, giving ``F^{+}_i = D_i : e^+`` and
``F^{\times}_i = D_i : e^\times`` (a 60° Michelson, hence the factor
``\sqrt{3}/2`` relative to a 90° detector is built in). The three
Michelson responses ``X, Y, Z`` are combined into the channels
``A = (Z - X)/\sqrt{2}``, ``E = (X - 2Y + Z)/\sqrt{6}`` of
[Baghi2026SGSConventions](@cite) (their Eq. 40), the combinations
[Prince2002](@cite) introduced for the Sagnac observables. For equal arms and
equal, mutually uncorrelated link noises the noise-correlation matrix of
``X, Y, Z`` has equal diagonal entries ``S_X`` and equal off-diagonal entries
``S_{XY}``, and ``A``, ``E`` and ``T = (X + Y + Z)/\sqrt{3}`` diagonalize it
with ``S_A = S_E = S_X - S_{XY}`` and ``S_T = S_X + 2S_{XY}``, the eigenvalues
[Prince2002](@cite) give for the Sagnac triple (their Eq. 16). In
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
``F_0`` with ``\tau(F_0) = t_c`` (0.56 mHz for ``\mathcal{M} = 0.1`` s and
``t_c = 0.63`` yr) and leaves the signal above about ``2F_0`` unchanged to
``10^{-10}``. Without the key the window is absent and ``T_{\mathrm{obs}}``
only sets the frequency resolution and the confusion-noise level. The
frequency grid is the analysis band of the detector, not the span of the
observation: the stationary-phase map
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
with the optical-metrology and test-mass acceleration noises
```math
P_{\mathrm{OMS}} = (1.5\times 10^{-11}\,\mathrm{m})^2\left[1 + \left(\frac{2\,\mathrm{mHz}}{f}\right)^{4}\right]\mathrm{Hz}^{-1}, \qquad
P_{\mathrm{acc}} = (3\times 10^{-15}\,\mathrm{m\,s^{-2}})^2\left[1 + \left(\frac{0.4\,\mathrm{mHz}}{f}\right)^{2}\right]\left[1 + \left(\frac{f}{8\,\mathrm{mHz}}\right)^{4}\right]\mathrm{Hz}^{-1},
```
their Eqs. 10–11 as given in the arXiv version of the paper
(arXiv:1803.01944); the journal version prints the two equations without the
factors in ``(2\,\mathrm{mHz}/f)^4`` and ``(f/8\,\mathrm{mHz})^4``. To this is
added the galactic confusion fit of Eq. 14,
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
between 0.5 and 4 yr; outside that range the nearest column is used.

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
- Precession, eccentricity, horizon absorption, phase terms beyond 4.5PN
  (beyond 4PN for the spins) and harmonics beyond 0.5PN are not included;
  the odd harmonics take ``k/2`` times the phase of the (2,2) harmonic at
  their frequency, without the phases of their own post-Newtonian amplitude
  corrections, and the stationary-phase times ``t_k`` are Newtonian at every
  phase order.
- The post-Newtonian series converges slowly near the innermost stable
  orbit. For sources that reach ``v \approx 0.4`` in the band the extent of
  the spin-plane zone along its weakest direction changes by factors of two to
  four from one phase order to the next (§2, Convergence in post-Newtonian
  order).

The band and the mass scale are campaign choices
([Run Parameters](parameters.md)); the geometric law itself makes
no assumption about them.
