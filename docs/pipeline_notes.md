# pipeline notes — open coronagraphic pointing stability sim

working notes. rough. being refined for a coronagraph instrument studies
workshop. this documents what's in src/ and why.

---

## plant model — where the physics lives

the plant is a 14-state rigid-body model: two bodies (bus + payload) coupled
through a 3-axis spring-damper isolation mount. 7 states per body: quaternion
(4) + angular velocity (3). scalar-last quaternion convention throughout.

RK4 integration at 1 kHz. quaternion renormalisation at every substage so |q|
drift stays below 1e-12. this matters — a quaternion that walks off the unit
sphere silently corrupts attitude estimation downstream.

dynamics come from Euler's equation in body frame:

    I * w_dot + w x (I*w) = sum(torques)

no orbital mechanics. no gravity gradient. no solar radiation pressure. this
is a pure pointing-in-space simulation — the regime that matters for
coronagraphic stability on the timescale of a science exposure (seconds to
minutes).

the mount is a diagonal spring-damper: K_i = I_pay_i * w0^2, C_i = 2*zeta*sqrt(K_i*I_pay_i).
f_iso = 1.0 Hz (baseline), zeta = 0.05. these came out of the frequency sweep
trade (run_sweep in run_sim.py) — 1.0 Hz gave the best jitter/contrast
compromise. below 0.5 Hz the mount stops attenuating RWA harmonics. above
2.0 Hz the controller runs out of authority because the payload decouples
too much from the bus actuators.

inertia tensors are diagonal and axisymmetric-ish: I_bus = diag(50000, 50000, 20000),
I_pay = diag(4225, 4225, 2000) [kg*m^2]. representative values for a ~6.5m
aperture telescope on a medium bus. included in the monte carlo dispersion set.

---

## disturbance — what the wheels do

reaction wheels are the dominant disturbance source for precision pointing.
Masterson (1999, MIT thesis) characterised them as discrete harmonic tones
plus broadband noise:

    tau(t) = tau_bb(t) + sum_i C_i * Omega^2 * sin(2*pi*h_i*f_wheel*t + phi_i)

C_i scales with Omega^2 — this is physically correct: static imbalance force
is m*omega^2*r (centripetal), so disturbance torque follows Omega^2. double
the wheel speed, quadruple the disturbance.

the harmonics we use are physically motivated but synthetic:

| h_i | freq at 1200 rpm | physical origin | C_i [N*m/(rad/s)^2] |
|:---:|:---|---:|:---|
| 0.88 | 17.6 Hz | bearing cage retainer | 1.0e-8 |
| 1.00 | 20.0 Hz | static imbalance | 5.0e-8 |
| 2.00 | 40.0 Hz | dynamic imbalance | 3.0e-8 |
| 3.00 | 60.0 Hz | 3rd harmonic | 1.5e-8 |
| 5.24 | 104.8 Hz | ball spin frequency | 0.8e-8 |

the harmonic ratios represent canonical bearing phenomena — cage retainer,
static imbalance (once-per-rev), dynamic imbalance (twice-per-rev), and ball
pass/spin frequencies. the C_i values are representative for a precision
reaction wheel at moderate speed. the broadband floor
(0.005 N*m RMS) dominates above ~50 Hz.

four wheels in a pyramid. uncorrelated random phases per wheel per harmonic
per axis — the tonal sum is divided by sqrt(4) because uncorrelated sinusoids
add in quadrature (RMS), not linearly.

---

## estimation — the MEKF

the multiplicative extended kalman filter is the standard spacecraft attitude
estimator. 6-state: attitude error (MRP, 3) + gyro bias (rad/s, 3).

gyro propagates at 100 Hz. FGS updates at 10 Hz. the gyro is fast but noisy
(ARW = 5e-7 rad/s/sqrt(Hz)); the FGS is slow but accurate (1-sigma noise =
5e-9 rad ~ 1 mas). the MEKF fuses them — the bias estimate converges in
~2 seconds.

joseph-form covariance update for numerical conditioning:

    P+ = (I - K*H)*P-*(I - K*H)^T + K*R*K^T

this guarantees P stays positive semi-definite even when the innovation
covariance S is near-singular. costs an extra matrix multiply per update
but prevents divergence.

Farrenkopf process noise model (Markley & Crassidis Eq 6.6): Q captures
gyro ARW (q11 ~ dt), bias instability (q22 ~ dt), and their coupling
(q12 ~ dt^2).

---

## control — H-infinity state feedback

per-axis 4-state plant: [theta_bus, omega_bus, theta_pay, omega_pay].
single control input (torque on bus). H-infinity synthesis via ARE +
gamma bisection:

    solve:  A^T*X + X*A - X*S*X + Q = 0
    where:  S = (1/R)*Bc*Bc^T - (1/gamma^2)*Bd*Bd^T
    K = (1/R)*Bc^T*X

gamma is the disturbance attenuation level. bisection finds gamma_min ~ 2.0
for each axis. we apply a 1.1x safety factor.

a Luenberger observer estimates the bus states from payload measurements
(MEKF output). observer poles at ~10x controller bandwidth via dual pole
placement. the observer assumes the disturbance input is unknown — it
treats the bus dynamics as a black box driven by control + unknown torques.
this decouples observer design from disturbance characterisation.

torque saturation at 1.0 N*m per axis. this is a hard limit — above it,
the controller clips and the linear analysis breaks. the MC trials that
hit saturation are treated as failures (they show up as jitter spikes).

---

## contrast — the thing we're actually trying to protect

the coronagraph physics in one sentence: pointing jitter displaces the star
on the focal plane mask, light leaks through, contrast degrades.

Nemati & Stahl (2020) developed a contrast budgeting methodology for the
HabEx coronagraph. we follow their approach — decompose contrast into
independent contributors, each with a sensitivity. beamwalk (the lateral
displacement of the beam on the coronagraph mask) depends on LOS jitter:

    beamwalk = S_beamwalk * theta_rms

S_beamwalk = 0.39 m/rad. this is the beamwalk sensitivity — the lateral
displacement of the coronagraph beam per radian of LOS jitter. the value
represents a typical coronagraph optical train at visible wavelengths.

the quadratic degradation model:

    C = C_floor * (1 + (beamwalk / bw_tol)^2)

follows from a second-order expansion of the coronagraph response to beam
displacement — the first-order term vanishes because the coronagraph is
nulled at zero displacement. Nemati & Stahl (2020) provide the overall
contrast budgeting framework.

bw_tol = lambda / 100 ~ 5.5 nm at 550 nm science wavelength. this follows
standard coronagraphic mask tolerancing at visible wavelengths.

C_floor = 1e-10. this is the raw contrast target for a space coronagraph
aiming to characterise earth-analogue exoplanets. the pipeline shows that
with 0.17 mas jitter, the beamwalk contribution is below the floor —
meaning jitter is not the limiting factor; wavefront stability and
coronagraph mask fabrication are.

---

## monte carlo — does it hold up?

500 trials. dispersions on: inertia tensors (+/-5%), isolation frequency
(+/-10%), damping (+/-20%), sensor noise (+/-15%), wheel speed (+/-10%),
RWA broadband noise (+/-20%), beamwalk sensitivity (+/-10%).

controller re-synthesised per trial — the H-infinity gains adapt to the
dispersed plant. this is physically realistic: you'd re-tune the controller
for the as-built spacecraft.

results: >95% pass rate on both jitter and contrast.
mean jitter = 0.17 mas. mean contrast = 1.2e-10.

the achieved jitter (0.17 mas RMS) is below published coronagraphic pointing
requirements of 0.3 mas (HabEx baseline, Morgan et al. 2023) by nearly a
factor of two.

---

## what's not modelled (yet)

- flexible body modes. the payload is rigid. real telescopes have structural
  resonances above ~10 Hz that amplify RWA disturbances.
- FSM (fast steering mirror). coronagraph designs include a tip/tilt correction
  mirror with ~2 Hz bandwidth. this could suppress residual jitter by
  an additional 40+ dB below 0.1 Hz.
- DFP (disturbance free payload). active isolation architectures can
  provide additional vibration suppression between bus and payload.
  the passive mount modelled here represents the simpler case.
- wavefront error dynamics. we model beamwalk only. real coronagraphs also
  degrade from WFE modes that couple jitter into higher-order aberrations.
- orbital environment. no gravity gradient, no magnetic torques, no thermal
  snap. these matter at longer timescales but are negligible during a
  science exposure.

---

## how to run

```
cd src
python run_sim.py
```

set flags at the bottom:
- run_baseline = True   —> single 5s sim + dashboard + V&V
- run_freq_sweep = True —> sweep f_iso [0.1..5.0] Hz
- run_mc_campaign = True —> 500-trial Monte Carlo (~several minutes)

dependencies: Basilisk, numpy, scipy, matplotlib.

---

*all processing was local. no data left the machine during pipeline execution*
*formatting and docstrings done via automated programmed pipeline*
*working draft*
