[1] Zhou, Doyle & Glover (1996), Ch. 4 (§4.7, Lemma 4.7), Ch. 13, Ch. 16 (§16.1-16.11)
[2] Wie, Liu & Bauer (1993), JGCD 16(6), doi:10.2514/3.21129
[3] Skogestad & Postlethwaite (2005), 2nd ed.
[4] Tits & Yang (1996), IEEE Trans. Auto. Control 41(10), pp. 1432-1452
[5] Masterson (2002), JSV 249(3), pp. 575-598, doi:10.1006/jsvi.2001.3868
[6] Masterson (1999), MIT thesis
[7] Markley & Crassidis (2014), Sec. 2.7 (Eqs. 2.82a-b, p. 37), Sec. 2.9.2 (Eq. 2.127, p. 46), Sec. 2.9.5 (p. 50), Sec. 6.1.3 (p. 239), Sec. 6.3 (p. 263), App. E (Eq. E.91, p. 453, Joseph form)
[8] Lefferts, Markley, Shuster (1982), JGCD 5(5), pp. 417-429, doi:10.2514/3.56190
[9] Farrenkopf (1978), JGCD 1(4), pp. 282-284, also [7] Sec. 6.3, p. 263
[10] Bierman (1977), Factorization Methods for Discrete Sequential Estimation
[11] Nemati & Stahl (2020), SPIE. Contrast model is second-order Taylor expansion of coronagraph PSF response to LOS displacement.

params.py — all physical constants, no Basilisk imports, builds the linear plant.

np.diag — Ib = np.diag([Ix,Iy,Iz]), principal axes assumed aligned, cross-axis inertias zero. 1D input builds diagonal matrix, 2D extracts. also for Q_hinf_diag in H∞ weights.

np.array — builds A matrix in build_single_axis_plant: 4×4 per-axis plant [theta_bus, omega_bus, theta_pay, omega_pay]. rows: [0,1,0,0], [-k/Ib, -c/Ib, k/Ib, c/Ib], [0,0,0,1], [k/Ip, c/Ip, -k/Ip, -c/Ip]. np.array([1,2,3]) is (3,) not (3,1). for column vector use np.array([[1],[2],[3]]).

build_single_axis_plant(Ib, Ip, freq_iso) — per-axis linear plant from inertias and isolation frequency. returns A(4×4), B(4×2). B has two columns: Bc (control) and Bd (disturbance), both acting on omega_bus_dot. used by the H∞ synthesis, pole placement, and the Luenberger observer in HinfCtrl.

quat_mult — Hamilton product. qv and qs from Markley & Crassidis Eqs. 2.82a-b, Sec. 2.7. vector part: qs*pv + ps*qv + cross(qv,pv). scalar part: qs*ps - dot(qv,pv). scalar-last convention: q=[qx,qy,qz,qs]. used for MEKF multiplicative reset, mount relative quaternion, and attitude propagation.

quat_inv — [-qx,-qy,-qz,qs] = q^-1. unit quaternion inverse equals conjugate. for error quaternion dq = q_pay ⊗ inv(q_bus) in mount torque, and FGS innovation in MEKF.

quat_normalize — divide by L2 norm. kinematics has no unit-norm constraint, float drift accumulates. normalise after every quaternion operation to keep q on S^3.

quat_to_mrp — q to MRP: sigma = qv/(1+qs). selects hemisphere with qs >= 0 to avoid 180° singularity. |sigma| > 1 falls back to shadow set. MRP is 3-parameter, avoids redundant 4th dimension of quaternion. for MEKF: z = 2*quat_to_mrp(q_err) gives delta_theta directly, so H = [I3, 0_3x3]. [7] Sec. 2.9.5, p. 50.

skew — 3×3 skew-symmetric matrix from 3-vector. [[0,-z,y],[z,0,-x],[-y,x,0]]. used in MEKF F-matrix and gyroscopic coupling.

RWA parameters — harmonics: [0.88, 1.0, 2.0, 3.0, 5.24]. ratios from [5][6] but C_i values are representative, not from a published wheel dataset. amplitudes scale Ω²: A_i = C_i × Ω². broadband noise floor from Masterson empirical data. 4-wheel pyramid, tonal divided by sqrt(2) (uncorrelated phases, RMS sum).

ctrl_bw, omega_ctrl — controller bandwidth 0.5 Hz (~10× below f_iso), enough separation for H∞ to shape. omega_ctrl = 2π × ctrl_bw used in Q_hinf_diag weighting.

Q_hinf_diag, R_hinf — per-axis LQ weights. Q diag = [0, 0, wb^2, wb*0.1], penalises payload states, ignores bus states. R = 1.0, control effort penalty. defines the same cost that the H∞ problem minimises for the worst-case disturbance.

hinf_gamma_lo, hi, tol, iters, safety — bisection bounds 0.5 to 100, tol 1e-3, 50 iterations. safety factor 1.1 applied to gamma_min. floor at 2.0, wide margin (gamma_min ~ 1.0 for all axes) means controller is near-LQR.

ctrl_poles_target, obs_poles_target — pole-placement benchmark uses complex conjugate pairs at [-3±1.5j, -4±1.0j] rad/s (~3× to 8× controller BW). observer poles [-25±8j, -30±5j] rad/s (~10× controller BW).

torque_limit — 1.0 N·m, per-axis. HinfCtrl saturates each axis independently.

coronagraph — S_beamwalk = 0.39 m/rad, maps LOS angle to beam displacement. bw_tol = wavelength/100. C_floor = 1e-10. contrast = C_floor × (1 + (beamwalk/bw_tol)²), second-order Taylor expansion of PSF response to LOS displacement [11].

timing — three rates: 1 kHz dynamics (RK4 integrator), 100 Hz FSW (MEKF + controller), 10 Hz FGS (sensor). 5 s simulation time (~5× settling time for 0.5 Hz controller BW).

jitter_req_mas = 7.0, contrast_req = 2e-10, est_err_req_mas = 3.0, gain_margin_req_dB = 6.0, phase_margin_req_deg = 30.0. requirements from the pointing spec, not mission-specific.

monte carlo dispersions — 9 parameters: I_bus ±5.3%, I_pay ±5.2%, f_iso ±8.7%, zeta_iso ±20%, sig_fgs ±15%, sig_gyro ±18%, rwa_speed ±9.4%, rwa_noise ±20%, S_beamwalk ±10%. 500 runs. controller re-built per trial (f_iso and inertias shift the plant, K must track).

structural dynamics — f_struct = 15 Hz (estimated 1st bending mode, 12-18 Hz typical). zeta_struct = 0.005 (MIL-STD-1540 typical for space structures). coupling_thresh = 2% proximity flag. if any RWA harmonic lands within threshold of the bending mode, flag as risk register item.

modules.py — five modules in closed-loop order, manual data flow via numpy attributes. SysModel base for Reset/UpdateState contract.

SysModel — Basilisk base class. contract: Reset(CurrentSimNanos) at t=0, UpdateState(CurrentSimNanos) each step. timestamps integer nanoseconds. Reset sets initial state and allocates, UpdateState reads inputs, computes, writes outputs. manual for-loop runner, SysModel provides interface contract only.

RwaDisturbance — Masterson tonal harmonics + filtered broadband. tonal: sum of A_i × sin(ratio_i × Ω × t + phase_i) across 5 harmonics and 4 wheels, divide by sqrt(4) for uncorrelated RMS sum. broadband: white noise → 50 ms box filter (first null at ~20 Hz), mimics bearing lubricant + housing attenuation. 50-sample buffer, np.mean(buf, axis=0).

np.random.randn — standard normal, one draw per axis per timestep for broadband RWA torque. sigma from Masterson coefficients. ~5000 draws per axis in 5 s at 1 kHz.

np.random.uniform — [0, 2π) at Reset(). tonal harmonic phases, shape (n_h, n_rwa, 3). independent per-harmonic, per-wheel, per-axis. uncorrelated across wheels avoids artificial constructive interference at t=0.

np.mean — 50 ms box filter. buffer N = 50 samples at 1 kHz. np.mean(buf, axis=0) collapses (50,3) to (3,). sinc-shaped frequency response, first null at ~20 Hz.

np.sin — tonal harmonic sum. argument = 2π × f_harmonic × t + phase. amplitudes from C_i × Ω², ratios from [5][6] empirical model.

np.sqrt — sqrt(n_wheels) for tonal RMS normalisation. also in mount damping: C_diag = 2ζ√(K×I).

TwoBodyPlant — 14-state two-body nonlinear plant. state: q_bus(4), w_bus(3), q_pay(4), w_pay(3). RK4 integrator at 1 kHz, quaternion renormalisation at each sub-stage. mount torque: τ = K×δθ + C×δω, δ computed from relative quaternion.

qdot — quaternion kinematics: dq = 0.5 × Ω(w) × q. Ω(w) is 4×4 matrix with upper-left = -skew(w). equivalent to dqv = 0.5(qs×w + cross(w,qv)), dqs = -0.5 dot(w,qv).

mount_torque — τ_m = K×δθ + C×δω. δ from relative quaternion dq = q_pay ⊗ inv(q_bus), δθ = 2×dq_vec (small-angle approximation, valid to ~5°). δω = ω_pay - ω_bus.

np.cross — gyroscopic term: ω × (I·ω). angular momentum rotates in body frame, cross product extracts torque to maintain that rotation. also in quaternion product vector part.

RK4 — 4-stage explicit Runge-Kutta at h = 0.001 s. quaternion normalisation at each sub-stage (k1→q2 = normalise(q + 0.5h·k1) etc.). 4th-order global truncation error O(h⁴), symplectic-like for angular momentum conservation. without renormalisation, quaternion drifts off S^3 after ~10 s.

FGS/gyro simulation — gyro: true rate + ARW noise per sample: σ_gyro × √(dt). FGS: attitude MRP + σ_fgs noise, valid at 10 Hz edges. FGS validity checked by integer nanosecond comparison: (CurrentSimNanos - last_fgs) ≥ dt_fgs_ns.

AttitudeMekf — 6-state multiplicative EKF. state: delta_theta(3) + gyro_bias(3). delta_theta is attitude error in tangent space of SO(3). multiplicative reset folds delta_theta into q_hat after each FGS update.

Farrenkopf Q model [9] — ARW sig_v = 5e-7 rad/s/√Hz, RRW sig_u = 1e-10 rad/s²/√Hz. discrete integrals at 100 Hz: q11 = sv²×dt + su²×dt³/3, q12 = -su²×dt²/2, q22 = su²×dt. zero cross-coupling between axes (independent per-channel).

time_update — at 100 Hz, every UpdateState call. F(6×6) upper-left = -skew(w_est), upper-right = -I3, lower rows zero. Phi = I + F×dt (first-order Euler for state transition). P = Phi×P×Phiᵀ + Q.

measurement_update — at 10 Hz, when fgs_valid flag set. z = 2×quat_to_mrp(q_err), H = [I3, 0_3×3]. innovation S = H×P×Hᵀ + R. K = P×Hᵀ×inv(S). dx = K×dz. delta_theta folded into q_hat via vec_to_quat, bias accumulated.

np.linalg.inv — innovation: K = P×Hᵀ×inv(S). S is 3×3, well-conditioned with current noise model (R = σ_fgs²×I3). could use solve(S, H×Pᵀ)ᵀ for better stability. [10] notes explicit inverses risky for poorly conditioned systems, not an issue at 10 Hz with tuned noise.

Joseph form — P⁺ = (I-KH)P⁻(I-KH)ᵀ + KRKᵀ. [7] App. E Eq. E.91, p. 453. guarantees P stays PSD. standard form (I-KH)P⁻ can lose symmetry under roundoff. two extra 6×6 multiplies at 10 Hz, negligible cost. 0.5×(P+Pᵀ) symmetrise after every update.

multiplicative approach [8] — additive form treats quaternion as R⁴ not S³, unit-norm violated by linear updates. multiplicative estimates 3-component error in tangent space of SO(3), folds correction via quaternion multiplication. constraint satisfied by construction.

HinfCtrl — per-axis state-feedback controller + Luenberger observer. K(3×4) set externally by runner after synthesis. observer estimates theta_bus, omega_bus from payload measurement (theta_pay, omega_pay) using payload truth at navigation rate.

Luenberger observer — ė = A ê + L(y - C ê). L from pole placement on (Aᵀ, Cᵀ) with obs_poles_target. observer propagates at 100 Hz. fallback L hard-coded for f_iso = 1 Hz if place_poles fails.

scipy.signal.place_poles — pole placement for observer: L = place_poles(Aᵀ, Cᵀ, obs_poles).gain_matrixᵀ. algorithm: [4]. dual to controller placement, same pole targets. observer poles ~10× controller BW for separation.

ContrastModel — beamwalk-to-contrast degradation, truth-side. 100-sample rolling RMS window. contrast = C_floor × (1 + (S_beamwalk × θ_los_rms / bw_tol)²). theta_los_rms = std of X and Y MRP over window.

rolling RMS — first 10 samples use instantaneous theta (window not full). after 10 samples, uses std of window buffer. circular buffer with _idx modulo window_size.

run_sim.py — manual multi-rate loop, controller synthesis, data reduction, margin analysis, plotting, Monte Carlo, V&V. the only file run directly.

H∞ synthesis — per-axis gamma bisection on the ARE. AᵀP + PA + Q - P(Bc Bcᵀ/R - Bd Bdᵀ/γ²)P = 0. Schur complement S = BcBcᵀ/R - BdBdᵀ/γ² must be SPD. Cholesky factor L from S → ARE with (A, L, Q). K = (1/R) Bcᵀ P. bisection: lo=0.5, hi=100, mid test, lo↑ if feasible (tightening), hi↓ if infeasible (relaxing). converges ~12-15 iterations at tol=1e-3. gamma_safety=1.1, floor=2.0.

scipy.linalg.solve_continuous_are — AᵀP + PA - P B R⁻¹ Bᵀ P + Q = 0. signature: (A, B, Q, R). B is the Cholesky factor L from the Schur complement, not the control input directly. gamma dependence through Bd/γ. [1] Ch. 16.

np.linalg.cholesky — Schur complement SPD check. LinAlgError means gamma too tight (BdBdᵀ/γ² term dominates, S loses positive definiteness). bisection responds by increasing gamma.

np.linalg.eigvalsh — checks P is PSD in _are_at_gamma. symmetric eigenvalue check, faster than eigvals for 4×4.

Pole placement benchmark — synth_place builds K per-axis using scipy.signal.place_poles with ctrl_poles_target. separate synthesis path, same plant model. H∞ outperforms pole placement on frequency-domain shaping (actively trades control vs disturbance rejection across frequency), but both yield stable closed-loop. used for sanity check that H∞ gains are in the same physical range.

margin computation — break loop at plant input. L(s) = K (sI - A)⁻¹ Bc. frequency sweep via np.logspace(-3, 3, 2000). Bode: |L(jω)| and angle(L(jω)). Nichols: phase vs magnitude. gain margin: -20 log10(|L|) at phase crossing (-180°). phase margin: 180 + angle(L) at gain crossing (0 dB). type-1 loops, GM = ∞ when no phase crossing below 0 dB.

np.logspace — freq sweep: np.logspace(-3, 3, 2000), 0.001 to 1000 rad/s. covers DC to beyond RWA tonal band.

np.log10 — Bode decade axis. GM: -20×log10(|L|) at phase crossover. negative sign intentional, GM positive for stable loops.

np.angle — phase of complex number. deg=True for Bode trace. np.unwrap for Nichols.

_crossings — linear interpolation across a threshold (0 dB or -180°). two-value local interpolation for sub-sample frequency accuracy. handles multiple crossings per axis.

np.interp — linear interpolation for dB or phase at crossover. finds magnitude at phase crossing and phase at gain crossing.

structural coupling check — _check_structural_risk compares RWA harmonic frequencies against f_struct = 15 Hz. if any harmonic within coupling_thresh (2%), flags as risk. separate risk flag, not baked into simulation output.

telemetry reduction — _reduce: trim first 0.5 s for settling. jitter: std of true MRP in X, Y axes, RSS of XY. estimation error: std of (true MRP - estimated MRP) at FSW rate. contrast: mean of C_vals after settle. beamwalk: mean of bw_vals after settle.

manual multi-rate loop — for step in range(n_dyn): 1 kHz disturbance + plant every step. 100 Hz MEKF + controller when step % n_fsw == 0. 10 Hz FGS implicit in plant (valid flag set internally). contrast at 100 Hz. all data flow visible in the loop body — each module's input attributes set explicitly, output attributes read explicitly. no messaging framework, no scheduler.

matplotlib — 2×2 subplot grid: (0,0) disturbance rejection Bode, (0,1) jitter timeseries, (1,0) contrast rolling RMS, (1,1) Nichols chart. dpi=200, tight_layout, savefig with bbox_inches='tight'.

frequency sweep — f_iso values [0.1, 0.3, 0.5, 1.0, 2.0, 5.0] Hz. 3 s per run, bar chart of contrast vs f_iso with jitter annotation. controller re-built per f_iso.

Monte Carlo — 500 trials, 3 s each, 3 dispersed parameters (f_iso, rwa_noise, rwa_speed), controller re-built per trial. pass/fail on jitter < 7.0 mas and contrast < 2e-10. scatter plot, histograms, box plot output. throughput ~2-3 runs/s.

V&V matrix — print_vv: 10 requirements (8 baseline + 2 MC). checks jitter, estimation error, contrast, beamwalk, gain margin (3×), phase margin (3×), MC jitter pass rate >95%, MC contrast pass rate >95%. formatted verification matrix to stdout.
