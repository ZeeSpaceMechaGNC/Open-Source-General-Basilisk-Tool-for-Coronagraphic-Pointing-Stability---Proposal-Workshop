[1] Zhou, Doyle & Glover (1996), Ch. 4 (§4.7), Ch. 13, Ch. 16
[2] Skogestad & Postlethwaite (2005), 2nd ed.
[3] Tits & Yang (1996), IEEE Trans. Auto. Control 41(10), pp. 1432-1452
[4] Masterson (2002), JSV 249(3), pp. 575-598
[5] Masterson (1999), MIT thesis
[6] Markley & Crassidis (2014), Sec. 2.7, 2.9.2, 2.9.5, 6.1.3, 6.3, App. E
[7] Lefferts, Markley, Shuster (1982), JGCD 5(5), pp. 417-429
[8] Farrenkopf (1978), JGCD 1(4), pp. 282-284
[9] Bierman (1977), Factorization Methods for Discrete Sequential Estimation
[10] Nemati & Stahl (2020), SPIE

params.py
  all physical constants. builds linear plant. no Basilisk imports.

- **inertias**: I_bus = diag([1000, 1000, 1200]) kg·m², I_pay = diag([100, 100, 120]) kg·m²

- **isolation mount**: f_iso = 1.0 Hz, ζ = 0.05. k_iso = I_pay·(2π·f_iso)², c_iso = 2ζ√(k_iso·I_pay)

- **controller**: ctrl_bw = 0.5 Hz, ω_b = 2π·ctrl_bw. Q_hinf = diag([0,0,ω_b²,0.1·ω_b]), R_hinf = 1.0. hinf_gamma lo=0.5, hi=100, tol=1e-3, safety=1.1, floor=2.0. ctrl_poles = [−3±1.5j, −4±1.0j] rad/s. obs_poles = [−25±8j, −30±5j] rad/s. torque_limit = 1.0 N·m per axis

- **RWA**: 4-wheel pyramid, Ω = 50 rad/s. 5 harmonics [0.88,1.0,2.0,3.0,5.24]×Ω. tonal A_i = C_i·Ω² from [4][5]. broadband σ_i from Masterson data

- **sensors**: gyro σ_v = 5e-7 rad/s/√Hz (ARW), σ_u = 1e-10 rad/s²/√Hz (RRW). FGS σ_fgs = 5e-9 rad (~1 mas) at 10 Hz

- **coronagraph**: S_bw = 0.39 m/rad, bw_tol = λ/100, C_floor = 1e-10

- **structural**: f_struct = 15 Hz, ζ_struct = 0.005, coupling threshold 2%

- **timing**: 1 kHz dynamics, 100 Hz FSW, 10 Hz FGS. Tsim = 5 s, dt = 1 ms

- **requirements**: jitter < 7.0 mas, contrast < 2e-10, est error < 3.0 mas, GM ≥ 6 dB, PM ≥ 30°

- **monte carlo dispersions (9 params)**: I_bus ±5.3%, I_pay ±5.2%, f_iso ±8.7%, ζ_iso ±20%, σ_fgs ±15%, σ_gyro ±18%, Ω_rwa ±9.4%, σ_rwa ±20%, S_bw ±10%. 500 trials, controller re-synthesised per trial

Modules

- **build_single_axis_plant(Ib, Ip, freq_iso)**: per-axis linear plant. returns A(4×4), B(4×2). A rows [0,1,0,0], [−k/Ib,−c/Ib,k/Ib,c/Ib], [0,0,0,1], [k/Ip,c/Ip,−k/Ip,−c/Ip]. B: Bc (control), Bd (disturbance), both on ω̇_bus

- **quaternion functions**: skew (3×3 cross-product), quat_mult (Hamilton product, scalar-last [6] 2.82a-b), quat_inv ([−qv,qs]), quat_normalise (L2 norm after every multiplication), quat_to_mrp (σ = qv/(1+qs), H = [I₃,0₃] for MEKF)

- **SysModel**: Basilisk base. Reset(CurrentSimNanos) at t=0, UpdateState each step. manual multi-rate loop

- **RwaDisturbance**: tonal Σ A_i sin(2π·h_i·Ω·t + φ_i), 5 harmonics × 4 wheels. broadband white noise → 50 ms box filter (1st null 20 Hz). phases random [0,2π) at Reset, independent per harmonic/wheel/axis. RMS ÷ √n_wheels

- **TwoBodyPlant**: 14-state, q_bus(4), ω_bus(3), q_pay(4), ω_pay(3). RK4 at 1 ms, quaternion normalised each sub-stage. mount torque τ_m = K_iso·δθ + C_iso·δω. δ from dq = q_pay ⊗ q_bus⁻¹, δθ ≈ 2·dq_vec. gyroscopic: ω × (I·ω)

- **AttitudeMekf**: 6-state MEKF, δθ(3) + δβ(3). propagate 100 Hz: F upper-left = −skew(ω_est), upper-right = −I₃. Φ = I + F·dt, P = Φ·P·Φᵀ + Q (Farrenkopf [8]). update 10 Hz FGS: z = 2·σ(q_err), K = P·Hᵀ·(H·P·Hᵀ+R)⁻¹. Joseph form P⁺ = (I−KH)P⁻(I−KH)ᵀ + KRKᵀ. multiplicative reset: δq → q̂ ← δq ⊗ q̂, errors zeroed

- **HinfCtrl**: per-axis K(3×4) set after synthesis. Luenberger observer x̂̇ = A x̂ + Bc u + L(y − C x̂). L from place_poles(Aᵀ,Cᵀ,obs_poles). receives MEKF nav (θ_pay, ω_pay). u = −K x̂, saturated ±torque_limit per axis

- **ContrastModel**: C = C_floor·(1 + (S_bw·θ_los_rms / bw_tol)²). θ_los_rms from 100-sample rolling window of X/Y MRP std. beamwalk = S_bw·θ_los_rms

Synthesis and Analysis (in run_sim.py)

- **H∞ synthesis (per axis)**: γ-bisection on ARE: AᵀP + PA + Q − P(BcBcᵀ/R − BdBdᵀ/γ²)P = 0. Schur complement SPD via Cholesky. converge ~12-15 iter, tol 1e-3. K = (1/R) Bcᵀ P, safety=1.1

- **pole placement (benchmark)**: K via place_poles with ctrl_poles_target. for comparison with H∞

- **stability margins**: break at plant input, L(s) = K(sI−A)⁻¹Bc. sweep 0.001-1000 rad/s. GM = −20 log₁₀|L| at −180°, PM = 180 + ∠L at 0 dB. linear interpolation for crossover

- **structural check**: flag if RWA harmonic within 2% of f_struct = 15 Hz

- **telemetry reduction**: trim 0.5 s settle. jitter = RSS of X/Y MRP std. est error = std(true−est MRP). contrast = mean C after settle

- **multi-rate loop**: each 1 ms: disturbance + plant. each 10 ms: MEKF + controller. each 100 ms: FGS flag set, MEKF update. contrast at 100 Hz

- **frequency sweep**: f_iso = [0.1,0.3,0.5,1.0,2.0,5.0] Hz, 3 s/run, controller rebuilt per f_iso. bar chart contrast vs f_iso

- **monte carlo**: 500 trials, 9 dispersed params, controller rebuilt per trial. pass: jitter < 7.0 mas, contrast < 2e-10. scatter, histograms, box plot

- **V&V matrix**: 10 requirements. checks jitter, est error, contrast, beamwalk, GM×3, PM×3, MC jitter >95%, MC contrast >95%

- **plotting**: 2×2: (0,0) disturbance Bode, (0,1) jitter timeseries, (1,0) contrast rolling RMS, (1,1) Nichols chart
