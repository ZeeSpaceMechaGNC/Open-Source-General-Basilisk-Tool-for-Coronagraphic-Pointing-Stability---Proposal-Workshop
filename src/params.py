# pointing sim params, fine-pointing stage
# Prototype 4. numbers from Masterson, Markley & Crassidis, and published reference data.

import numpy as np

# 1. unit conversions
MAS2RAD = 1.0 / 206265e3
RAD2MAS = 206265e3

# 2. telescope
D_aperture = 6.5                    # [m]
wavelength = 550e-9                 # [m], visible band centre

# 3. inertia
I_bus_diag = np.array([50000.0, 50000.0, 20000.0])     # [kg*m^2]
I_pay_diag = np.array([ 4225.0,  4225.0,  2000.0])     # [kg*m^2]

I_bus     = np.diag(I_bus_diag)
I_bus_inv = np.diag(1.0 / I_bus_diag)
I_pay     = np.diag(I_pay_diag)
I_pay_inv = np.diag(1.0 / I_pay_diag)

# 4. isolation mount
f_iso    = 1.0                      # [Hz]
zeta_iso = 0.05

w0_iso      = 2.0 * np.pi * f_iso
K_iso_diag  = I_pay_diag * w0_iso**2
C_iso_diag  = 2.0 * zeta_iso * np.sqrt(K_iso_diag * I_pay_diag)

# 5. per-axis plant builder

def build_single_axis_plant(Ib, Ip, freq_iso):
    """4-state linear plant. x = [theta_bus, omega_bus, theta_pay, omega_pay]"""
    w0 = 2.0 * np.pi * freq_iso
    k  = Ip * w0**2
    c  = 2.0 * zeta_iso * np.sqrt(k * Ip)
    A = np.array([
        [ 0.0,     1.0,     0.0,     0.0   ],
        [-k / Ib,  -c / Ib,  k / Ib,  c / Ib],
        [ 0.0,     0.0,     0.0,     1.0   ],
        [ k / Ip,   c / Ip,  -k / Ip, -c / Ip],
    ])
    B = np.array([
        [0.0,       0.0   ],
        [1.0 / Ib,  1.0 / Ib],
        [0.0,       0.0   ],
        [0.0,       0.0   ],
    ])
    return A, B

# 6. sensors
#   FGS (Fine Guidance Sensor): attitude at 10 Hz, ~1 mas noise
#   gyro: body rate at 100 Hz with ARW (Angle Random Walk) + RRW (Rate Random Walk, bias drift)

sig_fgs  = 5e-9                     # [rad], ~1 mas, 1-sigma per axis
sig_gyro = 5e-7                     # [rad/s/sqrt(Hz)], ARW
sig_bias = 1e-10                    # [rad/s^2/sqrt(Hz)], RRW. I think this is about right.
fgs_rate = 10.0                     # [Hz]

# 7. RWA (Masterson 1999, J. Sound Vib.)
#   RWA (Reaction Wheel Assembly), 4 wheels in pyramid. disturbance:
#   tonal harmonics + filtered broadband, per Masterson Table 4.2 methodology.

rwa_noise_broadband = 0.005         # [N*m]
rwa_speed_rpm       = 1200.0
rwa_speed_hz        = rwa_speed_rpm / 60.0
rwa_speed_rads      = rwa_speed_rpm * 2.0 * np.pi / 60.0

rwa_harmonic_ratios = np.array([0.88, 1.0, 2.0, 3.0, 5.24])
rwa_harmonic_Ci     = np.array([1.0e-8, 5.0e-8, 3.0e-8, 1.5e-8, 0.8e-8])

n_rwa = 4
rwa_amplitude_scale = np.sqrt(n_rwa)

# 8. control

ctrl_bw = 0.5                     # [Hz], ~10x below isolation, enough separation for hinf to shape
omega_ctrl = 2.0 * np.pi * ctrl_bw

Q_hinf_diag = np.array([0.0, 0.0, omega_ctrl**2, omega_ctrl * 0.1])
R_hinf      = 1.0

hinf_gamma_lo     = 0.5
hinf_gamma_hi     = 100.0
hinf_gamma_tol    = 1e-3
hinf_gamma_iters  = 50
hinf_gamma_safety = 1.1

ctrl_poles_target = np.array([-3.0+1.5j, -3.0-1.5j, -4.0+1.0j, -4.0-1.0j])
obs_poles_target  = np.array([-25.0+8.0j, -25.0-8.0j, -30.0+5.0j, -30.0-5.0j])

torque_limit = 1.0                  # [N*m]

# 9. coronagraph

S_beamwalk = 0.39                   # [m/rad]
bw_tol     = wavelength / 100.0
C_floor    = 1e-10

# 10. timing

dt_dyn = 0.001                      # 1 kHz
dt_fsw = 0.01                       # 100 Hz
dt_fgs = 1.0 / fgs_rate             # ~0.1 s
T_sim  = 5.0                        # ~5x settling time, enough to see steady-state

# 11. thresholds

jitter_req_mas       = 7.0
contrast_req         = 2e-10
bw_tol_pm            = bw_tol * 1e12
est_err_req_mas      = 3.0
gain_margin_req_dB   = 6.0
phase_margin_req_deg = 30.0

# 12. monte carlo

mc_nruns = 500
mc_dispersions = {
    'I_bus_scale':   0.053,
    'I_pay_scale':   0.052,
    'f_iso':         0.087,       # mount dynamics are the least certain, ~9% from vendor data
    'zeta_iso':      0.20,        # damping is hardest to set, 20% is a placeholder
    'sig_fgs':       0.15,
    'sig_gyro':      0.18,        # gyro ARW varies across the family. I think.
    'rwa_speed_rpm': 0.094,       # speed trim uncertainty, ~9% from Masterson nominal
    'rwa_noise':     0.20,
    'S_beamwalk':    0.10,
}

# 13. structural dynamics
#   RWA harmonics live in 10-100 Hz range (0.88 to 5.24 x omega_wheel at ~1200 RPM).
#   bus + payload first bending mode is somewhere in 12-18 Hz for this class
#   of spacecraft. if a harmonic lands inside that band, the mount cannot isolate,
#   payload sees direct structural transmission.
#   flag it, do not fix it, this is a risk register item, not a sim parameter.
#   f_struct ~ 15 Hz. rerun when the structural model is available.

f_struct = 15.0               # [Hz], estimated 1st bending mode (12-18 Hz typical)
zeta_struct = 0.005           # space structure damping, 0.5% from reference data, MIL-STD-1540 typical
coupling_thresh = 0.02        # 2% proximity = "close enough to worry". arbitrary; tighten later.

# 14. quaternion library (scalar-last: q = [q1, q2, q3, qs])
q_identity = np.array([0.0, 0.0, 0.0, 1.0])

def quat_mult(q, p):
    qv, qs = q[:3], q[3]
    pv, ps = p[:3], p[3]
    result_v = np.array([
        qs*pv[0] + ps*qv[0] + qv[1]*pv[2] - qv[2]*pv[1],
        qs*pv[1] + ps*qv[1] + qv[2]*pv[0] - qv[0]*pv[2],
        qs*pv[2] + ps*qv[2] + qv[0]*pv[1] - qv[1]*pv[0],
    ])
    result_s = qs*ps - qv[0]*pv[0] - qv[1]*pv[1] - qv[2]*pv[2]
    return np.array([result_v[0], result_v[1], result_v[2], result_s])

def quat_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_normalize(q):
    n = np.linalg.norm(q)
    if n > 1e-15:
        return q / n
    return q_identity.copy()

def quat_to_mrp(q):
    # q to MRP: sigma = qv / (1 + qs). see Markley & Crassidis Sec 2.6.
    #   canonical hemisphere first: q and -q are the same rotation.
    #   pick the one with qs >= 0 so sigma avoids the 180 deg singularity.
    if q[3] < 0:
        q = -q                                 # canonical hemisphere, same rotation, qs >= 0
    denom = 1.0 + q[3]
    if denom < 1e-12:
        d2 = 1.0 - q[3]                        # near 180 deg, qs ~ -1. denom ~ 0. shadow set: sigma' = -qv / (1 - qs)
        if abs(d2) < 1e-12:
            return np.zeros(3)                  # exactly 180 deg, degenerate, return zero MRP (identity)
        return -q[:3] / d2                      # shadow MRP, |sigma| < 1
    sigma = q[:3] / denom
    if np.dot(sigma, sigma) > 1.0:
        sigma = -q[:3] / (1.0 - q[3] + 1e-15)  # |sigma| > 1, swap to shadow set. +eps avoids /0 at qs ~ 1
    return sigma

def skew(v):
    return np.array([
        [ 0.0, -v[2],  v[1]],
        [ v[2],  0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])
