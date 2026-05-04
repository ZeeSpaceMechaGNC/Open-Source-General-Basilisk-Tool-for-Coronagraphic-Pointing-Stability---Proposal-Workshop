# pipeline modules, five in closed-loop order:
#   RwaDisturbance > TwoBodyPlant > AttitudeMekf > HinfCtrl > ContrastModel
# manual data flow via numpy attributes. SysModel base for Reset/UpdateState contract.

import numpy as np
from Basilisk.architecture.sysModel import SysModel
from scipy.signal import place_poles
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params import (
    I_bus, I_bus_inv, I_pay, I_pay_inv, I_bus_diag, I_pay_diag,
    f_iso, zeta_iso,
    sig_fgs, sig_gyro, sig_bias,
    rwa_noise_broadband, rwa_speed_rads, rwa_speed_hz,
    rwa_harmonic_ratios, rwa_harmonic_Ci, n_rwa,
    dt_dyn, dt_fsw, dt_fgs,
    S_beamwalk, bw_tol, C_floor,
    torque_limit, obs_poles_target,
    build_single_axis_plant,
    q_identity, quat_mult, quat_inv, quat_normalize, quat_to_mrp, skew,
)


# 1. RwaDisturbance, Masterson tonal + broadband model
#    50 ms moving-average box filter (~20 Hz cutoff). tonal = C_i*Omega^2 * sin.
#    phases are (n_h, n_wheels, 3), independent per-harmonic, per-wheel, per-axis.
#    tonal divided by sqrt(n_wheels), uncorrelated tones, amplitudes add in power.

class RwaDisturbance(SysModel):
    """Masterson tonal harmonics + filtered broadband."""

    def __init__(self):
        super().__init__()
        self.ModelTag = "rwaDisturbance"
        self.broadband_amplitude = rwa_noise_broadband
        self.wheel_speed_rads = rwa_speed_rads
        self.wheel_speed_hz = rwa_speed_hz
        self.harmonic_ratios = rwa_harmonic_ratios.copy()
        self.harmonic_Ci = rwa_harmonic_Ci.copy()
        self.n_wheels = n_rwa
        self.seed = 42
        self._filt_w = max(5, int(0.05 / dt_dyn))
        self.torque = np.zeros(3)       # output: (3,) [N*m]

    def Reset(self, CurrentSimNanos):
        np.random.seed(self.seed)
        n_h = len(self.harmonic_ratios)
        self._phases = np.random.uniform(0.0, 2.0*np.pi, (n_h, self.n_wheels, 3))
        self._amps = self.harmonic_Ci * self.wheel_speed_rads**2
        self._buf = np.zeros((self._filt_w, 3))
        self.torque = np.zeros(3)

    def UpdateState(self, CurrentSimNanos):
        t = CurrentSimNanos * 1e-9

        # broadband: white noise, 50 ms moving-average box filter
        #   mimics bearing noise datasheet, ~20 Hz cutoff. per-axis independent.
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = np.random.randn(3) * self.broadband_amplitude
        tau_bb = np.mean(self._buf, axis=0)

        tau_tone = np.zeros(3)
        n_h = len(self.harmonic_ratios)
        for i in range(n_h):
            f_i = self.harmonic_ratios[i] * self.wheel_speed_hz
            A_i = self._amps[i]
            arg_t = 2.0 * np.pi * f_i * t
            for w in range(self.n_wheels):
                ph = self._phases[i, w]
                tau_tone = tau_tone + A_i * np.sin(arg_t + ph)

        # RSS sum: uncorrelated tones across N wheels, divide by sqrt(N)
        #   assumes random phase, so amplitudes add in power
        self.torque = tau_bb + tau_tone / np.sqrt(self.n_wheels)


# 2. TwoBodyPlant, 14-state bus + payload, RK4 at 1 kHz

class TwoBodyPlant(SysModel):
    """14-state two-body plant. Quaternion kinematics + Euler. RK4 at 1 kHz."""

    def __init__(self):
        super().__init__()
        self.ModelTag = "twoBodyPlant"
        self.isolation_freq = f_iso
        self.dt = dt_dyn
        self.sensor_seed = 99
        self._last_fgs_nanos = -int(1e18)

        # inputs (set by runner before UpdateState)
        self._tau_dist = np.zeros(3)    # disturbance torque [N*m]
        self._tau_cmd  = np.zeros(3)    # control torque [N*m]

        # outputs (read by runner after UpdateState)
        self.q_pay     = q_identity.copy()    # (4,) truth quaternion
        self.w_pay     = np.zeros(3)          # (3,) truth rate
        self.gyro_meas = np.zeros(3)          # (3,) gyro measurement
        self.fgs_mrp   = np.zeros(3)          # (3,) FGS MRP measurement
        self.fgs_valid = False                # True on FGS samples
        self.fgs_time  = 0.0

    def Reset(self, CurrentSimNanos):
        self._q_bus = q_identity.copy()
        self._w_bus = np.zeros(3)
        self._q_pay = q_identity.copy()
        self._w_pay = np.zeros(3)
        np.random.seed(self.sensor_seed)
        self._last_fgs_nanos = -int(1e18)
        self.fgs_valid = False

        w0 = 2.0 * np.pi * self.isolation_freq
        K_diag = I_pay_diag * w0**2
        self._K_mount = np.diag(K_diag)
        self._C_mount = np.diag(2.0 * zeta_iso * np.sqrt(K_diag * I_pay_diag))

        self.q_pay = q_identity.copy()
        self.w_pay = np.zeros(3)
        self.gyro_meas = np.zeros(3)
        self.fgs_mrp = np.zeros(3)

    def _qdot(self, q, w):
        qv, qs = q[:3], q[3]
        dqv = 0.5 * (qs * w + np.cross(w, qv))
        dqs = -0.5 * np.dot(w, qv)
        return np.array([dqv[0], dqv[1], dqv[2], dqs])

    def _mount_torque(self, q_bus, w_bus, q_pay, w_pay):
        """tau = K*delta_theta + C*delta_omega. returns (3,) [N*m]."""
        dq = quat_mult(q_pay, quat_inv(q_bus))
        if dq[3] < 0.0:
            dq = -dq
        delta_angle = 2.0 * dq[:3]      # small-angle from MRP relation
        delta_rate  = w_pay - w_bus
        return self._K_mount @ delta_angle + self._C_mount @ delta_rate

    def _eom(self, q_bus, w_bus, q_pay, w_pay, u_ctrl, w_dist):
        tau_m = self._mount_torque(q_bus, w_bus, q_pay, w_pay)

        gyro_bus = np.cross(w_bus, I_bus @ w_bus)
        w_dot_bus = I_bus_inv @ (-gyro_bus + tau_m + u_ctrl + w_dist)

        gyro_pay = np.cross(w_pay, I_pay @ w_pay)
        w_dot_pay = I_pay_inv @ (-gyro_pay - tau_m)

        dq_bus = self._qdot(q_bus, w_bus)
        dq_pay = self._qdot(q_pay, w_pay)
        return dq_bus, w_dot_bus, dq_pay, w_dot_pay

    def _rk4_step(self, u_ctrl, w_dist):
        h = self.dt
        qb, wb, qp, wp = self._q_bus, self._w_bus, self._q_pay, self._w_pay

        k1b, k1w, k1p, k1wp = self._eom(qb, wb, qp, wp, u_ctrl, w_dist)

        qb2 = quat_normalize(qb + 0.5*h*k1b);  wb2 = wb + 0.5*h*k1w
        qp2 = quat_normalize(qp + 0.5*h*k1p);  wp2 = wp + 0.5*h*k1wp
        k2b, k2w, k2p, k2wp = self._eom(qb2, wb2, qp2, wp2, u_ctrl, w_dist)

        qb3 = quat_normalize(qb + 0.5*h*k2b);  wb3 = wb + 0.5*h*k2w
        qp3 = quat_normalize(qp + 0.5*h*k2p);  wp3 = wp + 0.5*h*k2wp
        k3b, k3w, k3p, k3wp = self._eom(qb3, wb3, qp3, wp3, u_ctrl, w_dist)

        qb4 = quat_normalize(qb + h*k3b);      wb4 = wb + h*k3w
        qp4 = quat_normalize(qp + h*k3p);      wp4 = wp + h*k3wp
        k4b, k4w, k4p, k4wp = self._eom(qb4, wb4, qp4, wp4, u_ctrl, w_dist)

        self._q_bus = quat_normalize(qb + (h/6.0)*(k1b + 2.0*k2b + 2.0*k3b + k4b))
        self._w_bus = wb         + (h/6.0)*(k1w + 2.0*k2w + 2.0*k3w + k4w)
        self._q_pay = quat_normalize(qp + (h/6.0)*(k1p + 2.0*k2p + 2.0*k3p + k4p))
        self._w_pay = wp         + (h/6.0)*(k1wp+ 2.0*k2wp+ 2.0*k3wp+ k4wp)

    def UpdateState(self, CurrentSimNanos):
        self._rk4_step(self._tau_cmd, self._tau_dist)

        self.q_pay = self._q_pay.copy()
        self.w_pay = self._w_pay.copy()

        # gyro output: true rate + ARW noise (per-sample: sig_gyro * sqrt(dt))
        self.gyro_meas = self._w_pay + np.random.randn(3) * sig_gyro * np.sqrt(self.dt)

        # FGS output: attitude + noise, only at 10 Hz
        self.fgs_valid = False
        dt_nanos = int(dt_fgs * 1e9)
        if (CurrentSimNanos - self._last_fgs_nanos) >= dt_nanos:
            sigma_pay = quat_to_mrp(self._q_pay)
            self.fgs_mrp = sigma_pay + np.random.randn(3) * sig_fgs
            self.fgs_valid = True
            self.fgs_time = CurrentSimNanos * 1e-9
            self._last_fgs_nanos = CurrentSimNanos


# 3. AttitudeMekf, 6-state multiplicative EKF (Markley & Crassidis)

class AttitudeMekf(SysModel):
    """6-state multiplicative EKF. Joseph form. Gyro + FGS."""

    def __init__(self):
        super().__init__()
        self.ModelTag = "mekfEstimator"
        self.dt = dt_fsw

        # inputs (set by runner before UpdateState)
        self.gyro_meas = np.zeros(3)
        self.fgs_mrp   = np.zeros(3)
        self.fgs_valid  = False

        # outputs (read by runner after UpdateState)
        self.q_nav = q_identity.copy()
        self.w_nav = np.zeros(3)

    def Reset(self, CurrentSimNanos):
        self._q_hat = q_identity.copy()
        self._bias = np.zeros(3)
        self._P = np.diag([1e-6, 1e-6, 1e-6, 1e-12, 1e-12, 1e-12])

        # Process noise Q (Farrenkopf model, M&C Eq 6.93)
        sv2 = sig_gyro**2
        su2 = sig_bias**2
        dt  = self.dt
        q11 = sv2 * dt + su2 * dt**3 / 3.0
        q12 = -su2 * dt**2 / 2.0
        q22 = su2 * dt

        self._Q = np.zeros((6, 6))
        for j in range(3):
            self._Q[j,   j]   = q11
            self._Q[j+3, j+3] = q22
            self._Q[j,   j+3] = q12
            self._Q[j+3, j]   = q12

        self._R = np.eye(3) * sig_fgs**2
        self._last_fgs_time = -1

        self.q_nav = q_identity.copy()
        self.w_nav = np.zeros(3)

    def _vec_to_quat(self, vec):
        ang = np.linalg.norm(vec)
        if ang > 1e-14:
            ax = vec / ang
            ha = ang / 2.0
            return np.array([ax[0]*np.sin(ha), ax[1]*np.sin(ha), ax[2]*np.sin(ha), np.cos(ha)])
        return np.array([0.0, 0.0, 0.0, 1.0])

    def time_update(self, w_gyro):
        w_corr = w_gyro - self._bias
        dq = self._vec_to_quat(w_corr * self.dt)
        self._q_hat = quat_normalize(quat_mult(dq, self._q_hat))

        F = np.zeros((6, 6))
        F[0:3, 0:3] = -skew(w_corr)
        F[0:3, 3:6] = -np.eye(3)
        Phi = np.eye(6) + F * self.dt
        self._P = Phi @ self._P @ Phi.T + self._Q
        self._P = 0.5 * (self._P + self._P.T)

    def measurement_update(self, sigma_meas):
        dz = sigma_meas - quat_to_mrp(self._q_hat)
        H = np.zeros((3, 6))
        H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 2] = 1.0

        S = H @ self._P @ H.T + self._R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self._P @ H.T @ S_inv
        dx = K @ dz

        dq = self._vec_to_quat(dx[0:3])
        self._q_hat = quat_normalize(quat_mult(dq, self._q_hat))
        self._bias = self._bias + dx[3:6]

        # Joseph form: P = (I-KH)P(I-KH)^T + KRK^T
        #   standard form (I-KH)P loses symmetry under roundoff,
        #   P drifts non-PSD, filter diverges. Joseph guarantees PSD.
        #   costs ~3x matmul, worth it for 1000 s sims.
        # NOTE: check Joseph PSD holds over 1000 s sims with worst-case noise draws.
        IKH = np.eye(6) - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ self._R @ K.T
        self._P = 0.5 * (self._P + self._P.T)

    def UpdateState(self, CurrentSimNanos):
        self.time_update(self.gyro_meas)

        if self.fgs_valid:
            self.measurement_update(self.fgs_mrp)

        self.q_nav = self._q_hat.copy()
        self.w_nav = self.gyro_meas - self._bias


# 4. HinfCtrl, per-axis state-feedback + Luenberger observer

class HinfCtrl(SysModel):
    """Per-axis H-inf + observer. K set externally by runner."""

    def __init__(self):
        super().__init__()
        self.ModelTag = "hinfController"
        self.K = np.zeros((3, 4))
        self.torque_limit = torque_limit
        self.isolation_freq = f_iso
        self.dt = dt_fsw
        self._C_obs = np.array([[0.0,0.0,1.0,0.0],
                                [0.0,0.0,0.0,1.0]])

        # inputs
        self.sigma_nav = np.zeros(3)
        self.omega_nav = np.zeros(3)

        # output
        self.cmd_torque = np.zeros(3)    # (3,) [N*m]

    def Reset(self, CurrentSimNanos):
        self._x_hat = np.zeros((3, 4))
        self._A = []
        self._Bc = []
        self._L = []

        for i in range(3):
            Ib = I_bus_diag[i]
            Ip = I_pay_diag[i]
            A_i, B_i = build_single_axis_plant(Ib, Ip, self.isolation_freq)
            Bc_i = B_i[:, 0]
            self._A.append(A_i)
            self._Bc.append(Bc_i)

            try:
                L_i = place_poles(A_i.T, self._C_obs.T, obs_poles_target).gain_matrix.T
            except Exception:
                # place_poles failed, hand-tuned L from f_iso=1 Hz nominal run
                # TODO: regenerate per axis on failure
                L_i = np.array([
                    [-30.0,  15.0],
                    [-400.0, 100.0],
                    [ 15.0,   1.0],
                    [-100.0,  50.0],
                ])
            self._L.append(L_i)

        self.cmd_torque = np.zeros(3)

    def UpdateState(self, CurrentSimNanos):
        u_cmd = np.zeros(3)
        for i in range(3):
            y_i = np.array([self.sigma_nav[i], self.omega_nav[i]])
            x_i = self._x_hat[i]
            innov = y_i - self._C_obs @ x_i

            u_i = -(self.K[i] @ x_i)
            if u_i > self.torque_limit:
                u_i = self.torque_limit
            if u_i < -self.torque_limit:
                u_i = -self.torque_limit
            u_cmd[i] = u_i

            x_dot = self._A[i] @ x_i + self._Bc[i] * u_i + self._L[i] @ innov
            self._x_hat[i] = x_i + x_dot * self.dt

        self.cmd_torque = u_cmd


# 5. ContrastModel, beamwalk-to-contrast, truth-side

class ContrastModel(SysModel):
    """Beamwalk-to-contrast degradation. Truth-side. Rolling RMS."""

    def __init__(self):
        super().__init__()
        self.ModelTag = "contrastModel"
        self.window_size = 100

        # input
        self.sigma_pay = np.zeros(3)

        # output
        self.contrast = 0.0
        self.beamwalk = 0.0
        self.theta_los = 0.0

    def Reset(self, CurrentSimNanos):
        self._hx = np.zeros(self.window_size)
        self._hy = np.zeros(self.window_size)
        self._idx = 0
        self._count = 0
        self.contrast = 0.0
        self.beamwalk = 0.0
        self.theta_los = 0.0

    def UpdateState(self, CurrentSimNanos):
        s = self.sigma_pay
        self._hx[self._idx] = s[0]
        self._hy[self._idx] = s[1]
        self._idx = (self._idx + 1) % self.window_size
        self._count = self._count + 1

        theta_los = np.sqrt(s[0]*s[0] + s[1]*s[1])

        if self._count >= 10:
            n = min(self._count, self.window_size)
            sx = np.std(self._hx[:n])
            sy = np.std(self._hy[:n])
            theta_rms = np.sqrt(sx*sx + sy*sy)
        else:
            theta_rms = theta_los

        self.theta_los = theta_los
        self.beamwalk = S_beamwalk * theta_rms
        self.contrast = C_floor * (1.0 + (self.beamwalk / bw_tol)**2)
