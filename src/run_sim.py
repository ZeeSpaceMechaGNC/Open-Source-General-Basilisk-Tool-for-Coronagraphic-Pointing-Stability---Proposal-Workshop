# pointing simulator, fine-pointing stage
# rough runner, expect rewiring as the FSW grows. pulls params, builds manual
# multi-rate loop, runs, reduces telemetry, prints v&v, plots. dashboard.png,
# sweep.png, Monte Carlo.png go to SCRIPT_DIR.
# rerun with different f_iso? call run_single(freq_iso=...) from python.

import numpy as np
import matplotlib
matplotlib.rcParams.update({'figure.dpi': 150, 'font.size': 9})
import matplotlib.pyplot as plt
import time
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import scipy.linalg
from scipy.signal import place_poles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params import (
    MAS2RAD, RAD2MAS,
    D_aperture, wavelength,
    I_bus_diag, I_pay_diag,
    f_iso, zeta_iso, w0_iso, K_iso_diag, C_iso_diag,
    build_single_axis_plant,
    sig_fgs, sig_gyro, sig_bias, fgs_rate,
    rwa_noise_broadband, rwa_speed_rpm, rwa_speed_hz, rwa_speed_rads,
    rwa_harmonic_ratios, rwa_harmonic_Ci, n_rwa, rwa_amplitude_scale,
    ctrl_bw, omega_ctrl, Q_hinf_diag, R_hinf,
    hinf_gamma_lo, hinf_gamma_hi, hinf_gamma_tol, hinf_gamma_iters, hinf_gamma_safety,
    ctrl_poles_target, obs_poles_target,
    torque_limit,
    S_beamwalk, bw_tol, C_floor,
    dt_dyn, dt_fsw, dt_fgs, T_sim,
    jitter_req_mas, contrast_req, bw_tol_pm,
    est_err_req_mas, gain_margin_req_dB, phase_margin_req_deg,
    mc_nruns, mc_dispersions,
    f_struct, zeta_struct, coupling_thresh,
    q_identity, quat_mult, quat_inv, quat_normalize, quat_to_mrp, skew,
)
from modules import (RwaDisturbance, TwoBodyPlant, AttitudeMekf,
                     HinfCtrl, ContrastModel)


# 1. H-infinity synthesis, per-axis via ARE bracket on gamma

def _are_at_gamma(A, Bc, Bd, Q, R, gamma):
    n = A.shape[0]
    S = (1.0/R)*(Bc @ Bc.T) - (1.0/(gamma*gamma))*(Bd @ Bd.T) + 1e-12*np.eye(n)
    if np.min(np.linalg.eigvalsh(S)) <= 0.0:
        return None, False
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return None, False
    try:
        X = scipy.linalg.solve_continuous_are(A, L, Q, np.eye(n))
    except Exception:
        return None, False
    if np.min(np.linalg.eigvalsh(X)) < -1e-10:
        return None, False
    return (1.0/R) * (Bc.T @ X), True


def _bracket_gamma_min(A, Bc, Bd, Q, R, lo, hi, tol, iters):
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        _, ok = _are_at_gamma(A, Bc, Bd, Q, R, mid)
        hi = mid if ok else hi
        lo = lo if ok else mid
        if hi - lo < tol:
            break
    return hi


def synth_hinf(f_iso):
    Q = np.diag(Q_hinf_diag)
    R = R_hinf
    K = np.zeros((3, 4))
    A_list, B_list = [], []
    gmin, gused = [], []

    for ax in range(3):
        A_i, B_i = build_single_axis_plant(I_bus_diag[ax], I_pay_diag[ax], f_iso)
        Bc = B_i[:, 0:1]
        Bd = B_i[:, 1:2]

        gamma_min = _bracket_gamma_min(A_i, Bc, Bd, Q, R,
                                       hinf_gamma_lo, hinf_gamma_hi,
                                       hinf_gamma_tol, hinf_gamma_iters)

        g = max(2.0, gamma_min * hinf_gamma_safety)
        K_ax, ok = _are_at_gamma(A_i, Bc, Bd, Q, R, g)
        if not ok:
            K_ax, ok = _are_at_gamma(A_i, Bc, Bd, Q, R, 50.0)
            g = 50.0
        if not ok:
            raise RuntimeError(
                "hinf synthesis failed on axis %d (gamma=%.1f)" % (ax, g))

        gmin.append(gamma_min);  gused.append(g)
        K[ax, :] = K_ax.flatten()
        A_list.append(A_i);  B_list.append(B_i)

    return K, {'method': 'hinf', 'A_list': A_list, 'B_list': B_list,
               'gamma_min': gmin, 'gamma_used': gused}


def synth_place(f_iso):
    K = np.zeros((3, 4))
    A_list, B_list = [], []
    for ax in range(3):
        A_i, B_i = build_single_axis_plant(
            I_bus_diag[ax], I_pay_diag[ax], f_iso)
        Bc = B_i[:, 0:1]
        res = place_poles(A_i, Bc, ctrl_poles_target)
        K[ax, :] = res.gain_matrix.flatten()
        A_list.append(A_i);  B_list.append(B_i)
    return K, {'method': 'place', 'A_list': A_list, 'B_list': B_list}


# 2. Loop margins from L(jw)

def _crossings(x, y, target):
    out = []
    for i in range(len(y)-1):
        if y[i] >= target and y[i+1] < target:
            t = (target - y[i]) / (y[i+1] - y[i])
            out.append(x[i] + t*(x[i+1] - x[i]))
    return out


def _per_axis_response(A, B, K_row):
    freqs = np.logspace(-3, 3, 2000)
    Bc = B[:, 0:1]
    Bd = B[:, 1:2]
    Cp = np.array([[0, 0, 1, 0]])
    K = K_row.reshape(1, 4)
    Acl = A - Bc @ K
    I = np.eye(4)
    n = len(freqs)
    H_ol, H_cl, L = np.zeros(n, complex), np.zeros(n, complex), np.zeros(n, complex)
    for j in range(n):
        s = 2j * np.pi * freqs[j]
        G    = np.linalg.inv(s*I - A)
        G_cl = np.linalg.inv(s*I - Acl)
        H_ol[j] = (Cp @ G @ Bd)[0, 0]
        H_cl[j] = (Cp @ G_cl @ Bd)[0, 0]
        L[j]    = (K @ G @ Bc)[0, 0]
    return freqs, H_ol, H_cl, L


def _margins(freqs, L):
    mag_db = 20*np.log10(np.abs(L) + 1e-30)
    phase_deg = np.unwrap(np.angle(L)) * 180/np.pi

    x_freqs = _crossings(freqs, mag_db, 0.0)
    pm = None
    if len(x_freqs):
        pm = np.min(180 + np.interp(x_freqs, freqs, phase_deg))

    gm_vals = []
    for tgt in [-180, -540]:
        for xf in _crossings(freqs, phase_deg, tgt):
            gm_vals.append(-np.interp(xf, freqs, mag_db))
    gm = min(gm_vals) if gm_vals else None

    return gm, pm


def _axis_margins(ax_idx, sinfo, K):
    A = sinfo['A_list'][ax_idx]
    B = sinfo['B_list'][ax_idx]
    fr, hol, hcl, Lv = _per_axis_response(A, B, K[ax_idx])
    gm, pm = _margins(fr, Lv)
    return gm, pm, fr, hol, hcl, Lv


# 3. Structural coupling check

def _check_structural_risk():
    harm_freqs = [h * rwa_speed_hz for h in rwa_harmonic_ratios]
    deviations = [abs(f - f_struct)/f_struct for f in harm_freqs]
    idx_min = np.argmin(deviations)
    harm_ratio = rwa_harmonic_ratios[idx_min]
    f_closest = harm_freqs[idx_min]
    delta = deviations[idx_min]
    at_risk = delta <= coupling_thresh
    return {'harmonic': harm_ratio, 'freq_hz': f_closest, 'delta_pct': delta*100,
            'at_risk': at_risk, 'zeta': zeta_struct,
            'all_freqs': [(rwa_harmonic_ratios[i], harm_freqs[i])
                          for i in range(len(harm_freqs))]}


# 4. Telemetry reduction

def _reduce(t_dyn, sig_tru, t_fsw, sig_est, t_opt, contrast, bw_vals):
    c_vals = contrast

    mask_dyn = t_dyn > 0.5
    mask_fsw = t_fsw > 0.5
    mask_opt = t_opt > 0.5

    jx = np.std(sig_tru[mask_dyn, 0]) * RAD2MAS
    jy = np.std(sig_tru[mask_dyn, 1]) * RAD2MAS
    jz = np.std(sig_tru[mask_dyn, 2]) * RAD2MAS
    j_rss = np.sqrt(jx*jx + jy*jy)

    ex = np.interp(t_fsw, t_dyn, sig_tru[:, 0]) - sig_est[:, 0]
    ey = np.interp(t_fsw, t_dyn, sig_tru[:, 1]) - sig_est[:, 1]
    ee_rss = np.sqrt(np.std(ex[mask_fsw])**2 + np.std(ey[mask_fsw])**2) * RAD2MAS

    bw_pm  = np.mean(bw_vals[mask_opt]) * 1e12
    C_mean = np.mean(c_vals[mask_opt])

    return jx, jy, jz, j_rss, ee_rss, bw_pm, C_mean


# 5. Print helpers

def _print_header(method, freq_iso, Tsim, sinfo, K):
    print("\n  %s  freq_iso=%.1f Hz  %.1f s" % (method, freq_iso, Tsim))
    if sinfo['method'] == 'hinf':
        for ax in range(3):
            nm = ['X','Y','Z'][ax]
            print("  %s: gmin=%.3f  gused=%.3f"
                  % (nm, sinfo['gamma_min'][ax], sinfo['gamma_used'][ax]))
        for ax in range(3):
            nm = ['X','Y','Z'][ax]
            print("  K_%s = [%.2f, %.0f, %.2f, %.1f]"
                  % (nm, K[ax,0], K[ax,1], K[ax,2], K[ax,3]))


def _print_metrics(jx, jy, jz, j_rss, ee_rss, bw_pm, C_mean, margins):
    print("  jitter X/Y/Z: %.3e / %.3e / %.3e mrad  rss: %.3e"
          % (jx / RAD2MAS * 1e3, jy / RAD2MAS * 1e3, jz / RAD2MAS * 1e3, j_rss / RAD2MAS * 1e3))
    print("  est err rss: %.3e mrad  beamwalk: %.0f pm  contrast: %.2e"
          % (ee_rss / RAD2MAS * 1e3, bw_pm, C_mean))
    for ax in range(3):
        m = margins[['X','Y','Z'][ax]]
        gs = "inf" if m['GM_dB'] is None else "%.1f dB" % m['GM_dB']
        ps = "N/A" if m['PM_deg'] is None else "%.1f deg" % m['PM_deg']
        print("  margins %s: gm=%s, pm=%s" % (['X','Y','Z'][ax], gs, ps))


def _print_error_budget(jx, jy, jz, j_rss, ee_rss, bw_pm, C_mean, struct):
    mech_contrast = max(C_mean - C_floor, 0.0)
    print("\n  budget")
    print("  contrast:      %.2e  /  req < %.0e  (mech %.2e, floor %.2e)"
          % (C_mean, contrast_req, mech_contrast, C_floor))
    print("  beamwalk:      %.0f pm  /  req < %.0f pm"
          % (bw_pm, bw_tol_pm))
    print("  jitter rss:    %.2f mas  /  req < %.1f mas  (X %.2f, Y %.2f, Z %.2f)"
          % (j_rss, jitter_req_mas, jx, jy, jz))
    print("  est err rss:   %.2f mas  /  req < %.1f mas"
          % (ee_rss, est_err_req_mas))
    if struct['at_risk']:
        print("  struct:        risk (harmonic %.2g @ %.1f Hz, %.1f%% from %.1f Hz, zeta=%.3f)"
              % (struct['harmonic'], struct['freq_hz'],
                 struct['delta_pct'], f_struct, zeta_struct))
    else:
        print("  struct:        ok (%.2g x omega @ %.1f Hz, %.1f%% from %.1f Hz)"
              % (struct['harmonic'], struct['freq_hz'],
                 struct['delta_pct'], f_struct))


# 6. Run single simulation (manual multi-rate loop)

def run_single(freq_iso=None, Tsim=None, show_plot=True,
               seed=None, mc_params=None, quiet=False, method='hinf'):

    freq_iso = freq_iso if freq_iso is not None else f_iso
    Tsim = Tsim if Tsim is not None else T_sim

    if method == 'hinf':
        K, sinfo = synth_hinf(freq_iso)
    else:
        K, sinfo = synth_place(freq_iso)

    if not quiet:
        _print_header(method, freq_iso, Tsim, sinfo, K)

    # build modules
    rwa   = RwaDisturbance()
    plant = TwoBodyPlant()
    mekf  = AttitudeMekf()
    ctrl  = HinfCtrl()
    cmod  = ContrastModel()

    plant.isolation_freq = freq_iso
    ctrl.K = K
    ctrl.isolation_freq = freq_iso

    if seed is not None:
        rwa.seed = seed
        plant.sensor_seed = seed + 1000
    if mc_params is not None:
        if 'rwa_noise' in mc_params:
            rwa.broadband_amplitude = mc_params['rwa_noise']
        if 'rwa_speed' in mc_params:
            rwa.wheel_speed_rads = mc_params['rwa_speed']
            rwa.wheel_speed_hz  = mc_params['rwa_speed'] / (2*np.pi)

    rwa.Reset(0)
    plant.Reset(0)
    mekf.Reset(0)
    ctrl.Reset(0)
    cmod.Reset(0)

    # manual multi-rate loop
    n_dyn  = int(Tsim / dt_dyn) + 1
    n_fsw  = int(dt_fsw / dt_dyn)         # 10
    n_fgs  = int(dt_fgs / dt_dyn)         # 100
    n_opt  = n_fsw                         # contrast at FSW rate

    t_dyn    = np.zeros(n_dyn)
    sig_tru  = np.zeros((n_dyn, 3))
    n_nav    = n_dyn // n_fsw + 1
    t_nav    = np.zeros(n_nav)
    sig_est  = np.zeros((n_nav, 3))
    n_contrast = n_dyn // n_opt + 1
    t_opt      = np.zeros(n_contrast)
    c_vals     = np.zeros(n_contrast)
    bw_vals    = np.zeros(n_contrast)
    idx_nav = 0
    idx_opt = 0

    for step in range(n_dyn):
        sim_time = step * dt_dyn
        sim_ns = int(sim_time * 1e9)

        # 1 kHz: disturbance + plant
        rwa.UpdateState(sim_ns)
        plant._tau_dist = rwa.torque.copy()
        plant._tau_cmd  = ctrl.cmd_torque.copy()
        plant.UpdateState(sim_ns)

        # recording at dynamics rate
        t_dyn[step]   = sim_time
        sigma_pay = quat_to_mrp(plant.q_pay)
        sig_tru[step] = sigma_pay

        # 100 Hz: MEKF + control
        if step % n_fsw == 0:
            mekf.gyro_meas = plant.gyro_meas.copy()
            mekf.fgs_mrp   = plant.fgs_mrp.copy()
            mekf.fgs_valid = plant.fgs_valid
            mekf.UpdateState(sim_ns)

            ctrl.sigma_nav = quat_to_mrp(mekf.q_nav)
            ctrl.omega_nav = mekf.w_nav.copy()
            ctrl.UpdateState(sim_ns)

            t_nav[idx_nav]   = sim_time
            sig_est[idx_nav] = quat_to_mrp(mekf.q_nav)
            idx_nav = idx_nav + 1

        # 100 Hz: contrast model (same rate as FSW)
        if step % n_opt == 0:
            cmod.sigma_pay = sigma_pay.copy()
            cmod.UpdateState(sim_ns)
            t_opt[idx_opt]   = sim_time
            c_vals[idx_opt]  = cmod.contrast
            bw_vals[idx_opt] = cmod.beamwalk
            idx_opt = idx_opt + 1

    # trim to filled
    t_nav = t_nav[:idx_nav]
    sig_est = sig_est[:idx_nav]
    t_opt = t_opt[:idx_opt]
    c_vals = c_vals[:idx_opt]
    bw_vals = bw_vals[:idx_opt]

    # reduce
    jx, jy, jz, j_rss, ee_rss, bw_pm, C_mean = _reduce(
        t_dyn, sig_tru, t_nav, sig_est, t_opt, c_vals, bw_vals)

    # margins
    margins = {}
    for ax in range(3):
        nm = ['X','Y','Z'][ax]
        gm, pm, fr, hol, hcl, Lv = _axis_margins(ax, sinfo, K)
        margins[nm] = {'GM_dB': gm, 'PM_deg': pm,
                       'freqs': fr, 'H_ol': hol, 'H_cl': hcl, 'L': Lv}

    if not quiet:
        _print_metrics(jx, jy, jz, j_rss, ee_rss, bw_pm, C_mean, margins)
        struct_risk = _check_structural_risk()
        _print_error_budget(jx, jy, jz, j_rss, ee_rss, bw_pm, C_mean, struct_risk)

    res = {
        'f_iso': freq_iso, 't_dyn': t_dyn, 'sig_tru': sig_tru,
        't_fsw': t_nav, 'sig_est': sig_est, 't_opt': t_opt,
        'c_vals': c_vals, 'bw_vals': bw_vals,
        'jx': jx, 'jy': jy, 'jz': jz, 'j_rss': j_rss,
        'ee_rss': ee_rss, 'bw_pm': bw_pm, 'C_mean': C_mean,
        'K': K, 'sinfo': sinfo, 'margins': margins, 'method': method,
    }
    if show_plot:
        _dashboard(res)
    return res


# 7. Dashboard plot

def save_fig(name):
    path = os.path.join(SCRIPT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print("  saved: %s" % path)
    plt.close()


def _dashboard(r):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('pointing sim', fontsize=13, fontweight='bold')

    # top-left: disturbance rejection bode, open vs closed-loop, X axis
    mx = r['margins']['X']
    ax[0,0].loglog(mx['freqs'], np.abs(mx['H_ol']), 'b', lw=1, label='open')
    ax[0,0].loglog(mx['freqs'], np.abs(mx['H_cl']), 'g', lw=1, label='closed')
    ax[0,0].axvline(r['f_iso'], ls='--', c='orange', alpha=0.7)
    ax[0,0].set(xlabel='freq [Hz]', ylabel='|H| [rad / N*m]',
                title='disturbance -> LOS (X)')
    ax[0,0].legend(fontsize=7);  ax[0,0].grid(True, alpha=0.2, which='both')

    # top-right: jitter timeseries, true attitude in mas, all 3 axes
    skip = max(1, len(r['t_dyn']) // 3000)
    for i, (c, lbl) in enumerate(zip(['b','r','g'], ['X','Y','Z'])):
        ax[0,1].plot(r['t_dyn'][::skip],
                     r['sig_tru'][::skip, i] * RAD2MAS,
                     c, lw=0.3, label=lbl)
    ax[0,1].axhline( jitter_req_mas, ls='--', c='r')
    ax[0,1].axhline(-jitter_req_mas, ls='--', c='r')
    ax[0,1].set(xlabel='time [s]', ylabel='mas', title='LOS jitter')
    ax[0,1].legend(fontsize=7);  ax[0,1].grid(True, alpha=0.2)

    # bottom-left: raw contrast rolling rms on log scale
    ax[1,0].semilogy(r['t_opt'], r['c_vals'], 'C3', lw=0.4)
    ax[1,0].axhline(contrast_req, ls='--', c='r', lw=1.2, label='req')
    ax[1,0].axhline(C_floor, ls=':', c='gray', lw=0.8, label='floor')
    ax[1,0].set(xlabel='time [s]', ylabel='contrast', title='raw contrast')
    ax[1,0].legend(fontsize=7);  ax[1,0].grid(True, alpha=0.2)

    # bottom-right: nichols chart, phase vs mag for all axes
    for i, (c, lbl) in enumerate(zip(['b','r','g'], ['X','Y','Z'])):
        Lv = r['margins'][lbl]['L']
        ph = np.unwrap(np.angle(Lv)) * 180/np.pi
        mg = 20*np.log10(np.abs(Lv) + 1e-30)
        ax[1,1].plot(ph, mg, c, lw=0.8, label=lbl)
    ax[1,1].axhline(0, c='k', lw=0.5);  ax[1,1].axvline(-180, c='k', lw=0.5)
    ax[1,1].set(xlabel='phase [deg]', ylabel='|L| [dB]',
                xlim=(-360, 0), ylim=(-60, 40), title='nichols')
    ax[1,1].legend(fontsize=7);  ax[1,1].grid(True, alpha=0.3)

    save_fig('dashboard.png')


# 8. Frequency sweep

def run_sweep(method='hinf'):
    flist = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    print("\n  isolation frequency sweep")
    sw = []
    for freq_iso in flist:
        r = run_single(freq_iso=freq_iso, Tsim=3.0, show_plot=False,
                       quiet=True, method=method)
        print("  f_iso=%.1f: jitter=%.2f mas  C=%.2e" % (freq_iso, r['j_rss'], r['C_mean']))
        sw.append(r)
    _plot_sweep(sw, flist)
    return sw


def _plot_sweep(sw, flist):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('f_iso sweep', fontsize=12, fontweight='bold')
    n = len(flist)
    contrasts = [sw[i]['C_mean'] for i in range(n)]
    jitters   = [sw[i]['j_rss'] for i in range(n)]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, n))
    for i in range(n):
        ax.bar(i, contrasts[i], color=colors[i], edgecolor='k', lw=0.7)
        ax.text(i, contrasts[i]*1.6, '%.1f' % jitters[i],
                ha='center', fontsize=6.5)
    ax.axhline(contrast_req, ls='--', c='r', lw=1.2, label='req')
    ax.set_yscale('log')
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(f) for f in flist])
    ax.set(xlabel='f_iso [Hz]', ylabel='contrast', title='contrast vs isolation')
    ax.legend(fontsize=7)
    save_fig('sweep.png')


# 9. Monte carlo

def run_montecarlo(N=None, method='hinf'):
    N = N or mc_nruns
    np.random.seed(2024)
    print("\n  monte carlo: %d runs" % N)

    keys = ['j_rss','jx','jy','jz','ee_rss','contrast','bw_pm']
    mc = {k: np.zeros(N) for k in keys}

    t0 = time.time()
    n_chunk = max(1, N//10)
    for trial in range(N):
        freq_iso_disp = max(0.05, f_iso*(1 + mc_dispersions['f_iso']*np.random.randn()))
        rwa_noise_disp = max(1e-4, rwa_noise_broadband*(1 + mc_dispersions['rwa_noise']*np.random.randn()))
        rwa_speed_disp = max(10.0, rwa_speed_rads*(1 + mc_dispersions['rwa_speed_rpm']*np.random.randn()))

        try:
            r = run_single(freq_iso=freq_iso_disp, Tsim=3.0, show_plot=False,
                           seed=trial, quiet=True, method=method,
                           mc_params={'rwa_noise': rwa_noise_disp, 'rwa_speed': rwa_speed_disp})
            mc['j_rss'][trial]    = r['j_rss']
            mc['jx'][trial]       = r['jx']
            mc['jy'][trial]       = r['jy']
            mc['jz'][trial]       = r['jz']
            mc['ee_rss'][trial]   = r['ee_rss']
            mc['contrast'][trial] = r['C_mean']
            mc['bw_pm'][trial]    = r['bw_pm']
        except Exception as e:
            print("  WARN trial %d: %s" % (trial, str(e)))
            mc['j_rss'][trial] = np.nan

        if (trial+1) % n_chunk == 0:
            dt = max(1e-6, time.time()-t0)
            print("  [%d/%d] j=%.2f  C=%.2e  (%.1f run/s)"
                  % (trial+1, N, mc['j_rss'][trial], mc['contrast'][trial], (trial+1)/dt))

    ok = ~np.isnan(mc['j_rss'])
    n_valid = np.sum(ok)
    pct_jitter  = 100*np.sum(mc['j_rss'][ok] < jitter_req_mas)/n_valid
    pct_contrast = 100*np.sum(mc['contrast'][ok] < contrast_req)/n_valid
    print("\n  %d valid  jitter: %.1f%%  contrast: %.1f%%" % (n_valid, pct_jitter, pct_contrast))
    _plot_mc(mc, n_valid)
    return mc


def _plot_mc(mc, n_valid):
    ok = ~np.isnan(mc['j_rss'])
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('mc (%d)' % n_valid, fontsize=13, fontweight='bold')

    ax[0,0].hist(mc['j_rss'][ok], bins=40, color='C0', edgecolor='k', lw=0.5, alpha=0.8)
    ax[0,0].axvline(jitter_req_mas, ls='--', c='r', lw=2, label='req=%.1f' % jitter_req_mas)
    ax[0,0].axvline(np.mean(mc['j_rss'][ok]), c='orange', lw=1.5)
    ax[0,0].set(xlabel='jitter rss [mas]', ylabel='count', title='jitter')
    ax[0,0].legend(fontsize=8)

    ax[0,1].hist(mc['contrast'][ok], bins=40, color='C3', edgecolor='k', lw=0.5, alpha=0.8)
    ax[0,1].axvline(contrast_req, ls='--', c='r', lw=2, label='req')
    ax[0,1].set(xlabel='contrast', ylabel='count', title='contrast')
    ax[0,1].legend(fontsize=8)

    ax[1,0].scatter(mc['j_rss'][ok], mc['contrast'][ok], s=6, alpha=0.5, c='C2')
    ax[1,0].axvline(jitter_req_mas, ls='--', c='r')
    ax[1,0].axhline(contrast_req, ls='--', c='r')
    ax[1,0].set(xlabel='jitter [mas]', ylabel='contrast',
                title='jitter vs contrast', yscale='log')

    box_data = [mc['jx'][ok], mc['jy'][ok], mc['jz'][ok], mc['j_rss'][ok]]
    bp = ax[1,1].boxplot(box_data, labels=['X','Y','Z','RSS'], patch_artist=True)
    for k, col in enumerate(['C0','C1','C4','C2']):
        bp['boxes'][k].set_facecolor(col);  bp['boxes'][k].set_alpha(0.5)
    ax[1,1].axhline(jitter_req_mas, ls='--', c='r')
    ax[1,1].set(ylabel='jitter [mas]', title='per-axis')

    save_fig('Monte Carlo.png')


# 10. Verification

def print_vv(baseline, mc=None):
    b = baseline
    checks = []

    checks.append(('los jitter rss', '< %.1f mas' % jitter_req_mas,
                   '%.2f mas' % b['j_rss'],
                   b['j_rss'] < jitter_req_mas))
    checks.append(('estimation error', '< %.1f mas' % est_err_req_mas,
                   '%.2f mas' % b['ee_rss'],
                   b['ee_rss'] < est_err_req_mas))
    checks.append(('contrast', '< %.0e' % contrast_req,
                   '%.2e' % b['C_mean'],
                   b['C_mean'] < contrast_req))
    checks.append(('beamwalk', '< %.0f pm' % bw_tol_pm,
                   '%.0f pm' % b['bw_pm'],
                   b['bw_pm'] < bw_tol_pm))

    for ax_n in ['X','Y','Z']:
        m = b['margins'][ax_n]
        gm, pm = m['GM_dB'], m['PM_deg']

        gm_str = 'inf' if gm is None else '%.1f dB' % gm
        gm_ok  = (gm is None) or (gm > gain_margin_req_dB)
        checks.append(('gain margin %s' % ax_n,
                       '> %.0f dB' % gain_margin_req_dB, gm_str, gm_ok))

        pm_str = 'N/A' if pm is None else '%.1f deg' % pm
        pm_ok  = (pm is not None) and (pm > phase_margin_req_deg)
        checks.append(('phase margin %s' % ax_n,
                       '> %.0f deg' % phase_margin_req_deg, pm_str, pm_ok))

    if mc is not None:
        ok = ~np.isnan(mc['j_rss'])
        n_valid = np.sum(ok)
        pct_jitter = 100*np.sum(mc['j_rss'][ok] < jitter_req_mas)/n_valid
        pct_contrast = 100*np.sum(mc['contrast'][ok] < contrast_req)/n_valid
        checks.append(('mc jitter pass (%d)' % n_valid,
                       '> 95%%', '%.1f%%' % pct_jitter, pct_jitter > 95))
        checks.append(('mc contrast pass (%d)' % n_valid,
                       '> 95%%', '%.1f%%' % pct_contrast, pct_contrast > 95))

    print("\n  checks")
    n_pass = 0
    for desc, thr, meas, ok in checks:
        n_pass += ok
        status = "pass" if ok else "FAIL"
        print("  %s:   %s  /  %s  [%s]" % (desc, meas, thr, status))
    print("  %d/%d" % (n_pass, len(checks)))


# 11. Entry

if __name__ == "__main__":
    method   = 'hinf'
    do_sweep = False
    do_mc    = True

    baseline = run_single(method=method)
    mc_results = None

    if do_sweep:
        run_sweep(method=method)
    if do_mc:
        mc_results = run_montecarlo(method=method)

    print_vv(baseline, mc_results)
    print("\n  done.")


# TODO
#   cross-axis inertia for misaligned mounts. small at f_iso=1 Hz, but check.
#   mount small-angle approx: 2*dq_vec for delta_theta, valid below ~5 deg.
#     allowable swing TBD.
#   damping hardest to set, 20% MC dispersion is placeholder.
#     comes from material + bonding, need vendor data.
#   linearised per-axis plant drops -w x (I*w) gyroscopic coupling.
#     valid at near-zero body rates.
#   energy conservation: torque-free should preserve |H| and T = 1/2*w'*I*w.
#     add explicit if-print (assert gets stripped under -O).
#   Joseph form: 0.5*(P + P.T) symmetrise suggests roundoff still bites.
#     test over 1000 s worst-case noise.
#   quaternion drift: renormalise every substage keeps |q| at machine precision.
#     test over 1000 s.
#   Kr for tracking: u = r*Kr - K*x_hat. r=0 currently.
#   structural coupling is scalar, f_struct ~ 15 Hz.
#     rerun when structural FEM arrives.
#   observer fallback in HinfCtrl set for f_iso=1 Hz.
#     regenerate per-axis if place_poles fails.
#   phase margin uses -180 and -540 crossings. may need -900.
#   0.5 s settle mask in _reduce. longer transients possible at low sweep end.
