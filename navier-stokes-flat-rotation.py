"""
navier-stokes-flat-rotation.py
------------------------------
Exact Finite-Difference and Analytical Verification of Flat Rotation Curves
in an Axisymmetric Navier-Stokes Suction Vortex (Zero Dark Matter).
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt


def solve_homogeneous_euler_cauchy_vortex(R_min=1.0, R_max=30.0, Nr=600,
                                          Re_s_list=(0.75, 1.00, 1.25, 1.50, 2.00)):
    """
    Solves the homogeneous 1D azimuthal Navier-Stokes ODE (RHS = 0):
        d^2 v_theta / dR^2 + ((1 + Re_s) / R) dv_theta / dR - ((1 - Re_s) / R^2) v_theta = 0
    via second-order central finite differences and compares against (R / R_min)^(1 - Re_s).
    """
    R = np.linspace(R_min, R_max, Nr)
    dR = R[1] - R[0]
    results = {}

    for Re_s in Re_s_list:
        main = np.zeros(Nr)
        upper = np.zeros(Nr - 1)
        lower = np.zeros(Nr - 1)
        rhs = np.zeros(Nr)

        main[0] = 1.0
        rhs[0] = 1.0

        for i in range(1, Nr - 1):
            Ri = R[i]
            adv = (1.0 + Re_s) / Ri
            pot = -(1.0 - Re_s) / (Ri ** 2)

            lower[i - 1] = (1.0 / dR ** 2) - (adv / (2.0 * dR))
            main[i] = (-2.0 / dR ** 2) + pot
            upper[i] = (1.0 / dR ** 2) + (adv / (2.0 * dR))
            rhs[i] = 0.0

        Rn = R[-1]
        adv_n = (1.0 + Re_s) / Rn
        pot_n = -(1.0 - Re_s) / (Rn ** 2)
        c_plus = (1.0 / dR ** 2) + (adv_n / (2.0 * dR))
        c_minus = (1.0 / dR ** 2) - (adv_n / (2.0 * dR))

        lower[Nr - 2] = c_minus + c_plus
        main[Nr - 1] = (-2.0 / dR ** 2) + pot_n + c_plus * (2.0 * dR * (1.0 - Re_s) / Rn)
        rhs[Nr - 1] = 0.0

        mat = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format='csr')
        v_fd = sla.spsolve(mat, rhs)
        v_exact = (R / R_min) ** (1.0 - Re_s)
        max_err = np.max(np.abs(v_fd - v_exact))
        results[Re_s] = (v_fd, v_exact, max_err)

    return R, results


def solve_nonsingular_galactic_composite(r_max_kpc=30.0, Nr=600, R_c_kpc=2.5, V_0_kms=200.0):
    """
    Integrates the non-singular distributed-core Navier-Stokes suction vortex (Re_s = 1.0)
    directly from R -> 0 using the analytical Corollary 2 central vorticity
    omega_0 = (V_0 / R_c) * exp(gamma_E / 2) without endpoint rescaling:
        d(omega_z)/dR = -(1 / R) * (1 - exp(-(R / R_c)^2)) * omega_z
        d(R * v_vortex)/dR = R * omega_z(R)
    """
    R = np.linspace(1e-3, r_max_kpc, Nr)
    gamma_euler = 0.5772156649015329
    omega_0 = (V_0_kms / R_c_kpc) * np.exp(0.5 * gamma_euler)

    def corollary2_ode(r, y):
        om = y[0]
        d_om = -(om / r) * (1.0 - np.exp(-((r / R_c_kpc) ** 2)))
        d_circ = r * om
        return [d_om, d_circ]

    sol = solve_ivp(
        corollary2_ode, (R[0], R[-1]),
        [omega_0, 0.5 * omega_0 * (R[0] ** 2)],
        t_eval=R, rtol=1e-11, atol=1e-13
    )
    v_vortex = sol.y[1] / sol.t

    M_baryon_enclosed = (R ** 3) / ((R ** 2 + R_c_kpc ** 2) ** 1.5)
    v_baryon = (0.92 * V_0_kms) * np.sqrt((2.0 * R_c_kpc * M_baryon_enclosed) / R)
    v_total = np.sqrt(v_baryon ** 2 + v_vortex ** 2)

    return R, v_baryon, v_vortex, v_total


def generate_publication_figure():
    R_ode, ode_results = solve_homogeneous_euler_cauchy_vortex()
    R_gal, v_bar, v_vor, v_tot = solve_nonsingular_galactic_composite()

    print("=" * 78)
    print(" NAVIER-STOKES SUCTION VORTEX VERIFICATION ( Analytical vs. Finite-Difference )")
    print("=" * 78)
    for Re_s, (_, _, max_err) in ode_results.items():
        exponent = 1.0 - Re_s
        print(f"  Suction Reynolds Re_s = {Re_s:4.2f}  |  Power Law R^({exponent:+5.2f})  |  Max FD Error = {max_err:.2e}")
    print("=" * 78)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), facecolor='#0b0f19')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0b0f19')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.grid(True, ls='--', alpha=0.25, color='white')

    palette = {
        0.75: ('#cc66ff', r'$Re_s = 0.75$ ($v_\theta \propto R^{+0.25}$ Rising Sheet)'),
        1.00: ('#00ffcc', r'$Re_s = 1.00$ ($\delta^* = R_c$: Exact Flat Curve $v_\theta = V_0$)'),
        1.25: ('#ffaa00', r'$Re_s = 1.25$ ($v_\theta \propto R^{-0.25}$ Mild Falloff)'),
        1.50: ('#3399ff', r'$Re_s = 1.50$ ($v_\theta \propto R^{-1/2}$ Keplerian-like Decay)'),
        2.00: ('#ff4444', r'$Re_s = 2.00$ ($v_\theta \propto R^{-1}$ Potential Vortex Decay)')
    }
    for Re_s, (v_fd, v_ex, _) in ode_results.items():
        col, lbl = palette[Re_s]
        ax1.plot(R_ode, v_fd, lw=2.6, color=col, label=lbl)
        ax1.plot(R_ode[::40], v_ex[::40], 'o', color=col, markersize=4)

    ax1.set_xlim([1.0, 30.0]); ax1.set_ylim([0.0, 1.45])
    ax1.set_title("(a) Homogeneous Navier-Stokes Suction Vortex ($v_\\theta \\propto R^{1 - Re_s}$)",
                  color='white', fontsize=11.5, fontweight='bold')
    ax1.set_xlabel("Dimensionless Radius $R / R_c$", color='white', fontsize=10.5)
    ax1.set_ylabel("Normalized Azimuthal Velocity $v_\\theta(R) / V_0$", color='white', fontsize=10.5)
    ax1.legend(loc='upper right', facecolor='#151c2c', edgecolor='cyan', labelcolor='white', fontsize=8.5)

    ax2.plot(R_gal, v_tot, color='#00ffcc', lw=3.0,
             label=r'Total Orbital Velocity $v_{\mathrm{total}}(R) = \sqrt{v_{\mathrm{baryon}}^2 + v_{\mathrm{vortex}}^2}$')
    ax2.plot(R_gal, v_vor, color='#3399ff', lw=2.2, ls='-.',
             label=r'Critical Navier-Stokes Suction Vortex $v_{\mathrm{vortex}}(R)$ ($Re_s = 1$)')
    ax2.plot(R_gal, v_bar, color='#ff4444', lw=2.2, ls='--',
             label=r'Newtonian Baryonic Core Alone $v_{\mathrm{baryon}}(R) \propto R^{-1/2}$')

    ax2.set_xlim([0.0, 30.0]); ax2.set_ylim([0.0, 260.0])
    ax2.set_title("(b) Galactic Rotation Curve at Critical Suction ($Re_s = R_c / \\delta^* = 1$)",
                  color='white', fontsize=11.5, fontweight='bold')
    ax2.set_xlabel("Galactocentric Radius $R$ (kpc)", color='white', fontsize=10.5)
    ax2.set_ylabel("Azimuthal Velocity $v_\\theta(R)$ (km/s)", color='white', fontsize=10.5)
    ax2.legend(loc='lower right', facecolor='#151c2c', edgecolor='cyan', labelcolor='white', fontsize=8.5)

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "Figure_1_Navier_Stokes_Rotation.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    print(f"Saved publication figure: {out_path}")
    plt.show()


if __name__ == "__main__":
    generate_publication_figure()
