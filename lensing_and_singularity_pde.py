"""
lensing_and_singularity_pde.py
------------------------------
Differential Equation Verification of:
  [Key 1] Acoustic Gravitational Lensing by the Re_s = 1 Navier-Stokes
          Suction Vortex Pressure Well (Delta_theta -> 2*pi * V_0^2 / c_0^2, Zero Dark Matter).
  [Key 2] Singularity Elimination & Gamma -> 1 Horizon Formation via the
          Stiffened Excluded-Volume Equation of State P(rho) = P_0 + beta_s*[rho/(rho_max - rho)]^2.
  [Key s] Save 300-DPI publication figure to script directory.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ==============================================================================
# NULL-GEODESIC RAY-TRACING THROUGH THE Re_s = 1 VORTEX PRESSURE WELL
# ==============================================================================
def compute_vortex_lensing_ode(V_0_kms=220.0, R_c_kpc=2.5, b_max_kpc=30.0):
    c_kms = 299792.458
    rad_to_arcsec = (180.0 / np.pi) * 3600.0
    gamma_euler = 0.5772156649015329

    # Integrate exact Corollary 2 ODE with analytical omega_0 = (V_0 / R_c) * exp(gamma_E / 2)
    omega_0 = (V_0_kms / R_c_kpc) * np.exp(0.5 * gamma_euler)

    def corollary2_ode(r, y):
        # y[0] = omega_z(r), y[1] = R * v_vortex(R) = int_0^R r' omega_z(r') dr'
        om = y[0]
        d_om = -(om / r) * (1.0 - np.exp(-((r / R_c_kpc) ** 2))) if r > 1e-8 else 0.0
        d_circ = r * om
        return [d_om, d_circ]

    r_grid = np.logspace(-3, 3.5, 3000)
    sol_vor = solve_ivp(
        corollary2_ode, (r_grid[0], r_grid[-1]),
        [omega_0, 0.5 * omega_0 * (r_grid[0] ** 2)],
        t_eval=r_grid, rtol=1e-10, atol=1e-12
    )
    v_vor_grid = sol_vor.y[1] / sol_vor.t

    def accel_over_R(R_val):
        M_enc_norm = (R_val ** 3) / ((R_val ** 2 + R_c_kpc ** 2) ** 1.5)
        v_bar_sq = ((0.92 * V_0_kms) ** 2) * (2.0 * R_c_kpc * M_enc_norm / R_val)
        g_bar_R = v_bar_sq / (R_val ** 2)
        v_vor_val = np.interp(R_val, r_grid, v_vor_grid)
        dP_vor_R = (v_vor_val ** 2) / (R_val ** 2)
        return g_bar_R, dP_vor_R

    b_vals = np.linspace(0.5, b_max_kpc, 50)
    theta_baryon_arcsec = np.zeros_like(b_vals)
    theta_vortex_arcsec = np.zeros_like(b_vals)
    theta_total_arcsec = np.zeros_like(b_vals)

    eta = np.linspace(-8.0, 8.0, 1600)
    d_eta = eta[1] - eta[0]

    for idx_b, b in enumerate(b_vals):
        x_line = b * np.sinh(eta)
        dx_deta = b * np.cosh(eta)
        R_path = np.sqrt(x_line ** 2 + b ** 2)

        g_bar_over_R, dP_vor_over_R = accel_over_R(R_path)

        defl_bar = (2.0 / (c_kms ** 2)) * np.sum(g_bar_over_R * b * dx_deta) * d_eta
        defl_vor = (2.0 / (c_kms ** 2)) * np.sum(dP_vor_over_R * b * dx_deta) * d_eta

        theta_baryon_arcsec[idx_b] = defl_bar * rad_to_arcsec
        theta_vortex_arcsec[idx_b] = defl_vor * rad_to_arcsec
        theta_total_arcsec[idx_b] = (defl_bar + defl_vor) * rad_to_arcsec

    theta_inf_arcsec = 2.0 * np.pi * ((V_0_kms / c_kms) ** 2) * rad_to_arcsec

    # Integrate actual null-geodesic ray ODE: d^2y/dx^2 = -M_vis * (2/c^2) * (dPhi_eff/dR) * (y/R)
    x_span = (-25.0, 25.0)
    x_ray = np.linspace(-25.0, 25.0, 300)
    vis_scale = 2.2e4  # Explicit visual magnification so ~0.7 arcsec deflection is visible at kpc scale
    b_rays = np.linspace(-18.0, 18.0, 18)
    ray_paths = []

    for b0 in b_rays:
        if abs(b0) < 0.8:
            continue

        def ray_ode(x, state):
            y_pos, dy_dx = state
            R_curr = np.sqrt(x ** 2 + y_pos ** 2 + 0.15 ** 2)
            g_b, dP_v = accel_over_R(R_curr)
            d2y_dx2 = -vis_scale * (2.0 / (c_kms ** 2)) * (g_b + dP_v) * y_pos
            return [dy_dx, d2y_dx2]

        sol_ray = solve_ivp(ray_ode, x_span, [b0, 0.0], t_eval=x_ray, rtol=1e-8, atol=1e-10)
        ray_paths.append((sol_ray.t, sol_ray.y[0]))

    return b_vals, theta_baryon_arcsec, theta_vortex_arcsec, theta_total_arcsec, theta_inf_arcsec, ray_paths


# ==============================================================================
# STIFFENED EQUATION OF STATE SINGULARITY ELIMINATION & Gamma -> 1
# ==============================================================================
def solve_stiffened_eos_singularity_arrest():
    R_outer = 5.0
    R_inner = 0.04
    R_span = (R_outer, R_inner)
    R_eval = np.linspace(R_outer, R_inner, 600)

    rho0_frac = 0.15
    u0 = rho0_frac / (1.0 - rho0_frac)
    C0 = u0 + 0.5 * (u0 ** 2)

    def stiffened_ode(R, y):
        rf = y[0]
        return [-((1.0 - rf) ** 3) / (R ** 2)]

    sol_stiff = solve_ivp(stiffened_ode, R_span, [rho0_frac], t_eval=R_eval, rtol=1e-11, atol=1e-13)
    R_vals = sol_stiff.t[::-1]
    rho_stiff_num = sol_stiff.y[0][::-1]

    Phi_tilde = C0 + (1.0 / R_vals) - (1.0 / R_outer)
    rho_stiff_exact = 1.0 - (1.0 / np.sqrt(1.0 + 2.0 * Phi_tilde))

    rho_unbounded_poly = rho0_frac + ((1.0 / R_vals) - (1.0 / R_outer))

    u_vals = rho_stiff_num / (1.0 - rho_stiff_num)
    Z_ratio = (u_vals / u0) ** 1.5
    Gamma_R = (Z_ratio - 1.0) / (Z_ratio + 1.0)

    max_ode_err = np.max(np.abs(rho_stiff_num - rho_stiff_exact))

    # Exact-radius evaluation for Table I (R/R_c in {5.00, 2.00, 1.00, 0.25, 0.04})
    R_table = np.array([5.00, 2.00, 1.00, 0.25, 0.04])
    sol_table = solve_ivp(stiffened_ode, R_span, [rho0_frac], t_eval=R_table, rtol=1e-11, atol=1e-13)
    table_rows = []
    for R_val, rho_rk in zip(R_table, sol_table.y[0]):
        dPhi = (1.0 / R_val) - (1.0 / R_outer)
        rho_poly_val = rho0_frac + dPhi
        Phi_tilde_val = C0 + dPhi
        rho_thm1_val = 1.0 - (1.0 / np.sqrt(1.0 + 2.0 * Phi_tilde_val))
        u_val = np.sqrt(1.0 + 2.0 * Phi_tilde_val) - 1.0
        Z_rat_val = (u_val / u0) ** 1.5
        gamma_val = (Z_rat_val - 1.0) / (Z_rat_val + 1.0)
        table_rows.append((R_val, rho_poly_val, rho_rk, rho_thm1_val, gamma_val))

    return R_vals, rho_stiff_num, rho_stiff_exact, rho_unbounded_poly, Z_ratio, Gamma_R, max_ode_err, table_rows


def launch_tests_3_and_4():
    b_vals, th_bar, th_vor, th_tot, th_inf, ray_paths = compute_vortex_lensing_ode()
    R_vals, rho_num, rho_ex, rho_poly, Z_rat, Gamma_R, ode_err, table_rows = solve_stiffened_eos_singularity_arrest()

    print("=" * 80)
    print(" ACOUSTIC LENSING & STIFFENED EOS SINGULARITY ELIMINATION")
    print("=" * 80)
    print(f"1. (Re_s = 1 Vortex Lensing at V_0 = 220 km/s):")
    print(f"   Analytical Asymptotic Lensing Limit Delta_theta_inf = 2*pi*(V_0/c)^2 : {th_inf:.4f} arcsec")
    print(f"   Numerical ODE Vortex Lensing at b = 25 kpc                           : {th_vor[-8]:.4f} arcsec")
    print(f"   Total Lensing (Baryons + Vortex) at b = 25 kpc                       : {th_tot[-8]:.4f} arcsec")
    print(f"   Newtonian Baryons Alone at b = 25 kpc                                : {th_bar[-8]:.4f} arcsec")
    print("-" * 80)
    print(f"2. (Stiffened EOS Singularity Arrest as R -> 0):")
    print(f"   Unbounded Polytrope Density at R = 0.04   : {rho_poly[0]:8.4f} * rho_max (BLOWS UP -> SINGULARITY)")
    print(f"   Stiffened EOS Density at R = 0.04         : {rho_num[0]:8.5f} * rho_max (BOUNDED < 1.00000)")
    print(f"   Acoustic Reflection Coefficient Gamma     : {Gamma_R[0]:8.5f}           (TOTAL INTERNAL REFLECTION)")
    print(f"   Max ODE vs. Analytical Theorem Error      : {ode_err:.2e}")
    print("-" * 80)
    print("   Table I Exact-Radius Verification (R/R_c | Polytrope | RK45 | Theorem 1 | Gamma):")
    for R_val, r_pol, r_rk, r_th, gam in table_rows:
        print(f"   {R_val:5.2f} | {r_pol:8.5f} | {r_rk:10.8f} | {r_th:10.8f} | {gam:7.5f}")
    print("=" * 80)
    print("Controls: Press [1] (Gravitational Lensing) | [2] (Singularity Arrest) | [s] Save PNG")

    fig = plt.figure(figsize=(15.5, 7.5), facecolor='#0b0f19')
    current_view = [1]

    def draw_view(mode=1):
        current_view[0] = mode
        fig.clf()
        ax1 = fig.add_subplot(1, 2, 1, facecolor='#0b0f19')
        ax2 = fig.add_subplot(1, 2, 2, facecolor='#0b0f19')

        if mode == 1:
            fig.suptitle(
                "Acoustic Gravitational Lensing by the $Re_s = 1$ Navier-Stokes Suction Vortex (Zero Dark Matter)\n"
                "Asymptotic Deflection Theorem: $\\Delta\\theta_{\\infty} = \\frac{2}{c_0^2}\\int_{-\\infty}^{\\infty} "
                "\\frac{V_0^2 b}{x^2 + b^2}dx = \\frac{2\\pi V_0^2}{c_0^2} = \\mathrm{const}$ | Keys: [1] Lensing, [2] Singularity Arrest, [s] Save",
                color='white', fontsize=11.5, fontweight='bold'
            )

            x_g = np.linspace(-25, 25, 200)
            y_g = np.linspace(-20, 20, 200)
            XX, YY = np.meshgrid(x_g, y_g)
            RR = np.sqrt(XX ** 2 + YY ** 2 + 2.5 ** 2)
            Phi_field = -np.log(RR / 30.0)
            ax1.contourf(XX, YY, Phi_field, levels=30, cmap='magma', alpha=0.55)

            for x_r, y_r in ray_paths:
                ax1.plot(x_r, y_r, color='#00ffcc', lw=1.5, alpha=0.85)

            circle = plt.Circle((0, 0), 2.5, color='yellow', fill=False, ls='--', lw=2.0, label='Core Horizon $R_c = 2.5$ kpc')
            ax1.add_patch(circle)
            ax1.set_xlim([-25, 25]); ax1.set_ylim([-19, 19])
            ax1.set_title("(a) Null-Geodesic Ray ODE Integration Across $P_{\\mathrm{vortex}}(R) \\propto \\ln(R/R_c)$",
                          color='white', fontsize=11, fontweight='bold')
            ax1.set_xlabel("Propagation Axis $x$ (kpc)", color='white', fontsize=10.5)
            ax1.set_ylabel("Impact Parameter Axis $y$ (kpc)", color='white', fontsize=10.5)
            ax1.legend(loc='upper right', facecolor='#151c2c', edgecolor='yellow', labelcolor='white', fontsize=8.5)

            ax2.plot(b_vals, th_tot, color='#00ffcc', lw=3.0,
                     label=r'Total Lensing Deflection $\Delta\theta_{\mathrm{total}}(b)$ (Baryons + $Re_s=1$ Vortex)')
            ax2.plot(b_vals, th_vor, color='#3399ff', lw=2.2, ls='-.',
                     label=r'Navier-Stokes Suction Vortex Alone $\Delta\theta_{\mathrm{vortex}}(b)$')
            ax2.plot(b_vals, th_bar, color='#ff4444', lw=2.2, ls='--',
                     label=r'Newtonian Baryons Alone $\Delta\theta_{\mathrm{baryon}}(b) \propto b^{-1}$')
            ax2.axhline(th_inf, color='yellow', ls=':', lw=1.8,
                        label=f'Exact Asymptotic Theorem $2\\pi(V_0/c_0)^2 = {th_inf:.3f}$ arcsec')

            ax2.set_xlim([0, 30]); ax2.set_ylim([0, 1.35])
            ax2.set_title("(b) Lensing Deflection Angle $\\Delta\\theta(b)$ vs. Impact Parameter $b$",
                          color='white', fontsize=11, fontweight='bold')
            ax2.set_xlabel("Impact Parameter $b$ (kpc)", color='white', fontsize=10.5)
            ax2.set_ylabel("Deflection Angle $\\Delta\\theta(b)$ (arcseconds)", color='white', fontsize=10.5)
            ax2.legend(loc='upper right', facecolor='#151c2c', edgecolor='cyan', labelcolor='white', fontsize=8.5)

        elif mode == 2:
            fig.suptitle(
                "Singularity Elimination & $\\Gamma \\to 1$ Horizon via Stiffened Excluded-Volume EOS\n"
                "$P(\\rho) = P_0 + \\beta_s[\\rho/(\\rho_{\\max}-\\rho)]^2 \\Rightarrow "
                "\\rho(R)/\\rho_{\\max} = 1 - [1 + 2\\tilde{\\Phi}(R)]^{-1/2} < 1$ | Keys: [1] Lensing, [2] Singularity Arrest, [s] Save",
                color='white', fontsize=11.5, fontweight='bold'
            )

            ax1.plot(R_vals, rho_poly, color='#ff4444', lw=2.5, ls='--',
                     label=r'Standard Unbounded Polytrope ($P \propto \rho^2 \Rightarrow \rho \to \infty$ Singularity)')
            ax1.plot(R_vals, rho_num, color='#00ffcc', lw=3.0,
                     label=r'Stiffened Excluded-Volume EOS ODE Solution ($\rho < \rho_{\max}$ Bounded)')
            ax1.plot(R_vals[::25], rho_ex[::25], 'o', color='yellow', markersize=4.5,
                     label=r'Exact Analytical Theorem $\rho/\rho_{\max} = 1 - (1 + 2\tilde{\Phi})^{-1/2}$')
            ax1.axhline(1.0, color='magenta', ls=':', lw=2.0, label=r'Maximum Density Threshold $\rho / \rho_{\max} = 1.0$')

            ax1.set_xlim([0.0, 5.0]); ax1.set_ylim([0.0, 2.2])
            ax1.set_title("(a) Elimination of Point Singularity as $R \\to 0$", color='white', fontsize=11, fontweight='bold')
            ax1.set_xlabel("Dimensionless Core Radius $R / R_c$", color='white', fontsize=10.5)
            ax1.set_ylabel("Normalized Fluid Density $\\rho(R) / \\rho_{\\max}$", color='white', fontsize=10.5)
            ax1.legend(loc='upper right', facecolor='#151c2c', edgecolor='cyan', labelcolor='white', fontsize=8.3)

            ax2.plot(R_vals, Gamma_R, color='#ff9933', lw=3.0,
                     label=r'Acoustic Reflection Coefficient $\Gamma(R) = \frac{Z(R) - Z_0}{Z(R) + Z_0}$')
            ax2.axhline(1.0, color='#00ffcc', ls='--', lw=1.8,
                        label=r'Total Internal Reflection Horizon ($\Gamma = 1.000$)')

            ax2.set_xlim([0.0, 5.0]); ax2.set_ylim([0.0, 1.08])
            ax2.set_title("(b) Emergence of the $\\Gamma = 1$ Total Reflection Horizon as $\\rho \\to \\rho_{\\max}$",
                          color='white', fontsize=11, fontweight='bold')
            ax2.set_xlabel("Dimensionless Core Radius $R / R_c$", color='white', fontsize=10.5)
            ax2.set_ylabel("Acoustic Reflection Coefficient $\\Gamma(R)$", color='white', fontsize=10.5)
            ax2.legend(loc='lower right', facecolor='#151c2c', edgecolor='cyan', labelcolor='white', fontsize=8.5)

        for ax in (ax1, ax2):
            ax.grid(True, ls='--', alpha=0.25, color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')

        plt.tight_layout()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ['1', '2']:
            draw_view(int(event.key))
        elif event.key == 's':
            script_dir = os.path.dirname(os.path.abspath(__file__))
            out_path = os.path.join(script_dir, f"Test_{current_view[0] + 2}_Result.png")
            fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
            print(f"Saved high-resolution figure: {out_path}")

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw_view(mode=1)
    plt.show()


if __name__ == "__main__":
    launch_tests_3_and_4()
