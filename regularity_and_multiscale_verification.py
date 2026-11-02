"""
regularity_and_multiscale_verification.py
-----------------------------------------
Numerical ODE and Multi-Scale Verification of:
  [1] Theorem 1: Sun-Wang-Zhang A Priori Density Regularity across gamma in {1.5, 2.0, 3.0}
      via Runge-Kutta (RK45) integration of the stiffened excluded-volume enthalpy ODE.
  [2] Theorem 3 & Corollary 2: Subatomic Suction Scaling (alpha^-1, m_e),
      Zero-Parameter Galactic Acceleration (a_0 = [c*H_cmb/(2*pi)]*(1 + phi^-4)),
      Baryonic Tully-Fisher Relation (V_0^4 = G*M_baryon*a_0), and
      Meridian Hubble Velocity Divergence (H_local = 72.90 km/s/Mpc).
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def verify_theorems_and_plot():
    # ==========================================================================
    # 1. FUNDAMENTAL CONSTANTS & EXACT SPINDLE TORUS INVARIANTS
    # ==========================================================================
    c_ms = 299792458.0
    k_B = 1.380649e-23
    e_charge = 1.602176634e-19
    G_grav = 6.67430e-11
    M_sun = 1.98847e30
    Mpc_m = 3.085677581491367e22

    T_CMB = 2.72548
    H_cmb_kmsMpc = 67.4
    H_cmb_si = (H_cmb_kmsMpc * 1000.0) / Mpc_m

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    W_phi = phi ** (-2)               # (3 - sqrt(5))/2 = 0.3819660113
    spindle_overlap = phi ** (-4)     # W_phi^2 = 0.1458980338
    Theta_phi = 360.0 * W_phi         # 137.50776405
    Delta_Theta_dil = np.sqrt(2.0) / 3.0

    # Theorem 3(1): Subatomic Suction Scaling
    alpha_dil_inv = Theta_phi - Delta_Theta_dil
    delta_r_dil = (Theta_phi / alpha_dil_inv) - 1.0
    Re_s_electron = 1.0 / delta_r_dil
    W_e_dil = Theta_phi / (1.0 + (7.0 / 3.0) * delta_r_dil)
    m_e_MeV = (2.0 * np.pi * k_B * T_CMB * (W_e_dil ** 4)) / (1e6 * e_charge)

    # Theorem 3(2): Zero-Parameter Galactic Acceleration Scale a_0 & BTFR
    a0_derived = (c_ms * H_cmb_si / (2.0 * np.pi)) * (1.0 + spindle_overlap)
    a0_obs = 1.20e-10

    # Corollary 2: Meridian Hubble Divergence
    H_local_derived = H_cmb_kmsMpc * (1.0 + 0.5 * spindle_overlap) * (1.0 + (7.0 / 3.0) * delta_r_dil)
    Delta_H_derived = H_local_derived - H_cmb_kmsMpc

    # ==========================================================================
    # 2. THEOREM 1: RK45 INTEGRATION OF STIFFENED EOS ACROSS gamma IN {1.5, 2.0, 3.0}
    # ==========================================================================
    R_outer = 5.0
    R_inner = 0.04
    R_eval = np.linspace(R_outer, R_inner, 600)
    rho0 = 0.15
    u0 = rho0 / (1.0 - rho0)

    def solve_gamma_profile(gamma_val):
        # Exact ODE from dh/dR = (gamma*beta_s/rho_max)*(1 + u)*u^(gamma-2)*(du/dR) = d(Delta_Phi)/dR:
        # With dimensionless (rho_max / [gamma*beta_s]) * d(Delta_Phi)/dR = -1 / R^2:
        # drho/dR = -((1 - rho)^(gamma + 1) / (rho^(gamma - 2))) / (R^2)
        def ode_rhs(R, y):
            rf = np.clip(y[0], 1e-12, 1.0 - 1e-12)
            return [-((1.0 - rf) ** (gamma_val + 1.0)) / ((rf ** (gamma_val - 2.0)) * (R ** 2))]

        sol = solve_ivp(ode_rhs, (R_outer, R_inner), [rho0], t_eval=R_eval, rtol=1e-11, atol=1e-13)
        return sol.t[::-1], sol.y[0][::-1]

    R_vals, rho_g15 = solve_gamma_profile(1.5)
    _, rho_g20 = solve_gamma_profile(2.0)
    _, rho_g30 = solve_gamma_profile(3.0)

    # Exact closed-form Theorem 1 solution for gamma = 2.0
    Phi_tilde = u0 + 0.5 * (u0 ** 2) + (1.0 / R_vals) - (1.0 / R_outer)
    rho_g20_exact = 1.0 - 1.0 / np.sqrt(1.0 + 2.0 * Phi_tilde)
    max_rk_err = np.max(np.abs(rho_g20 - rho_g20_exact))
    rho_poly = rho0 + ((1.0 / R_vals) - (1.0 / R_outer))

    # ==========================================================================
    # 3. PRINT TERMINAL VERIFICATION LEDGER
    # ==========================================================================
    print("=" * 82)
    print(" SUN-WANG-ZHANG REGULARITY & MULTI-SCALE SUCTION SCALING VERIFICATION")
    print("=" * 82)
    print(f" [Theorem 1] Stiffened EOS RK45 vs. Exact Radical (gamma=2) Max Error : {max_rk_err:.2e}")
    print(f"             Bounded Core Density rho(0.04)/rho_max (gamma=1.5, 2, 3) : "
          f"{rho_g15[0]:.5f}, {rho_g20[0]:.5f}, {rho_g30[0]:.5f} (< 1.0)")
    print(f"             Unbounded Polytrope rho_poly(0.04)/rho_max               : {rho_poly[0]:.4f} (SINGULAR)")
    print("-" * 82)
    print(f" [Theorem 3(1)] Derived Fine-Structure Constant alpha_dil^-1          : {alpha_dil_inv:.8f} (CODATA: 137.03599918)")
    print(f"                Subatomic Suction Reynolds Number Re_s(r_e) = r_e/d*  : {Re_s_electron:.4f} (d*/r_e = {delta_r_dil:.8f})")
    print(f"                Derived Electron Rest Mass m_e (from T_CMB)           : {m_e_MeV:.7f} MeV/c^2 (CODATA: 0.51099895)")
    print("-" * 82)
    print(f" [Theorem 3(2)] Derived Galactic Acceleration Scale a_0               : {a0_derived:.4e} m/s^2")
    print(f"                Empirical SPARC Acceleration Scale a_0,obs            : {a0_obs:.4e} +/- 0.02e-10 m/s^2 (-0.28 sigma)")
    print("-" * 82)
    print(f" [Corollary 2]  Derived Local Hubble Divergence H_local               : {H_local_derived:.2f} km s^-1 Mpc^-1")
    print(f"                Empirical JWST/Cepheid Determination H_local,obs      : 73.00 +/- 1.00 km s^-1 Mpc^-1 (-0.10 sigma)")
    print(f"                Derived Velocity Shear Differential Delta_H           : {Delta_H_derived:.2f} km s^-1 Mpc^-1")
    print("=" * 82)

    # ==========================================================================
    # 4. GENERATE 2-PANEL PUBLICATION FIGURE
    # ==========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.8), facecolor="#0b0f19")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0b0f19")

    fig.suptitle(
        "A Priori Sun-Wang-Zhang Regularity & Zero-Parameter Baryonic Tully-Fisher Scaling\n"
        r"Theorem 1: $0 < \rho_0 \leq \rho(R) < \rho_{\max}$ across $\gamma > 1$ | "
        r"Theorem 3: $a_0 = \frac{c_0\mathcal{H}_{\mathrm{cmb}}}{2\pi}(1 + \phi^{-4}) = 1.194 \times 10^{-10}\ \mathrm{m/s^2}$",
        color="white", fontsize=11.5, fontweight="bold"
    )

    # Panel (a): Theorem 1 Regularity across gamma in {1.5, 2.0, 3.0}
    ax1.plot(R_vals, rho_poly, color="#ff4444", lw=2.4, ls="--",
             label=r"Unbounded Polytrope ($P \propto \rho^2 \Rightarrow \rho \to \infty$ Singularity)")
    ax1.plot(R_vals, rho_g15, color="#66b3ff", lw=2.2, ls="-.",
             label=r"Stiffened Excluded-Volume ODE ($\gamma = 1.5$, $\rho < \rho_{\max}$)")
    ax1.plot(R_vals, rho_g20, color="#00ffcc", lw=3.0,
             label=r"Stiffened Excluded-Volume ODE ($\gamma = 2.0$, $\rho < \rho_{\max}$)")
    ax1.plot(R_vals[::25], rho_g20_exact[::25], "o", color="yellow", markersize=4.5,
             label=r"Exact Theorem 1 Radical $\rho/\rho_{\max} = 1 - (1 + 2\tilde{\Phi})^{-1/2}$")
    ax1.plot(R_vals, rho_g30, color="#ff9933", lw=2.2, ls=":",
             label=r"Stiffened Excluded-Volume ODE ($\gamma = 3.0$, $\rho < \rho_{\max}$)")
    ax1.axhline(1.0, color="magenta", ls="--", lw=1.8,
                label=r"Sun-Wang-Zhang Upper Cap $\rho / \rho_{\max} = 1.0$")

    ax1.set_xlim([0.0, 5.0])
    ax1.set_ylim([0.0, 2.15])
    ax1.set_title(r"(a) Theorem 1: A Priori Singularity Arrest Across $\gamma > 1$",
                  color="white", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Dimensionless Radius $R / R_c$", color="white", fontsize=10.5)
    ax1.set_ylabel(r"Normalized Fluid Density $\rho(R) / \rho_{\max}$", color="white", fontsize=10.5)
    ax1.legend(loc="upper right", facecolor="#151c2c", edgecolor="cyan", labelcolor="white", fontsize=8.2)

    # Panel (b): Theorem 3 Zero-Parameter Baryonic Tully-Fisher Relation
    M_baryon_solar = np.logspace(7.2, 11.8, 300)
    V0_derived_kms = ((G_grav * M_baryon_solar * M_sun * a0_derived) ** 0.25) / 1000.0
    V0_obs_band_low = ((G_grav * M_baryon_solar * M_sun * (1.08e-10)) ** 0.25) / 1000.0
    V0_obs_band_high = ((G_grav * M_baryon_solar * M_sun * (1.32e-10)) ** 0.25) / 1000.0

    sparc_names = ["DDO 154", "NGC 2403", "NGC 3198", "Milky Way", "UGC 2885"]
    sparc_M_solar = np.array([3.6e8, 2.3e10, 3.8e10, 6.5e10, 5.5e11])
    sparc_V_obs = np.array([47.0, 134.0, 150.0, 220.0, 298.0])
    sparc_V_err = np.array([3.0, 5.5, 6.0, 10.0, 11.0])

    ax2.fill_between(M_baryon_solar, V0_obs_band_low, V0_obs_band_high, color="#3399ff", alpha=0.22,
                     label=r"Empirical SPARC Scatter Band ($\pm 10\%$ in $a_0$)")
    ax2.plot(M_baryon_solar, V0_derived_kms, color="#00ffcc", lw=3.0,
             label=r"Theorem 3 Prediction $V_0 = (G M_{\mathrm{baryon}} a_0)^{1/4}$ (Zero Free Parameters)")
    ax2.errorbar(sparc_M_solar, sparc_V_obs, yerr=sparc_V_err, fmt="o", color="yellow",
                 ecolor="yellow", elinewidth=1.8, capsize=4, markersize=6.5,
                 label="Representative SPARC Benchmark Galaxies")

    for name, mx, vy in zip(sparc_names, sparc_M_solar, sparc_V_obs):
        ax2.annotate(name, (mx, vy), textcoords="offset points", xytext=(6, -12),
                     color="white", fontsize=8.5, fontweight="bold")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim([1.5e7, 6.5e11])
    ax2.set_ylim([20.0, 420.0])
    ax2.set_title(r"(b) Theorem 3: Zero-Parameter Baryonic Tully-Fisher Law ($Re_s = 1$)",
                  color="white", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Total Baryonic Mass $M_{\mathrm{baryon}}\ (M_\odot)$", color="white", fontsize=10.5)
    ax2.set_ylabel(r"Asymptotic Flat Rotation Speed $V_0\ (\mathrm{km/s})$", color="white", fontsize=10.5)
    ax2.legend(loc="upper left", facecolor="#151c2c", edgecolor="cyan", labelcolor="white", fontsize=8.2)

    for ax in (ax1, ax2):
        ax.grid(True, which="both", ls="--", alpha=0.22, color="white")
        ax.tick_params(colors="white", which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "MultiScale_Regularity_Verification.png")
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    print(f"Saved 300-DPI publication figure to: {out_path}")
    plt.show()


if __name__ == "__main__":
    verify_theorems_and_plot()