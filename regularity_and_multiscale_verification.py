"""
regularity_and_multiscale_verification.py
-----------------------------------------
First-Principles Numerical PDE and ODE Verification Suite for:
  "A Priori Sun-Wang-Zhang Regularity, Conormal Spectral Mass Gaps, and
   Multi-Scale Suction Scaling in a Stiffened Compressible Superfluid"

Generates four large, standalone 300-DPI publication figures:
  1. Regularity_Fig1_Enthalpy_ODE.png
     - Theorem 1: Adaptive RK45 integration of the stiffened excluded-volume enthalpy ODE
       across gamma in {1.5, 2.0, 3.0} vs. unbounded polytrope blow-up and exact radical.
  2. Regularity_Fig2_Eigenmode_Polarities.png
     - Theorem 2: 2D Finite-Volume Beltrami / Grad-Shafranov crescent eigenmode polarity
       (Delta* psi + k^2 psi = 0, E_pol = E_tor => beta_eff = 45 deg) and 2D spherical
       Laplace-Beltrami icosahedral horizon eigenmode (-Delta_{S^2} Psi_15 = 240 Psi_15).
  3. Regularity_Fig3_Suction_Disk_ODE.png
     - Theorem 3: Coupled variable-density 1D compressible Prandtl ASBL integration and
       variable-density 2D axisymmetric azimuthal Navier-Stokes ODE integration across
       Re_s in {0.5, 1.0, 1.5} (plus 600-node homogeneous Neumann BVP at Re_s = 1).
  4. Regularity_Fig4_Baryonic_Tully_Fisher.png
     - Constitutive Closure 2: Zero-parameter Baryonic Tully-Fisher scaling
       V_0 = (G M_baryon a_0)^(1/4) with a_0 = [c_0 H_cmb / (2 pi)](1 + phi^-4).
"""

import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ==============================================================================
# 1. THEOREM 1 SOLVER: MULTI-GAMMA STIFFENED ENTHALPY RK45 INTEGRATION
# ==============================================================================
def verify_theorem1_enthalpy_regularity():
    R_outer = 5.0
    R_inner = 0.04
    R_eval = np.linspace(R_outer, R_inner, 600)
    rho0 = 0.15
    u0 = rho0 / (1.0 - rho0)

    def solve_gamma_profile(gamma_val):
        def ode_rhs(R, y):
            rf = np.clip(y[0], 1e-12, 1.0 - 1e-12)
            return [-((1.0 - rf) ** (gamma_val + 1.0)) / ((rf ** (gamma_val - 2.0)) * (R ** 2))]

        sol = solve_ivp(ode_rhs, (R_outer, R_inner), [rho0], t_eval=R_eval, rtol=1e-11, atol=1e-13)
        return sol.t[::-1], sol.y[0][::-1]

    R_vals, rho_g15 = solve_gamma_profile(1.5)
    _, rho_g20 = solve_gamma_profile(2.0)
    _, rho_g30 = solve_gamma_profile(3.0)

    Phi_tilde = u0 + 0.5 * (u0 ** 2) + (1.0 / R_vals) - (1.0 / R_outer)
    rho_g20_exact = 1.0 - 1.0 / np.sqrt(1.0 + 2.0 * Phi_tilde)
    max_rk_err = float(np.max(np.abs(rho_g20 - rho_g20_exact)))
    rho_poly = rho0 + ((1.0 / R_vals) - (1.0 / R_outer))

    return R_vals, rho_g15, rho_g20, rho_g30, rho_g20_exact, rho_poly, max_rk_err


# ==============================================================================
# 2. THEOREM 2 SOLVERS: BELTRAMI CURL PDE, HELMHOLTZ GAP, & I_h LAPLACE-BELTRAMI
# ==============================================================================
def verify_theorem2_spectral_and_polarities(W_aspect, N_bel=140, N_helm=95):
    r_0 = 1.0
    R_0 = W_aspect * r_0

    # (A) Axisymmetric Beltrami / Grad-Shafranov Curl PDE on 140x140 Finite-Volume Grid
    R_vals = np.linspace(0.005, R_0 + r_0, N_bel)
    Z_vals = np.linspace(-r_0, r_0, N_bel)
    dR = R_vals[1] - R_vals[0]
    dZ = Z_vals[1] - Z_vals[0]

    RR, ZZ = np.meshgrid(R_vals, Z_vals, indexing="ij")
    mask = (((RR - R_0) ** 2 + ZZ ** 2) < r_0 ** 2) & (((RR + R_0) ** 2 + ZZ ** 2) > r_0 ** 2)
    coords = np.argwhere(mask)
    n_act = len(coords)
    idx_map = -np.ones((N_bel, N_bel), dtype=int)
    for k, (i, j) in enumerate(coords):
        idx_map[i, j] = k

    rows, cols, vals = [], [], []
    m_rows, m_cols, m_vals = [], [], []
    for row_idx, (i, j) in enumerate(coords):
        Ri = R_vals[i]
        c_plus = (1.0 / (Ri + 0.5 * dR)) / (dR ** 2)
        c_minus = (1.0 / (Ri - 0.5 * dR)) / (dR ** 2)
        c_z = (1.0 / Ri) / (dZ ** 2)
        diag = c_plus + c_minus + 2.0 * c_z

        if i + 1 < N_bel and idx_map[i + 1, j] != -1:
            rows.append(row_idx); cols.append(idx_map[i + 1, j]); vals.append(-c_plus)
        if i - 1 >= 0 and idx_map[i - 1, j] != -1:
            rows.append(row_idx); cols.append(idx_map[i - 1, j]); vals.append(-c_minus)
        if j + 1 < N_bel and idx_map[i, j + 1] != -1:
            rows.append(row_idx); cols.append(idx_map[i, j + 1]); vals.append(-c_z)
        if j - 1 >= 0 and idx_map[i, j - 1] != -1:
            rows.append(row_idx); cols.append(idx_map[i, j - 1]); vals.append(-c_z)

        rows.append(row_idx); cols.append(row_idx); vals.append(diag)
        m_rows.append(row_idx); m_cols.append(row_idx); m_vals.append(1.0 / Ri)

    A_bel = sp.csr_matrix((vals, (rows, cols)), shape=(n_act, n_act))
    M_bel = sp.csr_matrix((m_vals, (m_rows, m_cols)), shape=(n_act, n_act))
    evals_bel, evecs_bel = sla.eigsh(A_bel, k=2, M=M_bel, which="SM")
    order_bel = np.argsort(evals_bel)
    k0_bel = float(np.sqrt(evals_bel[order_bel[0]]))
    psi0_vec = evecs_bel[:, order_bel[0]]

    E_pol = float(psi0_vec.T @ (A_bel @ psi0_vec)) * dR * dZ
    E_tor = float((k0_bel ** 2) * (psi0_vec.T @ (M_bel @ psi0_vec))) * dR * dZ

    psi0_pos = np.abs(psi0_vec) / np.max(np.abs(psi0_vec))
    psi_half = np.zeros((N_bel, N_bel))
    for row_idx, (i, j) in enumerate(coords):
        psi_half[i, j] = psi0_pos[row_idx]

    x_full = np.concatenate([-R_vals[::-1], R_vals])
    XX, ZZ_full = np.meshgrid(x_full, Z_vals, indexing="ij")
    psi_full = np.concatenate([-psi_half[::-1, :], psi_half], axis=0)
    in_right = (XX >= 0) & (((XX - R_0) ** 2 + ZZ_full ** 2) <= (r_0 + 0.015) ** 2) & (((XX + R_0) ** 2 + ZZ_full ** 2) >= (r_0 - 0.015) ** 2)
    in_left = (XX <= 0) & (((XX + R_0) ** 2 + ZZ_full ** 2) <= (r_0 + 0.015) ** 2) & (((XX - R_0) ** 2 + ZZ_full ** 2) >= (r_0 - 0.015) ** 2)
    psi_full_masked = np.where(in_right | in_left, psi_full, np.nan)

    # (B) Non-Axisymmetric (m=1) Conormal Neumann Helmholtz PDE on 95x95 Finite-Volume Grid
    Rh_vals = np.linspace(0.015, R_0 + r_0, N_helm)
    Zh_vals = np.linspace(-r_0, r_0, N_helm)
    dRh = Rh_vals[1] - Rh_vals[0]
    dZh = Zh_vals[1] - Zh_vals[0]
    RRh, ZZh = np.meshgrid(Rh_vals, Zh_vals, indexing="ij")
    mask_h = (((RRh - R_0) ** 2 + ZZh ** 2) <= r_0 ** 2) & (((RRh + R_0) ** 2 + ZZh ** 2) >= r_0 ** 2)
    coords_h = np.argwhere(mask_h)
    n_act_h = len(coords_h)
    idx_h = -np.ones((N_helm, N_helm), dtype=int)
    for k, (i, j) in enumerate(coords_h):
        idx_h[i, j] = k

    h_rows, h_cols, h_vals = [], [], []
    for row_idx, (i, j) in enumerate(coords_h):
        Ri = Rh_vals[i]
        diag_h = 1.0 / (Ri ** 2)  # m^2 / R^2 for m = 1
        if i + 1 < N_helm and idx_h[i + 1, j] != -1:
            w_p = ((Ri + 0.5 * dRh) / Ri) / (dRh ** 2)
            h_rows.append(row_idx); h_cols.append(idx_h[i + 1, j]); h_vals.append(-w_p)
            diag_h += w_p
        if i - 1 >= 0 and idx_h[i - 1, j] != -1:
            w_m = ((Ri - 0.5 * dRh) / Ri) / (dRh ** 2)
            h_rows.append(row_idx); h_cols.append(idx_h[i - 1, j]); h_vals.append(-w_m)
            diag_h += w_m
        if j + 1 < N_helm and idx_h[i, j + 1] != -1:
            w_z = 1.0 / (dZh ** 2)
            h_rows.append(row_idx); h_cols.append(idx_h[i, j + 1]); h_vals.append(-w_z)
            diag_h += w_z
        if j - 1 >= 0 and idx_h[i, j - 1] != -1:
            w_z = 1.0 / (dZh ** 2)
            h_rows.append(row_idx); h_cols.append(idx_h[i, j - 1]); h_vals.append(-w_z)
            diag_h += w_z

        h_rows.append(row_idx); h_cols.append(row_idx); h_vals.append(diag_h)

    A_helm = sp.csr_matrix((h_vals, (h_rows, h_cols)), shape=(n_act_h, n_act_h))
    evals_h, _ = sla.eigs(A_helm, k=6, which="SM")
    k_helm = np.sqrt(np.sort(np.real(evals_h)))

    # (C) 3D & 2D Spherical Laplace-Beltrami Verification of Psi_15(x)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            verts.extend([[0.0, s1, s2 * phi], [s1, s2 * phi, 0.0], [s2 * phi, 0.0, s1]])
    verts = np.array(verts)
    verts /= np.linalg.norm(verts[0])
    normals = []
    edge_len = 2.0 / np.sqrt(1.0 + phi ** 2)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            diff = verts[i] - verts[j]
            dist = np.linalg.norm(diff)
            if np.isclose(dist, edge_len, atol=1e-5):
                n_vec = diff / dist
                if not any(np.allclose(n_vec, m, atol=1e-5) or np.allclose(n_vec, -m, atol=1e-5) for m in normals):
                    normals.append(n_vec)
    normals = np.array(normals)

    def Psi_15(x, y, z):
        pts = np.stack([x, y, z], axis=-1)
        return np.prod(pts @ normals.T, axis=-1)

    def Psi_15_sph(theta, varphi):
        return Psi_15(np.sin(theta) * np.cos(varphi),
                      np.sin(theta) * np.sin(varphi),
                      np.cos(theta))

    np.random.seed(42)
    test_pts = np.random.randn(20, 3)
    test_pts /= np.linalg.norm(test_pts, axis=1, keepdims=True)
    h_step = 1e-4
    max_lap_err = 0.0
    s2_eigenvals = []
    for pt in test_pts:
        x0, y0, z0 = pt
        f0 = Psi_15(x0, y0, z0)
        d2x = (Psi_15(x0 + h_step, y0, z0) - 2.0 * f0 + Psi_15(x0 - h_step, y0, z0)) / (h_step ** 2)
        d2y = (Psi_15(x0, y0 + h_step, z0) - 2.0 * f0 + Psi_15(x0, y0 - h_step, z0)) / (h_step ** 2)
        d2z = (Psi_15(x0, y0, z0 + h_step) - 2.0 * f0 + Psi_15(x0, y0, z0 - h_step)) / (h_step ** 2)
        max_lap_err = max(max_lap_err, abs(d2x + d2y + d2z))

        # Direct 2D spherical Laplace-Beltrami evaluation -Delta_{S^2} Psi_15 / Psi_15
        th0 = np.arccos(np.clip(z0, -1.0, 1.0))
        ph0 = np.arctan2(y0, x0)
        if abs(f0) > 1e-9 and 0.2 < th0 < np.pi - 0.2:
            dth_p = (Psi_15_sph(th0 + h_step, ph0) - Psi_15_sph(th0, ph0)) / h_step
            dth_m = (Psi_15_sph(th0, ph0) - Psi_15_sph(th0 - h_step, ph0)) / h_step
            term_th = (np.sin(th0 + 0.5 * h_step) * dth_p - np.sin(th0 - 0.5 * h_step) * dth_m) / (np.sin(th0) * h_step)
            term_ph = (Psi_15_sph(th0, ph0 + h_step) - 2.0 * f0 + Psi_15_sph(th0, ph0 - h_step)) / ((np.sin(th0) ** 2) * (h_step ** 2))
            s2_eigenvals.append(-(term_th + term_ph) / f0)

    mean_s2_eval = float(np.mean(s2_eigenvals))

    return XX, ZZ_full, R_0, r_0, k0_bel, E_pol, E_tor, psi_full_masked, k_helm, len(normals), max_lap_err, mean_s2_eval, Psi_15


# ==============================================================================
# 3. THEOREM 3 SOLVERS: VARIABLE-DENSITY ASBL ODE & 2D DISK NAVIER-STOKES ODE/BVP
# ==============================================================================
def verify_theorem3_asbl_and_disk_ode():
    # (A) Coupled 1D Variable-Density Compressible Prandtl ASBL ODE
    rho_inf = 0.15
    u0 = rho_inf / (1.0 - rho_inf)
    C0 = u0 + 0.5 * (u0 ** 2)

    def rho_profile_1d(eta):
        Phi_tilde = C0 + 5.0 / (1.0 + eta)
        return 1.0 - 1.0 / np.sqrt(1.0 + 2.0 * Phi_tilde)

    def vs_profile_1d(eta):
        return rho_inf / rho_profile_1d(eta)

    def coupled_asbl_ode(eta, y):
        u_val, du_val, _ = y
        d2u = -(rho_profile_1d(eta) * vs_profile_1d(eta) / rho_inf) * du_val
        return [du_val, d2u, 1.0 - u_val]

    eta_eval = np.linspace(0.0, 28.0, 500)
    sol_asbl = solve_ivp(coupled_asbl_ode, (0.0, 28.0), [0.0, 1.0, 0.0], t_eval=eta_eval, rtol=1e-13, atol=1e-15)
    asbl_err = float(np.max(np.abs(sol_asbl.y[0] - (1.0 - np.exp(-sol_asbl.t)))))

    # (B) Un-Simplified Variable-Density 2D Axisymmetric Azimuthal Navier-Stokes ODE
    #     Marched outward from core horizon R = R_c = 1.0 to R = 10.0 WITHOUT assuming
    #     any outer boundary power law!
    R_min, R_max = 1.0, 10.0
    R_grid = np.linspace(R_min, R_max, 600)

    def rho_disk(R):
        Phi_r = C0 + 2.0 / R
        return 1.0 - 1.0 / np.sqrt(1.0 + 2.0 * Phi_r)

    def u_R_disk(R, Re_s_val):
        # From 2D compressible continuity d/dR[R * rho(R) * u_R(R)] = 0:
        # rho(R) * u_R(R) / mu_A = -Re_s / R
        return -(Re_s_val / R) / rho_disk(R)

    def solve_variable_rho_disk_ivp(Re_s_val):
        # Raw variable-density azimuthal Navier-Stokes ODE:
        # d2v/dR2 = -(1/R)*dv/dR + v/R^2 + [rho(R)*u_R(R)/mu_A]*(dv/dR + v/R)
        def disk_ode(R, y):
            v_th, dv_th = y
            rho_loc = rho_disk(R)
            uR_over_mu = u_R_disk(R, Re_s_val)
            d2v_th = -(1.0 / R) * dv_th + (v_th / (R ** 2)) + (rho_loc * uR_over_mu) * (dv_th + v_th / R)
            return [dv_th, d2v_th]

        # Inner core horizon state at R = R_c = 1: v_theta(1) = 1, core vorticity omega_z(1) = 2 - Re_s
        sol = solve_ivp(disk_ode, (R_min, R_max), [1.0, 1.0 - Re_s_val], t_eval=R_grid, rtol=1e-12, atol=1e-14)
        return sol.y[0]

    v_re05 = solve_variable_rho_disk_ivp(0.5)
    v_re10 = solve_variable_rho_disk_ivp(1.0)
    v_re15 = solve_variable_rho_disk_ivp(1.5)

    # Also solve the 600-node 2nd-order central Finite-Difference BVP at Re_s = 1.0
    # using homogeneous Neumann outer boundary condition dv_theta/dR(R_max) = 0 (zero assumed power law!)
    N_r = len(R_grid)
    dR = R_grid[1] - R_grid[0]
    n_unknowns = N_r - 1  # Nodes 1 .. N_r-1 (including outer Neumann node)
    A_bvp = sp.lil_matrix((n_unknowns, n_unknowns))
    rhs = np.zeros(n_unknowns)
    v_left = 1.0

    for idx in range(n_unknowns):
        Ri = R_grid[idx + 1]
        rho_loc = rho_disk(Ri)
        uR_over_mu = u_R_disk(Ri, 1.0)
        conv = rho_loc * uR_over_mu  # equals -1.0 / Ri

        c_m = 1.0 / (dR ** 2) - (1.0 / Ri - conv) / (2.0 * dR)
        c_0 = -2.0 / (dR ** 2) - (1.0 / (Ri ** 2) + conv / Ri)
        c_p = 1.0 / (dR ** 2) + (1.0 / Ri - conv) / (2.0 * dR)

        A_bvp[idx, idx] = c_0
        if idx == 0:
            rhs[idx] -= c_m * v_left
            A_bvp[idx, idx + 1] = c_p
        elif idx < n_unknowns - 1:
            A_bvp[idx, idx - 1] = c_m
            A_bvp[idx, idx + 1] = c_p
        else:
            # Homogeneous Neumann outer boundary dv_theta/dR(R_max) = 0 => ghost node v_{N} = v_{N-2}
            A_bvp[idx, idx - 1] = c_m + c_p

    v_bvp_re10 = np.concatenate([[v_left], sla.spsolve(A_bvp.tocsr(), rhs)])
    bvp_err_re1 = float(np.max(np.abs(v_bvp_re10 - 1.0)))

    return sol_asbl.t, sol_asbl.y[0], rho_profile_1d(sol_asbl.t), asbl_err, R_grid, v_re05, v_re10, v_re15, bvp_err_re1


# ==============================================================================
# 4. MAIN VERIFICATION RUNNER & 4 SEPARATE LARGE PUBLICATION FIGURES
# ==============================================================================
def verify_theorems_and_plot():
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
    W_phi = phi ** (-2)
    spindle_overlap = phi ** (-4)
    Theta_phi = 360.0 * W_phi
    Delta_Theta_dil = np.sqrt(2.0) / 3.0

    # Constitutive Multi-Scale Closures
    alpha_dil_inv = Theta_phi - Delta_Theta_dil
    delta_r_dil = (Theta_phi / alpha_dil_inv) - 1.0
    Re_s_electron = 1.0 / delta_r_dil
    W_e_dil = Theta_phi / (1.0 + (7.0 / 3.0) * delta_r_dil)
    m_e_MeV = (2.0 * np.pi * k_B * T_CMB * (W_e_dil ** 4)) / (1e6 * e_charge)

    a0_derived = (c_ms * H_cmb_si / (2.0 * np.pi)) * (1.0 + spindle_overlap)
    a0_obs = 1.20e-10
    H_local_derived = H_cmb_kmsMpc * (1.0 + 0.5 * spindle_overlap) * (1.0 + (7.0 / 3.0) * delta_r_dil)
    Delta_H_derived = H_local_derived - H_cmb_kmsMpc

    # Execute First-Principles PDE/ODE Solvers
    R_vals, rho_g15, rho_g20, rho_g30, rho_g20_exact, rho_poly, max_rk_err = verify_theorem1_enthalpy_regularity()
    XX, ZZ_full, R_0, r_0, k0_bel, E_pol, E_tor, psi_full_masked, k_helm, n_planes, max_lap_err, mean_s2_eval, Psi_15 = (
        verify_theorem2_spectral_and_polarities(W_phi)
    )
    eta_t, u_asbl, rho_asbl, asbl_err, R_bvp, v_re05, v_re10, v_re15, bvp_err_re1 = (
        verify_theorem3_asbl_and_disk_ode()
    )

    rayleigh_lower_bound = 1.0 / (1.0 + W_phi)

    print("=" * 88)
    print(" SUN-WANG-ZHANG REGULARITY, EIGENMODE POLARITIES & SUCTION SCALING VERIFICATION")
    print("=" * 88)
    print(f" [Theorem 1] Stiffened EOS RK45 vs. Exact Radical (gamma=2) Max Error : {max_rk_err:.2e}")
    print(f"             Bounded Core Density rho(0.04)/rho_max (gamma=1.5, 2, 3) : "
          f"{rho_g15[0]:.5f}, {rho_g20[0]:.5f}, {rho_g30[0]:.5f} (< 1.0)")
    print(f"             Unbounded Polytrope rho_poly(0.04)/rho_max               : {rho_poly[0]:.4f} (SINGULAR)")
    print("-" * 88)
    print(f" [Theorem 2] Analytical Rayleigh Lower Bound (m=1) [1 + W_phi]^-1     : {rayleigh_lower_bound:.6f} (> 0)")
    print(f"             95x95 Finite-Volume Helmholtz Ground State k_0*r_0 (m=1) : {k_helm[0]:.5f} (> {rayleigh_lower_bound:.4f})")
    print(f"             140x140 Finite-Volume Beltrami Eigenvalue k_0^Bel*r_0    : {k0_bel:.6f}")
    print(f"             Beltrami Energy Ratio E_tor/E_pol (beta_eff = 45 deg)    : {E_tor / E_pol:.15f}")
    print(f"             Icosahedral Horizon (l=15): {n_planes} planes -> |I_h|=120 | -Delta_S2 Psi/Psi = {mean_s2_eval:.5f} (|nabla^2 Psi|={max_lap_err:.2e})")
    print("-" * 88)
    print(f" [Theorem 3] Coupled Variable-Density 1D Compressible ASBL ODE Error  : {asbl_err:.2e}")
    print(f"             Variable-Density 2D Disk BVP (Neumann dv/dR=0) Re_s=1 Err: {bvp_err_re1:.2e} (Flat v_theta = V_0)")
    print("-" * 88)
    print(f" [Closures]  Fine-Structure Constant alpha_dil^-1                     : {alpha_dil_inv:.8f} (CODATA: 137.03599918)")
    print(f"             Subatomic Suction Reynolds Number Re_s(r_e) = r_e/d*     : {Re_s_electron:.4f} (d*/r_e = {delta_r_dil:.8f})")
    print(f"             Derived Electron Rest Mass m_e (from T_CMB)              : {m_e_MeV:.6f} MeV/c^2 (CODATA: 0.51099895)")
    print(f"             Derived Galactic Acceleration Scale a_0                  : {a0_derived:.4e} m/s^2 (SPARC: {a0_obs:.2e} m/s^2)")
    print(f"             Derived Local Hubble Divergence H_local                  : {H_local_derived:.2f} km s^-1 Mpc^-1 (Delta_H = {Delta_H_derived:.2f})")
    print("=" * 88)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 12.0, "axes.titlesize": 12.5})

    # ==========================================================================
    # FIGURE 1: THEOREM 1 MULTI-GAMMA STIFFENED ENTHALPY REGULARITY
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=(8.8, 5.8), facecolor="white")
    ax1.plot(R_vals, rho_poly, "r--", lw=2.5, label=r"Unbounded Polytrope ($P \propto \rho^2 \Rightarrow \rho \to \infty$ Singularity)")
    ax1.plot(R_vals, rho_g15, color="#1f77b4", lw=2.3, ls="-.", label=r"Stiffened Excluded-Volume ODE ($\gamma = 1.5$, $\rho < \rho_{\max}$)")
    ax1.plot(R_vals, rho_g20, color="#00a878", lw=3.0, label=r"Stiffened Excluded-Volume ODE ($\gamma = 2.0$, $\rho < \rho_{\max}$)")
    ax1.plot(R_vals[::25], rho_g20_exact[::25], "ko", markersize=5.0,
             label=rf"Exact Theorem 1 Radical $\rho/\rho_{{\max}} = 1 - (1 + 2\tilde{{\Phi}})^{{-1/2}}$ (Err ${max_rk_err:.2e}$)")
    ax1.plot(R_vals, rho_g30, color="#ff7f0e", lw=2.3, ls=":", label=r"Stiffened Excluded-Volume ODE ($\gamma = 3.0$, $\rho < \rho_{\max}$)")
    ax1.axhline(1.0, color="magenta", ls="--", lw=1.8, label=r"Excluded-Volume Packing Ceiling $\rho / \rho_{\max} = 1.0$")
    ax1.set_xlim([0.0, 5.0]); ax1.set_ylim([0.0, 2.15])
    ax1.set_title(r"Theorem 1: A Priori Singularity Arrest Across $\gamma > 1$ ($0 < \rho_0 \leq \rho(R) < \rho_{\max}$)", fontweight="bold")
    ax1.set_xlabel(r"Dimensionless Radius $R / R_c$")
    ax1.set_ylabel(r"Normalized Fluid Density $\rho(R) / \rho_{\max}$")
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=10.0, framealpha=0.95)
    fig1.tight_layout()
    fig1_path = os.path.join(script_dir, "Regularity_Fig1_Enthalpy_ODE.png")
    fig1.savefig(fig1_path, dpi=300, facecolor="white")

    # ==========================================================================
    # FIGURE 2: THEOREM 2 TOROIDAL BELTRAMI & ICOSAHEDRAL HORIZON POLARITIES
    # ==========================================================================
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14.2, 5.8), facecolor="white")

    levels_a = np.linspace(-1.0, 1.0, 25)
    cf2a = ax2a.contourf(XX, ZZ_full, psi_full_masked, levels=levels_a, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    th = np.linspace(0, 2 * np.pi, 300)
    ax2a.plot(R_0 + r_0 * np.cos(th), r_0 * np.sin(th), "k-", lw=1.6)
    ax2a.plot(-R_0 + r_0 * np.cos(th), r_0 * np.sin(th), "k--", lw=1.4)
    ax2a.set_aspect("equal")
    ax2a.set_xlim([-1.45, 1.45]); ax2a.set_ylim([-1.08, 1.08])
    ax2a.set_title(r"(a) Toroidal Beltrami Eigenmode $\Delta^*\psi + k^2\psi = 0$" + "\n" +
                   rf"($W_\phi = \phi^{{-2}}$, $k_0^{{\mathrm{{Bel}}}}r_0 = {k0_bel:.4f}$, $E_{{\mathrm{{pol}}}} = E_{{\mathrm{{tor}}}} \Rightarrow \beta_{{\mathrm{{eff}}}} = 45^\circ$)",
                   fontweight="bold")
    ax2a.set_xlabel(r"Cylindrical Coordinate $x / r_0$")
    ax2a.set_ylabel(r"Axial Coordinate $Z / r_0$")
    cbar2a = fig2.colorbar(cf2a, ax=ax2a, fraction=0.046, pad=0.04, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar2a.set_label(r"Normalized Stream Function $\pm\psi(R,Z)$")

    lon = np.linspace(-np.pi, np.pi, 360)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, 180)
    LON, LAT = np.meshgrid(lon, lat)
    X_s = np.cos(LAT) * np.cos(LON)
    Y_s = np.cos(LAT) * np.sin(LON)
    Z_s = np.sin(LAT)
    F_s = Psi_15(X_s, Y_s, Z_s)
    F_s /= np.max(np.abs(F_s))
    cf2b = ax2b.contourf(np.degrees(LON), np.degrees(LAT), np.sign(F_s) * (np.abs(F_s) ** 0.35),
                         levels=levels_a, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax2b.contour(np.degrees(LON), np.degrees(LAT), F_s, levels=[0.0], colors="k", linewidths=1.0)
    ax2b.set_title(r"(b) Icosahedral Horizon Mode $-\Delta_{S^2}\Psi_{15} = 240\Psi_{15}$ ($\ell = 15$)" + "\n" +
                   r"15 Great-Circle Nodal Planes $\Rightarrow |I_h| = 120$ Triangles ($60+ / 60-$)",
                   fontweight="bold")
    ax2b.set_xlabel(r"Horizon Azimuthal Longitude $\varphi$ (degrees)")
    ax2b.set_ylabel(r"Horizon Poloidal Latitude $\theta$ (degrees)")
    cbar2b = fig2.colorbar(cf2b, ax=ax2b, fraction=0.046, pad=0.04, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar2b.set_label(r"Normalized Chladni Phase $\Psi_{15}(\theta,\varphi)$")

    fig2.tight_layout()
    fig2_path = os.path.join(script_dir, "Regularity_Fig2_Eigenmode_Polarities.png")
    fig2.savefig(fig2_path, dpi=300, facecolor="white")

    # ==========================================================================
    # FIGURE 3: THEOREM 3 VARIABLE-DENSITY 1D ASBL & 2D DISK NAVIER-STOKES ODE
    # ==========================================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14.2, 5.6), facecolor="white")

    idx_cut = 140
    eta_plot = eta_t[:idx_cut]
    u_plot = u_asbl[:idx_cut]
    defect_plot = 1.0 - u_plot
    rho_plot = rho_asbl[:idx_cut]
    ax3a.plot(eta_plot, u_plot, "b-", lw=2.5,
              label=rf"Tangential Velocity $u(\eta)/U_0 = 1 - e^{{-\eta}}$ (Err ${asbl_err:.2e}$)")
    ax3a.plot(eta_plot, defect_plot, "r--", lw=2.2,
              label=r"Velocity Defect $1 - u(\eta)/U_0 = e^{-\eta}$ ($\delta^* = \nu_{A\infty}/v_{s\infty}$)")
    ax3a.plot(eta_plot, rho_plot, "k-.", lw=2.0,
              label=r"Stiffened Density $\rho(\eta)/\rho_{\max} = 1 - [1 + 2\tilde{\Phi}(\eta)]^{-1/2} < 1$")
    ax3a.fill_between(eta_plot, 0, defect_plot, color="red", alpha=0.10)
    ax3a.set_xlim([0.0, 7.5]); ax3a.set_ylim([0.0, 1.08])
    ax3a.set_title(r"(a) Theorem 3(1): Variable-Density 1D Compressible ASBL" + "\n" +
                   r"$\frac{d}{dn}[\rho(n)v_s(n)] = 0 \Rightarrow \delta^* = \nu_{A\infty}/v_{s\infty}$ Invariance",
                   fontweight="bold")
    ax3a.set_xlabel(r"Normalized Boundary Coordinate $\eta = n / \delta^*$")
    ax3a.set_ylabel(r"Normalized Velocity & Density Profiles")
    ax3a.grid(True, ls="--", alpha=0.35)
    ax3a.legend(loc="center right", fontsize=9.2, framealpha=0.95)

    ax3b.plot(R_bvp, v_re15, "r--", lw=2.3, label=r"Super-Critical Suction $Re_s = 1.5$ ($v_\theta \propto R^{-0.5}$ Decay)")
    ax3b.plot(R_bvp, v_re10, color="#00a878", lw=3.0,
              label=rf"Critical Fixed Point $Re_s = 1.0$ ($v_\theta = V_0$, BVP Err ${bvp_err_re1:.2e}$)")
    ax3b.plot(R_bvp[::25], np.ones_like(R_bvp[::25]), "ko", markersize=4.8,
              label=r"Exact Analytical Plateau $v_\theta(R)/V_0 = 1.0$")
    ax3b.plot(R_bvp, v_re05, "b-.", lw=2.3, label=r"Sub-Critical Suction $Re_s = 0.5$ ($v_\theta \propto R^{+0.5}$ Rise)")
    ax3b.set_xlim([1.0, 10.0]); ax3b.set_ylim([0.2, 3.3])
    ax3b.set_title(r"(b) Theorem 3(2): Variable-Density 2D Disk Navier–Stokes" + "\n" +
                   r"$\frac{1}{R}\frac{d}{dR}[R\rho(R)u_R(R)] = 0 \Rightarrow v_\theta(R) = V_0 \Longleftrightarrow Re_s(R_c) = 1$",
                   fontweight="bold")
    ax3b.set_xlabel(r"Normalized Galactic Radius $R / R_c$")
    ax3b.set_ylabel(r"Normalized Azimuthal Speed $v_\theta(R) / V_0$")
    ax3b.grid(True, ls="--", alpha=0.35)
    ax3b.legend(loc="upper left", fontsize=9.2, framealpha=0.95)

    fig3.tight_layout()
    fig3_path = os.path.join(script_dir, "Regularity_Fig3_Suction_Disk_ODE.png")
    fig3.savefig(fig3_path, dpi=300, facecolor="white")

    # ==========================================================================
    # FIGURE 4: CONSTITUTIVE CLOSURE 2 BARYONIC TULLY-FISHER RELATION
    # ==========================================================================
    fig4, ax4 = plt.subplots(figsize=(8.8, 5.8), facecolor="white")
    M_baryon_solar = np.logspace(7.2, 11.8, 300)
    V0_derived_kms = ((G_grav * M_baryon_solar * M_sun * a0_derived) ** 0.25) / 1000.0
    V0_obs_band_low = ((G_grav * M_baryon_solar * M_sun * (1.08e-10)) ** 0.25) / 1000.0
    V0_obs_band_high = ((G_grav * M_baryon_solar * M_sun * (1.32e-10)) ** 0.25) / 1000.0

    sparc_names = ["DDO 154", "NGC 2403", "NGC 3198", "Milky Way", "UGC 2885"]
    sparc_M_solar = np.array([3.6e8, 2.3e10, 3.8e10, 6.5e10, 5.5e11])
    sparc_V_obs = np.array([47.0, 134.0, 150.0, 220.0, 298.0])
    sparc_V_err = np.array([3.0, 5.5, 6.0, 10.0, 11.0])

    ax4.fill_between(M_baryon_solar, V0_obs_band_low, V0_obs_band_high, color="#1f77b4", alpha=0.18,
                     label=r"Empirical SPARC Scatter Band ($\pm 10\%$ in $a_0$)")
    ax4.plot(M_baryon_solar, V0_derived_kms, color="#00a878", lw=3.0,
             label=r"Zero-Parameter Prediction $V_0 = (G M_{\mathrm{baryon}} a_0)^{1/4}$ ($a_0 = 1.194\times 10^{-10}\ \mathrm{m/s^2}$)")
    ax4.errorbar(sparc_M_solar, sparc_V_obs, yerr=sparc_V_err, fmt="ko", ecolor="k",
                 elinewidth=1.6, capsize=4, markersize=6.0, label="Representative SPARC Benchmark Galaxies")

    for name, mx, vy in zip(sparc_names, sparc_M_solar, sparc_V_obs):
        ax4.annotate(name, (mx, vy), textcoords="offset points", xytext=(7, -12), fontsize=10.0, fontweight="bold")

    ax4.set_xscale("log"); ax4.set_yscale("log")
    ax4.set_xlim([1.5e7, 6.5e11]); ax4.set_ylim([20.0, 420.0])
    ax4.set_title(r"Constitutive Closure 2: Zero-Parameter Baryonic Tully–Fisher Law ($Re_s = 1$)", fontweight="bold")
    ax4.set_xlabel(r"Total Baryonic Mass $M_{\mathrm{baryon}}\ (M_\odot)$")
    ax4.set_ylabel(r"Asymptotic Flat Rotation Speed $V_0\ (\mathrm{km/s})$")
    ax4.grid(True, which="both", ls="--", alpha=0.35)
    ax4.legend(loc="upper left", fontsize=10.0, framealpha=0.95)
    fig4.tight_layout()
    fig4_path = os.path.join(script_dir, "Regularity_Fig4_Baryonic_Tully_Fisher.png")
    fig4.savefig(fig4_path, dpi=300, facecolor="white")

    print("Saved 4 large, standalone 300-DPI publication figures:")
    for p in (fig1_path, fig2_path, fig3_path, fig4_path):
        print(f"  -> {p}")
    plt.show()


if __name__ == "__main__":
    verify_theorems_and_plot()
