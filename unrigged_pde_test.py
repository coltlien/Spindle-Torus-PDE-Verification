import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from matplotlib import cm


def solve_exact_cylindrical_spindle_helmholtz(W_aspect, N_grid=90, m_toroidal=1, n_modes=6):
    """
    Solves the EXACT 3D cylindrical Helmholtz PDE:
        (1/R) d/dR (R d_psi/dR) + d^2_psi/dZ^2 - (m^2 / R^2) psi = -k^2 psi
    on the true geometric Spindle Torus exterior crescent domain:
        Omega = { (R, Z) : (R - R_0)^2 + Z^2 <= r_0^2  AND  (R + R_0)^2 + Z^2 >= r_0^2 }
    using a conservative finite-volume discretization that automatically enforces
    the exact homogeneous Neumann boundary condition (d_psi/dn = 0) on all curved boundaries.
    """
    r_0 = 1.0
    R_0 = W_aspect * r_0

    R_min = max(1e-3, R_0 - r_0)
    R_max = R_0 + r_0
    Z_min = -r_0
    Z_max = r_0

    R_vals = np.linspace(R_min + 0.01, R_max, N_grid)
    Z_vals = np.linspace(Z_min, Z_max, N_grid)
    dR = R_vals[1] - R_vals[0]
    dZ = Z_vals[1] - Z_vals[0]

    RR, ZZ = np.meshgrid(R_vals, Z_vals, indexing='ij')

    in_outer_tube = ((RR - R_0) ** 2 + ZZ ** 2) <= (r_0 ** 2)
    outside_inner_spindle = ((RR + R_0) ** 2 + ZZ ** 2) >= (r_0 ** 2)
    mask = in_outer_tube & outside_inner_spindle

    active_coords = np.argwhere(mask)
    n_active = len(active_coords)
    index_map = -np.ones((N_grid, N_grid), dtype=int)
    for idx_val, (i, j) in enumerate(active_coords):
        index_map[i, j] = idx_val

    rows, cols, vals = [], [], []

    for row_idx, (i, j) in enumerate(active_coords):
        Ri = R_vals[i]
        diag = (m_toroidal ** 2) / (Ri ** 2)

        if i + 1 < N_grid and index_map[i + 1, j] != -1:
            R_half = Ri + 0.5 * dR
            c_plus = (R_half / Ri) / (dR ** 2)
            rows.append(row_idx); cols.append(index_map[i + 1, j]); vals.append(-c_plus)
            diag += c_plus

        if i - 1 >= 0 and index_map[i - 1, j] != -1:
            R_half = Ri - 0.5 * dR
            c_minus = (R_half / Ri) / (dR ** 2)
            rows.append(row_idx); cols.append(index_map[i - 1, j]); vals.append(-c_minus)
            diag += c_minus

        if j + 1 < N_grid and index_map[i, j + 1] != -1:
            c_z = 1.0 / (dZ ** 2)
            rows.append(row_idx); cols.append(index_map[i, j + 1]); vals.append(-c_z)
            diag += c_z

        if j - 1 >= 0 and index_map[i, j - 1] != -1:
            c_z = 1.0 / (dZ ** 2)
            rows.append(row_idx); cols.append(index_map[i, j - 1]); vals.append(-c_z)
            diag += c_z

        rows.append(row_idx); cols.append(row_idx); vals.append(diag)

    H_op = sp.csr_matrix((vals, (rows, cols)), shape=(n_active, n_active))

    evals, evecs = sla.eigs(H_op, k=n_modes, which='SM')
    k_sq = np.abs(np.real(evals))
    order = np.argsort(k_sq)
    k_vals = np.sqrt(k_sq[order])
    evecs = np.real(evecs[:, order])

    fields_2d = []
    for m_idx in range(n_modes):
        grid_field = np.full((N_grid, N_grid), np.nan)
        vec = evecs[:, m_idx]
        vec = vec / np.max(np.abs(vec))
        for row_idx, (i, j) in enumerate(active_coords):
            grid_field[i, j] = vec[row_idx]
        fields_2d.append(grid_field)

    return R_vals, Z_vals, R_0, r_0, k_vals, fields_2d


def sweep_aspect_ratios(W_list, N_grid=65, m_toroidal=1, n_modes=4):
    all_k = np.zeros((len(W_list), n_modes))
    for idx_w, W in enumerate(W_list):
        _, _, _, _, k_vals, _ = solve_exact_cylindrical_spindle_helmholtz(
            W_aspect=W, N_grid=N_grid, m_toroidal=m_toroidal, n_modes=n_modes
        )
        all_k[idx_w, :] = k_vals
    return all_k


def solve_homogeneous_navier_stokes_suction_vortex(R_min=1.0, R_max=30.0, Nr=500, Re_s_list=(0.75, 1.0, 1.25, 1.5, 2.0)):
    """
    Solves the EXACT homogeneous 1D cylindrical Navier-Stokes azimuthal momentum ODE (RHS = 0):
        d^2v/dR^2 + ((1 + Re_s)/R) dv/dR - ((1 - Re_s)/R^2) v = 0
    where Re_s = (v_s * R_c) / nu_A = R_c / delta^* is the suction Reynolds number.
    Characteristic roots: (p + 1)(p + Re_s - 1) = 0  =>  p_1 = -1,  p_2 = 1 - Re_s.
    """
    R = np.linspace(R_min, R_max, Nr)
    dR = R[1] - R[0]
    profiles = {}

    for Re_s in Re_s_list:
        main = np.zeros(Nr)
        upper = np.zeros(Nr - 1)
        lower = np.zeros(Nr - 1)
        rhs = np.zeros(Nr)

        # Inner core horizon boundary condition: v(R_min) = 1.0
        main[0] = 1.0
        rhs[0] = 1.0

        # Interior homogeneous Navier-Stokes operator (RHS = 0)
        for i in range(1, Nr - 1):
            Ri = R[i]
            adv_coeff = (1.0 + Re_s) / Ri
            pot_coeff = -(1.0 - Re_s) / (Ri ** 2)

            lower[i - 1] = (1.0 / dR ** 2) - (adv_coeff / (2.0 * dR))
            main[i] = (-2.0 / dR ** 2) + pot_coeff
            upper[i] = (1.0 / dR ** 2) + (adv_coeff / (2.0 * dR))
            rhs[i] = 0.0

        # Asymptotic outflow boundary condition at R = R_max: R * dv/dR - (1 - Re_s) * v = 0
        lower[Nr - 2] = -1.0
        main[Nr - 1] = 1.0 - (1.0 - Re_s) * (dR / R[-1])
        rhs[Nr - 1] = 0.0

        mat = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format='csr')
        v_sol = sla.spsolve(mat, rhs)
        profiles[Re_s] = v_sol

    return R, profiles


def launch_unrigged_suite():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    W_phi = phi ** -2

    print("=" * 75)
    print(" PDE SUITE: EXACT CYLINDRICAL HELMHOLTZ & HOMOGENEOUS NAVIER-STOKES")
    print("=" * 75)
    R_vals, Z_vals, R_0, r_0, k_vals, fields_2d = solve_exact_cylindrical_spindle_helmholtz(
        W_aspect=W_phi, N_grid=95, m_toroidal=1, n_modes=6
    )
    for m_idx, kv in enumerate(k_vals):
        print(f"   Mode n = {m_idx}:  k_{m_idx} = {kv:8.5f}   (Ratio k_{m_idx}/k_0 = {kv / k_vals[0]:.5f})")

    print("\nRunning blind aspect-ratio sweep W = R_0 / r_0 in [0.10, 0.95]...")
    W_list = np.linspace(0.10, 0.95, 28)
    sweep_k = sweep_aspect_ratios(W_list, N_grid=60, m_toroidal=1, n_modes=4)

    print("Solving homogeneous Navier-Stokes radial-suction vortex PDE (RHS = 0)...")
    R_ns, ns_profiles = solve_homogeneous_navier_stokes_suction_vortex()
    print("=" * 75)

    fig = plt.figure(figsize=(16, 7.6), facecolor='#0b0f19')
    ax3d = fig.add_axes([0.02, 0.06, 0.43, 0.80], projection='3d', facecolor='#0b0f19')
    ax2d = fig.add_axes([0.50, 0.10, 0.40, 0.75], facecolor='#0b0f19')
    cax  = fig.add_axes([0.915, 0.10, 0.018, 0.75], facecolor='#0b0f19')

    RR, ZZ = np.meshgrid(R_vals, Z_vals, indexing='ij')
    th_circle = np.linspace(0, 2 * np.pi, 300)

    def draw_3d_from_actual_pde(mode_idx):
        ax3d.cla()
        ax3d.set_facecolor('#0b0f19')

        u_core = np.arccos(-R_0 / r_0)
        u_ext = np.linspace(-u_core + 0.06, u_core - 0.06, 45)
        v_tor = np.linspace(0, 2 * np.pi, 55)
        U, V = np.meshgrid(u_ext, v_tor, indexing='ij')

        R_surf = R_0 + r_0 * np.cos(U)
        Z_surf = r_0 * np.sin(U)
        X_surf = R_surf * np.cos(V)
        Y_surf = R_surf * np.sin(V)

        field = fields_2d[mode_idx]
        psi_boundary = np.zeros_like(u_ext)
        for idx_u, uu in enumerate(u_ext):
            r_target = R_0 + r_0 * np.cos(uu)
            z_target = r_0 * np.sin(uu)
            i_near = np.argmin(np.abs(R_vals - r_target))
            j_near = np.argmin(np.abs(Z_vals - z_target))
            sub = field[max(0, i_near-2):min(len(R_vals), i_near+3),
                        max(0, j_near-2):min(len(Z_vals), j_near+3)]
            valid = sub[~np.isnan(sub)]
            psi_boundary[idx_u] = np.mean(valid) if len(valid) > 0 else 0.0

        wave_3d = psi_boundary[:, None] * np.cos(1.0 * V)
        w_max = max(np.max(np.abs(wave_3d)), 1e-6)
        norm_wave = 0.5 + 0.5 * (wave_3d / w_max)
        face_colors = cm.coolwarm(norm_wave)
        face_colors[..., 3] = 0.45

        ax3d.plot_surface(X_surf, Y_surf, Z_surf, facecolors=face_colors,
                          rstride=1, cstride=1, linewidth=0.15, edgecolor='cyan')

        u_in = np.linspace(u_core, 2 * np.pi - u_core, 35)
        U_in, V_in = np.meshgrid(u_in, v_tor, indexing='ij')
        X_in = (R_0 + r_0 * np.cos(U_in)) * np.cos(V_in)
        Y_in = (R_0 + r_0 * np.cos(U_in)) * np.sin(V_in)
        Z_in = r_0 * np.sin(U_in)
        ax3d.plot_surface(X_in, Y_in, Z_in, color='#ff3366', alpha=0.65,
                          rstride=1, cstride=2, edgecolor='yellow', linewidth=0.2)

        ax3d.set_title(f"True 3D PDE Eigenfunction $\\psi(R,Z)\\cos\\varphi$ (Mode n={mode_idx})",
                       color='white', fontsize=11)
        ax3d.set_xlim([-1.3, 1.3]); ax3d.set_ylim([-1.3, 1.3]); ax3d.set_zlim([-1.1, 1.1])
        ax3d.set_axis_off()

    def show_helmholtz_mode(mode_idx):
        draw_3d_from_actual_pde(mode_idx)
        ax2d.cla(); cax.cla()
        ax2d.set_facecolor('#0b0f19')
        cax.set_visible(True)

        field = fields_2d[mode_idx]
        ratio = k_vals[mode_idx] / k_vals[0]

        fig.suptitle(
            f"Cylindrical Helmholtz PDE on Exact Spindle Torus Crescent ($W = R_0/r_0 = {W_phi:.4f}$)\n"
            f"Mode n = {mode_idx}: $k_{mode_idx}$ = {k_vals[mode_idx]:.5f} ($k_{mode_idx}/k_0$ = {ratio:.4f}) | "
            f"Keys: [0]-[5] Modes | [w] Aspect Sweep | [n] Navier-Stokes",
            color='white', fontsize=12, fontweight='bold'
        )

        cf = ax2d.contourf(RR, ZZ, field, levels=28, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        ax2d.contourf(-RR, ZZ, -field, levels=28, cmap='coolwarm', vmin=-1.0, vmax=1.0)

        cb = fig.colorbar(cf, cax=cax)
        cb.set_label("Solved Eigenvector $\\psi(R, Z)$", color='white')
        cax.tick_params(colors='white')
        for spine in cax.spines.values():
            spine.set_edgecolor('white')

        ax2d.plot(R_0 + r_0 * np.cos(th_circle), r_0 * np.sin(th_circle), 'c-', lw=1.8, label='Outer Torus Boundary')
        ax2d.plot(-R_0 + r_0 * np.cos(th_circle), r_0 * np.sin(th_circle), 'y--', lw=1.6, label='Inner Spindle Core Boundary')

        ax2d.set_aspect('equal')
        ax2d.set_xlim([-1.45, 1.45]); ax2d.set_ylim([-1.1, 1.1])
        ax2d.set_title(f"[Key {mode_idx}] Exact Cylindrical Helmholtz Eigenmode n = {mode_idx}", color='white', fontsize=11)
        ax2d.set_xlabel("Cylindrical Radius $x / r_0$", color='white')
        ax2d.set_ylabel("Axial Coordinate $Z / r_0$", color='white')
        ax2d.tick_params(colors='white')
        for spine in ax2d.spines.values():
            spine.set_edgecolor('white')
        ax2d.legend(loc='upper right', facecolor='#151c2c', edgecolor='cyan', labelcolor='white', fontsize=8)
        fig.canvas.draw_idle()

    def show_aspect_sweep():
        ax2d.cla(); cax.cla()
        cax.set_visible(False)
        ax2d.set_aspect('auto')  # Reset aspect ratio so plot fills the panel!
        ax2d.set_facecolor('#0b0f19')

        colors = ['#00ffcc', '#ff9933', '#3399ff', '#ff3366']
        for m_idx in range(1, 4):
            ratios = sweep_k[:, m_idx] / sweep_k[:, 0]
            ax2d.plot(W_list, ratios, marker='o', markersize=4.5, lw=2.2, color=colors[m_idx],
                      label=f'Solved PDE Ratio $k_{m_idx}(W) / k_0(W)$')

        ax2d.axvline(phi ** -2, color='yellow', ls='--', lw=1.8, label=f'Spindle Peak $W_\\phi = \\phi^{{-2}} \\approx {phi**-2:.4f}$')
        ax2d.axvline(phi ** -1, color='cyan', ls=':', lw=1.8, label=f'Conjugate Trough $\\phi^{{-1}} \\approx {phi**-1:.4f}$')

        ax2d.set_xlim([0.10, 0.95]); ax2d.set_ylim([1.75, 3.75])
        ax2d.set_title("[Key w] Blind Aspect-Ratio Sweep: Eigenvalue Ratios $k_n(W) / k_0(W)$ vs. $W = R_0/r_0$",
                       color='white', fontsize=11, fontweight='bold')
        ax2d.set_xlabel("Torus Aspect Ratio $W = R_0 / r_0$", color='white')
        ax2d.set_ylabel("Harmonic Eigenvalue Ratio $k_n(W) / k_0(W)$", color='white')
        ax2d.grid(True, ls='--', alpha=0.25, color='white')
        ax2d.tick_params(colors='white')
        for spine in ax2d.spines.values():
            spine.set_edgecolor('white')
        ax2d.legend(loc='upper left', facecolor='#151c2c', edgecolor='cyan', labelcolor='white', fontsize=8.5)
        fig.canvas.draw_idle()

    def show_navier_stokes():
        ax2d.cla(); cax.cla()
        cax.set_visible(False)
        ax2d.set_aspect('auto')  # Reset aspect ratio so plot fills the panel!
        ax2d.set_facecolor('#0b0f19')

        palette = {
            0.75: ('#cc66ff', r'$Re_s = 0.75$ ($v_\theta \propto R^{+0.25}$ Rising Sheet)'),
            1.0:  ('#00ffcc', r'$Re_s = 1.00$ ($\delta^* = R_c$: Exact Flat Rotation $v_\theta = \mathrm{const}$)'),
            1.25: ('#ffaa00', r'$Re_s = 1.25$ ($v_\theta \propto R^{-0.25}$ Mild Falloff)'),
            1.5:  ('#3399ff', r'$Re_s = 1.50$ ($v_\theta \propto R^{-1/2}$ Keplerian-like Decay)'),
            2.0:  ('#ff4444', r'$Re_s = 2.00$ ($v_\theta \propto R^{-1}$ Potential Vortex Decay)')
        }
        for Re_s, v_prof in ns_profiles.items():
            col, lbl = palette[Re_s]
            ax2d.plot(R_ns, v_prof, lw=2.6, color=col, label=lbl)

        ax2d.set_xlim([1.0, 30.0]); ax2d.set_ylim([0.0, 1.45])
        ax2d.set_title("[Key n] Exact Homogeneous Navier-Stokes Suction Vortex ($v_\\theta \\propto R^{1 - Re_s}$, RHS = 0)",
                       color='white', fontsize=11, fontweight='bold')
        ax2d.set_xlabel("Dimensionless Radius $R / R_c$", color='white')
        ax2d.set_ylabel("Normalized Azimuthal Velocity $v_\\theta(R) / V_{\\mathrm{core}}$", color='white')
        ax2d.grid(True, ls='--', alpha=0.25, color='white')
        ax2d.tick_params(colors='white')
        for spine in ax2d.spines.values():
            spine.set_edgecolor('white')
        ax2d.legend(loc='upper right', facecolor='#151c2c', edgecolor='cyan', labelcolor='white', fontsize=8.5)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ['0', '1', '2', '3', '4', '5']:
            show_helmholtz_mode(int(event.key))
        elif event.key == 'w':
            show_aspect_sweep()
        elif event.key == 'n':
            show_navier_stokes()

    fig.canvas.mpl_connect('key_press_event', on_key)
    show_helmholtz_mode(0)
    plt.show()


if __name__ == "__main__":
    launch_unrigged_suite()