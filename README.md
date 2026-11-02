# Spindle Torus PDE & Multi-Scale Continuum Verification Suite

Standalone Python numerical verification scripts (Conservative Finite-Volume, Central Finite-Difference, and Adaptive RK45 Runge-Kutta solvers) accompanying the mathematical physics manuscripts by **Colt Camron Lien** (University of Arkansas at Monticello).

## Installation

```bash
pip install -r requirements.txt
```

## Manuscript-to-Script Verification Map

| Script | Verified Manuscript & Theorems | Numerical Method | Max Error vs. Exact Theorem |
| :--- | :--- | :--- | :--- |
| `unrigged_pde_test.py` | *Exact Non-Singular $\Gamma \to 1$ Acoustic Horizons and Poloidal-to-Radial Eigenmode Transitions in a Compressible Spindle Torus* (Proposition 1, Table II, Figs. 2–3) | $95 \times 95$ Conservative Finite-Volume Helmholtz Solver (`scipy.sparse.linalg.eigs`) | Exact Neumann Zero-Flux ($\partial\psi/\partial n = 0$) |
| `lensing_and_singularity_pde.py` | *Exact Non-Singular $\Gamma \to 1$ Acoustic Horizons...* (Theorem 1, Table I, Fig. 1) & *Exact Flat Rotation Curves and Asymptotic Lensing...* (Corollary 3, Fig. 3) | Adaptive RK45 ODE Solver (`solve_ivp`, `rtol=1e-11`) & Fermat Ray Integration | $2.75 \times 10^{-12}$ |
| `navier-stokes-flat-rotation.py` | *Exact Flat Rotation Curves and Asymptotic Lensing in an Axisymmetric Navier-Stokes Suction Vortex* (Theorem 1, Table I, Fig. 1) | 2nd-Order Central Finite-Difference Euler-Cauchy Solver ($N_r = 600$) | $3.21 \times 10^{-12}$ at $Re_s = 1$ |
| `sparc_and_running_pde.py` | *Exact Flat Rotation Curves and Asymptotic Lensing...* (Corollaries 1–2, Table II, Fig. 2) | Regularized Burgers-Rott Distributed Sink Integration across SPARC Galaxies | Matches SPARC within $1\sigma$ |
| `regularity_and_multiscale_verification.py` | *A Priori Sun-Wang-Zhang Regularity, Conormal Spectral Mass Gaps, and Multi-Scale Suction Scaling in a Stiffened Compressible Superfluid* (Theorems 1–3, Corollaries 1–2, Table I, Fig. 1) | Multi-$\gamma$ Stiffened Enthalpy RK45 Solver & Zero-Parameter BTFR ($a_0$) Verification | $2.75 \times 10^{-12}$ |

## Usage

Run any script directly from the terminal or VS Code to print the numerical verification table and generate the 300-DPI publication figures:

```bash
python regularity_and_multiscale_verification.py
python lensing_and_singularity_pde.py
python unrigged_pde_test.py
python navier-stokes-flat-rotation.py
python sparc_and_running_pde.py
```

## License

Released under the MIT License.
