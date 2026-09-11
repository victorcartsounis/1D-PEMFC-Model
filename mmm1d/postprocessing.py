"""
Post-processing -- spatial profile extraction and generation of the
potentials, fluxes and polarization curve plots.

Equivalent to the original MMM1D_postprocessing.m.
"""
import numpy as np
import matplotlib.pyplot as plt

from .model import NSTATE, NREGION, DOMAINS, VAR_NAMES

FIG_NAMES = ["Potentials", "Fluxes"]

UNIT_SCALE = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1e-4, 1e-4, 1e-4, 1e2, 1e2, 1e2, 1e2, 1e2],
])

QUANTITY = [
    [r"$\phi_e\ [\mathrm{V}]$", r"$\phi_p\ [\mathrm{V}]$", r"$T\ [\mathrm{K}]$",
     r"$\lambda\ [-]$", r"$w_{H_2O}\ [-]$", r"$w_{O_2}\ [-]$", r"$s\ [-]$",
     r"$P_\mathrm{gas}\ [\mathrm{Pa}]$"],
    [r"$j_e\ [\mathrm{A/cm}^2]$", r"$j_p\ [\mathrm{A/cm}^2]$", r"$j_T\ [\mathrm{W/cm}^2]$",
     r"$j_\lambda\ [\mu\mathrm{mol/cm}^2\mathrm{s}]$", r"$j_{H_2O}\ [\mu\mathrm{g/cm}^2\mathrm{s}]$",
     r"$j_{O_2}\ [\mu\mathrm{g/cm}^2\mathrm{s}]$", r"$j_s\ [\mu\mathrm{mol/cm}^2\mathrm{s}]$",
     r"$\rho_\mathrm{gas}\cdot u_\mathrm{gas}\ [\mathrm{\mu g}/\mathrm{cm}^2\mathrm{s}]$"],
]


def extract_profiles(sol, p, n_dense=201):
    """Extracts continuous, physical spatial profiles from the stacked
    (80, m) solve_bvp solution, filling with NaN the regions where each
    of the 8 variables is not physically active (equivalent to the
    "fill solution on inactive domains with NaN" step in the original
    MATLAB code).

    Returns
    -------
    x_all : np.ndarray, shape (5*n_dense,)
        Physical position [m], concatenated region by region.
    y_all : np.ndarray, shape (16, 5*n_dense)
        The 16 variables (potential+flux interleaved) in the same order
        as VAR_NAMES; NaN where the variable is inactive in that region.
    """
    s_dense = np.linspace(0.0, 1.0, n_dense)
    Yfull = sol.sol(s_dense)  # (80, n_dense)

    x_all = np.empty(NREGION * n_dense)
    y_all = np.empty((NSTATE, NREGION * n_dense))

    for d in range(1, NREGION + 1):
        cols = slice((d - 1) * n_dense, d * n_dense)
        x_all[cols] = p.Lsum[d - 1] + s_dense * p.L[d - 1]
        block = Yfull[(d - 1) * NSTATE: d * NSTATE, :]
        for var in range(len(VAR_NAMES)):
            active = DOMAINS[var, d - 1]
            rows = slice(2 * var, 2 * var + 2)
            y_all[rows, cols] = block[rows, :] if active else np.nan

    return x_all, y_all


def plot_potentials_and_fluxes(result, n_dense=201, figsize=(14, 5)):
    """Generates the 'Potentials' and 'Fluxes' figures (one curve per
    cell voltage, one subplot per each of the 8 variables), equivalent
    to the first block of MMM1D_postprocessing.m."""
    p = result.params
    profiles = [extract_profiles(sol, p, n_dense) for sol in result.SOL]
    cmap = plt.get_cmap("jet")
    colors = [cmap(k / max(1, result.Np - 1)) for k in range(result.Np)]

    figs = []
    for m in range(2):  # 0: potentials, 1: fluxes
        fig, axes = plt.subplots(2, 4, figsize=figsize, num=FIG_NAMES[m])
        axes = axes.ravel()
        for n in range(8):
            ax = axes[n]
            us = UNIT_SCALE[m, n]
            active_regions = np.where(DOMAINS[n, :])[0]
            x_lo = p.Lsum[active_regions[0]]
            x_hi = p.Lsum[active_regions[-1] + 1]
            for k in range(result.Np):
                x_all, y_all = profiles[k]
                row = 2 * n + m
                ax.plot(x_all * 1e6, y_all[row, :] * us, color=colors[k],
                        label=f"{result.U[k]:.2f} V")
            ax.set_xlim(x_lo * 1e6, x_hi * 1e6)
            ax.set_xlabel("x [um]")
            ax.set_ylabel(QUANTITY[m][n])
            for xi in p.Lsum[1:-1]:
                ax.axvline(xi * 1e6, color="k", linewidth=0.8)
            ax.grid(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=1)
        fig.suptitle(FIG_NAMES[m])
        fig.tight_layout(rect=[0, 0, 0.94, 0.96])
        figs.append(fig)
    return figs


def plot_polarization_curve(result, figsize=(7, 5)):
    """Generates the polarization curve (cell voltage and power density
    vs. current density), equivalent to the second block of
    MMM1D_postprocessing.m."""
    I, U = result.I, result.U
    P = I * U

    fig, ax1 = plt.subplots(figsize=figsize, num="Polarization curve")
    ax1.plot(I, U, "b-o", label="Cell voltage")
    ax1.set_xlabel("Current density [A/cm$^2$]")
    ax1.set_ylabel("Cell voltage [V]", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_xlim(0, max(I.max(), 1e-12))
    ax1.set_ylim(0, max(U.max(), P.max()))

    ax2 = ax1.twinx()
    ax2.plot(I, P, "r-s", label="Power density")
    ax2.set_ylabel("Power density [W/cm$^2$]", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(0, max(U.max(), P.max()))

    fig.tight_layout()
    return fig
