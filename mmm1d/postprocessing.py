"""Spatial profile extraction and the potentials, fluxes and polarization plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .model import SweepResult
from .params import Params
from .state import (ACTIVE_REGIONS, N_REGIONS, N_STATE, Quantity, Region,
                    block_rows)

#: Axis labels for the quantity itself and for its flux.
VALUE_LABELS: dict[Quantity, str] = {
    Quantity.PHI_E: r"$\phi_e\ [\mathrm{V}]$",
    Quantity.PHI_P: r"$\phi_p\ [\mathrm{V}]$",
    Quantity.T: r"$T\ [\mathrm{K}]$",
    Quantity.LAMBDA: r"$\lambda\ [-]$",
    Quantity.W_H2O: r"$w_{H_2O}\ [-]$",
    Quantity.W_O2: r"$w_{O_2}\ [-]$",
    Quantity.SATURATION: r"$s\ [-]$",
    Quantity.P_GAS: r"$P_\mathrm{gas}\ [\mathrm{Pa}]$",
}

FLUX_LABELS: dict[Quantity, str] = {
    Quantity.PHI_E: r"$j_e\ [\mathrm{A/cm}^2]$",
    Quantity.PHI_P: r"$j_p\ [\mathrm{A/cm}^2]$",
    Quantity.T: r"$j_T\ [\mathrm{W/cm}^2]$",
    Quantity.LAMBDA: r"$j_\lambda\ [\mu\mathrm{mol/cm}^2\mathrm{s}]$",
    Quantity.W_H2O: r"$j_{H_2O}\ [\mu\mathrm{g/cm}^2\mathrm{s}]$",
    Quantity.W_O2: r"$j_{O_2}\ [\mu\mathrm{g/cm}^2\mathrm{s}]$",
    Quantity.SATURATION: r"$j_s\ [\mu\mathrm{mol/cm}^2\mathrm{s}]$",
    Quantity.P_GAS: r"$\rho_\mathrm{gas}\cdot u_\mathrm{gas}\ "
                    r"[\mathrm{\mu g}/\mathrm{cm}^2\mathrm{s}]$",
}

#: SI -> plot units. Quantities are plotted as-is; fluxes are rescaled.
VALUE_SCALE: dict[Quantity, float] = {q: 1.0 for q in Quantity}
FLUX_SCALE: dict[Quantity, float] = {
    Quantity.PHI_E: 1e-4, Quantity.PHI_P: 1e-4, Quantity.T: 1e-4,
    Quantity.LAMBDA: 1e2, Quantity.W_H2O: 1e2, Quantity.W_O2: 1e2,
    Quantity.SATURATION: 1e2, Quantity.P_GAS: 1e2,
}


def extract_profiles(sol, params: Params,
                     n_dense: int = 201) -> tuple[np.ndarray, np.ndarray]:
    """Continuous spatial profiles from a stacked solve_bvp solution.

    Returns physical positions [m] and the 16 state rows, concatenated region by
    region, with NaN wherever a quantity is not physically defined.
    """
    s_dense = np.linspace(0.0, 1.0, n_dense)
    stacked = sol.sol(s_dense)

    positions = np.empty(N_REGIONS * n_dense)
    profiles = np.empty((N_STATE, N_REGIONS * n_dense))

    for region in Region:
        columns = slice(region * n_dense, (region + 1) * n_dense)
        positions[columns] = params.Lsum[region] + s_dense * params.L[region]
        block = stacked[block_rows(region)]
        for quantity in Quantity:
            active = region in ACTIVE_REGIONS[quantity]
            profiles[quantity.rows, columns] = block[quantity.rows] if active else np.nan

    return positions, profiles


def plot_potentials_and_fluxes(result: SweepResult, n_dense: int = 201,
                               figsize: tuple[float, float] = (14, 5)) -> list:
    """One figure of the eight quantities and one of their eight fluxes."""
    params = result.params
    profiles = [extract_profiles(sol, params, n_dense) for sol in result.solutions]
    colormap = plt.get_cmap("jet")
    colors = [colormap(i / max(1, result.n_voltages - 1))
              for i in range(result.n_voltages)]

    figures = []
    for name, labels, scales, row_of in (
        ("Potentials", VALUE_LABELS, VALUE_SCALE, lambda q: q.value_row),
        ("Fluxes", FLUX_LABELS, FLUX_SCALE, lambda q: q.flux_row),
    ):
        figure, axes = plt.subplots(2, 4, figsize=figsize, num=name)
        for quantity, axis in zip(Quantity, axes.ravel()):
            active = ACTIVE_REGIONS[quantity]
            row = row_of(quantity)
            for i, (voltage, (positions, values)) in enumerate(zip(result.voltages,
                                                                   profiles)):
                axis.plot(positions * 1e6, values[row] * scales[quantity],
                          color=colors[i], label=f"{voltage:.2f} V")
            axis.set_xlim(params.Lsum[active[0]] * 1e6,
                          params.Lsum[active[-1] + 1] * 1e6)
            axis.set_xlabel("x [um]")
            axis.set_ylabel(labels[quantity])
            for interface in params.Lsum[1:-1]:
                axis.axvline(interface * 1e6, color="k", linewidth=0.8)
            axis.grid(False)
        handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
        figure.legend(handles, legend_labels, loc="upper right", fontsize=8, ncol=1)
        figure.suptitle(name)
        figure.tight_layout(rect=[0, 0, 0.94, 0.96])
        figures.append(figure)
    return figures


def plot_polarization_curve(result: SweepResult,
                            figsize: tuple[float, float] = (7, 5)):
    """Cell voltage and power density against current density."""
    current = result.current_densities
    voltage = result.voltages
    power = result.power_densities
    upper = max(voltage.max(), power.max())

    figure, voltage_axis = plt.subplots(figsize=figsize, num="Polarization curve")
    voltage_axis.plot(current, voltage, "b-o", label="Cell voltage")
    voltage_axis.set_xlabel("Current density [A/cm$^2$]")
    voltage_axis.set_ylabel("Cell voltage [V]", color="b")
    voltage_axis.tick_params(axis="y", labelcolor="b")
    voltage_axis.set_xlim(0, max(current.max(), 1e-12))
    voltage_axis.set_ylim(0, upper)

    power_axis = voltage_axis.twinx()
    power_axis.plot(current, power, "r-s", label="Power density")
    power_axis.set_ylabel("Power density [W/cm$^2$]", color="r")
    power_axis.tick_params(axis="y", labelcolor="r")
    power_axis.set_ylim(0, upper)

    figure.tight_layout()
    return figure
