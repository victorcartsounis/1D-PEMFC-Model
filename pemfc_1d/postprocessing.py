"""Spatial profile extraction and the potentials, fluxes, sensitivity and
polarization plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .model import SweepResult
from .params import Params
from .state import (ACTIVE_REGIONS, N_REGIONS, N_STATE, Quantity, Region,
                    block_rows)

#: LaTeX symbol and unit of the quantity itself and of its flux, kept apart so
#: that a derived figure can build its own axis label -- the sensitivity plots
#: label an axis ``d(phi_e)/d(theta) [V]`` -- without a second table of units
#: to fall out of step with this one.
VALUE_SYMBOLS: dict[Quantity, str] = {
    Quantity.PHI_E: r"\phi_e",
    Quantity.PHI_P: r"\phi_p",
    Quantity.T: r"T",
    Quantity.LAMBDA: r"\lambda",
    Quantity.W_H2O: r"w_{H_2O}",
    Quantity.W_O2: r"w_{O_2}",
    Quantity.SATURATION: r"s",
    Quantity.P_GAS: r"P_\mathrm{gas}",
}

VALUE_UNITS: dict[Quantity, str] = {
    Quantity.PHI_E: r"\mathrm{V}",
    Quantity.PHI_P: r"\mathrm{V}",
    Quantity.T: r"\mathrm{K}",
    Quantity.LAMBDA: r"-",
    Quantity.W_H2O: r"-",
    Quantity.W_O2: r"-",
    Quantity.SATURATION: r"-",
    Quantity.P_GAS: r"\mathrm{Pa}",
}

FLUX_SYMBOLS: dict[Quantity, str] = {
    Quantity.PHI_E: r"j_e",
    Quantity.PHI_P: r"j_p",
    Quantity.T: r"j_T",
    Quantity.LAMBDA: r"j_\lambda",
    Quantity.W_H2O: r"j_{H_2O}",
    Quantity.W_O2: r"j_{O_2}",
    Quantity.SATURATION: r"j_s",
    Quantity.P_GAS: r"\rho_\mathrm{gas}\cdot u_\mathrm{gas}",
}

FLUX_UNITS: dict[Quantity, str] = {
    Quantity.PHI_E: r"\mathrm{A/cm}^2",
    Quantity.PHI_P: r"\mathrm{A/cm}^2",
    Quantity.T: r"\mathrm{W/cm}^2",
    Quantity.LAMBDA: r"\mu\mathrm{mol/cm}^2\mathrm{s}",
    Quantity.W_H2O: r"\mu\mathrm{g/cm}^2\mathrm{s}",
    Quantity.W_O2: r"\mu\mathrm{g/cm}^2\mathrm{s}",
    Quantity.SATURATION: r"\mu\mathrm{mol/cm}^2\mathrm{s}",
    Quantity.P_GAS: r"\mathrm{\mu g}/\mathrm{cm}^2\mathrm{s}",
}

#: Axis labels for the quantity itself and for its flux.
VALUE_LABELS: dict[Quantity, str] = {
    quantity: rf"${VALUE_SYMBOLS[quantity]}\ [{VALUE_UNITS[quantity]}]$"
    for quantity in Quantity
}

FLUX_LABELS: dict[Quantity, str] = {
    quantity: rf"${FLUX_SYMBOLS[quantity]}\ [{FLUX_UNITS[quantity]}]$"
    for quantity in Quantity
}

#: SI -> plot units. Quantities are plotted as-is; fluxes are rescaled.
VALUE_SCALE: dict[Quantity, float] = {q: 1.0 for q in Quantity}
FLUX_SCALE: dict[Quantity, float] = {
    Quantity.PHI_E: 1e-4, Quantity.PHI_P: 1e-4, Quantity.T: 1e-4,
    Quantity.LAMBDA: 1e2, Quantity.W_H2O: 1e2, Quantity.W_O2: 1e2,
    Quantity.SATURATION: 1e2, Quantity.P_GAS: 1e2,
}


def extract_profiles(sol, params: Params, n_dense: int = 201,
                     row_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Continuous spatial profiles from a stacked solve_bvp solution.

    Returns physical positions [m] and the 16 state rows, concatenated region by
    region, with NaN wherever a quantity is not physically defined.

    ``row_offset`` selects which 80-row half of an augmented solution to read.
    It is 0 for a plain solve and ``N_TOTAL`` for the sensitivity block of a
    solution from :mod:`pemfc_1d.sensitivity`, whose rows carry ``dY/dtheta``
    in the same layout -- so both are unpacked by this one function rather than
    by two that could disagree about where a layer starts.
    """
    s_dense = np.linspace(0.0, 1.0, n_dense)
    stacked = sol.sol(s_dense)

    positions = np.empty(N_REGIONS * n_dense)
    profiles = np.empty((N_STATE, N_REGIONS * n_dense))

    for region in Region:
        columns = slice(region * n_dense, (region + 1) * n_dense)
        positions[columns] = params.Lsum[region] + s_dense * params.L[region]
        rows = block_rows(region)
        block = stacked[row_offset + rows.start:row_offset + rows.stop]
        for quantity in Quantity:
            active = region in ACTIVE_REGIONS[quantity]
            profiles[quantity.rows, columns] = block[quantity.rows] if active else np.nan

    return positions, profiles


def _voltage_colors(n_voltages: int) -> list:
    """Dark blue at the first voltage through dark red at the last."""
    colormap = plt.get_cmap("jet")
    return [colormap(i / max(1, n_voltages - 1)) for i in range(n_voltages)]


def _draw_profiles(axis, quantity: Quantity, row: int, label: str, scale: float,
                   params: Params, voltages, profiles, colors) -> None:
    """One panel: every voltage's profile of one row, over the layers it lives on."""
    active = ACTIVE_REGIONS[quantity]
    for index, (voltage, (positions, values)) in enumerate(zip(voltages, profiles)):
        axis.plot(positions * 1e6, values[row] * scale,
                  color=colors[index], label=f"{voltage:.2f} V")
    axis.set_xlim(params.Lsum[active[0]] * 1e6, params.Lsum[active[-1] + 1] * 1e6)
    axis.set_xlabel("x [um]")
    axis.set_ylabel(label)
    for interface in params.Lsum[1:-1]:
        axis.axvline(interface * 1e6, color="k", linewidth=0.8)
    axis.grid(False)


def plot_potentials_and_fluxes(result: SweepResult, n_dense: int = 201,
                               figsize: tuple[float, float] = (14, 5)) -> list:
    """One figure of the eight quantities and one of their eight fluxes."""
    params = result.params
    profiles = [extract_profiles(sol, params, n_dense) for sol in result.solutions]
    colors = _voltage_colors(result.n_voltages)

    figures = []
    for name, labels, scales, row_of in (
        ("Potentials", VALUE_LABELS, VALUE_SCALE, lambda q: q.value_row),
        ("Fluxes", FLUX_LABELS, FLUX_SCALE, lambda q: q.flux_row),
    ):
        figure, axes = plt.subplots(2, 4, figsize=figsize, num=name)
        for quantity, axis in zip(Quantity, axes.ravel()):
            _draw_profiles(axis, quantity, row_of(quantity), labels[quantity],
                           scales[quantity], params, result.voltages, profiles,
                           colors)
        handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
        figure.legend(handles, legend_labels, loc="upper right", fontsize=8, ncol=1)
        figure.suptitle(name)
        figure.tight_layout(rect=[0, 0, 0.94, 0.96])
        figures.append(figure)
    return figures


def plot_sensitivity_profiles(result, n_dense: int = 201,
                              figsize: tuple[float, float] = (14, 10)):
    """``dY/dtheta`` across the MEA for one material parameter.

    Takes a :class:`pemfc_1d.sensitivity.SensitivityResult` and lays its
    sixteen rows out exactly as :func:`plot_potentials_and_fluxes` lays out the
    solution itself -- the eight quantities on the top two rows of panels, their
    eight fluxes below, one colour per cell voltage, each panel spanning only
    the layers where its quantity is defined. What changes is the row read
    (the sensitivity half of the augmented solution) and the axis labels.

    One figure rather than two, because a parameter's sensitivity is saved as a
    single file per parameter.
    """
    parameter = result.parameter
    params = result.params
    from .state import N_TOTAL  # local: keeps the plotting import graph flat
    profiles = [extract_profiles(sol, params, n_dense, row_offset=N_TOTAL)
                for sol in result.solutions]
    colors = _voltage_colors(result.n_voltages)

    name = f"Sensitivity to {parameter.name}"
    figure, axes = plt.subplots(4, 4, figsize=figsize, num=name)
    panels = iter(axes.ravel())
    for symbols, units, scales, row_of in (
        (VALUE_SYMBOLS, VALUE_UNITS, VALUE_SCALE, lambda q: q.value_row),
        (FLUX_SYMBOLS, FLUX_UNITS, FLUX_SCALE, lambda q: q.flux_row),
    ):
        for quantity in Quantity:
            label = (rf"$\partial {symbols[quantity]}/\partial\theta\ "
                     rf"[{units[quantity]}]$")
            _draw_profiles(next(panels), quantity, row_of(quantity), label,
                           scales[quantity], params, result.voltages, profiles,
                           colors)

    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper right", fontsize=8, ncol=1)
    figure.suptitle(
        rf"Sensitivity to ${parameter.symbol}$  ({parameter.description})"
        "\n"
        r"$\theta$ is a dimensionless multiplier on the nominal value, "
        r"so $\partial y/\partial\theta$ is the shift for a 100% increase")
    figure.tight_layout(rect=[0, 0, 0.94, 0.94])
    return figure


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
