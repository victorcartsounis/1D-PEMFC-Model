"""Tabular export of the forward sensitivity analysis.

The figures written by :func:`pemfc_1d.postprocessing.plot_sensitivity_profiles`
are the only record a run used to leave of ``dY/dtheta``, and a figure cannot
be integrated, sorted or ranked. This module writes the same arrays the figure
is drawn from to two CSV files:

``sensitivity_raw_<run_id>.csv``
    long/tidy, one row per mesh point: which layer, which variable, where in
    the MEA, which parameter was perturbed, at which cell voltage, and the
    value of ``dy/dtheta`` there.

``sensitivity_by_layer_<run_id>.csv``
    one row per ``(layer, variable, param, voltage)``, holding that profile
    integrated across the layer, with and without its sign.

What ``theta`` is, and why the column is constant
-------------------------------------------------
There is no sweep over ``theta`` anywhere in this project, and the ``theta``
column is 1.0 in every row. :mod:`pemfc_1d.sensitivity` does not estimate a
derivative by re-solving at perturbed parameter values; it differentiates the
model symbolically and solves the variational equations alongside it, which
yields ``dY/dtheta`` *exactly at the nominal parameter set*, i.e. at
``theta = 1``. The column is carried anyway so that a reader of the file does
not have to know that, and so that the schema survives if a run over several
``theta`` is ever added.

The sweep that does exist in these files is over cell voltage: the default run
solves 1.15 V down to 0.40 V in 50 mV steps, and every parameter is
differentiated at each of those points.

Which mesh the numbers come from
--------------------------------
The solver's own converged mesh, read straight out of ``solution.x`` and
``solution.y`` -- not the 201-point interpolation the figures are drawn on.
That mesh is the base sweep's: the augmented system is solved on the mesh the
model converged on and is not refined further (see
``SensitivitySettings.max_nodes``), and the mesh-refinement study in
:func:`pemfc_1d.metrics.convergence_metrics` is a separate, once-per-run
re-solve whose result is discarded where it is computed. Nothing from it can
reach these files.

Units
-----
The values are in the units of the figures, not SI, so that a number read off
the CSV and a number read off the PNG beside it agree. That means the current
sensitivities are in A/cm^2 and the mass fluxes in ug/cm^2/s, as
``SensitivityResult.current_sensitivities`` and the run log also report them;
the scale factors are read off :mod:`pemfc_1d.postprocessing` rather than
restated here. Every row carries its unit, and ``theta`` is dimensionless, so
the unit of ``dydtheta`` is the unit of the variable itself.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from scipy.integrate import trapezoid

from .state import (ACTIVE_REGIONS, N_TOTAL, Quantity, Region, State,
                    block_rows)

__all__ = ["RAW_COLUMNS", "BY_LAYER_COLUMNS", "NOMINAL_COLUMNS",
           "POLARIZATION_COLUMNS", "VALUE_NAMES", "FLUX_NAMES",
           "SensitivityExport", "sensitivity_rows", "layer_integral_rows",
           "nominal_rows", "polarization_rows", "write_sensitivity_exports",
           "write_nominal_exports", "run_id_from_directory"]

RAW_COLUMNS = ("layer", "variable", "x_um", "param", "theta", "voltage",
               "dydtheta", "unit")

BY_LAYER_COLUMNS = ("layer", "variable", "param", "theta", "voltage",
                    "integral_abs", "integral_signed", "unit", "n_points")

NOMINAL_COLUMNS = ("layer", "variable", "x_um", "voltage", "value", "unit")

POLARIZATION_COLUMNS = ("voltage", "variable", "value", "unit", "n_points")

#: Column name of each quantity, and of its flux. Deliberately plain ASCII
#: rather than the LaTeX of the figures: these are spreadsheet column values.
#: ``Ps`` is the liquid-water pressure row, which the figures label ``s``
#: because it is what the saturation is read from; it is a pressure in Pa.
VALUE_NAMES: dict[Quantity, str] = {
    Quantity.PHI_E: "phi_e",
    Quantity.PHI_P: "phi_p",
    Quantity.T: "T",
    Quantity.LAMBDA: "lambda",
    Quantity.W_H2O: "wH2O",
    Quantity.W_O2: "wO2",
    Quantity.SATURATION: "Ps",
    Quantity.P_GAS: "Pgas",
}

FLUX_NAMES: dict[Quantity, str] = {
    Quantity.PHI_E: "j_e",
    Quantity.PHI_P: "j_p",
    Quantity.T: "j_T",
    Quantity.LAMBDA: "j_lambda",
    Quantity.W_H2O: "j_H2O",
    Quantity.W_O2: "j_O2",
    Quantity.SATURATION: "j_s",
    Quantity.P_GAS: "rho_u_gas",
}

#: Units the values are written in, matching the scale factors in
#: :mod:`pemfc_1d.postprocessing` rather than SI. See the module docstring.
VALUE_UNITS: dict[Quantity, str] = {
    Quantity.PHI_E: "V",
    Quantity.PHI_P: "V",
    Quantity.T: "K",
    Quantity.LAMBDA: "-",
    Quantity.W_H2O: "-",
    Quantity.W_O2: "-",
    Quantity.SATURATION: "Pa",
    Quantity.P_GAS: "Pa",
}

FLUX_UNITS: dict[Quantity, str] = {
    Quantity.PHI_E: "A/cm^2",
    Quantity.PHI_P: "A/cm^2",
    Quantity.T: "W/cm^2",
    Quantity.LAMBDA: "umol/cm^2/s",
    Quantity.W_H2O: "ug/cm^2/s",
    Quantity.W_O2: "ug/cm^2/s",
    Quantity.SATURATION: "umol/cm^2/s",
    Quantity.P_GAS: "ug/cm^2/s",
}


def _plot_scales() -> tuple[dict, dict]:
    """The SI -> plot-unit factors, read off the plotting module.

    Imported here rather than at the top of the file for two reasons. The
    package keeps matplotlib off the path of anyone who only wants numbers
    (see ``pemfc_1d/__init__.py``), and importing it at module level would put
    a plotting stack behind an import of this one. And the factors belong
    written down once: a CSV whose numbers disagreed with the figure drawn
    beside it from the same array would be worse than no CSV at all.
    """
    from .postprocessing import FLUX_SCALE, VALUE_SCALE
    return VALUE_SCALE, FLUX_SCALE


def _variables(quantity: Quantity) -> tuple[tuple[str, State, str], ...]:
    """The two exported columns of one quantity: itself, then its flux."""
    return ((VALUE_NAMES[quantity], quantity.value_row, VALUE_UNITS[quantity]),
            (FLUX_NAMES[quantity], quantity.flux_row, FLUX_UNITS[quantity]))


def run_id_from_directory(directory: str | Path) -> str:
    """``run_20260919_070531`` -> ``20260919_070531``.

    The run directory already says ``run_``; repeating it inside a file that
    lives in it would only make the name longer.
    """
    name = Path(directory).name
    return name[len("run_"):] if name.startswith("run_") else name


# =============================================================================
# THE ROWS
# =============================================================================

def _profiles(result, voltage_index: int, row_offset: int = N_TOTAL
              ) -> Iterator[tuple[Region, np.ndarray, str, str, np.ndarray]]:
    """Every exportable ``(region, x [um], variable, unit, values)`` of one voltage.

    ``row_offset`` selects which half of an augmented solution is read, the same
    convention :func:`pemfc_1d.postprocessing.extract_profiles` uses: ``N_TOTAL``
    (the default) is the ``dY/dtheta`` half of a :class:`SensitivityResult`, and
    ``0`` is the model half -- which is the whole of a plain
    :class:`~pemfc_1d.model.SweepResult` solution, and is what the nominal
    export below writes.

    The layer is taken from the block of the stacked vector a row belongs to,
    not inferred by comparing ``x`` against ``params.Lsum``. The two agree
    everywhere except at the four internal interfaces, where they cannot: each
    interface position is both the last point of one layer and the first of the
    next, and the solution holds two genuinely different values there -- that
    discontinuity is what the interface boundary conditions are about. The
    block says which of the two a row is, and a comparison against Lsum would
    have to guess.

    Rows for a quantity that is not physically defined in a layer are dropped
    rather than written as NaN, which is what ``ACTIVE_REGIONS`` says and what
    the figures show as an empty stretch of axis.
    """
    value_scale, flux_scale = _plot_scales()
    params = result.params
    solution = result.solutions[voltage_index]
    s_mesh = solution.x                     # normalised, shared by all layers
    rows = solution.y[row_offset:row_offset + N_TOTAL]

    for region in Region:
        x_um = (params.Lsum[region] + s_mesh * params.L[region]) * 1e6
        block = rows[block_rows(region)]
        for quantity in Quantity:
            if region not in ACTIVE_REGIONS[quantity]:
                continue
            scales = (value_scale[quantity], flux_scale[quantity])
            for (name, row, unit), scale in zip(_variables(quantity), scales):
                yield region, x_um, name, unit, block[row] * scale


def sensitivity_rows(result) -> Iterator[tuple]:
    """Long/tidy rows of one :class:`~pemfc_1d.sensitivity.SensitivityResult`.

    One row per mesh point, per variable, per voltage, in ``RAW_COLUMNS``
    order.
    """
    param = result.parameter.name
    for index, voltage in enumerate(result.voltages):
        for region, x_um, name, unit, values in _profiles(result, index):
            layer = region.name
            for position, value in zip(x_um, values):
                yield (layer, name, f"{position:.6f}", param, "1.0",
                       f"{voltage:.3f}", f"{value:.9e}", unit)


def layer_integral_rows(result) -> Iterator[tuple]:
    """One row per ``(layer, variable, param, voltage)``, in ``BY_LAYER_COLUMNS`` order.

    ``integral_abs`` is the integral of ``|dy/dtheta|`` across the layer and
    ``integral_signed`` the integral of ``dy/dtheta`` itself, both by the
    trapezoidal rule over the solver's own mesh points inside that layer. They
    are reported side by side because they answer different questions: the
    signed integral is the net displacement of the profile, and cancels where
    a parameter pushes one end of a layer up and the other down, while the
    absolute integral measures how much the profile moved at all. A variable
    whose two integrals differ sharply in a layer is one the parameter
    reshapes rather than shifts.

    Both are in the variable's own unit times um, because ``x_um`` is the
    integration variable -- so a layer's figure is comparable across variables
    of the same kind but is not a thickness-weighted average. Divide by the
    layer thickness for that.

    No normalisation by a ``theta`` offset enters, because there is no offset:
    ``dy/dtheta`` is the derivative at the nominal parameter set (see the
    module docstring), and every voltage of the sweep gets its own row rather
    than the sweep being reduced to its endpoints.
    """
    param = result.parameter.name
    for index, voltage in enumerate(result.voltages):
        for region, x_um, name, unit, values in _profiles(result, index):
            yield (region.name, name, param, "1.0", f"{voltage:.3f}",
                   f"{trapezoid(np.abs(values), x_um):.9e}",
                   f"{trapezoid(values, x_um):.9e}",
                   f"{unit}*um", str(len(x_um)))


# =============================================================================
# WRITING
# =============================================================================

class SensitivityExport:
    """Writes both CSV files as the sensitivity results are produced.

    A run solves one parameter at a time and throws the result away once its
    figure is saved, so this accumulates nothing it does not have to: the raw
    rows are streamed to disk as each parameter arrives, and only the per-layer
    integrals -- a few thousand rows for a whole run -- are held until the end.

    Used as a context manager::

        with SensitivityExport(directory) as export:
            for parameter in studied:
                sensitivity = solve_sensitivity(result, parameter, settings)
                export.add(sensitivity)
                ...                      # the figures, from the same object
    """

    def __init__(self, directory: str | Path, run_id: str | None = None) -> None:
        self.directory = Path(directory)
        self.run_id = run_id_from_directory(self.directory) if run_id is None else run_id
        self.raw_path = self.directory / f"sensitivity_raw_{self.run_id}.csv"
        self.by_layer_path = self.directory / f"sensitivity_by_layer_{self.run_id}.csv"
        self.n_raw_rows = 0
        self._by_layer: list[tuple] = []
        self._handle = None
        self._writer = None

    def __enter__(self) -> "SensitivityExport":
        self._handle = self.raw_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(RAW_COLUMNS)
        return self

    def add(self, result) -> None:
        """Append one parameter's rows, reusing the arrays it already holds."""
        if self._writer is None:
            raise RuntimeError("use SensitivityExport as a context manager")
        for row in sensitivity_rows(result):
            self._writer.writerow(row)
            self.n_raw_rows += 1
        self._by_layer.extend(layer_integral_rows(result))

    def __exit__(self, *exception) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = self._writer = None
        # Written even if a parameter raised: the parameters that did finish
        # are complete rows, and losing them would mean re-running the sweep.
        with self.by_layer_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(BY_LAYER_COLUMNS)
            writer.writerows(self._by_layer)

    @property
    def n_by_layer_rows(self) -> int:
        return len(self._by_layer)


def write_sensitivity_exports(results: Sequence, directory: str | Path,
                              run_id: str | None = None) -> SensitivityExport:
    """Write both files for an already-collected list of results.

    The batch counterpart of :class:`SensitivityExport`, for a caller that
    holds every parameter's result at once -- a notebook, or a re-export of a
    sweep solved earlier -- rather than producing them one at a time.
    """
    with SensitivityExport(directory, run_id) as export:
        for result in results:
            export.add(result)
    return export


# =============================================================================
# THE NOMINAL SOLUTION
# =============================================================================
#
# Why these two files exist
# -------------------------
# The sensitivity files hold ``dY/dtheta`` and nothing else, which is enough to
# rank parameters against each other but not to normalise them. ``theta`` is a
# dimensionless multiplier whose nominal value is 1 (see the module docstring
# of :mod:`pemfc_1d.sensitivity`), so
#
#     theta * dy/dtheta = dy/dln(theta)   at theta = 1
#
# and the elasticity of ``y`` with respect to the parameter is therefore just
# ``(dy/dtheta) / y``. The parameter's own nominal value never enters. What does
# enter is ``y`` itself, and until these files existed a run left no record of
# it outside ``potentials.png`` and ``fluxes.png`` -- a figure again, which
# cannot be divided by.
#
# ``nominal_profiles_<run_id>.csv`` is written on the solver's own converged
# mesh, in the same units and under the same variable names as
# ``sensitivity_raw_<run_id>.csv``, so the two join on
# ``(layer, variable, x_um, voltage)`` with no interpolation and no unit
# conversion in between. ``polarization_<run_id>.csv`` carries the cell current
# and power, which are the denominators for the headline ``dI/dtheta``.


def nominal_rows(result) -> Iterator[tuple]:
    """Long/tidy rows of the nominal solution ``Y`` of a :class:`SweepResult`.

    One row per mesh point, per variable, per voltage, in ``NOMINAL_COLUMNS``
    order -- the same shape, mesh, units and variable names as
    :func:`sensitivity_rows`, minus the ``param``/``theta`` columns, which a
    solution taken at the nominal parameter set has no use for.
    """
    for index, voltage in enumerate(result.voltages):
        for region, x_um, name, unit, values in _profiles(result, index, row_offset=0):
            layer = region.name
            for position, value in zip(x_um, values):
                yield (layer, name, f"{position:.6f}", f"{voltage:.3f}",
                       f"{value:.9e}", unit)


def polarization_rows(result) -> Iterator[tuple]:
    """The cell current and power at each voltage, in ``POLARIZATION_COLUMNS`` order.

    Long rather than wide so that the unit travels with the number, as it does
    in every other file here. ``n_points`` is the size of the converged mesh at
    that voltage, which is what lets a reader line a row up against the
    ``nodes`` column of ``metrics.log``.
    """
    for index, voltage in enumerate(result.voltages):
        n_points = len(result.solutions[index].x)
        yield (f"{voltage:.3f}", "I", f"{result.current_densities[index]:.9e}",
               "A/cm^2", str(n_points))
        yield (f"{voltage:.3f}", "P", f"{result.power_densities[index]:.9e}",
               "W/cm^2", str(n_points))


def write_nominal_exports(result, directory: str | Path,
                          run_id: str | None = None) -> tuple[Path, Path]:
    """Write ``nominal_profiles_<run_id>.csv`` and ``polarization_<run_id>.csv``.

    Takes a :class:`~pemfc_1d.model.SweepResult`, not a sensitivity result: the
    nominal solution is the sweep, and this costs no solving at all. Returns
    the two paths written.
    """
    directory = Path(directory)
    run_id = run_id_from_directory(directory) if run_id is None else run_id
    written = []
    for name, columns, rows in (
            ("nominal_profiles", NOMINAL_COLUMNS, nominal_rows(result)),
            ("polarization", POLARIZATION_COLUMNS, polarization_rows(result))):
        path = directory / f"{name}_{run_id}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        written.append(path)
    return tuple(written)
