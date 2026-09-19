"""The CSV export of the sensitivity analysis.

What is worth pinning here is not that a file appears but that the numbers in
it are the ones the rest of the project reports. Three things could quietly go
wrong and would not show up as an error:

* the layer column could drift from the block of the stacked vector a row
  actually came from, which at the four internal interfaces is not recoverable
  from ``x`` alone;
* the values could be written in SI while the figures and the run log are in
  A/cm^2 and ug/cm^2/s, putting a factor of 1e4 between a number read off the
  CSV and the same number read off the PNG beside it;
* the per-layer integral could be taken over the wrong span of mesh.

Each is checked against the source it has to agree with -- ``ACTIVE_REGIONS``,
``postprocessing``'s own scale factors, and ``current_sensitivities`` -- rather
than against a recorded value, so the check cannot go stale.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest
from scipy.integrate import trapezoid

from pemfc_1d import solve
from pemfc_1d.export import (BY_LAYER_COLUMNS, FLUX_NAMES, RAW_COLUMNS,
                             VALUE_NAMES, SensitivityExport,
                             layer_integral_rows, run_id_from_directory,
                             sensitivity_rows, write_sensitivity_exports)
from pemfc_1d.postprocessing import FLUX_SCALE, VALUE_SCALE
from pemfc_1d.sensitivity import SensitivitySettings, solve_sensitivity
from pemfc_1d.state import (ACTIVE_REGIONS, N_TOTAL, Quantity, Region, State,
                            block_rows, stacked_index)

#: One voltage and one parameter: the export does the same thing to every
#: parameter and every voltage, and each one costs an augmented 160-equation
#: solve. ``sigma_e`` is the cheapest of them to differentiate.
VOLTAGES = (1.15,)
PARAMETER = "sigma_e"


@pytest.fixture(scope="module")
def sensitivity():
    """One solved parameter, for the whole module."""
    swept = solve(voltages=VOLTAGES)
    settings = SensitivitySettings(parameters=(PARAMETER,))
    return solve_sensitivity(swept, PARAMETER, settings)


@pytest.fixture(scope="module")
def raw(sensitivity):
    return list(sensitivity_rows(sensitivity))


@pytest.fixture(scope="module")
def by_layer(sensitivity):
    return list(layer_integral_rows(sensitivity))


def _as_dicts(rows, columns):
    return [dict(zip(columns, row)) for row in rows]


# =============================================================================
# SHAPE OF THE TABLES
# =============================================================================

def test_every_active_quantity_and_flux_appears_in_every_layer_it_lives_on(raw):
    """The rows present are exactly what ``ACTIVE_REGIONS`` says they should be."""
    seen = {(row[0], row[1]) for row in raw}
    expected = {(region.name, name)
                for quantity in Quantity
                for region in ACTIVE_REGIONS[quantity]
                for name in (VALUE_NAMES[quantity], FLUX_NAMES[quantity])}
    assert seen == expected


def test_one_raw_row_per_mesh_point(raw, sensitivity):
    """No interpolation: the mesh in the file is the solver's own.

    The figures are drawn on 201 interpolated points per layer; the export is
    not, so the row count follows ``solution.x`` exactly.
    """
    n_nodes = len(sensitivity.solutions[0].x)
    n_columns = 2 * sum(len(ACTIVE_REGIONS[quantity]) for quantity in Quantity)
    assert len(raw) == n_nodes * n_columns


def test_one_by_layer_row_per_layer_variable_param_voltage(by_layer, raw):
    keys = {(row[0], row[1], row[2], row[4]) for row in by_layer}
    assert len(keys) == len(by_layer)
    assert keys == {(row[0], row[1], row[3], row[5]) for row in raw}


def test_theta_is_one_everywhere(raw, by_layer):
    """There is no sweep over theta: the derivative is taken at the nominal set."""
    assert {row[RAW_COLUMNS.index("theta")] for row in raw} == {"1.0"}
    assert {row[BY_LAYER_COLUMNS.index("theta")] for row in by_layer} == {"1.0"}


# =============================================================================
# THE NUMBERS
# =============================================================================

def test_layer_comes_from_the_block_not_from_comparing_x(raw, sensitivity):
    """Each layer's rows span exactly that layer, interfaces included.

    An interface position belongs to two layers -- it is the last point of one
    and the first of the next -- so a row landing on one is only correctly
    labelled if the label came from the block it was read out of.
    """
    params = sensitivity.params
    rows = _as_dicts(raw, RAW_COLUMNS)
    for region in Region:
        positions = np.array([float(row["x_um"]) for row in rows
                              if row["layer"] == region.name])
        assert positions.min() == pytest.approx(params.Lsum[region] * 1e6)
        assert positions.max() == pytest.approx(params.Lsum[region + 1] * 1e6)

    # and the interfaces really do carry two different values, which is why
    # the label cannot be recovered from x.
    interface = params.Lsum[1] * 1e6
    touching = {row["layer"] for row in rows
                if float(row["x_um"]) == pytest.approx(interface)}
    assert touching == {Region.AGDL.name, Region.ACL.name}


def test_values_are_in_the_units_of_the_figures(raw, sensitivity):
    """A number in the CSV equals the number the figure beside it plots.

    Checked on ``dphi_e/dtheta`` (unscaled) and ``dj_e/dtheta`` (scaled by
    1e-4 into A/cm^2), against the solution rows the plot reads.
    """
    solution = sensitivity.solutions[0]
    rows = _as_dicts(raw, RAW_COLUMNS)
    for quantity, names, scales, row_of in (
        (Quantity.PHI_E, VALUE_NAMES, VALUE_SCALE, lambda q: q.value_row),
        (Quantity.PHI_E, FLUX_NAMES, FLUX_SCALE, lambda q: q.flux_row),
    ):
        exported = np.array([float(row["dydtheta"]) for row in rows
                             if row["layer"] == Region.CGDL.name
                             and row["variable"] == names[quantity]])
        expected = (solution.y[N_TOTAL + stacked_index(row_of(quantity), Region.CGDL)]
                    * scales[quantity])
        assert exported == pytest.approx(expected, rel=1e-8, abs=1e-30)


def test_the_cell_current_sensitivity_is_recoverable(raw, sensitivity):
    """``dI/dtheta``, the headline number, is the last CGDL ``j_e`` row.

    This is what makes the file usable as the macroscopic result: the value
    the run log prints has to be findable in it without a unit conversion.
    """
    rows = _as_dicts(raw, RAW_COLUMNS)
    cgdl = [row for row in rows if row["layer"] == Region.CGDL.name
            and row["variable"] == FLUX_NAMES[Quantity.PHI_E]]
    outlet = max(cgdl, key=lambda row: float(row["x_um"]))
    assert float(outlet["dydtheta"]) == pytest.approx(
        sensitivity.current_sensitivities[0], rel=1e-8)
    assert outlet["unit"] == "A/cm^2"


def test_integrals_are_the_trapezoid_rule_over_that_layer(raw, by_layer):
    """The aggregate is reproducible from the raw file, row for row."""
    rows = _as_dicts(raw, RAW_COLUMNS)
    for aggregate in _as_dicts(by_layer, BY_LAYER_COLUMNS):
        matching = [row for row in rows
                    if row["layer"] == aggregate["layer"]
                    and row["variable"] == aggregate["variable"]
                    and row["voltage"] == aggregate["voltage"]]
        x_um = np.array([float(row["x_um"]) for row in matching])
        values = np.array([float(row["dydtheta"]) for row in matching])
        order = np.argsort(x_um)
        x_um, values = x_um[order], values[order]

        assert int(aggregate["n_points"]) == len(x_um)
        assert float(aggregate["integral_abs"]) == pytest.approx(
            trapezoid(np.abs(values), x_um), rel=1e-6, abs=1e-30)
        assert float(aggregate["integral_signed"]) == pytest.approx(
            trapezoid(values, x_um), rel=1e-6, abs=1e-30)
        assert aggregate["unit"].endswith("*um")


def test_the_absolute_integral_dominates_the_signed_one(by_layer):
    """|integral of f| <= integral of |f|, which is the point of having both."""
    for aggregate in _as_dicts(by_layer, BY_LAYER_COLUMNS):
        assert (abs(float(aggregate["integral_signed"]))
                <= float(aggregate["integral_abs"]) * (1 + 1e-9))


# =============================================================================
# THE FILES
# =============================================================================

def test_writes_both_files_with_their_headers(sensitivity, tmp_path):
    directory = tmp_path / "run_20260919_070531"
    directory.mkdir()
    export = write_sensitivity_exports([sensitivity], directory)

    assert export.raw_path.name == "sensitivity_raw_20260919_070531.csv"
    assert export.by_layer_path.name == "sensitivity_by_layer_20260919_070531.csv"
    for path, columns, count in ((export.raw_path, RAW_COLUMNS, export.n_raw_rows),
                                 (export.by_layer_path, BY_LAYER_COLUMNS,
                                  export.n_by_layer_rows)):
        with path.open(newline="", encoding="utf-8") as handle:
            written = list(csv.reader(handle))
        assert tuple(written[0]) == columns
        assert len(written) == count + 1


def test_run_id_strips_the_directory_prefix(tmp_path):
    assert run_id_from_directory("results/run_20260919_070531") == "20260919_070531"
    assert run_id_from_directory("somewhere/else") == "else"


def test_the_by_layer_file_survives_a_failing_parameter(sensitivity, tmp_path):
    """A parameter that raises must not cost the ones that already finished.

    A full run is eight augmented solves over sixteen voltages; losing the
    aggregate of the first seven because the eighth failed would mean redoing
    all of it.
    """
    directory = tmp_path / "run_20260919_070531"
    directory.mkdir()
    with pytest.raises(RuntimeError):
        with SensitivityExport(directory) as export:
            export.add(sensitivity)
            raise RuntimeError("the next parameter blew up")
    assert export.by_layer_path.exists()
    assert export.n_by_layer_rows > 0
