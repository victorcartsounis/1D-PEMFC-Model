"""The solve path pinned to the golden file.

``tests/data/reference_solution.npz`` holds validated output. These tests are
what make it load-bearing: a refactor that perturbs a number fails here rather
than passing quietly and being discovered in a figure much later.

The golden file also carries the inputs it was produced from -- the collocation
state the right-hand side was evaluated on, and the edge states the boundary
residuals were formed from -- so ``full_ode`` and ``boundary_conditions`` are
pinned against stored inputs rather than against whatever an initial guess
happens to produce today.

**Regenerate the golden file only when the physics is deliberately changed** --
never to make a failing refactor pass.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pemfc_1d import Params, boundary_conditions, full_ode, solve
from pemfc_1d.postprocessing import extract_profiles

GOLDEN = Path(__file__).parent / "data" / "reference_solution.npz"

#: Mesh density ``extract_profiles`` was called with when the file was written.
#: The stored profiles are (16, 5 * N_DENSE), so this is not free to change.
N_DENSE = 11

#: Tolerance for quantities that come back out of ``solve_bvp``.
#:
#: The solver runs to ``tol=1e-4``, so this is five orders tighter than the
#: answer is claimed to be accurate: it is not a physics tolerance but a
#: "did this refactor move anything?" tolerance. It is not tighter still
#: because the collocation solve is not bit-reproducible across BLAS builds
#: and vectorisation choices -- a few elements drift in the last few digits
#: between machines, which is not a regression and must not read as one.
SOLVE_RTOL = 1e-9

#: Tolerance for pure evaluations on inputs stored in the golden file. These
#: run the same arithmetic on the same bits every time, so they are held an
#: order tighter than the solver's output.
EVAL_RTOL = 1e-10


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN)


@pytest.fixture(scope="module")
def swept():
    """The default sweep, solved once for every test in this module."""
    return solve()


def test_polarization_curve_matches_reference(swept, golden):
    np.testing.assert_allclose(swept.voltages, golden["U"], rtol=SOLVE_RTOL, atol=0)
    np.testing.assert_allclose(swept.current_densities, golden["I"],
                               rtol=SOLVE_RTOL, atol=0)


def test_spatial_profiles_match_reference(swept, golden):
    """Every state row of every solved voltage, positions included.

    NaN marks a quantity that is not defined in a layer, so the comparison has
    to treat NaN as a value -- a quantity silently becoming defined where it
    was not is exactly the kind of regression this catches.

    Each state row is compared against its own scale rather than element by
    element. The 16 rows span potentials of order 1 V and fluxes of order
    1e-8, and several rows pass through zero, where a relative tolerance
    compares two different roundings of nothing and an absolute tolerance
    meaningful for volts would wave through a flux entirely. Requiring each row
    to agree to SOLVE_RTOL of that row's own largest value says what is
    actually meant: nine significant digits of the quantity being pinned.
    """
    for index in range(swept.n_voltages):
        positions, profiles = extract_profiles(swept.solutions[index],
                                               swept.params, n_dense=N_DENSE)
        expected = golden[f"y_profile_{index}"]

        np.testing.assert_allclose(positions, golden["x_profile"],
                                   rtol=SOLVE_RTOL, atol=0,
                                   err_msg=f"positions differ at voltage {index}")

        assert profiles.shape == expected.shape
        for row in range(expected.shape[0]):
            scale = np.nanmax(np.abs(expected[row]))
            floor = 0.0 if not np.isfinite(scale) else SOLVE_RTOL * scale
            np.testing.assert_allclose(
                profiles[row], expected[row], rtol=SOLVE_RTOL, atol=floor,
                equal_nan=True,
                err_msg=f"state row {row} differs at voltage {index}")


def test_full_ode_matches_reference(golden):
    dYds = full_ode(golden["ode_s"], golden["ode_Y"], Params())
    np.testing.assert_allclose(dYds, golden["ode_dYds"], rtol=EVAL_RTOL, atol=0)


def test_boundary_conditions_match_reference(golden):
    residuals = boundary_conditions(golden["bc_Ya"], golden["bc_Yb"],
                                    float(golden["bc_U"]), Params())
    np.testing.assert_allclose(residuals, golden["bc_res"], rtol=EVAL_RTOL, atol=0)
