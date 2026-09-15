"""Parameters, constitutive relations and the saturation inversion, pinned to
the golden file.

Where ``test_regression.py`` pins the assembled solve, this pins the pieces it
is assembled from, so that a change to a correlation is reported as a change to
that correlation rather than as a shifted polarization curve.

On the input grids
------------------
The golden file stores the *outputs* of each relation but not the grids they
were evaluated on. Those grids were recovered by inverting the stored values
against the relations themselves, and every one asserted below reproduces its
stored array exactly. The three gas diffusivities are the exception: their grid
could not be recovered, so what the file still certifies about them -- the
fixed ratios between the three species coefficients -- is pinned instead, and
the arrays themselves are not. Regenerating the golden file is the opportunity
to close that gap.

**Regenerate the golden file only when the physics is deliberately changed** --
never to make a failing refactor pass.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pemfc_1d import Params
from pemfc_1d.saturation import _lookup_table
from pemfc_1d.saturation import saturation_from_capillary_pressure

GOLDEN = Path(__file__).parent / "data" / "reference_solution.npz"

#: Evaluation grids recovered from the stored outputs. They are not arbitrary:
#: changing one stops the corresponding array reproducing.
T_GRID = np.linspace(300.0, 370.0, 9)      # [K]
RH_GRID = np.linspace(0.05, 1.4, 9)        # [-], deliberately past saturation
S_GRID = np.linspace(0.12, 0.85, 9)        # [-]
LAMBDA_GRID = np.linspace(0.5, 14.0, 9)    # [-]

#: Equilibrium water content that ``k_ad`` switches on. Any value between
#: ``LAMBDA_GRID[3]`` and ``LAMBDA_GRID[4]`` reproduces the stored array, since
#: only which side of the switch each point falls on is observable.
LAMBDA_EQ = 6.0

#: Capillary pressures below, straddling and above the tabulated range. The
#: table bottoms out near -1875 Pa; "mixed" is chosen so exactly its first two
#: points fall below, which is what makes loose and strict disagree.
PC_BELOW = np.linspace(-4000.0, -2000.0, 11)
PC_MIXED = np.linspace(-2500.0, 2500.0, 11)
PC_ABOVE = np.linspace(-1000.0, 5000.0, 11)

RTOL = 1e-12


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN)


@pytest.fixture(scope="module")
def params():
    return Params()


# =============================================================================
# PARAMETERS
# =============================================================================

def test_parameter_values_match_reference(golden, params):
    """Every physical constant, exactly -- no tolerance.

    These are not computed, they are written down, so a difference means an
    edit rather than arithmetic drift.
    """
    stored = {key[len("param_"):]: golden[key]
              for key in golden.files if key.startswith("param_")}
    assert stored, "the golden file carries no parameters"

    wrong = {name: (getattr(params, name), value) for name, value in stored.items()
             if not np.array_equal(np.asarray(getattr(params, name), dtype=float), value)}
    assert not wrong, f"parameters differ from the reference: {sorted(wrong)}"


def test_every_stored_parameter_still_exists(golden, params):
    absent = [key[len("param_"):] for key in golden.files
              if key.startswith("param_") and not hasattr(params, key[len("param_"):])]
    assert not absent, f"the reference pins parameters Params no longer has: {absent}"


def test_params_is_frozen_and_replaceable(params):
    """A running model's constants cannot be mutated, but a variant can be made."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.T_A_celsius = 80.0

    hotter = dataclasses.replace(params, T_A_celsius=80.0)
    assert hotter.T_A == pytest.approx(80.0 + 273.15)
    assert params.T_A == pytest.approx(70.0 + 273.15), "the original was mutated"


# =============================================================================
# CONSTITUTIVE RELATIONS
# =============================================================================

def test_saturation_pressure_matches_reference(golden, params):
    np.testing.assert_allclose(params.P_sat(T_GRID), golden["P_sat"],
                               rtol=RTOL, atol=0)


def test_sorption_isotherm_matches_reference(golden, params):
    np.testing.assert_allclose(params.sorption(RH_GRID), golden["sorption"],
                               rtol=RTOL, atol=0)


def test_proton_conductivity_matches_reference(golden, params):
    np.testing.assert_allclose(
        params.sigma_p(params.eps_i_CL, LAMBDA_GRID, T_GRID),
        golden["sigma_p"], rtol=RTOL, atol=0)


def test_membrane_water_diffusivity_matches_reference(golden, params):
    np.testing.assert_allclose(
        params.D_lambda(params.eps_i_CL, LAMBDA_GRID, T_GRID),
        golden["D_lambda"], rtol=RTOL, atol=0)


def test_sorption_rate_matches_reference(golden, params):
    np.testing.assert_allclose(
        params.k_ad(LAMBDA_GRID, LAMBDA_EQ, T_GRID),
        golden["k_ad"], rtol=RTOL, atol=0)


def test_relative_liquid_permeability_matches_reference(golden, params):
    np.testing.assert_allclose(params.kappa_rel_liq(S_GRID),
                               golden["kappa_rel_liq"], rtol=RTOL, atol=0)


def test_gas_diffusivities_keep_their_fixed_ratios(golden):
    """The three species share one scaling and differ only by a coefficient.

    This is what the golden file still certifies about them now that the grid
    they were evaluated on is lost: whatever that grid was, the ratios are
    fixed by the coefficients alone, so a change to the shared scaling cancels
    and a change to one coefficient does not.
    """
    np.testing.assert_allclose(golden["D_H2O_A"] / golden["D_O2"],
                               1.24 / 0.28, rtol=RTOL, atol=0)
    np.testing.assert_allclose(golden["D_H2O_C"] / golden["D_O2"],
                               0.36 / 0.28, rtol=RTOL, atol=0)


def test_the_same_ratios_hold_for_the_relations_today(params):
    """The companion to the test above, on live values rather than stored ones."""
    args = (params.eps_p_GDL, params.tau_GDL, S_GRID, T_GRID, params.P_C)
    d_o2 = params.D_O2(*args)
    np.testing.assert_allclose(params.D_H2O_A(*args) / d_o2, 1.24 / 0.28,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(params.D_H2O_C(*args) / d_o2, 0.36 / 0.28,
                               rtol=RTOL, atol=0)


# =============================================================================
# SATURATION INVERSION
# =============================================================================

@pytest.mark.parametrize("strict", [False, True], ids=["loose", "strict"])
@pytest.mark.parametrize("where, pc", [("below", PC_BELOW),
                                       ("mixed", PC_MIXED),
                                       ("above", PC_ABOVE)])
def test_saturation_matches_reference(golden, params, where, pc, strict):
    """The two out-of-range modes, on pressures below, across and above the table.

    ``loose`` consults the inversion only when *every* pressure is below the
    table; ``strict`` decides element by element. The two therefore agree on
    the all-below and all-above cases and differ only on the mixed one, which
    is the whole reason both are pinned.
    """
    key = f"sat_{where}_{'strict' if strict else 'loose'}"
    result = saturation_from_capillary_pressure(pc, params.s_im, strict=strict)
    np.testing.assert_allclose(result, golden[key], rtol=RTOL, atol=0)


def test_saturation_of_a_scalar_is_still_an_array(golden, params):
    result = saturation_from_capillary_pressure(float(PC_BELOW[0]), params.s_im)
    assert result.shape == (1,)
    np.testing.assert_allclose(result, golden["sat_scalar"], rtol=RTOL, atol=0)


def test_saturation_never_falls_below_the_immobile_floor(params):
    pc = np.linspace(-5000.0, 8000.0, 201)
    for strict in (False, True):
        result = saturation_from_capillary_pressure(pc, params.s_im, strict=strict)
        assert np.all(result >= params.s_im)


def test_lookup_table_is_monotonic_and_shared(params):
    """Cached per ``s_im`` and handed out read-only, so a caller cannot corrupt it."""
    s_grid, pc_grid = _lookup_table(params.s_im)

    assert np.all(np.diff(pc_grid) > 0), "the table must be searchable"
    assert s_grid[0] == pytest.approx(params.s_im)
    assert _lookup_table(params.s_im)[0] is s_grid, "the table should be cached"

    with pytest.raises(ValueError):
        pc_grid[0] = 0.0
