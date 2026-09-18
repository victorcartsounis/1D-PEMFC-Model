"""Forward sensitivity analysis, validated against central differences.

The sensitivity equations are only worth as much as the derivatives behind
them, and those are produced by differentiating the model's own right-hand
side symbolically (:mod:`pemfc_1d.symbolic`). Nothing about that is
self-evidently correct: the trace could silently miss a term, the frozen
saturation could stop matching what the lookup table returns, or a boundary
condition could acquire a dependence on the parameter under study. Every one
of those is checked here by perturbing the parameter and re-solving.

The checks come in two layers:

* on the right-hand side, where ``dF/dY`` and ``dF/dtheta`` are compared with
  central differences of ``full_ode`` itself. These are cheap and sharp -- the
  same arithmetic on the same inputs -- so they localise an error to a term.
* on the solution, where ``Z = dY/dtheta`` from the augmented solve is compared
  with ``(Y(theta+eps) - Y(theta-eps)) / (2*eps)`` from two extra sweeps. This
  is the check that matters, because it exercises the augmented system, its
  boundary conditions and the solver together, and it is what the second layer
  cannot catch on its own: a correct Jacobian assembled into the wrong system.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from pemfc_1d import Params, boundary_conditions, full_ode, solve
from pemfc_1d.saturation import saturation_from_capillary_pressure
from pemfc_1d.sensitivity import (SENSITIVITY_PARAMETERS, ScaledParams,
                                  SensitivitySettings,
                                  augmented_boundary_conditions,
                                  frozen_saturation, scaled_params,
                                  solve_sensitivity)
from pemfc_1d.state import (N_STATE, N_TOTAL, Region, State, block_rows)
from pemfc_1d.symbolic import (STATE_SYMBOLS, compile_sensitivity_terms,
                               trace_right_hand_side)

PARAMETERS = tuple(SENSITIVITY_PARAMETERS)

#: Voltages the tests solve on. Two points of the default sweep are enough to
#: exercise every equation, and each parameter costs three sweeps plus an
#: augmented solve, so the sweep is kept short deliberately.
VOLTAGES = (1.15, 1.10)

#: Step for the central differences taken on ``full_ode``, as a fraction of the
#: nominal parameter.
#:
#: The error of a central difference is the sum of a truncation term growing as
#: ``eps**2`` and a roundoff term falling as ``1/eps``; for this model the two
#: cross near 1e-4, where the derivatives of ``full_ode`` come back to seven or
#: eight digits. Ten times smaller and roundoff in the Butler-Volmer
#: exponentials dominates; ten times larger and curvature does.
FD_STEP = 1e-4

#: Step for the central differences taken on the *solution*, which is ten times
#: larger.
#:
#: The noise being divided by ``2*eps`` is different there: not the last bits of
#: an arithmetic expression but the digits two separate boundary-value solves
#: never agreed on, which is many orders larger. Measured over all eight
#: parameters, 1e-3 is where the comparison is cleanest -- 1e-4 leaves several
#: rows dominated by that noise, and 1e-2 starts to show curvature.
SOLUTION_FD_STEP = 1e-3

#: How large a sensitivity has to be, as a fraction of the largest value its
#: quantity takes anywhere in the MEA, before a central difference can confirm
#: it.
#:
#: Below the threshold ``Y(theta+eps)`` and ``Y(theta-eps)`` differ only in the
#: digits the solver never settled, so dividing by ``2*eps`` measures the
#: solver's noise rather than a derivative; most of the 80 rows are there for
#: most parameters, and the measured disagreement in them runs to 100% with
#: both sides comparing noise.
#:
#: The scale is the quantity's magnitude across the whole cell, not within the
#: layer being looked at. A heat flux of 1e-3 W/m^2 in the cathode GDL near
#: open circuit is a heat flux of zero, and judging its sensitivity against
#: itself demands agreement on a number the model does not carry to that many
#: digits; judging it against the ~1e3 W/m^2 the same flux reaches in the
#: catalyst layer says what is meant.
#:
#: Rows under the threshold are checked the other way round, by requiring the
#: symbolic answer to be just as negligible, rather than skipped: a sensitivity
#: the model does not show when the parameter is actually perturbed is still a
#: failure.
RESOLVABLE = 1e-6

#: How closely the solved sensitivity has to match the central difference.
#:
#: Measured worst case over the eight parameters, on the sweep above: 4.1e-4
#: (sigma_e), 5.3e-4 (D_H2), 1.7e-3 (eps_p_over_tau2, D_lambda, D_O2), 8.7e-3
#: (k), 8.9e-3 (kappa), 2.6e-2 (sigma_p). The tolerance sits above the worst of
#: those with room to spare, because what sets them is not the differentiation.
#:
#: Two things do. The augmented system is solved on the mesh the sweep
#: converged on and is not refined further (see
#: ``SensitivitySettings.max_nodes``), so Z carries that mesh's discretisation
#: error; letting the solver refine measurably makes the agreement *worse*,
#: since it then chases the ``k_ad`` kink. And the same kink makes dY/dtheta a
#: one-sided derivative wherever the solution rests on it, which is over
#: stretches of both catalyst layers, while the finite difference straddles it.
#: Both are properties of the model -- NOTES.md items 5 and 7 -- and the sharp
#: statement about the derivatives themselves is made by the two checks above,
#: which hold to 1e-4 and 1e-6.
SOLUTION_TOLERANCE = 5e-2


@pytest.fixture(scope="module")
def swept():
    """The short sweep, solved once for the whole module."""
    return solve(voltages=VOLTAGES)


@pytest.fixture(scope="module")
def state(swept):
    """A physically reachable stacked state: the converged solution's own mesh.

    The whole mesh, and not a thinned sample of it, because the saturation
    lookup is not pointwise -- see
    ``test_the_saturation_lookup_depends_on_the_whole_mesh``. Evaluating
    ``full_ode`` on a subset of the columns the trace was frozen against would
    compare the model with itself under a different constant.
    """
    return swept.solutions[-1].y


# =============================================================================
# THE SCALED PARAMETER SET
# =============================================================================

@pytest.mark.parametrize("name", PARAMETERS)
def test_a_multiplier_of_one_reproduces_the_model_exactly(name, state):
    """``theta = 1`` must be the nominal model, bit for bit.

    Everything else rests on this: the finite-difference check straddles
    ``theta = 1``, and the augmented system evaluates the model at it, so a
    scaled parameter set that merely came close would quietly bias both.
    """
    params = Params()
    expected = full_ode(0.0, state, params)
    actual = full_ode(0.0, state, scaled_params(params, name, 1.0))
    assert np.array_equal(expected, actual)


@pytest.mark.parametrize("name", PARAMETERS)
def test_a_multiplier_actually_changes_the_model(name, state):
    """A parameter that is scaled but never read would sail through every
    other test in this file, reporting a sensitivity of exactly zero."""
    params = Params()
    moved = full_ode(0.0, state, scaled_params(params, name, 1.5))
    assert not np.array_equal(full_ode(0.0, state, params), moved)


def test_scaling_an_already_scaled_parameter_set_is_refused():
    params = scaled_params(Params(), "sigma_e", 2.0)
    with pytest.raises(TypeError):
        scaled_params(params, "sigma_e", 2.0)


# =============================================================================
# THE FROZEN SATURATION
# =============================================================================

def test_the_saturation_lookup_depends_on_the_whole_mesh(swept):
    """The lookup is not a pointwise function, and this is where that is recorded.

    ``saturation_from_capillary_pressure`` decides whether to consult its table
    at all by testing whether the *entire* capillary-pressure vector falls below
    the tabulated range, and the solution sits exactly on that threshold: the
    cathode GDL's capillary pressure equals the table's first entry to within
    rounding, so dropping a mesh point can flip the answer between ``s_im`` and
    half a table step above it.

    Two consequences, both of which the sensitivity code has to live with.
    ``full_ode`` is mesh-dependent in the cathode GDL, so it is not, strictly, a
    pointwise right-hand side; and a symbolic trace of it has to freeze one of
    the two values, which :func:`frozen_saturation` takes from the converged
    mesh -- the vector the solver itself evaluates. The half-step between them
    is 0.4% of ``(1-s)**3``, so it is visible in the gas diffusivities and
    nowhere else.

    This is a symptom of the inactive saturation inversion recorded in NOTES.md,
    not something this module introduced, and it disappears when that is fixed.
    """
    solution = swept.solutions[-1]
    block = solution.y[block_rows(Region.CGDL)]
    capillary = block[State.P_LIQ] - block[State.P_GAS]
    whole = saturation_from_capillary_pressure(capillary, swept.params.s_im)
    thinned = saturation_from_capillary_pressure(capillary[::5], swept.params.s_im)
    assert np.unique(whole) != np.unique(thinned), (
        "the saturation lookup has become pointwise, which is good news: "
        "frozen_saturation and the note in pemfc_1d.symbolic can be simplified")


def test_the_saturation_lookup_is_flat_across_the_solved_sweep(swept):
    """The symbolic trace replaces the saturation table with a constant.

    That is only legitimate while the table returns one value per layer over
    the whole solution, which is the case for as long as the two-phase
    behaviour stays inactive. When it is switched on this test fails, and the
    sensitivity equations will need a genuine ``ds/dp_c`` term before they can
    be trusted again.
    """
    saturation = frozen_saturation(swept)
    for region in (Region.CCL, Region.CGDL):
        for solution in swept.solutions:
            block = solution.y[block_rows(region)]
            values = saturation_from_capillary_pressure(
                block[State.P_LIQ] - block[State.P_GAS], swept.params.s_im)
            assert np.all(values == saturation[region])
        assert saturation[region] == pytest.approx(swept.params.s_im, abs=1e-3)


def test_the_traced_expressions_reproduce_full_ode(swept, state):
    """The trace has to be the model, not a plausible copy of it.

    ``full_ode`` is evaluated twice: once as itself, and once as the sympy
    expressions the trace recorded, compiled back to numpy. Any term the shim
    mishandled -- a ``where`` that collapsed to one branch, a frozen saturation
    taken from the wrong layer -- shows up as a mismatch here.
    """
    expressions = trace_right_hand_side(Params(), frozen_saturation(swept))
    compiled = sp.lambdify(STATE_SYMBOLS, expressions, modules="numpy", cse=True)
    # broadcast against a zero row so that rows which came out constant -- the
    # layers where a quantity is inactive -- arrive as columns, not scalars
    traced = np.array(np.broadcast_arrays(*compiled(*state),
                                          np.zeros(state.shape[1])))[:-1]
    # Not tighter than 1e-5 because sympy reassociates as it records: the
    # sorption isotherm arrives expanded, so the near-cancellation in
    # ``S_ad ~ lambda_eq - lam`` is summed in a different order and loses a few
    # digits to it. A missing or wrong term is an O(1) discrepancy, not this.
    np.testing.assert_allclose(traced, full_ode(0.0, state, Params()),
                               rtol=1e-5, atol=0)


# =============================================================================
# THE DERIVATIVES OF THE RIGHT-HAND SIDE
# =============================================================================

def _terms(name, swept):
    theta = sp.Symbol("theta", real=True, positive=True)
    return compile_sensitivity_terms(scaled_params(swept.params, name, theta),
                                     theta, name, frozen_saturation(swept))


@pytest.mark.parametrize("name", PARAMETERS)
def test_dF_dtheta_matches_central_differences(name, swept, state):
    _, dF_dtheta = _terms(name, swept)(state)
    difference = (full_ode(0.0, state, scaled_params(swept.params, name, 1 + FD_STEP))
                  - full_ode(0.0, state, scaled_params(swept.params, name, 1 - FD_STEP))
                  ) / (2 * FD_STEP)

    # Row by row against the row's own size: the 80 rows span potentials of
    # order 1 and fluxes of order 1e9, so one global scale would say nothing.
    scale = np.abs(difference).max(axis=1, keepdims=True)
    interesting = scale[:, 0] > 0
    error = np.abs(dF_dtheta - difference)[interesting] / scale[interesting]
    # 1e-4 is the floor of the central difference, not of the derivative: the
    # Butler-Volmer exponentials make F large where dF/dtheta is small, so the
    # difference of the two runs loses digits to cancellation. Halving or
    # doubling FD_STEP moves this number; the symbolic answer does not.
    assert error.max() < 1e-4
    assert np.all(dF_dtheta[~interesting] == 0)


#: State rows a perturbation may move when the Jacobian is checked by finite
#: differences: the potentials and every flux.
#:
#: The six excluded rows -- T, LAMBDA, W_H2O, W_O2, P_LIQ and P_GAS -- are the
#: ones that move ``lam - lambda_eq``, and the sorption rate ``k_ad`` switches
#: by a factor of four exactly where that crosses zero. The solution sits on
#: the crossing, because sorption is what drives it there, so a central
#: difference along those directions straddles a kink and returns the average
#: of two one-sided slopes rather than a derivative. That non-smoothness is the
#: same one the mesh refinement cannot resolve either (see
#: ``model.DEFAULT_MAX_NODES``); it is a property of the model, not of the
#: differentiation, and the end-to-end check below covers those columns instead.
SMOOTH_DIRECTIONS = (State.PHI_E, State.J_E, State.PHI_P, State.J_P, State.J_T,
                     State.J_LAMBDA, State.J_H2O, State.J_O2, State.RHO_U_LIQ,
                     State.RHO_U_GAS)

#: Step for the Jacobian check, smaller than ``FD_STEP`` because it perturbs
#: the state rather than a parameter and the Butler-Volmer exponentials are far
#: more sensitive to that. The error falls as ``eps**2`` from here, which is how
#: one can tell it is truncation rather than a wrong derivative.
JACOBIAN_FD_STEP = 1e-5


def test_the_jacobian_matches_central_differences(swept, state):
    """``dF/dY`` along random directions.

    A directional derivative rather than 6400 one-at-a-time differences: it is
    what the sensitivity equations actually form -- ``J`` only ever appears as
    ``J @ Z`` -- and one well-scaled difference per direction is far better
    conditioned than 6400 whose step has to suit every row at once.
    """
    jacobian, _ = _terms("sigma_e", swept)(state)
    allowed = np.zeros((state.shape[0], 1))
    for row in SMOOTH_DIRECTIONS:
        allowed[row::N_STATE] = 1.0

    generator = np.random.default_rng(20260918)
    for _ in range(5):
        direction = (generator.normal(size=state.shape) * allowed
                     * np.abs(state).max(axis=1, keepdims=True))
        step = JACOBIAN_FD_STEP * direction
        difference = (full_ode(0.0, state + step, swept.params)
                      - full_ode(0.0, state - step, swept.params)
                      ) / (2 * JACOBIAN_FD_STEP)
        product = np.einsum("ijk,jk->ik", jacobian, direction)

        # Scaled by the size of the terms that were summed, not by the size of
        # the sum. ``J @ v`` cancels heavily -- the electron and proton terms of
        # a heat equation are large and nearly opposite -- and a tolerance on
        # the result would be a tolerance on that cancellation rather than on
        # the Jacobian.
        terms = np.einsum("ijk,jk->ik", np.abs(jacobian), np.abs(direction))
        error = np.abs(product - difference)[terms > 0] / terms[terms > 0]
        assert error.max() < 1e-6


# =============================================================================
# THE BOUNDARY CONDITIONS
# =============================================================================

def _edge_states(seed):
    generator = np.random.default_rng(seed)
    return generator.normal(size=N_TOTAL), generator.normal(size=N_TOTAL)


def test_the_boundary_residuals_are_affine_in_the_state():
    """``augmented_boundary_conditions`` differentiates the residuals by
    subtracting their constant part, which is only their derivative if every
    residual is affine. A residual that acquired a product or a power of the
    state would break the sensitivity boundary conditions silently."""
    params, U_cell = Params(), 1.1
    zero = np.zeros(N_TOTAL)
    constant = boundary_conditions(zero, zero, U_cell, params)

    first_left, first_right = _edge_states(1)
    second_left, second_right = _edge_states(2)

    def linear(left, right):
        return boundary_conditions(left, right, U_cell, params) - constant

    # atol is set from the size of the constants that were subtracted off:
    # the channel pressures are of order 1e5, so cancelling them leaves a few
    # ulps behind, and demanding exact equality would be demanding exact
    # floating-point subtraction rather than affinity.
    floor = 1e-10 * np.abs(constant).max()
    np.testing.assert_allclose(
        linear(first_left + second_left, first_right + second_right),
        linear(first_left, first_right) + linear(second_left, second_right),
        rtol=1e-12, atol=floor)
    np.testing.assert_allclose(linear(3.0 * first_left, 3.0 * first_right),
                               3.0 * linear(first_left, first_right),
                               rtol=1e-12, atol=floor)


@pytest.mark.parametrize("name", PARAMETERS)
def test_no_scaled_parameter_reaches_a_boundary_condition(name):
    """``dbc/dtheta = 0``, which is why the sensitivity residuals are purely
    the linearised ones. Scaling a property that did appear in a channel
    condition would need the extra term, and this is where that is noticed."""
    params = Params()
    left, right = _edge_states(3)
    np.testing.assert_array_equal(
        boundary_conditions(left, right, 1.1, params),
        boundary_conditions(left, right, 1.1, scaled_params(params, name, 1.7)))


def test_the_augmented_residuals_extend_the_models_own():
    """The first 80 residuals of the augmented problem must still be the
    model's, unchanged, so that an augmented solve cannot drift away from the
    solution it started from."""
    params, U_cell = Params(), 1.1
    left, right = _edge_states(4)
    augmented = augmented_boundary_conditions(
        np.concatenate((left, np.zeros(N_TOTAL))),
        np.concatenate((right, np.zeros(N_TOTAL))), U_cell, params)
    np.testing.assert_array_equal(
        augmented[:N_TOTAL], boundary_conditions(left, right, U_cell, params))
    np.testing.assert_array_equal(augmented[N_TOTAL:], np.zeros(N_TOTAL))


# =============================================================================
# THE SOLUTION
# =============================================================================

@pytest.mark.parametrize("name", PARAMETERS)
def test_the_solved_sensitivity_matches_central_differences(name, swept):
    """``Z(x)`` against ``(Y(theta+eps) - Y(theta-eps)) / (2*eps)``.

    This is the end-to-end check: the symbolic Jacobian, the parameter
    derivative, the linearised boundary conditions and the augmented solve are
    all in it, and it is compared against two runs of the unmodified model.
    """
    sensitivity = solve_sensitivity(swept, name)
    plus = solve(voltages=VOLTAGES,
                 params=scaled_params(swept.params, name, 1 + SOLUTION_FD_STEP))
    minus = solve(voltages=VOLTAGES,
                  params=scaled_params(swept.params, name, 1 - SOLUTION_FD_STEP))
    assert len(plus.voltages) == len(minus.voltages) == swept.n_voltages

    grid = np.linspace(0.0, 1.0, 201)
    worst = 0.0
    for index in range(swept.n_voltages):
        state = swept.solutions[index].sol(grid)
        symbolic = sensitivity.solutions[index].sol(grid)[N_TOTAL:]
        difference = (plus.solutions[index].sol(grid)
                      - minus.solutions[index].sol(grid)) / (2 * SOLUTION_FD_STEP)
        # the magnitude of each of the 16 quantities across all five layers
        cell_scale = np.abs(state).reshape(-1, N_STATE, grid.size).max(axis=(0, 2))

        for row in range(N_TOTAL):
            floor = RESOLVABLE * cell_scale[row % N_STATE]
            if np.abs(difference[row]).max() <= floor:
                # Nothing to compare against; require the symbolic answer to
                # be just as negligible rather than let the row off.
                assert np.abs(symbolic[row]).max() <= floor, (
                    f"row {row} has a symbolic sensitivity the model does not "
                    "show when the parameter is actually perturbed")
                continue
            worst = max(worst, np.abs(symbolic[row] - difference[row]).max()
                        / np.abs(difference[row]).max())

    assert worst < SOLUTION_TOLERANCE, f"worst relative disagreement {worst:.2e}"
