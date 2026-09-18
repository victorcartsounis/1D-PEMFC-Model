"""Smoke test: run the default voltage sweep and check basic sanity.

The default sweep is the whole polarization curve, 1.15 V down to 0.40 V, so
this covers all three of its regions rather than just the activation region
near open circuit.

Run with: python -m pytest tests/
"""
import numpy as np

from pemfc_1d import Region, State, solve
from pemfc_1d.state import stacked_index

#: Above this voltage ``solve_bvp`` is expected to meet its tolerance.
#:
#: Below roughly 0.75 V it does not, and that is not a failure: ``k_ad``
#: switches discontinuously where ``lam`` crosses ``lambda_eq``, no mesh can
#: drive the collocation residual there below ``tol``, and the sweep carries on
#: from the solution exactly as the reference implementation does. See
#: ``model.DEFAULT_MAX_NODES``. The threshold is set with margin, because which
#: side of it a marginal point lands on is not stable across BLAS builds.
TOLERANCE_MET_ABOVE = 0.80


def test_default_sweep_converges_and_is_monotonic():
    result = solve(tol=1e-4, n_per_region=15)

    assert len(result.voltages) == len(result.params.U_list), \
        "the sweep stopped early instead of solving every voltage"

    # current must rise monotonically as the cell voltage drops
    assert np.all(np.diff(result.current_densities) > 0), \
        "current is not monotonically increasing as the voltage drops"

    # the kinetic and ohmic regions must still meet the solver's tolerance
    missed = [voltage for voltage, solution in zip(result.voltages, result.solutions)
              if voltage >= TOLERANCE_MET_ABOVE and not solution.success]
    assert not missed, f"tolerance missed above {TOLERANCE_MET_ABOVE} V at {missed}"

    # temperature at the cathode plate must match its boundary condition
    index = stacked_index(State.T, Region.CGDL)
    for sol in result.solutions:
        assert abs(sol.y[index, -1] - result.params.T_C) < 1e-3, \
            "temperature boundary condition not satisfied"


def test_the_sweep_reaches_the_mass_transport_limited_plateau():
    """The point of taking the default sweep past 1.00 V.

    Near open circuit the curve is all activation loss and the current is
    microamps; the plateau is where oxygen transport to the cathode catalyst
    layer starts to limit the cell, and it is the part a polarization curve is
    usually read for.
    """
    result = solve()

    assert result.current_densities[-1] > 2.0, \
        "the sweep no longer reaches the mass-transport-limited plateau"
    # the curve has to bend over: the last 50 mV step must buy less current
    # than an earlier one of the same size did
    steps = np.diff(result.current_densities)
    assert steps[-1] < steps[len(steps) // 2], \
        "the polarization curve is not bending over at high current"
