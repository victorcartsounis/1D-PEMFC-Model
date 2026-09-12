"""Smoke test: run the default voltage sweep and check basic sanity.

Run with: python -m pytest tests/
"""
import numpy as np

from mmm1d import Region, State, solve
from mmm1d.state import stacked_index


def test_default_sweep_converges_and_is_monotonic():
    result = solve(tol=1e-4, n_per_region=15, max_nodes=200000)

    assert result.converged, "not all solutions converged"

    # current must rise monotonically as the cell voltage drops
    assert np.all(np.diff(result.current_densities) > 0), \
        "current is not monotonically increasing as the voltage drops"

    # temperature at the cathode plate must match its boundary condition
    index = stacked_index(State.T, Region.CGDL)
    for sol in result.solutions:
        assert abs(sol.y[index, -1] - result.params.T_C) < 1e-3, \
            "temperature boundary condition not satisfied"
