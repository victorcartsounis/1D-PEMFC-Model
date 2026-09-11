"""Smoke test: runs the model for the default voltage sweep and checks
basic sanity of the results (convergence, monotonic current, and the
temperature boundary condition).

Run with: python -m pytest tests/  (or simply: python tests/test_smoke.py)
"""
import numpy as np

from mmm1d.model import solve


def test_default_sweep_converges_and_is_monotonic():
    result = solve(tol=1e-4, n_per_region=15, max_nodes=200000)

    assert all(sol.success for sol in result.SOL), "not all solutions converged"

    # current must increase monotonically as the cell voltage drops
    assert np.all(np.diff(result.I) > 0), "current is not monotonically increasing as the voltage drops"

    # temperature at the end of the CGDL must match the T_C boundary condition
    from mmm1d.model import _yidx
    for sol in result.SOL:
        T_end = sol.y[_yidx(5, 5), -1]
        assert abs(T_end - result.params.T_C) < 1e-3, "temperature boundary condition not satisfied"

    print("OK:", result.U, result.I)


if __name__ == "__main__":
    test_default_sweep_converges_and_is_monotonic()
    print("Smoke test passed.")
