"""Inversion of the GDL capillary-pressure / saturation relationship.

Given a capillary pressure ``p_c = P_liq - P_gas``, return the liquid water
saturation ``s`` consistent with the constitutive curve

    p_c(s) = -0.00011*exp(-44.02*(s-0.496)) + 278.3*exp(8.103*(s-0.496)) - 191.8

The curve is strictly increasing on the tabulated range, so the inversion is a
binary search (``np.searchsorted``) over a lookup table. The table depends only
on ``s_im``, so it is built once and cached rather than rebuilt on each of the
thousands of calls the solver makes.

Physics reference:

R. Vetter and J. O. Schumacher, "Free open reference implementation of a
two-phase PEM fuel cell model", Computer Physics Communications 234 (2019)
223-234. https://doi.org/10.1016/j.cpc.2018.07.023
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

_S_MAX = 0.9      # [-] upper end of the tabulated saturation range
_S_STEP = 0.001   # [-] tabulation step


def capillary_pressure(s: np.ndarray | float) -> np.ndarray | float:
    """[Pa] capillary pressure at liquid saturation ``s``."""
    return (-0.00011 * np.exp(-44.02 * (s - 0.496))
            + 278.3 * np.exp(8.103 * (s - 0.496)) - 191.8)


@lru_cache(maxsize=None)
def _lookup_table(s_im: float) -> tuple[np.ndarray, np.ndarray]:
    """Saturation grid and the capillary pressures on it, cached per ``s_im``.

    Returned read-only so a caller cannot corrupt the shared cached arrays.
    """
    s_grid = np.arange(s_im, _S_MAX + 1e-12, _S_STEP)
    pc_grid = capillary_pressure(s_grid)
    s_grid.flags.writeable = False
    pc_grid.flags.writeable = False
    return s_grid, pc_grid


def saturation_from_capillary_pressure(
    pc_in: np.ndarray | float, s_im: float, strict: bool = False
) -> np.ndarray:
    """Liquid water saturation corresponding to capillary pressure ``pc_in``.

    Parameters
    ----------
    pc_in
        Local capillary pressure ``P_liq - P_gas`` [Pa].
    s_im
        Immobile liquid water saturation, the floor of the result.
    strict
        How the "outside the tabulated range" test is applied. ``False``
        (default) requires every element of ``pc_in`` to fall below the table
        before the inversion is consulted; ``True`` tests each element on its
        own.

        Both modes return ``s_im`` for every physically reachable capillary
        pressure, because the test selects pressures *below* the tabulated
        range, where the saturation is ``s_im`` regardless. Liquid water is
        consequently immobile in the current model. Genuinely inverting the
        curve means dropping the test altogether -- a physics change, left to
        the planned two-phase work rather than made silently here.

    Returns
    -------
    np.ndarray
        Liquid water saturation, at least 1-D, same shape as ``pc_in``.
    """
    pc_in = np.atleast_1d(np.asarray(pc_in, dtype=float))
    s_grid, pc_grid = _lookup_table(float(s_im))

    # index of the first tabulated point whose pressure reaches pc_in
    crossing = np.searchsorted(pc_grid, pc_in, side="left")
    upper = np.clip(crossing, 1, s_grid.size - 1)
    bracketed = np.maximum(s_im, (s_grid[upper] + s_grid[upper - 1]) / 2.0)

    below_table = pc_in < pc_grid.min()
    use_bracketed = below_table if strict else np.full(pc_in.shape, below_table.all())
    return np.where(use_bracketed, bracketed, s_im)
