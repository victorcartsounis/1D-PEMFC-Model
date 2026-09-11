"""
mmm1d -- 1D PEM fuel cell model (steady-state, non-isothermal,
two-phase, macro-homogeneous), based on the MMM1D reference
implementation by Vetter & Schumacher (2019).

Quick start:
    from mmm1d.model import solve
    result = solve()
    print(result.U, result.I)
"""

__version__ = "0.1.0"

from .model import solve, MMM1DResult  # noqa: F401
