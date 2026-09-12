"""mmm1d -- a 1D PEM fuel cell model: steady-state, non-isothermal, two-phase,
macro-homogeneous membrane electrode assembly.

Physics and constitutive relations follow:

    R. Vetter and J. O. Schumacher, "Free open reference implementation of a
    two-phase PEM fuel cell model", Computer Physics Communications 234 (2019)
    223-234. https://doi.org/10.1016/j.cpc.2018.07.023

Quick start::

    from mmm1d import solve
    result = solve()
    print(result.voltages, result.current_densities)
"""
from .model import SweepResult, boundary_conditions, full_ode, solve
from .params import Params
from .saturation import saturation_from_capillary_pressure
from .state import ACTIVE_REGIONS, Quantity, Region, State

__version__ = "0.2.0"

__all__ = [
    "solve", "SweepResult", "Params",
    "full_ode", "boundary_conditions", "saturation_from_capillary_pressure",
    "Region", "State", "Quantity", "ACTIVE_REGIONS",
]
