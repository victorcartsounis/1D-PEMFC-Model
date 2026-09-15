"""pemfc_1d -- a 1D PEM fuel cell model: steady-state, non-isothermal, two-phase,
macro-homogeneous membrane electrode assembly.

Physics and constitutive relations follow:

    R. Vetter and J. O. Schumacher, "Free open reference implementation of a
    two-phase PEM fuel cell model", Computer Physics Communications 234 (2019)
    223-234. https://doi.org/10.1016/j.cpc.2018.07.023

Quick start::

    from pemfc_1d import solve
    result = solve()
    print(result.voltages, result.current_densities)

What this namespace promises
----------------------------
The names below are the package's supported surface: solving, the parameter
set and state layout, and the run metrics and log. They are what a separate
front end is entitled to import, and what may not change without a version
bump -- the desktop interface in the 1D-PEMFC-Model-GUI repository consumes
exactly this list.

Plotting is deliberately *not* re-exported here. ``pemfc_1d.postprocessing``
imports matplotlib at module level, and hoisting it would put a plotting stack
behind every ``import pemfc_1d``, including runs that only want numbers. Import
it directly when you want figures::

    from pemfc_1d.postprocessing import plot_polarization_curve
"""
#: Defined before the submodule imports below, because ``metrics`` reads it back
#: off this package while this module is still initialising.
__version__ = "0.2.0"

from .metrics import (GB_PER_1000_NODES, REFINEMENT_NODE_CEILING, RunMetrics,
                      SolverSettings, convergence_metrics, create_run_directory,
                      git_revision, sweep_metrics, write_run_log)
from .model import (DEFAULT_MAX_NODES, SweepResult, boundary_conditions,
                    full_ode, solve)
from .params import Params
from .saturation import saturation_from_capillary_pressure
from .state import ACTIVE_REGIONS, Quantity, Region, State

__all__ = [
    # solving
    "solve", "SweepResult", "full_ode", "boundary_conditions",
    "DEFAULT_MAX_NODES",
    # parameters, state layout and constitutive relations
    "Params", "Region", "State", "Quantity", "ACTIVE_REGIONS",
    "saturation_from_capillary_pressure",
    # run metrics, provenance and the on-disk log
    "RunMetrics", "SolverSettings", "sweep_metrics", "convergence_metrics",
    "create_run_directory", "write_run_log", "git_revision",
    "GB_PER_1000_NODES", "REFINEMENT_NODE_CEILING",
]
