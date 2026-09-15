"""The package's public surface, as a separate repository consumes it.

The desktop front end now lives in 1D-PEMFC-Model-GUI and imports this package
as an installed dependency, so a name quietly dropped from ``pemfc_1d`` breaks
a program this repository's own test suite never runs. These tests pin the
surface here, where a change to it is visible in the diff that causes it.
"""
from __future__ import annotations

import subprocess
import sys

import pemfc_1d

#: Every name the graphical front end imports from this package. Removing or
#: renaming one is a breaking change and needs a version bump.
CONSUMED_BY_THE_GUI = (
    "__version__",
    "solve", "SweepResult", "DEFAULT_MAX_NODES",
    "Params", "Region", "Quantity", "ACTIVE_REGIONS",
    "RunMetrics", "SolverSettings", "sweep_metrics", "convergence_metrics",
    "create_run_directory", "write_run_log",
    "GB_PER_1000_NODES", "REFINEMENT_NODE_CEILING",
)


def test_every_name_the_front_end_imports_is_present():
    missing = [name for name in CONSUMED_BY_THE_GUI
               if not hasattr(pemfc_1d, name)]
    assert not missing, f"the GUI imports names this package no longer has: {missing}"


def test_all_is_importable():
    missing = [name for name in pemfc_1d.__all__ if not hasattr(pemfc_1d, name)]
    assert not missing, f"__all__ promises names that do not exist: {missing}"


def test_importing_the_package_does_not_pull_in_matplotlib():
    """Plotting stays behind ``pemfc_1d.postprocessing``.

    A headless sweep should not pay for a plotting stack, and importing
    matplotlib has side effects of its own (backend selection). Run in a
    subprocess because the rest of the suite imports matplotlib for real.
    """
    probe = ("import sys; import pemfc_1d; "
             "sys.exit(1 if 'matplotlib' in sys.modules else 0)")
    assert subprocess.run((sys.executable, "-c", probe)).returncode == 0
