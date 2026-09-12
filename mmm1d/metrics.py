"""Run metrics and the on-disk run log.

Every invocation of ``run_example.py`` writes one timestamped directory under
``results/`` holding the three figures and a ``metrics.log`` recording what was
run, when, against which revision of the code, and how good the answer is.

The point of separating the metrics from the figures is that "how hard the
solver worked" and "how good the answer is" are different questions, and they
can disagree: a collocation solver can miss its tolerance on a solution that is
converged to several digits, and can meet it on a mesh too coarse for the
answer to have settled. Elapsed time, node count and whether the tolerance was
met are therefore all recorded as *cost and diagnostics*, while the quantity
that actually certifies the result is the mesh-convergence measure in
``convergence_metrics``: how far the current density moves when the mesh is
refined.

Validation against experimental data is the other half of the picture and is
deliberately not attempted here -- it needs measurements this repository does
not yet carry.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from .model import DEFAULT_MAX_NODES, SweepResult, solve
from .params import Params

__all__ = ["RunMetrics", "SolverSettings", "create_run_directory",
           "sweep_metrics", "convergence_metrics", "write_run_log"]

#: Peak memory ``solve_bvp`` needs per 1000 mesh nodes for this 80-state
#: system, measured on the collocation Jacobian's LU factorization.
GB_PER_1000_NODES = 0.85

#: Ceiling applied to the *refined* solve of a convergence check, so that
#: asking for a check can never be what exhausts memory. See GB_PER_1000_NODES.
REFINEMENT_NODE_CEILING = 4_000


# =============================================================================
# PROVENANCE
# =============================================================================

def _git_revision() -> str:
    """Short commit hash of the working tree, marked if it has changes."""
    def git(*args: str) -> str:
        return subprocess.run(("git", *args), capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent,
                              timeout=10).stdout.strip()
    try:
        revision = git("rev-parse", "--short", "HEAD")
        if not revision:
            return "unknown (not a git repository)"
        return f"{revision} (dirty)" if git("status", "--porcelain") else revision
    except (OSError, subprocess.SubprocessError):
        return "unknown (git unavailable)"


@dataclass(frozen=True)
class SolverSettings:
    """The solver knobs a run was invoked with, as recorded in the log."""

    tol: float
    n_per_region: int
    max_nodes: int
    voltages: np.ndarray


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class RunMetrics:
    """Everything the log reports about one sweep."""

    settings: SolverSettings
    #: per voltage: current [A/cm^2], power [W/cm^2], nodes, max rms residual,
    #: and whether solve_bvp met its tolerance
    current_densities: np.ndarray
    power_densities: np.ndarray
    node_counts: np.ndarray
    max_residuals: np.ndarray
    tolerance_met: np.ndarray
    elapsed_seconds: float
    #: filled in by ``convergence_metrics`` when a check was run
    refined_max_nodes: int | None = None
    refined_currents: np.ndarray | None = None
    relative_current_change: np.ndarray | None = None
    convergence_note: str = "not run"
    started: datetime = field(default_factory=lambda: datetime.now().astimezone())

    @property
    def voltages(self) -> np.ndarray:
        return self.settings.voltages

    @property
    def peak_power(self) -> tuple[float, float]:
        """[W/cm^2] peak power density and the [V] it occurs at."""
        if not len(self.power_densities):
            return float("nan"), float("nan")
        best = int(np.argmax(self.power_densities))
        return float(self.power_densities[best]), float(self.voltages[best])

    @property
    def limiting_current(self) -> float:
        """[A/cm^2] current at the lowest voltage solved.

        A lower bound on the true limiting current rather than the plateau
        itself, unless the sweep was taken far enough for the curve to flatten.
        """
        return float(self.current_densities[-1]) if len(self.current_densities) else float("nan")

    @property
    def worst_relative_change(self) -> float:
        """[-] largest ``|dI|/I`` over the sweep, or NaN if no check was run."""
        if self.relative_current_change is None:
            return float("nan")
        finite = self.relative_current_change[np.isfinite(self.relative_current_change)]
        return float(finite.max()) if finite.size else float("nan")


def sweep_metrics(result: SweepResult, settings: SolverSettings,
                  elapsed_seconds: float) -> RunMetrics:
    """Collect the per-voltage cost and diagnostic metrics of one sweep."""
    solutions = result.solutions
    return RunMetrics(
        settings=SolverSettings(tol=settings.tol,
                                n_per_region=settings.n_per_region,
                                max_nodes=settings.max_nodes,
                                voltages=result.voltages),
        current_densities=result.current_densities,
        power_densities=result.power_densities,
        node_counts=np.array([len(sol.x) for sol in solutions]),
        max_residuals=np.array([float(np.max(sol.rms_residuals)) for sol in solutions]),
        tolerance_met=np.array([bool(sol.success) for sol in solutions]),
        elapsed_seconds=elapsed_seconds,
    )


def convergence_metrics(metrics: RunMetrics, refine_factor: int,
                        params: Params | None = None) -> RunMetrics:
    """Re-solve on a finer mesh and record how far the current density moved.

    This is the metric that certifies a result. ``solve_bvp``'s own success
    flag cannot: it reports failure wherever the mesh cannot resolve the
    ``k_ad`` discontinuity, however close to converged the current density is.

    The refined solve is capped at ``REFINEMENT_NODE_CEILING`` so that running
    a check is never itself what exhausts memory, and a ``MemoryError`` is
    caught rather than being allowed to discard the sweep that has already
    been computed.
    """
    settings = metrics.settings
    refined_max_nodes = min(refine_factor * settings.max_nodes, REFINEMENT_NODE_CEILING)

    if refined_max_nodes <= settings.max_nodes:
        metrics.convergence_note = (
            f"skipped: --max-nodes {settings.max_nodes} is already at or above "
            f"the {REFINEMENT_NODE_CEILING}-node refinement ceiling "
            f"(~{REFINEMENT_NODE_CEILING * GB_PER_1000_NODES / 1000:.1f} GB)")
        return metrics
    if not len(settings.voltages):
        metrics.convergence_note = "skipped: no voltages converged in the base sweep"
        return metrics

    try:
        refined = solve(voltages=settings.voltages, tol=settings.tol,
                        n_per_region=settings.n_per_region,
                        max_nodes=refined_max_nodes, params=params)
    except MemoryError:
        metrics.convergence_note = (
            f"failed: ran out of memory refining to {refined_max_nodes} nodes")
        return metrics

    # The refined sweep can stop early where the base one did not, so compare
    # only the voltages both of them reached.
    shared = min(len(refined.current_densities), len(metrics.current_densities))
    base = metrics.current_densities[:shared]
    fine = refined.current_densities[:shared]
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(np.abs(fine) > 1e-6, np.abs(base - fine) / np.abs(fine), np.nan)

    metrics.refined_max_nodes = refined_max_nodes
    metrics.refined_currents = fine
    metrics.relative_current_change = relative
    metrics.convergence_note = (
        f"{settings.max_nodes} vs {refined_max_nodes} nodes over "
        f"{shared} voltage{'s' if shared != 1 else ''}"
        + ("" if shared == len(metrics.current_densities)
           else f" ({len(metrics.current_densities) - shared} voltage(s) the "
                "refined sweep did not reach are excluded)"))
    return metrics


# =============================================================================
# RUN DIRECTORY AND LOG
# =============================================================================

def create_run_directory(base: str | Path = "results") -> Path:
    """Make a fresh timestamped directory under ``base`` and return it.

    Named ``run_YYYYmmdd_HHMMSS`` so runs sort chronologically. A numeric
    suffix disambiguates two runs started in the same second.
    """
    base = Path(base)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    directory = base / stamp
    suffix = 1
    while directory.exists():
        suffix += 1
        directory = base / f"{stamp}_{suffix}"
    directory.mkdir(parents=True)
    return directory


def _format_table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
              for i in range(len(header))]
    lines = ["  ".join(name.rjust(width) for name, width in zip(header, widths)),
             "  ".join("-" * width for width in widths)]
    lines += ["  ".join(cell.rjust(width) for cell, width in zip(row, widths))
              for row in rows]
    return lines


def _log_text(metrics: RunMetrics, directory: Path, command: str) -> str:
    settings = metrics.settings
    finished = datetime.now().astimezone()
    lines = [
        "=" * 72,
        "1D-PEMFC-Model run log",
        "=" * 72,
        "",
        "## Provenance",
        f"Run directory  : {directory}",
        f"Started        : {metrics.started.isoformat(timespec='seconds')}",
        f"Finished       : {finished.isoformat(timespec='seconds')}",
        f"Git revision   : {_git_revision()}",
        f"Command        : {command}",
        f"Working dir    : {Path.cwd()}",
        f"Host           : {platform.node()} ({platform.platform()})",
        f"Python         : {platform.python_version()} ({sys.executable})",
        f"numpy / scipy  : {np.__version__} / {scipy.__version__}",
        "",
        "## Solver settings",
        f"tol            : {settings.tol:g}",
        f"n_per_region   : {settings.n_per_region}",
        f"max_nodes      : {settings.max_nodes}"
        + (f"  (MATLAB bvp4c's floor(10000/n))" if settings.max_nodes == DEFAULT_MAX_NODES else ""),
        f"voltages [V]   : {np.array2string(settings.voltages, precision=3, separator=' ', max_line_width=200)}",
        "",
        "## Cost",
        f"Elapsed        : {metrics.elapsed_seconds:.1f} s",
        f"Voltages solved: {len(settings.voltages)}",
        f"Peak mesh      : {int(metrics.node_counts.max()) if metrics.node_counts.size else 0} nodes"
        + (f"  (~{metrics.node_counts.max() * GB_PER_1000_NODES / 1000:.2f} GB)"
           if metrics.node_counts.size else ""),
        "",
        "## Per-voltage results and diagnostics",
    ]

    rows = [(f"{voltage:.3f}", f"{current:.4f}", f"{power:.4f}", str(int(nodes)),
             f"{residual:.2e}", "yes" if met else "NO")
            for voltage, current, power, nodes, residual, met
            in zip(settings.voltages, metrics.current_densities,
                   metrics.power_densities, metrics.node_counts,
                   metrics.max_residuals, metrics.tolerance_met)]
    lines += _format_table(("U [V]", "I [A/cm^2]", "P [W/cm^2]", "nodes",
                            "max rms res", "tol met"), rows)

    if not metrics.tolerance_met.all():
        missed = settings.voltages[~metrics.tolerance_met]
        lines += [
            "",
            f"NOTE: solve_bvp did not meet tol at {len(missed)} voltage(s): "
            f"{np.array2string(missed, precision=3, separator=' ', max_line_width=200)}.",
            "      A missed tolerance is a diagnostic, not a verdict: judge these",
            "      points by the convergence section below, which reports how far",
            "      the current density moves when the mesh is refined.",
        ]

    lines += [
        "",
        "## Figures of merit",
    ]
    peak_power, peak_voltage = metrics.peak_power
    lines += [
        f"Peak power density  : {peak_power:.4f} W/cm^2 at U = {peak_voltage:.3f} V",
        f"Current at lowest U : {metrics.limiting_current:.4f} A/cm^2 "
        f"(U = {settings.voltages[-1]:.3f} V)" if len(settings.voltages) else
        "Current at lowest U : n/a",
    ]

    lines += ["", "## Verification: mesh convergence of the current density",
              f"Check          : {metrics.convergence_note}"]
    if metrics.relative_current_change is not None:
        rows = [(f"{voltage:.3f}", f"{base:.4f}", f"{fine:.4f}",
                 "n/a" if not np.isfinite(change) else f"{change * 100:.3f}%")
                for voltage, base, fine, change
                in zip(settings.voltages, metrics.current_densities,
                       metrics.refined_currents, metrics.relative_current_change)]
        lines += _format_table((
            "U [V]", f"I @{settings.max_nodes}", f"I @{metrics.refined_max_nodes}",
            "|dI|/I"), rows)
        worst = metrics.worst_relative_change
        lines += ["",
                  f"Worst |dI|/I   : {worst * 100:.3f}%" if np.isfinite(worst)
                  else "Worst |dI|/I   : n/a",
                  "This is the number that certifies the sweep: it says how much",
                  "refining the mesh changes the answer. Report it alongside any",
                  "'tol met: NO' rows above.",
                  ]

    lines += [
        "",
        "## Not measured here",
        "Validation against experimental data. Agreement with measurements is a",
        "separate question from the numerical convergence above, and is what",
        "decides whether a change to the physics is an improvement. Record it",
        "per region -- kinetic (>0.8 V), ohmic (0.8-0.6 V), mass transport",
        "(<0.6 V) -- since a whole-curve RMSE hides a missed limiting current.",
        "",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


def write_run_log(metrics: RunMetrics, directory: Path,
                  command: str | None = None, filename: str = "metrics.log") -> Path:
    """Write ``metrics.log`` into ``directory`` and return its path."""
    command = " ".join(sys.argv) if command is None else command
    path = Path(directory) / filename
    path.write_text(_log_text(metrics, Path(directory), command), encoding="utf-8")
    return path
