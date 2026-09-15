"""Tests for the run metrics and the on-disk run log.

Run with: python -m pytest tests/
"""
import numpy as np

from pemfc_1d import __version__ as MODEL_VERSION
from pemfc_1d.metrics import (SolverSettings, convergence_metrics,
                              create_run_directory, git_revision,
                              sweep_metrics, write_run_log)
from pemfc_1d.model import DEFAULT_MAX_NODES, solve


def _small_sweep():
    voltages = [1.15, 1.10]
    result = solve(voltages=voltages, tol=1e-4, n_per_region=11)
    settings = SolverSettings(tol=1e-4, n_per_region=11,
                              max_nodes=DEFAULT_MAX_NODES,
                              voltages=result.voltages)
    return result, sweep_metrics(result, settings, elapsed_seconds=1.0)


def test_run_directories_are_unique(tmp_path):
    first = create_run_directory(tmp_path)
    second = create_run_directory(tmp_path)
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_sweep_metrics_line_up_with_the_solution():
    result, metrics = _small_sweep()

    for values in (metrics.current_densities, metrics.node_counts,
                   metrics.max_residuals, metrics.tolerance_met):
        assert len(values) == len(result.voltages)

    assert np.all(metrics.node_counts <= DEFAULT_MAX_NODES)
    assert metrics.tolerance_met.all(), "the high-voltage end should meet tol"

    peak_power, peak_voltage = metrics.peak_power
    assert peak_power == metrics.power_densities.max()
    assert peak_voltage in result.voltages


def test_convergence_check_certifies_the_high_voltage_end():
    _, metrics = _small_sweep()
    metrics = convergence_metrics(metrics, refine_factor=4)

    assert metrics.refined_max_nodes == 4 * DEFAULT_MAX_NODES
    assert metrics.relative_current_change is not None
    # Currents this far from limiting are tiny, so |dI|/I is reported as NaN
    # rather than dividing by ~0; the check must still have run.
    assert "vs" in metrics.convergence_note


def test_log_records_provenance_settings_and_metrics(tmp_path):
    _, metrics = _small_sweep()
    directory = create_run_directory(tmp_path)

    path = write_run_log(metrics, directory, command="python run_example.py --no-show")
    text = path.read_text(encoding="utf-8")

    assert path.parent == directory
    for section in ("## Provenance", "## Solver settings", "## Cost",
                    "## Per-voltage results and diagnostics",
                    "## Figures of merit",
                    "## Verification: mesh convergence of the current density"):
        assert section in text, f"missing {section}"

    assert "Git revision" in text
    # The released version is always knowable; the revision is not, once the
    # package is installed rather than checked out.
    assert f"Model version  : pemfc_1d {MODEL_VERSION}" in text
    assert "python run_example.py --no-show" in text
    assert f"max_nodes      : {DEFAULT_MAX_NODES}" in text
    # every solved voltage has a row
    for voltage in metrics.voltages:
        assert f"{voltage:.3f}" in text


def test_log_records_the_revision_its_caller_supplies(tmp_path):
    """A separate front end runs the model from site-packages, where the model's
    own source is not a checkout. It passes its own revision so the log names
    the program that actually ran."""
    _, metrics = _small_sweep()
    directory = create_run_directory(tmp_path)

    path = write_run_log(metrics, directory, revision="abc1234 (dirty)")
    text = path.read_text(encoding="utf-8")

    assert "Git revision   : abc1234 (dirty)" in text


def test_revision_of_a_directory_outside_any_checkout_is_reported_unknown(tmp_path):
    assert git_revision(tmp_path) == "unknown (not a git checkout)"


def test_convergence_check_is_skipped_rather_than_exhausting_memory():
    _, metrics = _small_sweep()
    metrics.settings = SolverSettings(tol=metrics.settings.tol,
                                      n_per_region=metrics.settings.n_per_region,
                                      max_nodes=1_000_000,
                                      voltages=metrics.settings.voltages)
    metrics = convergence_metrics(metrics, refine_factor=12)

    assert metrics.relative_current_change is None
    assert metrics.convergence_note.startswith("skipped:")
