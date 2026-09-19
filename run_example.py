"""Command-line entry point: solves the model for a sweep of cell
voltages, prints the resulting current densities, and writes one
timestamped run directory under results/ holding the potentials, fluxes,
sensitivity and polarization curve figures, the sensitivity analysis as
two CSV tables, plus a metrics.log.

The log records what was run, when, against which revision of the code,
and how good the answer is -- see pemfc_1d/metrics.py for why solver cost
and solution quality are reported as separate things.

Examples
--------
    python run_example.py
    python run_example.py --voltages 1.15 1.10 1.05 1.00 --no-show   # just the top
    python run_example.py --no-convergence-check          # skip the refined solve
    python run_example.py --no-sensitivity                # skip the sensitivity study
    python run_example.py --sensitivity-parameters sigma_p D_O2
"""
import argparse
import time

import matplotlib

# =============================================================================
# CONFIGURATION
# =============================================================================

#: Run the forward sensitivity analysis (pemfc_1d/sensitivity.py) as part of a
#: run, writing one ``sensitivity_<parameter>.png`` per material parameter
#: alongside the other figures, and the underlying dY/dtheta as
#: ``sensitivity_raw_<run_id>.csv`` and ``sensitivity_by_layer_<run_id>.csv``
#: (pemfc_1d/export.py).
#:
#: Set this to False to switch the whole analysis off. Nothing of it then runs:
#: the module is not even imported, so neither sympy nor the symbolic
#: differentiation is paid for and a run costs exactly what it did before the
#: feature existed. ``--sensitivity`` / ``--no-sensitivity`` overrides it for a
#: single run.
#:
#: It is worth having a switch because the analysis is the expensive half of a
#: run: one augmented 160-equation boundary value problem per parameter and per
#: voltage, against one 80-equation problem per voltage for the sweep itself.
SENSITIVITY_ENABLED = True

#: Which material parameters to analyse. None means every parameter
#: pemfc_1d.sensitivity knows about; naming a few is the way to keep a run
#: short while still looking at the ones that matter.
SENSITIVITY_PARAMETERS = None

from pemfc_1d.metrics import (SolverSettings, convergence_metrics,
                              create_run_directory, sweep_metrics, write_run_log)
from pemfc_1d.model import DEFAULT_MAX_NODES, solve
from pemfc_1d.postprocessing import plot_polarization_curve, plot_potentials_and_fluxes


def run_sensitivity_analysis(result, parameters, directory):
    """Solve the sensitivity equations, then export and plot each parameter.

    Both outputs come off the same solved object: ``export.add`` and
    ``plot_sensitivity_profiles`` read the arrays ``solve_sensitivity``
    returned, so writing the CSVs costs no extra solving.

    Imported here rather than at module level so that a run with the analysis
    switched off never loads sympy or the symbolic machinery.
    """
    from pemfc_1d.export import SensitivityExport
    from pemfc_1d.postprocessing import plot_sensitivity_profiles
    from pemfc_1d.sensitivity import SensitivitySettings, solve_sensitivity

    settings = SensitivitySettings(
        **({} if parameters is None else {"parameters": tuple(parameters)}))
    studied = settings.resolved_parameters()
    print(f"Solving the sensitivity equations for {len(studied)} parameter(s) "
          "(--no-sensitivity to skip)...")

    print("parameter           dI/dtheta [A/cm^2] per voltage")
    with SensitivityExport(directory) as export:
        for parameter in studied:
            started = time.perf_counter()
            sensitivity = solve_sensitivity(result, parameter, settings)
            export.add(sensitivity)
            figure = plot_sensitivity_profiles(sensitivity)
            figure.savefig(directory / f"sensitivity_{parameter.name}.png", dpi=150)
            currents = "  ".join(f"{value:+.3e}"
                                 for value in sensitivity.current_sensitivities)
            print(f"{parameter.name:18s}  {currents}   "
                  f"({time.perf_counter() - started:.1f} s)")

    print(f"Sensitivity data: {export.raw_path.name} "
          f"({export.n_raw_rows} rows) and {export.by_layer_path.name} "
          f"({export.n_by_layer_rows} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voltages", type=float, nargs="+", default=None,
                        help="cell voltages [V] to sweep over (default: 1.15 to 0.40 in 50 mV steps, "
                             "the whole polarization curve)")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="tolerance passed to scipy.integrate.solve_bvp (default: 1e-4, "
                             "the RelTol of the MATLAB reference implementation)")
    parser.add_argument("--n-per-region", type=int, default=11,
                        help="number of initial mesh points per region (default: 11)")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES,
                        help=f"maximum number of mesh nodes (default: {DEFAULT_MAX_NODES}, "
                             "matching MATLAB bvp4c's floor(10000/n); raising it costs "
                             "about 0.85 GB of memory per 1000 nodes)")
    parser.add_argument("--convergence-check", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="re-solve on a finer mesh and log how far the current density "
                             "moves; this is the metric that certifies the run (default: on)")
    parser.add_argument("--refine-factor", type=int, default=12,
                        help="mesh refinement factor for the convergence check (default: 12)")
    parser.add_argument("--sensitivity", action=argparse.BooleanOptionalAction,
                        default=SENSITIVITY_ENABLED,
                        help="solve the forward sensitivity equations for each material "
                             "parameter and write one figure per parameter "
                             f"(default: {'on' if SENSITIVITY_ENABLED else 'off'}, "
                             "set by SENSITIVITY_ENABLED in this file)")
    parser.add_argument("--sensitivity-parameters", nargs="+", default=SENSITIVITY_PARAMETERS,
                        metavar="NAME",
                        help="material parameters to analyse (default: all of them); "
                             "an unknown name lists the ones that exist")
    parser.add_argument("--outdir", default="results",
                        help="directory the per-run folders are created in (default: results)")
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening an interactive window")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    result = solve(voltages=args.voltages, tol=args.tol,
                   n_per_region=args.n_per_region, max_nodes=args.max_nodes)
    elapsed = time.perf_counter() - started

    requested = SolverSettings(tol=args.tol, n_per_region=args.n_per_region,
                               max_nodes=args.max_nodes, voltages=result.voltages)
    metrics = sweep_metrics(result, requested, elapsed)

    print("U [V]      I [A/cm^2]   P [W/cm^2]")
    for voltage, current, power in zip(result.voltages, result.current_densities,
                                       result.power_densities):
        print(f"{voltage:6.3f}     {current:10.4f}   {power:10.4f}")

    if args.convergence_check:
        print("Refining the mesh to measure convergence "
              "(--no-convergence-check to skip)...")
        metrics = convergence_metrics(metrics, args.refine_factor, result.params)
        worst = metrics.worst_relative_change
        print(f"Worst |dI|/I under refinement: "
              f"{worst * 100:.3f}%" if worst == worst else
              f"Convergence check: {metrics.convergence_note}")

    directory = create_run_directory(args.outdir)
    figure_potentials, figure_fluxes = plot_potentials_and_fluxes(result)
    figure_polarization = plot_polarization_curve(result)
    figure_potentials.savefig(directory / "potentials.png", dpi=150)
    figure_fluxes.savefig(directory / "fluxes.png", dpi=150)
    figure_polarization.savefig(directory / "polarization_curve.png", dpi=150)
    log_path = write_run_log(metrics, directory)

    if args.sensitivity and len(result.voltages):
        run_sensitivity_analysis(result, args.sensitivity_parameters, directory)

    print(f"Run saved to {directory}/ (figures and {log_path.name})")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
