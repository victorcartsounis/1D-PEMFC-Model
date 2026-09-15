"""Command-line entry point: solves the model for a sweep of cell
voltages, prints the resulting current densities, and writes one
timestamped run directory under results/ holding the potentials, fluxes
and polarization curve figures plus a metrics.log.

The log records what was run, when, against which revision of the code,
and how good the answer is -- see pemfc_1d/metrics.py for why solver cost
and solution quality are reported as separate things.

Examples
--------
    python run_example.py
    python run_example.py --voltages 1.15 1.10 1.05 1.00 --no-show
    python run_example.py --no-convergence-check          # skip the refined solve
"""
import argparse
import time

import matplotlib

from pemfc_1d.metrics import (SolverSettings, convergence_metrics,
                              create_run_directory, sweep_metrics, write_run_log)
from pemfc_1d.model import DEFAULT_MAX_NODES, solve
from pemfc_1d.postprocessing import plot_polarization_curve, plot_potentials_and_fluxes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voltages", type=float, nargs="+", default=None,
                        help="cell voltages [V] to sweep over (default: 1.15 to 1.00 in 50 mV steps)")
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
    print(f"Run saved to {directory}/ (figures and {log_path.name})")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
