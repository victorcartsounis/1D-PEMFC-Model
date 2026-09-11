"""Command-line entry point: solves the model for a sweep of cell
voltages, prints the resulting current densities and saves the
potentials, fluxes and polarization curve figures to figs/.

Examples
--------
    python run_example.py
    python run_example.py --voltages 1.15 1.10 1.05 1.00 --no-show
"""
import argparse
import os

import matplotlib

from mmm1d.model import solve
from mmm1d.postprocessing import plot_potentials_and_fluxes, plot_polarization_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voltages", type=float, nargs="+", default=None,
                        help="cell voltages [V] to sweep over (default: 1.15 to 1.00 in 50 mV steps)")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="tolerance passed to scipy.integrate.solve_bvp (default: 1e-4)")
    parser.add_argument("--n-per-region", type=int, default=11,
                        help="number of initial mesh points per region (default: 11)")
    parser.add_argument("--max-nodes", type=int, default=200000,
                        help="maximum number of mesh nodes (default: 200000)")
    parser.add_argument("--outdir", default="figs",
                        help="directory the figures are written to (default: figs)")
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening an interactive window")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = solve(voltages=args.voltages, tol=args.tol,
                   n_per_region=args.n_per_region, max_nodes=args.max_nodes)

    print("U [V]      I [A/cm^2]")
    for U, I in zip(result.U, result.I):
        print(f"{U:6.3f}     {I:10.4f}")

    os.makedirs(args.outdir, exist_ok=True)
    fig_pot, fig_flux = plot_potentials_and_fluxes(result)
    fig_pol = plot_polarization_curve(result)
    fig_pot.savefig(os.path.join(args.outdir, "potentials.png"), dpi=150)
    fig_flux.savefig(os.path.join(args.outdir, "fluxes.png"), dpi=150)
    fig_pol.savefig(os.path.join(args.outdir, "polarization_curve.png"), dpi=150)
    print(f"Figures saved to {args.outdir}/")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
