# 1D PEM Fuel Cell Model

A Python implementation of a one-dimensional, steady-state, non-isothermal,
two-phase, macro-homogeneous membrane electrode assembly (MEA) model for
polymer electrolyte membrane fuel cells.

Written by **Victor Constantin Cartsounis** as the codebase for a master's
thesis at **CEFET-MG**, supervised by **Sidney Nicodemos da Silva**.

The model resolves eight coupled quantities across the five MEA layers — anode
GDL, anode catalyst layer, membrane, cathode catalyst layer, cathode GDL — and
will be extended over the course of the thesis.

## Physics reference

The governing equations, constitutive relations and parameter values
implemented here follow:

> R. Vetter, J. O. Schumacher, *Free open reference implementation of a
> two-phase PEM fuel cell model*, Computer Physics Communications **234**
> (2019) 223–234. <https://doi.org/10.1016/j.cpc.2018.07.023>

Cite that paper for the model formulation, and this repository for the
implementation and the extensions developed during the thesis.

The authors' MATLAB reference implementation, published alongside the paper,
was also consulted during development:
<https://github.com/Isomorph-Electrochemical-Cells/PEMFC-1DMMM>.

### License and attribution

This project is released under the **BSD-3-Clause** license (see `LICENSE`).
The reference implementation consulted during development is distributed under
the same license by the Institute of Computational Physics, ZHAW (© 2017–2020),
and their copyright notice is retained in `LICENSE` accordingly. Clause 3 of
that license also means their names must not be used to imply that they endorse
this work or its extensions.

## Model structure

Each of the eight quantities obeys a second-order transport equation, written
as a potential/flux pair — 16 first-order equations per layer, 80 in total.

| Quantity | Symbol | Active layers |
|---|---|---|
| Electron potential | `phi_e` | GDLs and CLs |
| Proton potential | `phi_p` | CLs and membrane |
| Temperature | `T` | all |
| Dissolved water content | `lambda` | CLs and membrane |
| Water vapour mass fraction | `w_H2O` | GDLs and CLs |
| Oxygen mass fraction | `w_O2` | cathode side |
| Liquid water pressure | `P_liq` | cathode side |
| Gas pressure | `P_gas` | GDLs and CLs |

## Solver design

Every layer carries its own equation set, coupled to its neighbours by
continuity conditions at the four internal interfaces.
`scipy.integrate.solve_bvp` requires a strictly increasing mesh and has no
notion of separate regions, so the five layers are **stacked** into a single
system of `5 × 16 = 80` first-order ODEs over one normalised coordinate
`s ∈ [0, 1]`, with a copy of `s` per layer. The chain rule maps it back onto
physical position:

```
x_region(s) = Lsum[region] + s * L[region]      =>      dY/ds = L[region] * dY/dx
```

Interface continuity and gas-channel conditions are then imposed as boundary
conditions on the stacked system, which is equivalent to solving the five
layers as a coupled multi-region problem. See `mmm1d/model.py`.

## Layout

```
mmm1d/
├── state.py            # state-vector layout: Region / State / Quantity enums
├── params.py           # constants, operating conditions, constitutive relations
├── saturation.py       # capillary pressure -> liquid water saturation
├── model.py            # per-layer physics, boundary conditions, solve loop
└── postprocessing.py   # profile extraction and plots
tests/
├── data/
│   └── reference_solution.npz   # validated reference output (golden file)
├── test_smoke.py                # convergence and monotonicity
├── test_regression.py           # full solution pinned to the golden file
└── test_constitutive.py         # parameters and saturation inversion
run_example.py                   # command-line entry point
```

Quantities and layers are addressed by name rather than by index: `State`,
`Region` and `Quantity` are `IntEnum`s, so `block[State.W_O2]` and
`residuals[Region.CCL, State.J_E]` index arrays directly while remaining
readable. `Params` is a frozen dataclass, so the physical constants of a
running model cannot be mutated by accident.

## Installation

```bash
poetry install
```

## Running

```bash
python run_example.py
```

This solves the default voltage sweep (1.15 to 1.00 V in 50 mV steps), prints
the current and power density at each voltage, and writes one timestamped run
directory under `results/`:

```
results/run_20260912_102554/
    potentials.png
    fluxes.png
    polarization_curve.png
    metrics.log
```

Every run gets its own directory, so results are never overwritten and any
figure can be traced back to the settings that produced it.

For a full polarization curve including the mass-transport-limited region:

```bash
python run_example.py --no-show --voltages 1.15 1.10 1.05 1.00 0.95 0.90 0.85 \
    0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40
```

At low voltage the mesh grows quickly and `solve_bvp` may report that it could
not meet `--tol`, warning once per voltage. `--max-nodes` defaults to
`10000 // 80` = 125, the ceiling MATLAB's `bvp4c` imposes on a system of this
size (`NMax = floor(10000/n)`), so the port stops refining where the reference
implementation stops and carries on from the same solution. Raising it is
rarely the fix: where the adaptive mesh clusters around a feature the
collocation method cannot resolve, the solver exhausts memory before the error
estimate drops. Judge those points by the convergence check described below
rather than by the tolerance flag.

### The run log

`metrics.log` records the run's provenance (timestamp, git revision, exact
command, host, library versions), the solver settings, and two kinds of
measurement that are deliberately kept apart:

* **Cost and diagnostics** — elapsed time, node count per voltage, the largest
  `rms_residual`, and whether `solve_bvp` met its tolerance. These say how hard
  the solver worked, not whether the answer is right.
* **Verification** — the current density re-solved on a mesh `--refine-factor`
  times finer, and `|dI|/I` between the two. This is the number that certifies
  a run: it says how much refining the mesh changes the answer.

The distinction matters because the two can disagree. A solver can miss its
tolerance on a solution that is converged to several digits — a discontinuity
in a constitutive relation is enough to hold the local residual up no matter
how fine the mesh gets — and can equally meet it on a mesh too coarse for the
answer to have settled. So the tolerance column is read as a diagnostic, and
the refinement comparison is what certifies a point.

The refined solve roughly triples the runtime; pass `--no-convergence-check` to
skip it. It is capped so that requesting a check can never be what exhausts
memory.

Not measured: agreement with experimental data. That is a separate question
from numerical convergence, and it is the one that decides whether a change to
the physics is an improvement.

From Python:

```python
from mmm1d import solve
from mmm1d.postprocessing import plot_potentials_and_fluxes, plot_polarization_curve

result = solve()                      # default sweep
print(result.voltages)                # [V]
print(result.current_densities)       # [A/cm^2]
print(result.power_densities)         # [W/cm^2]
assert result.converged

fig_potentials, fig_fluxes = plot_potentials_and_fluxes(result)
fig_polarization = plot_polarization_curve(result)
```

A sweep is run by continuation — each voltage starts from the previous
converged solution — so voltages should be ordered from high (low current)
downwards.

## Tests

```bash
python -m pytest tests/
```

`tests/data/reference_solution.npz` is a golden file holding validated output:
current densities, spatial profiles, ODE right-hand sides, boundary-condition
residuals, every parameter value and the saturation inversion. The suite pins
the model to it, so a refactor that perturbs the numbers fails loudly.

**Regenerate the golden file only when the physics is deliberately changed** —
never to make a failing refactor pass.

Some two-phase behaviour is intentionally inactive in this version; the tests
pin it so it cannot change unnoticed.

## Performance and convergence

* The default sweep (1.15–1.00 V, 4 points) solves in about 3 seconds.
* Tested down to 0.40 V (~2.3 A/cm², in the mass-transport-limited plateau);
  the full 1.15–0.40 V sweep takes a couple of minutes, as the adaptive mesh
  grows considerably at low voltage.
* If `solve_bvp` fails to converge, increase `n_per_region` or `max_nodes`,
  narrow the voltage step for smoother continuation, or relax `tol`.

## VS Code integration

The `.vscode/` folder ships ready-to-use configuration:

* **`launch.json`** — run the default sweep, run the full polarization curve,
  debug the open file, or debug the test suite. Breakpoints work in
  `mmm1d/model.py` as usual; open the Run and Debug panel (`Ctrl+Shift+D`).
* **`tasks.json`** — shell tasks (`Ctrl+Shift+P` → *Tasks: Run Task*) for
  `poetry install`, running the tests, and running the default sweep.
* **`settings.json`** — enables the Testing panel with pytest auto-discovery.
* **`extensions.json`** — recommends Python, Pylance, debugpy, Even Better TOML
  and Jupyter.

## Credits

Implementation, extensions and subsequent development: **Victor Constantin
Cartsounis**, CEFET-MG, 2026.

Model formulation and constitutive relations: R. Vetter and J. O. Schumacher,
Institute of Computational Physics, ZHAW — see
[Physics reference](#physics-reference).
