# 1D PEM Fuel Cell Model

Python implementation of a 1D, steady-state, non-isothermal,
macro-homogeneous two-phase MEA model for PEM fuel cells, developed by
**Victor Constantin Cartsounis** as the codebase for a master's thesis
at **CEFET-MG**, under the supervision of **Sidney Nicodemos da
Silva**.

This project starts from the physical formulation and numerical
structure published by Vetter & Schumacher (see
[Reference](#reference)) and will be extended throughout the thesis
— *the exact scope is still to be written up here (e.g. additional
submodels, degradation, transient operation, validation against
experimental data, parameter optimization).*

## Reference

The physical and numerical starting point of this code is:

> R. Vetter, J.O. Schumacher, *Free open reference implementation of a
> two-phase PEM fuel cell model*, Computer Physics Communications 234
> (2019) 223–234. https://doi.org/10.1016/j.cpc.2018.07.023

Original source code (MATLAB, reference version from the paper, with
the v2 modifications by Dr. Robert Herrendörfer and Prof. Dr. Jürgen
O. Schumacher — Institute of Computational Physics, ZHAW):
https://github.com/Isomorph-Electrochemical-Cells/PEMFC-1DMMM
(mirrors the same files originally archived on Mendeley Data,
https://doi.org/10.17632/2msdd4j84c.1).

When citing this work in the thesis, cite the paper above for the
original model formulation, and reference this repository for the
extensions developed during the thesis.

### License

The original code is distributed under the **BSD-3-Clause** license
(ZHAW-ICP, 2017–2020). This project inherits substantial parts of that
implementation and therefore **keeps the same license** (see `LICENSE`) —
clause 1 of the BSD-3 requires retaining the original copyright notice
in redistributions, even with modifications. This does not prevent you
from presenting this repository as your thesis code; it only requires
keeping credit to the original authors for the parts derived from
their work (clause 3 also forbids using their name to endorse your
work without permission, so avoid implying that they endorse your
extensions).

---

## Structure

```
.vscode/                   # VS Code integration (see below)
mmm1d/
├── params.py          # constants, operating conditions and constitutive relations
├── model.py             # per-subdomain physics (odefun), boundary conditions
│                         # (bcfun), initial guess (yinit), and the solve loop
├── pc_s.py               # capillary-pressure -> saturation inversion
└── postprocessing.py     # profile extraction + plots (potentials, fluxes, polarization)
run_example.py             # command-line script
pyproject.toml
.gitignore
LICENSE
```

## VS Code integration

The `.vscode/` folder ships ready-to-use configuration:

* **`launch.json`** — debug configurations: run the default voltage
  sweep, run the full 1.15–0.40 V polarization curve, debug the
  currently open file, or debug the smoke test (breakpoints work in
  `mmm1d/model.py` as usual). Open the "Run and Debug" panel (`Ctrl+Shift+D`
  / `Cmd+Shift+D`) and pick a configuration.
* **`tasks.json`** — shell tasks (`Ctrl+Shift+P` → *Tasks: Run Task*)
  for `poetry install`, running the smoke test, and running the
  default sweep without opening a terminal manually.
* **`settings.json`** — enables the Testing panel (pytest,
  auto-discovers `tests/`), sensible editor defaults.
* **`extensions.json`** — recommends the Python, Pylance, debugpy,
  Even Better TOML and Jupyter extensions (VS Code will prompt you to
  install them on first open).


## Installation

```bash
poetry install
```

## Running

```bash
python run_example.py
```

This solves the model for the default voltage sweep (1.15 to 1.00 V,
in 50 mV steps), prints the current density obtained at each voltage,
and saves three figures to `figs/`: `potentials.png`, `fluxes.png` and
`polarization_curve.png`.

For a full polarization curve (including the mass-transport-limited
region):

```bash
python run_example.py --voltages 1.15 1.10 1.05 1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 --no-show
```

Lower voltages (higher currents) require more mesh nodes and more CPU
time — see the performance note below.

Direct use from Python code:

```python
from mmm1d.model import solve
from mmm1d.postprocessing import plot_potentials_and_fluxes, plot_polarization_curve

result = solve()                 # default sweep
print(result.U, result.I)        # cell voltages [V] and current densities [A/cm^2]

fig_pot, fig_flux = plot_potentials_and_fluxes(result)
fig_pol = plot_polarization_curve(result)
```

## Tests

```bash
pip install -e .
python -m pytest tests/
```

## Numerical origin of the solver

The original MATLAB code solves a **multi-region boundary value
problem** with `bvp4c`: 5 MEA layers (AGDL, ACL, PEM, CCL, CGDL), each
with its own set of ODEs, coupled through continuity conditions at the
interfaces. `scipy.integrate.solve_bvp` does not natively support
multiple regions (it requires a strictly increasing mesh, with no
repeated nodes), so here the 5 regions are **stacked** into a single
system of `5 × 16 = 80` first-order ODEs, solved simultaneously over a
normalized independent variable `s ∈ [0, 1]` shared by all regions:

```
x_region_d(s) = Lsum[d-1] + s * L[d-1]      =>      dY/ds = L[d-1] * dY/dx
```

The boundary conditions (`bcfun`) reproduce those of the reference
code, including potential/flux continuity at the 4 internal interfaces
and the Dirichlet conditions at the gas channels. See the docstring at
the top of `mmm1d/model.py` for details.

## Performance and convergence

* The default sweep (1.15–1.00 V, 4 points) converges in a few
  seconds.
* Tested down to 0.40 V (current ~2.3 A/cm², already in the
  mass-transport-limited plateau region) — the full sweep from 1.15 to
  0.40 V takes about 2–3 minutes, since the adaptive mesh grows quite
  a lot at the lower voltages.
* If `solve_bvp` fails to converge (`sol.success == False`): increase
  `n_per_region`, increase `max_nodes`, reduce the voltage step between
  consecutive sweep points (smoother continuation), or relax `tol`.

## Issues identified during code review, and next steps

While reviewing the reference MATLAB implementation in detail before
extending it for this thesis, I identified a few points in the
original code that I plan to address as next steps:

1. **The evaporation/condensation source term (`S_ec`) is always
   zero.** In the original MATLAB, inside the CCL and CGDL cases, the
   line is
   `S_ec = zeros(size(s)); gamma_ec(x_H2O,x_sat,s,T).*C.*(x_H2O-x_sat);`
   — the second part of the line (after the `;`) is never assigned to
   anything, i.e., it is dead code. This turns off the
   evaporation/condensation coupling between vapor and liquid water in
   the default model. Reproduced faithfully in `mmm1d/model.py`
   (`S_ec = np.zeros_like(s)`).
   **Next step:** reactivate and validate this term as part of the
   two-phase water transport extensions.

2. **`pc_s.m` compares the whole vector instead of the loop element**
   (`if pc_in<min(pc)` instead of `if pc_in(i)<min(pc)`), which makes
   the saturation almost always fall into the `s = s_im` branch. In the
   default configuration this is partly masked because
   `s_im == s_C` (both 0.12). See the docstring in `mmm1d/pc_s.py`
   (there is a `strict=True` parameter for the element-wise
   comparison).
   **Next step:** decide whether to keep the
   original behavior, switch to `strict=True`, or rewrite the
   inversion altogether once `s_im` and `s_C` are no longer forced to
   be equal.

3. **Permeability used in `dP_gas`/`dP_liq` in the CCL.** In the CCL
   case, Darcy's law uses `kappa_GDL` instead of `kappa_CL` in the
   original code — reproduced as-is, with a note in `mmm1d/model.py`
   (function `_ccl`).
   **Next step:** confirm against the reference paper whether this is
   intentional, and correct to `kappa_CL` if not.

4. **The `"s [-]"` label in the potentials plot.** The panel labeled
   as saturation `s` actually plots the state variable `P_liq` [Pa]
   (saturation itself is a local algebraic variable, not a state
   variable of the BVP).
   **Next step:** fix the label (or add a derived saturation curve)
   when reworking the plotting code.

## Credits

Physical formulation and reference numerical implementation: Dr. Roman
Vetter, Prof. Dr. Jürgen O. Schumacher (v1); Dr. Robert Herrendörfer,
Prof. Dr. Jürgen O. Schumacher (v2) — Institute of Computational
Physics (ICP), ZHAW.

Extension, Python port and subsequent development:
**Victor Constantin Cartsounis**, **CEFET-MG**, 2026.
