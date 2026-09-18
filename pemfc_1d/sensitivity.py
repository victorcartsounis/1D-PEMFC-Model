"""Forward (direct) sensitivity analysis of the MEA model.

For a material parameter ``theta`` the solution ``Y(x)`` of the boundary value
problem moves as ``Z = dY/dtheta``, and ``Z`` obeys the variational equations

    Z' = (dF/dY)(x, Y; theta) * Z + dF/dtheta

obtained by differentiating ``Y' = F(x, Y; theta)`` with respect to ``theta``.
This module stacks those onto the model itself and hands the result to the same
``solve_bvp``: 80 original equations plus 80 sensitivity equations, on the mesh
the base solution converged on. ``dF/dY`` and ``dF/dtheta`` are differentiated
symbolically from the model's own right-hand side -- see :mod:`pemfc_1d.symbolic`
for how -- so they cannot drift away from the physics that is actually solved.

What ``theta`` is
-----------------
Every parameter here is a *dimensionless multiplier* on a material property,
equal to 1 at the nominal parameter set, rather than the property's own value.
There are three reasons:

* several of the properties are not single numbers. "Electrical conductivity"
  is ``sigma_e_GDL`` and ``sigma_e_CL``, two different values of one material
  property, and ``d/dsigma_e`` would have to say which; a multiplier scales the
  pair, as a change of electrode material would.
* two of them are not parameters at all but factors inside a constitutive
  relation -- the pore/tortuosity factor ``eps_p/tau^2`` shared by every gas
  diffusivity, and the diffusivity prefactors -- and have no field to perturb.
* ``dY/dtheta`` is then in the units of ``Y`` for every parameter, so the eight
  figures are directly comparable: each reads as "how far does the solution
  move if this property is 1% larger".

Cost
----
One augmented solve per parameter and per voltage. The collocation Jacobian of
a 160-equation system holds four times the non-zeros of the 80-equation one, so
the memory per mesh node is about four times ``metrics.GB_PER_1000_NODES``. The
sensitivity equations are linear in ``Z`` and start from a converged ``Y``,
though, so the solver rarely has to refine the mesh it is given.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import partial
from typing import Mapping, Sequence

import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp

from .model import DEFAULT_MAX_NODES, SweepResult, boundary_conditions, full_ode
from .params import ArrayLike, Params
from .saturation import saturation_from_capillary_pressure
from .state import N_TOTAL, Region, State, block_rows, stacked_index
from .symbolic import CompiledSensitivityTerms, compile_sensitivity_terms

__all__ = ["SENSITIVITY_PARAMETERS", "SensitivityParameter", "SensitivityResult",
           "SensitivitySettings", "ScaledParams", "scaled_params",
           "frozen_saturation", "solve_sensitivity", "sweep_sensitivities",
           "augmented_ode", "augmented_boundary_conditions", "GB_PER_1000_NODES"]

#: Peak memory ``solve_bvp`` needs per 1000 mesh nodes for the 160-equation
#: augmented system. The collocation Jacobian holds ``2 * 160**2`` non-zeros
#: per interval against ``2 * 80**2`` for the model alone, so this is four
#: times ``metrics.GB_PER_1000_NODES``.
GB_PER_1000_NODES = 3.4


# =============================================================================
# THE PARAMETERS UNDER STUDY
# =============================================================================

@dataclass(frozen=True)
class SensitivityParameter:
    """One material property, and how a multiplier is applied to it.

    A property is scaled either through the ``Params`` fields that hold it or
    through the constitutive method that computes it -- the pore/tortuosity
    factor and the diffusivity prefactors exist only inside a method.
    """

    name: str                       #: key, and the figure's filename stem
    symbol: str                     #: LaTeX symbol, for figure titles
    description: str                #: one line, for the run log
    scaled_fields: tuple[str, ...] = ()
    scaled_methods: tuple[str, ...] = ()


#: The eight material properties this module can differentiate with respect to.
#: Extending it needs a name and a place to hang the multiplier, nothing more:
#: the Jacobian and ``dF/dtheta`` follow from the model's own expressions.
SENSITIVITY_PARAMETERS: dict[str, SensitivityParameter] = {
    parameter.name: parameter for parameter in (
        SensitivityParameter(
            "sigma_e", r"\sigma_e", "electrical conductivity of GDL and CL",
            scaled_fields=("sigma_e_GDL", "sigma_e_CL")),
        SensitivityParameter(
            "sigma_p", r"\sigma_p", "protonic conductivity of the ionomer",
            scaled_methods=("sigma_p",)),
        SensitivityParameter(
            "k", r"k", "thermal conductivity of GDL, CL and membrane",
            scaled_fields=("k_GDL", "k_CL", "k_PEM")),
        SensitivityParameter(
            "D_lambda", r"D_\lambda", "diffusivity of dissolved water in the ionomer",
            scaled_methods=("D_lambda",)),
        SensitivityParameter(
            "eps_p_over_tau2", r"\varepsilon_p/\tau^2",
            "pore/tortuosity factor shared by every gas diffusivity",
            scaled_methods=("_diffusivity_scaling",)),
        SensitivityParameter(
            "D_H2", r"D_{H_2}",
            "hydrogen diffusivity, i.e. the anode H2/H2O binary coefficient",
            scaled_methods=("D_H2O_A",)),
        SensitivityParameter(
            "D_O2", r"D_{O_2}", "oxygen diffusivity in the cathode gas",
            scaled_methods=("D_O2",)),
        SensitivityParameter(
            "kappa", r"\kappa",
            "absolute hydraulic permeability, acting through p_c(s) in Darcy's law",
            scaled_fields=("kappa_GDL", "kappa_CL")),
    )
}


@dataclass(frozen=True, eq=False)
class ScaledParams(Params):
    """A parameter set with one material property multiplied by ``theta``.

    ``theta`` is a plain float for a finite-difference run and a sympy symbol
    for the symbolic trace; nothing here cares which, because the model only
    ever multiplies and adds it.

    Field scaling happens *after* ``Params.__post_init__``, which is safe
    because none of the derived quantities that method computes -- channel
    compositions, the capillary pressure at the cathode, layer positions --
    read any of the fields that are scaled here.

    Build one with :func:`scaled_params` rather than directly: the fields hold
    already-multiplied values, so constructing a ``ScaledParams`` from another
    one would apply the multiplier twice.
    """

    theta: ArrayLike = 1.0
    scaled_fields: tuple[str, ...] = ()
    scaled_methods: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in self.scaled_fields:
            object.__setattr__(self, name, self.theta * getattr(self, name))

    def _scaled(self, name: str, value: ArrayLike) -> ArrayLike:
        return self.theta * value if name in self.scaled_methods else value

    # Each override multiplies the inherited relation rather than restating it,
    # so the physics stays written once, in Params.
    def _diffusivity_scaling(self, *args):
        return self._scaled("_diffusivity_scaling", super()._diffusivity_scaling(*args))

    def D_O2(self, *args):
        return self._scaled("D_O2", super().D_O2(*args))

    def D_H2O_A(self, *args):
        return self._scaled("D_H2O_A", super().D_H2O_A(*args))

    def D_H2O_C(self, *args):
        return self._scaled("D_H2O_C", super().D_H2O_C(*args))

    def D_lambda(self, *args):
        return self._scaled("D_lambda", super().D_lambda(*args))

    def sigma_p(self, *args):
        return self._scaled("sigma_p", super().sigma_p(*args))


def scaled_params(base: Params, parameter: str | SensitivityParameter,
                  theta: ArrayLike) -> ScaledParams:
    """``base`` with the named property multiplied by ``theta``.

    ``theta = 1`` reproduces ``base`` exactly, which is what makes the same
    construction usable for the symbolic trace and for the finite-difference
    check that validates it.
    """
    if isinstance(base, ScaledParams):
        raise TypeError("scale a plain Params: scaling an already scaled "
                        "parameter set would apply both multipliers")
    parameter = (SENSITIVITY_PARAMETERS[parameter] if isinstance(parameter, str)
                 else parameter)
    inherited = {f.name: getattr(base, f.name) for f in fields(base) if f.init}
    return ScaledParams(**inherited, theta=theta,
                        scaled_fields=parameter.scaled_fields,
                        scaled_methods=frozenset(parameter.scaled_methods))


def frozen_saturation(result: SweepResult) -> dict[Region, float]:
    """The liquid saturation the lookup table returns in each layer of ``result``.

    The symbolic trace needs a number here rather than the table itself (see
    :mod:`pemfc_1d.symbolic`), and the number is not always ``s_im``: the
    lookup picks its branch on whether the whole capillary-pressure vector
    falls below the tabulated range, and the solution sits on that threshold to
    within rounding, so the two cathode layers can land on either side of it.

    A layer whose saturation is *not* constant over the sweep would mean the
    inversion has genuinely been reached and the sensitivity equations are
    missing a ``ds/dp_c`` term, so that is an error rather than an average.
    """
    saturation = {region: float(result.params.s_im) for region in Region}
    for region in (Region.CCL, Region.CGDL):
        values = set()
        for solution in result.solutions:
            block = solution.y[block_rows(region)]
            values.update(np.unique(saturation_from_capillary_pressure(
                block[State.P_LIQ] - block[State.P_GAS], result.params.s_im)))
        if len(values) != 1:
            raise ValueError(
                f"liquid saturation is no longer constant in the {region.name}: "
                f"the solved sweep spans {sorted(values)}. The sensitivity "
                "equations assume the saturation lookup is flat, which is only "
                "true while the two-phase behaviour is inactive.")
        saturation[region] = values.pop()
    return saturation


# =============================================================================
# THE AUGMENTED SYSTEM
# =============================================================================

def augmented_ode(s: np.ndarray, W: np.ndarray, params: Params,
                  terms: CompiledSensitivityTerms) -> np.ndarray:
    """``d/ds`` of the 160-equation system: the model, then its sensitivity.

    The first 80 rows are evaluated by ``full_ode`` itself, not by the traced
    copy of it, so the model half of an augmented solve is the same arithmetic
    as a plain solve.
    """
    Y, Z = W[:N_TOTAL], W[N_TOTAL:]
    jacobian, dF_dtheta = terms(Y)
    return np.vstack((full_ode(s, Y, params),
                      np.einsum("ijk,jk->ik", jacobian, Z) + dF_dtheta))


def augmented_boundary_conditions(W_left: np.ndarray, W_right: np.ndarray,
                                  U_cell: float, params: Params) -> np.ndarray:
    """The model's own residuals, followed by their derivative with respect to ``theta``.

    Differentiating ``bc(Y_left, Y_right) = 0`` gives

        dbc/dY_left * Z_left + dbc/dY_right * Z_right + dbc/dtheta = 0

    Every residual in :func:`pemfc_1d.model.boundary_conditions` is affine in
    the state -- an interface difference, or an edge value against a fixed
    channel condition -- so its derivative is the same function applied to
    ``Z`` with the constants removed, which is what subtracting ``bc(0, 0)``
    does. None of the constants involved is one of the scaled material
    properties, so ``dbc/dtheta`` is zero; both facts are pinned by
    ``tests/test_sensitivity.py`` rather than assumed.
    """
    zero = np.zeros(N_TOTAL)
    constants = boundary_conditions(zero, zero, U_cell, params)
    return np.concatenate((
        boundary_conditions(W_left[:N_TOTAL], W_right[:N_TOTAL], U_cell, params),
        boundary_conditions(W_left[N_TOTAL:], W_right[N_TOTAL:], U_cell, params)
        - constants,
    ))


# =============================================================================
# SOLVING
# =============================================================================

@dataclass(frozen=True)
class SensitivitySettings:
    """Whether to run sensitivity analysis, and how.

    ``enabled`` is the switch: with it off nothing in this module is imported
    or executed by a run, and the sweep costs exactly what it did before.
    """

    enabled: bool = True
    #: Which of ``SENSITIVITY_PARAMETERS`` to analyse; all eight by default.
    parameters: tuple[str, ...] = tuple(SENSITIVITY_PARAMETERS)
    tol: float = 1e-4
    #: Ceiling on mesh refinement for the augmented solve. The base mesh is
    #: reused as-is, so this only bounds any further refinement; see
    #: ``GB_PER_1000_NODES``, which is four times the model's own figure.
    max_nodes: int = DEFAULT_MAX_NODES
    verbose: int = 0

    def resolved_parameters(self) -> tuple[SensitivityParameter, ...]:
        unknown = [name for name in self.parameters
                   if name not in SENSITIVITY_PARAMETERS]
        if unknown:
            raise KeyError(f"unknown sensitivity parameter(s): {unknown}; "
                           f"known: {sorted(SENSITIVITY_PARAMETERS)}")
        return tuple(SENSITIVITY_PARAMETERS[name] for name in self.parameters)


@dataclass(frozen=True, eq=False)
class SensitivityResult:
    """``dY/dtheta`` across the MEA, for one parameter over a voltage sweep."""

    parameter: SensitivityParameter
    voltages: np.ndarray
    solutions: list          #: augmented solve_bvp solution per voltage
    params: Params           #: the nominal parameter set, theta = 1
    n_terms: int             #: non-zero derivative expressions behind it

    @property
    def n_voltages(self) -> int:
        return len(self.voltages)

    @property
    def converged(self) -> bool:
        return all(solution.success for solution in self.solutions)

    @property
    def current_sensitivities(self) -> np.ndarray:
        """[A/cm^2] ``dI/dtheta`` at each voltage: the headline number.

        The cell current is the electron flux leaving the cathode GDL, so its
        sensitivity is the matching row of ``Z`` at the same edge.
        """
        row = N_TOTAL + stacked_index(State.J_E, Region.CGDL)
        return np.array([solution.y[row, -1] / 1e4 for solution in self.solutions])


def solve_sensitivity(result: SweepResult, parameter: str | SensitivityParameter,
                      settings: SensitivitySettings | None = None
                      ) -> SensitivityResult:
    """Solve the augmented system for one parameter, over a solved sweep.

    ``result`` supplies both the mesh and the initial guess: the model half
    starts from the converged solution and the sensitivity half from zero, so
    each voltage begins at a point where 80 of the 160 equations are already
    satisfied.
    """
    settings = settings if settings is not None else SensitivitySettings()
    parameter = (SENSITIVITY_PARAMETERS[parameter] if isinstance(parameter, str)
                 else parameter)
    params = result.params

    theta = sp.Symbol("theta", real=True, positive=True)
    terms = compile_sensitivity_terms(scaled_params(params, parameter, theta),
                                      theta, parameter.name,
                                      saturation=frozen_saturation(result))
    ode = partial(augmented_ode, params=params, terms=terms)

    solutions = []
    for U_cell, base in zip(result.voltages, result.solutions):
        guess = np.vstack((base.y, np.zeros_like(base.y)))
        bc = partial(augmented_boundary_conditions, U_cell=U_cell, params=params)
        solutions.append(solve_bvp(
            ode, bc, base.x, guess, tol=settings.tol,
            max_nodes=max(settings.max_nodes, base.x.size),
            verbose=settings.verbose))

    return SensitivityResult(parameter=parameter, voltages=result.voltages,
                             solutions=solutions, params=params,
                             n_terms=terms.n_terms)


def sweep_sensitivities(result: SweepResult,
                        settings: SensitivitySettings | None = None
                        ) -> list[SensitivityResult]:
    """One :func:`solve_sensitivity` per parameter named in ``settings``."""
    settings = settings if settings is not None else SensitivitySettings()
    return [solve_sensitivity(result, parameter, settings)
            for parameter in settings.resolved_parameters()]
