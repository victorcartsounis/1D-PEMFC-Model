"""Symbolic differentiation of the model's *own* right-hand side.

Forward sensitivity analysis needs the Jacobian ``dF/dY`` of the stacked
system and the partial derivative ``dF/dtheta`` with respect to a material
parameter. Both are obtained here by differentiating the expressions that
:func:`pemfc_1d.model.full_ode` already evaluates, rather than by writing the
physics out a second time -- a hand-derived Jacobian would be a second copy of
the model, free to drift away from the one that actually runs.

How the expressions are obtained
--------------------------------
:func:`trace_right_hand_side` calls ``full_ode`` itself, with two
substitutions in place:

* the ``numpy`` name inside :mod:`pemfc_1d.model` and :mod:`pemfc_1d.params`
  is swapped for :class:`_SymbolicArray`, a shim that forwards everything to
  numpy unless one of its arguments is a sympy expression, in which case it
  builds the sympy counterpart (``exp``, ``log``, ``sqrt``) or a
  ``Piecewise`` (``where``, ``maximum``);
* the state vector handed in is 80 sympy symbols instead of 80 numbers.

Python's arithmetic operators need no help: ``-j_e / params.sigma_e_GDL``
builds a sympy expression as readily as it multiplies two floats. What comes
back is therefore the same arithmetic the solver performs, recorded instead of
evaluated.

The one frozen term
-------------------
``saturation_from_capillary_pressure`` is a table lookup, not an expression,
and cannot be traced. It does not need to be: in this version of the model it
is constant in each layer for every capillary pressure the solution reaches
(see :mod:`pemfc_1d.saturation`), so the trace substitutes that constant, whose
derivative is zero -- which is also the derivative of the lookup.

The constant is taken per layer, and is not simply ``s_im``. The lookup selects
its branch on whether the *whole* capillary-pressure vector falls below the
tabulated range, and the solution sits on that threshold to within rounding:
the cathode catalyst layer comes out half a table step above the immobile
saturation and the cathode GDL exactly at it. Half a table step is a 0.4%
difference in ``(1-s)**3``, which is large enough to show up against a
finite-difference check, so :func:`trace_right_hand_side` takes the value the
running model actually returns in each layer rather than assuming ``s_im``.
:func:`pemfc_1d.sensitivity.frozen_saturation` reads those values off a solved
sweep, and ``tests/test_sensitivity.py`` pins both the constancy and the values,
so the day the inversion is genuinely reached -- when the sensitivity equations
would need a real ``ds/dp_c`` term -- the tests say so.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import sympy as sp

from . import model as _model
from . import params as _params
from .params import Params
from .state import N_STATE, N_TOTAL, Region

__all__ = ["STATE_SYMBOLS", "CompiledSensitivityTerms",
           "trace_right_hand_side", "compile_sensitivity_terms", "symbolic_numpy"]

#: The 80 symbols standing in for the stacked state vector, in its own order,
#: so that ``STATE_SYMBOLS[stacked_index(State.J_E, Region.CGDL)]`` names the
#: cell current exactly as the numeric code indexes it.
STATE_SYMBOLS: tuple[sp.Symbol, ...] = tuple(sp.symbols(f"Y0:{N_TOTAL}", real=True))


# =============================================================================
# THE NUMPY SHIM
# =============================================================================

def _is_symbolic(*values: object) -> bool:
    return any(isinstance(value, sp.Basic)
               or (isinstance(value, np.ndarray) and value.dtype == object)
               for value in values)


def _elementwise(symbolic: Callable, numeric: Callable, value: object) -> object:
    """Apply ``symbolic`` to sympy input and ``numeric`` to everything else."""
    if isinstance(value, np.ndarray) and value.dtype == object:
        return np.array([symbolic(sp.sympify(item)) for item in value.ravel()],
                        dtype=object).reshape(value.shape)
    if isinstance(value, sp.Basic):
        return symbolic(value)
    return numeric(value)


class _SymbolicArray:
    """A stand-in for the ``numpy`` module that also understands sympy.

    Anything not named below is forwarded to numpy untouched, so constants
    (``np.pi``) and the array plumbing the parameter set uses at construction
    time (``np.array``, ``np.cumsum``) keep working while a trace is running.
    """

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)

    def exp(self, x): return _elementwise(sp.exp, np.exp, x)

    def log(self, x): return _elementwise(sp.log, np.log, x)

    def sqrt(self, x): return _elementwise(sp.sqrt, np.sqrt, x)

    def where(self, condition, if_true, if_false):
        """``np.where`` as a two-branch ``Piecewise``.

        The model uses it for genuine switches -- the sorption rate ``k_ad``
        flips where ``lam`` crosses ``lambda_eq`` -- so the symbolic form has
        to keep both branches and the condition that selects them.
        """
        if not _is_symbolic(condition, if_true, if_false):
            return np.where(condition, if_true, if_false)
        return sp.Piecewise((sp.sympify(if_true), condition),
                            (sp.sympify(if_false), True))

    def maximum(self, a, b):
        """``np.maximum`` as ``sympy.Max``, deliberately not as a ``Piecewise``.

        The model clamps with it -- ``sigma_p`` raises ``max(0, ...)`` to the
        power 3/2 -- and a ``Piecewise`` distributes a power into its branches,
        so the clamped branch becomes ``0`` and differentiating a term with
        ``sigma_p`` in a denominator yields ``zoo`` from a branch that is never
        taken. ``Max`` stays intact under ``Pow`` and differentiates into a
        ``Heaviside``, which ``lambdify`` renders as ``numpy.select``.
        """
        if not _is_symbolic(a, b):
            return np.maximum(a, b)
        return sp.Max(sp.sympify(a), sp.sympify(b))

    def zeros_like(self, x):
        if isinstance(x, np.ndarray):
            return np.full(x.shape, sp.Integer(0), dtype=object)
        return sp.Integer(0)

    #: ``full_ode`` writes every row it allocates, but an uninitialised object
    #: array would hold ``None`` and fail loudly rather than symbolically.
    empty_like = zeros_like


#: Which layer each state symbol belongs to, so that a frozen saturation can be
#: looked up from the expression it was asked about.
_REGION_OF_SYMBOL: dict[sp.Symbol, Region] = {
    symbol: Region(index // N_STATE) for index, symbol in enumerate(STATE_SYMBOLS)
}


def _region_of(expression: object) -> Region:
    """The layer whose state ``expression`` is built from.

    The capillary pressure handed to the saturation lookup is
    ``P_liq - P_gas`` of one layer, so its free symbols identify that layer;
    this is how the trace tells the cathode catalyst layer's call apart from
    the cathode GDL's when both reach the same module-level function.
    """
    regions = {_REGION_OF_SYMBOL[symbol]
               for symbol in sp.sympify(expression).free_symbols
               if symbol in _REGION_OF_SYMBOL}
    if len(regions) != 1:
        raise ValueError("cannot attribute the capillary pressure to a single "
                         f"layer; it involves {sorted(r.name for r in regions)}")
    return regions.pop()


@contextlib.contextmanager
def symbolic_numpy(saturation: Mapping[Region, float]):
    """Run the model's own code against sympy expressions.

    Rebinding the module-global ``np`` is what makes the trace possible
    without a parallel implementation of the physics. It is confined to this
    context manager and restored on the way out, including on exception, so no
    numeric caller can ever observe the shim.

    ``saturation`` supplies the frozen liquid saturation of each layer, which
    replaces the table lookup for the reasons in the module docstring.
    """
    shim = _SymbolicArray()

    def frozen_saturation(pc: object, s_im: float, strict: bool = False) -> float:
        return saturation[_region_of(pc)]

    saved_numpy = {module: module.np for module in (_model, _params)}
    saved_saturation = _model.saturation_from_capillary_pressure
    try:
        for module in saved_numpy:
            module.np = shim
        _model.saturation_from_capillary_pressure = frozen_saturation
        yield shim
    finally:
        for module, original in saved_numpy.items():
            module.np = original
        _model.saturation_from_capillary_pressure = saved_saturation


# =============================================================================
# TRACING AND DIFFERENTIATION
# =============================================================================

def trace_right_hand_side(params: Params,
                          saturation: Mapping[Region, float] | None = None
                          ) -> list[sp.Expr]:
    """The 80 expressions of ``dY/ds``, as :func:`pemfc_1d.model.full_ode` forms them.

    ``params`` may carry sympy expressions in place of numbers -- that is how a
    material parameter is made differentiable (see
    :class:`pemfc_1d.sensitivity.ScaledParams`); any it carries appear as free
    symbols in the result alongside :data:`STATE_SYMBOLS`.

    ``saturation`` gives the frozen liquid saturation per layer and defaults to
    ``s_im`` everywhere. Pass the values the solved model actually uses --
    :func:`pemfc_1d.sensitivity.frozen_saturation` reads them off a sweep -- or
    the traced diffusivities are a fraction of a percent off the ones the
    solver evaluates.
    """
    if saturation is None:
        saturation = {region: params.s_im for region in Region}
    state = np.array(STATE_SYMBOLS, dtype=object)
    with symbolic_numpy(saturation):
        derivatives = _model.full_ode(sp.Integer(0), state, params)
    return [sp.sympify(expression) for expression in derivatives]


def _sparse_jacobian(expressions: Sequence[sp.Expr]
                     ) -> tuple[list[tuple[int, int]], list[sp.Expr]]:
    """Non-zero entries of ``d(expressions)/d(STATE_SYMBOLS)``.

    Only the entries whose symbol actually occurs in the row are differentiated.
    The Jacobian of this system is block-diagonal by layer and sparse within a
    block -- some 5% of its 6400 entries are non-zero -- so testing membership
    first is what keeps both the differentiation and the compiled code small.
    """
    indices: list[tuple[int, int]] = []
    entries: list[sp.Expr] = []
    for row, expression in enumerate(expressions):
        present = expression.free_symbols
        for column, symbol in enumerate(STATE_SYMBOLS):
            if symbol not in present:
                continue
            derivative = sp.diff(expression, symbol)
            if derivative == 0:
                continue
            indices.append((row, column))
            entries.append(derivative)
    return indices, entries


@dataclass(frozen=True)
class CompiledSensitivityTerms:
    """``dF/dY`` and ``dF/dtheta`` as numpy-evaluable functions of the state.

    Both are produced by one compiled callable, so the common subexpressions
    they share -- the reaction rate, the gas density, the sorption rate -- are
    computed once per mesh point rather than once per entry.
    """

    parameter: str
    jacobian_indices: tuple[tuple[int, int], ...]
    _evaluate: Callable
    _n_jacobian: int
    #: expression count, reported by the run log as a size of the derived system
    n_terms: int

    def __call__(self, Y: np.ndarray, theta: float = 1.0
                 ) -> tuple[np.ndarray, np.ndarray]:
        """``(dF/dY, dF/dtheta)`` on a mesh: shapes ``(80, 80, m)`` and ``(80, m)``.

        ``Y`` is the stacked state ``(80, m)``. ``theta`` is passed as a full
        column rather than a scalar because a compiled ``Piecewise`` becomes
        ``numpy.select``, which needs its conditions to be arrays.
        """
        Y = np.atleast_2d(Y)
        columns = Y.shape[1]
        values = self._evaluate(*Y, np.full(columns, float(theta)))

        jacobian = np.zeros((N_TOTAL, N_TOTAL, columns))
        for (row, column), value in zip(self.jacobian_indices,
                                        values[:self._n_jacobian]):
            jacobian[row, column] = value          # broadcasts constant entries
        dF_dtheta = np.zeros((N_TOTAL, columns))
        for row, value in enumerate(values[self._n_jacobian:]):
            dF_dtheta[row] = value
        return jacobian, dF_dtheta


def compile_sensitivity_terms(params: Params, theta: sp.Symbol, parameter: str,
                              saturation: Mapping[Region, float] | None = None
                              ) -> CompiledSensitivityTerms:
    """Differentiate the traced right-hand side and compile the result.

    ``params`` must already carry ``theta`` in place of the material property
    under study, so that tracing produces expressions that depend on it.
    ``saturation`` is passed straight through to :func:`trace_right_hand_side`.
    """
    expressions = trace_right_hand_side(params, saturation)
    indices, jacobian_entries = _sparse_jacobian(expressions)
    theta_entries = [sp.diff(expression, theta) for expression in expressions]

    arguments = (*STATE_SYMBOLS, theta)
    evaluate = sp.lambdify(arguments, [*jacobian_entries, *theta_entries],
                           modules="numpy", cse=True)
    return CompiledSensitivityTerms(
        parameter=parameter,
        jacobian_indices=tuple(indices),
        _evaluate=evaluate,
        _n_jacobian=len(jacobian_entries),
        n_terms=len(jacobian_entries) + sum(1 for e in theta_entries if e != 0),
    )
