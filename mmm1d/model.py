"""1D PEM fuel cell model: steady-state, non-isothermal, two-phase,
macro-homogeneous membrane electrode assembly.

Eight coupled quantities are resolved across the five MEA layers (anode GDL,
anode catalyst layer, membrane, cathode catalyst layer, cathode GDL): electron
and proton potentials, temperature, dissolved water content in the ionomer,
water vapour and oxygen mass fractions, and liquid and gas pressures. Each obeys
a second-order transport equation, written here as a potential/flux pair, giving
16 first-order equations per layer.

Physics and constitutive relations follow:

    R. Vetter and J. O. Schumacher, "Free open reference implementation of a
    two-phase PEM fuel cell model", Computer Physics Communications 234 (2019)
    223-234. https://doi.org/10.1016/j.cpc.2018.07.023

Solver design
-------------
Every layer carries its own equation set, coupled to its neighbours by
continuity conditions at the four internal interfaces. ``scipy.integrate.solve_bvp``
needs a strictly increasing mesh and has no notion of separate regions, so the
five layers are *stacked* into a single system of ``N_REGIONS * N_STATE = 80``
first-order ODEs over one normalised coordinate ``s`` in [0, 1], with a copy of
``s`` per layer. The chain rule maps it back onto physical position:

    x_region(s) = Lsum[region] + s*L[region]   =>   dY/ds = L[region] * dY/dx

Interface continuity and gas-channel conditions are then imposed as boundary
conditions on the stacked system (see :func:`boundary_conditions`), which is
equivalent to solving the five layers as a coupled multi-region problem.

Current simplifications
-----------------------
Two modelling choices are active in this version and are flagged where they
occur, because they shape how the results should be read:

* The evaporation/condensation source ``S_ec`` is set to zero, so phase change
  between water vapour and liquid water is disabled.
* Darcy's law in the cathode catalyst layer uses the GDL permeability
  ``kappa_GDL`` rather than ``kappa_CL`` (see :func:`_ccl`).

Together with the saturation floor described in :mod:`mmm1d.saturation`, these
leave the two-phase behaviour inactive: liquid water neither forms nor moves.
Re-enabling it is planned work; the regression tests pin the present behaviour
so that change cannot happen unnoticed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_bvp

from .params import Params
from .saturation import saturation_from_capillary_pressure
from .state import (ACTIVE_REGIONS, N_QUANTITIES, N_REGIONS, N_STATE, N_TOTAL,
                    Quantity, Region, State, block_rows, stacked_index)

__all__ = ["solve", "SweepResult", "full_ode", "boundary_conditions",
           "build_initial_guess", "Region", "State", "Quantity"]


# =============================================================================
# PER-REGION PHYSICS
#
# Each function takes one (N_STATE, m) block and returns its d/dx. Derivatives
# are written into a zero-filled array by name, so a quantity can never land in
# the wrong row; rows never written stay zero, which is how a layer declares a
# quantity inactive.
# =============================================================================

RegionODE = Callable[[np.ndarray, Params], np.ndarray]


def _agdl(block: np.ndarray, params: Params) -> np.ndarray:
    """Anode gas diffusion layer: electron, heat, vapour and gas transport."""
    j_e = block[State.J_E]
    T, j_T = block[State.T], block[State.J_T]
    w_H2O, j_H2O = block[State.W_H2O], block[State.J_H2O]
    P_gas, rho_u_gas = block[State.P_GAS], block[State.RHO_U_GAS]

    w_H2 = 1 - w_H2O
    Mn = 1.0 / (w_H2O / params.M_H2O + w_H2 / params.M_H2)
    C = P_gas / (params.R * T)
    rho_gas = Mn * C
    u_gas = rho_u_gas / rho_gas
    s = np.zeros_like(T)

    d_phi_e = -j_e / params.sigma_e_GDL

    d = np.zeros_like(block)
    d[State.P_GAS] = -u_gas / (params.kappa_GDL / params.mu_gas)
    d[State.PHI_E] = d_phi_e
    d[State.T] = -j_T / params.k_GDL
    d[State.W_H2O] = -(j_H2O - rho_gas * w_H2O * u_gas) / (
        rho_gas * params.D_H2O_A(params.eps_p_GDL, params.tau_GDL, s, T, P_gas))
    d[State.J_T] = -j_e * d_phi_e
    return d


def _acl(block: np.ndarray, params: Params) -> np.ndarray:
    """Anode catalyst layer: hydrogen oxidation plus sorption into the ionomer."""
    phi_e, j_e = block[State.PHI_E], block[State.J_E]
    phi_p, j_p = block[State.PHI_P], block[State.J_P]
    T, j_T = block[State.T], block[State.J_T]
    lam, j_lam = block[State.LAMBDA], block[State.J_LAMBDA]
    w_H2O, j_H2O = block[State.W_H2O], block[State.J_H2O]
    P_gas, rho_u_gas = block[State.P_GAS], block[State.RHO_U_GAS]

    w_H2 = 1 - w_H2O
    Mn = 1.0 / (w_H2O / params.M_H2O + w_H2 / params.M_H2)
    x_H2O = w_H2O / params.M_H2O * Mn
    x_H2 = 1 - x_H2O
    C = P_gas / (params.R * T)
    rho_gas = Mn * C
    x_sat = params.P_sat(T) / P_gas
    lambda_eq = params.sorption(x_H2O / x_sat)

    L_ACL = params.L[Region.ACL]
    S_ad = params.k_ad(lam, lambda_eq, T) / (L_ACL * params.V_m) * (lambda_eq - lam)
    P_H2 = x_H2 * P_gas
    eta = (phi_e - phi_p + T * params.DeltaS_HOR / (2 * params.F)
           + params.R * T / (2 * params.F) * np.log(P_H2 / params.P_ref))
    i = params.butler_volmer(params.i_0_HOR(T), params.a_ACL, T, params.beta_HOR, eta)
    S_F = i / (2 * params.F)

    u_gas = rho_u_gas / rho_gas
    s = np.zeros_like(T)
    d_phi_e = -j_e / params.sigma_e_CL
    d_phi_p = -j_p / params.sigma_p(params.eps_i_CL, lam, T)

    d = np.zeros_like(block)
    d[State.P_GAS] = -u_gas / (params.kappa_CL / params.mu_gas)
    d[State.PHI_E] = d_phi_e
    d[State.PHI_P] = d_phi_p
    d[State.T] = -j_T / params.k_CL
    d[State.LAMBDA] = ((-j_lam + params.electro_osmotic_drag(lam) / params.F * j_p)
                       * params.V_m / params.D_lambda(params.eps_i_CL, lam, T))
    d[State.W_H2O] = -(j_H2O - rho_gas * w_H2O * u_gas) / (
        rho_gas * params.D_H2O_A(params.eps_p_CL, params.tau_CL, s, T, P_gas))
    d[State.J_E] = -i
    d[State.J_P] = i
    d[State.J_T] = (-j_e * d_phi_e - j_p * d_phi_p + i * eta
                    - S_F * T * params.DeltaS_HOR + params.H_ad * S_ad)
    d[State.J_LAMBDA] = S_ad
    d[State.J_H2O] = -S_ad * params.M_H2O
    d[State.RHO_U_GAS] = -params.M_H2O * S_ad - params.M_H2 * S_F
    return d


def _pem(block: np.ndarray, params: Params) -> np.ndarray:
    """Membrane: proton, heat and dissolved water transport only."""
    j_p = block[State.J_P]
    T, j_T = block[State.T], block[State.J_T]
    lam, j_lam = block[State.LAMBDA], block[State.J_LAMBDA]

    d_phi_p = -j_p / params.sigma_p(1.0, lam, T)

    d = np.zeros_like(block)
    d[State.PHI_P] = d_phi_p
    d[State.T] = -j_T / params.k_PEM
    d[State.LAMBDA] = ((-j_lam + params.electro_osmotic_drag(lam) / params.F * j_p)
                       * params.V_m / params.D_lambda(1.0, lam, T))
    d[State.J_T] = -j_p * d_phi_p
    return d


def _ccl(block: np.ndarray, params: Params) -> np.ndarray:
    """Cathode catalyst layer: oxygen reduction, sorption and two-phase flow.

    Darcy's law below uses ``kappa_GDL`` rather than ``kappa_CL``; see the
    "Current simplifications" note in this module's docstring.
    """
    phi_e, j_e = block[State.PHI_E], block[State.J_E]
    phi_p, j_p = block[State.PHI_P], block[State.J_P]
    T, j_T = block[State.T], block[State.J_T]
    lam, j_lam = block[State.LAMBDA], block[State.J_LAMBDA]
    w_H2O, j_H2O = block[State.W_H2O], block[State.J_H2O]
    w_O2, j_O2 = block[State.W_O2], block[State.J_O2]
    P_liq, rho_u_liq = block[State.P_LIQ], block[State.RHO_U_LIQ]
    P_gas, rho_u_gas = block[State.P_GAS], block[State.RHO_U_GAS]

    w_N2 = 1 - w_H2O - w_O2
    Mn = 1.0 / (w_H2O / params.M_H2O + w_O2 / params.M_O2 + w_N2 / params.M_N2)
    x_H2O = w_H2O / params.M_H2O * Mn
    x_O2 = w_O2 / params.M_O2 * Mn
    C = P_gas / (params.R * T)
    rho_gas = Mn * C
    s = saturation_from_capillary_pressure(P_liq - P_gas, params.s_im)
    x_sat = params.P_sat(T) / P_gas
    S_ec = np.zeros_like(s)  # phase change is inactive; see module docstring
    lambda_eq = params.sorption(x_H2O / x_sat)

    L_CCL = params.L[Region.CCL]
    S_ad = params.k_ad(lam, lambda_eq, T) / (L_CCL * params.V_m) * (lambda_eq - lam)
    P_O2 = x_O2 * P_gas
    eta = (-(params.DeltaH - T * params.DeltaS_ORR) / (2 * params.F)
           + params.R * T / (4 * params.F) * np.log(P_O2 / params.P_ref)
           - (phi_e - phi_p))
    i = params.butler_volmer(params.i_0_ORR(T, P_O2), params.a_CCL, T,
                             params.beta_ORR, eta)
    S_F = i / (2 * params.F)

    u_gas = rho_u_gas / rho_gas
    u_liq = rho_u_liq / params.rho_liq
    d_phi_e = -j_e / params.sigma_e_CL
    d_phi_p = -j_p / params.sigma_p(params.eps_i_CL, lam, T)

    d = np.zeros_like(block)
    d[State.P_GAS] = -u_gas / (params.kappa_GDL / params.mu_gas)
    d[State.P_LIQ] = -u_liq / (params.kappa_GDL * params.kappa_rel_liq(s)
                               / params.mu_liq(T))
    d[State.PHI_E] = d_phi_e
    d[State.PHI_P] = d_phi_p
    d[State.T] = -j_T / params.k_CL
    d[State.LAMBDA] = ((-j_lam + params.electro_osmotic_drag(lam) / params.F * j_p)
                       * params.V_m / params.D_lambda(params.eps_i_CL, lam, T))
    d[State.W_H2O] = -(j_H2O - rho_gas * w_H2O * u_gas) / (
        rho_gas * params.D_H2O_C(params.eps_p_CL, params.tau_CL, s, T, P_gas))
    d[State.W_O2] = -(j_O2 - rho_gas * w_O2 * u_gas) / (
        rho_gas * params.D_O2(params.eps_p_CL, params.tau_CL, s, T, P_gas))
    d[State.J_E] = i
    d[State.J_P] = -i
    d[State.J_T] = (-j_e * d_phi_e - j_p * d_phi_p + i * eta
                    - S_F * T * params.DeltaS_ORR + params.H_ad * S_ad
                    + params.H_ec * S_ec)
    d[State.J_LAMBDA] = S_F + S_ad
    d[State.J_H2O] = -params.M_H2O * (S_ec + S_ad)
    d[State.J_O2] = -params.M_O2 * S_F / 2
    d[State.RHO_U_LIQ] = params.M_H2O * S_ec
    d[State.RHO_U_GAS] = -params.M_H2O * (S_ec + S_ad) - params.M_O2 * S_F / 2
    return d


def _cgdl(block: np.ndarray, params: Params) -> np.ndarray:
    """Cathode gas diffusion layer: electron, heat and two-phase transport."""
    j_e = block[State.J_E]
    T, j_T = block[State.T], block[State.J_T]
    w_H2O, j_H2O = block[State.W_H2O], block[State.J_H2O]
    w_O2, j_O2 = block[State.W_O2], block[State.J_O2]
    P_liq, rho_u_liq = block[State.P_LIQ], block[State.RHO_U_LIQ]
    P_gas, rho_u_gas = block[State.P_GAS], block[State.RHO_U_GAS]

    w_N2 = 1 - w_H2O - w_O2
    Mn = 1.0 / (w_H2O / params.M_H2O + w_O2 / params.M_O2 + w_N2 / params.M_N2)
    C = P_gas / (params.R * T)
    rho_gas = Mn * C

    u_gas = rho_u_gas / rho_gas
    u_liq = rho_u_liq / params.rho_liq
    s = saturation_from_capillary_pressure(P_liq - P_gas, params.s_im)
    S_ec = np.zeros_like(s)  # phase change is inactive; see module docstring

    d_phi_e = -j_e / params.sigma_e_GDL

    d = np.zeros_like(block)
    d[State.P_GAS] = -u_gas / (params.kappa_GDL / params.mu_gas)
    d[State.P_LIQ] = -u_liq / (params.kappa_GDL * params.kappa_rel_liq(s)
                               / params.mu_liq(T))
    d[State.PHI_E] = d_phi_e
    d[State.T] = -j_T / params.k_GDL
    d[State.W_H2O] = -(j_H2O - rho_gas * w_H2O * u_gas) / (
        rho_gas * params.D_H2O_C(params.eps_p_GDL, params.tau_GDL, s, T, P_gas))
    d[State.W_O2] = -(j_O2 - rho_gas * w_O2 * u_gas) / (
        rho_gas * params.D_O2(params.eps_p_GDL, params.tau_GDL, s, T, P_gas))
    d[State.J_T] = -j_e * d_phi_e + params.H_ec * S_ec
    d[State.J_H2O] = -params.M_H2O * S_ec
    d[State.RHO_U_LIQ] = params.M_H2O * S_ec
    d[State.RHO_U_GAS] = -params.M_H2O * S_ec
    return d


REGION_ODES: dict[Region, RegionODE] = {
    Region.AGDL: _agdl,
    Region.ACL: _acl,
    Region.PEM: _pem,
    Region.CCL: _ccl,
    Region.CGDL: _cgdl,
}


def full_ode(s: np.ndarray, Y: np.ndarray, params: Params) -> np.ndarray:
    """dY/ds of the stacked 80-equation system. ``Y`` is (80, m)."""
    dYds = np.empty_like(Y)
    for region in Region:
        rows = block_rows(region)
        dYds[rows] = params.L[region] * REGION_ODES[region](Y[rows], params)
    return dYds


# =============================================================================
# BOUNDARY AND INTERFACE CONDITIONS
# =============================================================================

def boundary_conditions(Y_left: np.ndarray, Y_right: np.ndarray,
                        U_cell: float, params: Params) -> np.ndarray:
    """Residuals that solve_bvp drives to zero.

    ``Y_left`` and ``Y_right`` hold the state at s=0 and s=1 of every region.
    All three vectors are viewed as a (region, state) grid, so
    ``left[Region.CCL, State.W_O2]`` reads as "the oxygen mass fraction at the
    left edge of the cathode catalyst layer". Every (region, state) pair owns
    exactly one residual slot, holding the condition that closes it.
    """
    left = Y_left.reshape(N_REGIONS, N_STATE)
    right = Y_right.reshape(N_REGIONS, N_STATE)
    # slots not assigned below keep a homogeneous condition
    residuals = Y_left.copy().reshape(N_REGIONS, N_STATE)

    AGDL, ACL, PEM, CCL, CGDL = Region

    # ELECTRONS -- grounded at the anode plate, driven to U_cell at the cathode
    residuals[AGDL, State.PHI_E] = left[AGDL, State.PHI_E]
    residuals[AGDL, State.J_E] = left[ACL, State.J_E] - right[AGDL, State.J_E]
    residuals[ACL, State.PHI_E] = left[ACL, State.PHI_E] - right[AGDL, State.PHI_E]
    residuals[ACL, State.J_E] = right[ACL, State.J_E]
    residuals[CCL, State.PHI_E] = left[CGDL, State.PHI_E] - right[CCL, State.PHI_E]
    residuals[CCL, State.J_E] = left[CCL, State.J_E]
    residuals[CGDL, State.PHI_E] = right[CGDL, State.PHI_E] - U_cell
    residuals[CGDL, State.J_E] = left[CGDL, State.J_E] - right[CCL, State.J_E]

    # PROTONS -- confined to the ionomer, no flux at either catalyst boundary
    residuals[ACL, State.PHI_P] = left[CCL, State.J_P] - right[PEM, State.J_P]
    residuals[ACL, State.J_P] = left[ACL, State.J_P]
    residuals[PEM, State.PHI_P] = left[PEM, State.PHI_P] - right[ACL, State.PHI_P]
    residuals[PEM, State.J_P] = left[PEM, State.J_P] - right[ACL, State.J_P]
    residuals[CCL, State.PHI_P] = left[CCL, State.PHI_P] - right[PEM, State.PHI_P]
    residuals[CCL, State.J_P] = right[CCL, State.J_P]

    # TEMPERATURE -- continuous everywhere, Dirichlet at both plates
    for region in list(Region)[1:]:
        previous = Region(region - 1)
        residuals[region, State.T] = left[region, State.T] - right[previous, State.T]
        residuals[region, State.J_T] = left[region, State.J_T] - right[previous, State.J_T]
    residuals[AGDL, State.T] = left[AGDL, State.T] - params.T_A
    residuals[AGDL, State.J_T] = right[CGDL, State.T] - params.T_C

    # DISSOLVED WATER -- confined to the ionomer
    residuals[ACL, State.LAMBDA] = left[CCL, State.J_LAMBDA] - right[PEM, State.J_LAMBDA]
    residuals[ACL, State.J_LAMBDA] = left[ACL, State.J_LAMBDA]
    residuals[PEM, State.LAMBDA] = left[PEM, State.LAMBDA] - right[ACL, State.LAMBDA]
    residuals[PEM, State.J_LAMBDA] = left[PEM, State.J_LAMBDA] - right[ACL, State.J_LAMBDA]
    residuals[CCL, State.LAMBDA] = left[CCL, State.LAMBDA] - right[PEM, State.LAMBDA]
    residuals[CCL, State.J_LAMBDA] = right[CCL, State.J_LAMBDA]

    # WATER VAPOUR -- Dirichlet at both gas channels
    residuals[AGDL, State.W_H2O] = left[AGDL, State.W_H2O] - params.w_H2O_A
    residuals[AGDL, State.J_H2O] = left[ACL, State.J_H2O] - right[AGDL, State.J_H2O]
    residuals[ACL, State.W_H2O] = left[ACL, State.W_H2O] - right[AGDL, State.W_H2O]
    residuals[ACL, State.J_H2O] = right[ACL, State.J_H2O]
    residuals[CCL, State.W_H2O] = left[CGDL, State.W_H2O] - right[CCL, State.W_H2O]
    residuals[CCL, State.J_H2O] = left[CCL, State.J_H2O]
    residuals[CGDL, State.W_H2O] = right[CGDL, State.W_H2O] - params.w_H2O_C
    residuals[CGDL, State.J_H2O] = left[CGDL, State.J_H2O] - right[CCL, State.J_H2O]

    # OXYGEN -- cathode side only
    residuals[CCL, State.W_O2] = left[CGDL, State.W_O2] - right[CCL, State.W_O2]
    residuals[CCL, State.J_O2] = left[CCL, State.J_O2]
    residuals[CGDL, State.W_O2] = right[CGDL, State.W_O2] - params.w_O2_C
    residuals[CGDL, State.J_O2] = left[CGDL, State.J_O2] - right[CCL, State.J_O2]

    # LIQUID WATER -- cathode side only
    residuals[CCL, State.P_LIQ] = left[CGDL, State.P_LIQ] - right[CCL, State.P_LIQ]
    residuals[CCL, State.RHO_U_LIQ] = left[CCL, State.RHO_U_LIQ]
    residuals[CGDL, State.P_LIQ] = right[CGDL, State.P_LIQ] - params.P_liq_C
    residuals[CGDL, State.RHO_U_LIQ] = (left[CGDL, State.RHO_U_LIQ]
                                       - right[CCL, State.RHO_U_LIQ])

    # GAS -- Dirichlet at both gas channels
    residuals[AGDL, State.P_GAS] = left[AGDL, State.P_GAS] - params.P_A
    residuals[AGDL, State.RHO_U_GAS] = (left[ACL, State.RHO_U_GAS]
                                        - right[AGDL, State.RHO_U_GAS])
    residuals[ACL, State.P_GAS] = left[ACL, State.P_GAS] - right[AGDL, State.P_GAS]
    residuals[ACL, State.RHO_U_GAS] = right[ACL, State.RHO_U_GAS]
    residuals[CCL, State.P_GAS] = left[CGDL, State.P_GAS] - right[CCL, State.P_GAS]
    residuals[CCL, State.RHO_U_GAS] = left[CCL, State.RHO_U_GAS]
    residuals[CGDL, State.P_GAS] = right[CGDL, State.P_GAS] - params.P_C
    residuals[CGDL, State.RHO_U_GAS] = (left[CGDL, State.RHO_U_GAS]
                                        - right[CCL, State.RHO_U_GAS])

    return residuals.ravel()


# =============================================================================
# INITIAL GUESS
# =============================================================================

def _initial_state_for_region(region: Region, params: Params,
                              U_cell: float) -> np.ndarray:
    """Flat starting profile for one region: anode-side values before the
    membrane, cathode-side values after it, zero where a quantity is inactive."""
    anode_side = region < Region.PEM
    cathode_side = region > Region.PEM

    state = np.zeros(N_STATE)
    state[State.PHI_E] = U_cell if cathode_side else 0.0
    state[State.T] = (params.T_C + params.T_A) / 2
    if Region.ACL <= region <= Region.CCL:
        state[State.LAMBDA] = params.sorption(1.0)
    if anode_side:
        state[State.W_H2O] = params.w_H2O_A
        state[State.P_GAS] = params.P_A
    elif cathode_side:
        state[State.W_H2O] = params.w_H2O_C
        state[State.W_O2] = params.w_O2_C
        state[State.P_LIQ] = params.P_liq_C
        state[State.P_GAS] = params.P_C
    return state


def build_initial_guess(params: Params, U_cell: float,
                        s_mesh: np.ndarray) -> np.ndarray:
    """(80, len(s_mesh)) starting guess, constant along ``s`` in every region."""
    guess = np.zeros((N_TOTAL, len(s_mesh)))
    for region in Region:
        state = _initial_state_for_region(region, params, U_cell)
        guess[block_rows(region)] = state[:, np.newaxis]
    return guess


# =============================================================================
# SOLVE
# =============================================================================

#: Default ceiling on adaptive mesh refinement, mirroring MATLAB ``bvp4c``.
#:
#: ``bvp4c`` caps its mesh at ``NMax = floor(10000/n)``, which is 125 points
#: for this ``n = 80`` system, and the reference implementation ran on that
#: default. The cap is doing real work here: ``k_ad`` switches discontinuously
#: where ``lam`` crosses ``lambda_eq`` inside the CCL (``Params.k_ad``), so no
#: mesh can drive the collocation residual there below ``tol``. ``bvp4c`` hits
#: NMax, warns that the tolerance was not met, and returns a usable solution
#: the sweep continues from; below about 0.75 V that is what the published
#: curve rests on.
#:
#: Translating that ceiling as 200000 was the one part of the port that was
#: not faithful, and it is why sweeps below 0.55 V died. ``solve_bvp``
#: factorizes a collocation Jacobian holding ``2 * N_TOTAL ** 2 = 12800``
#: nonzeros per interval -- roughly 0.85 GB of peak memory per 1000 nodes --
#: so chasing the discontinuity to 200000 nodes would need some 170 GB. The
#: process dies inside SuperLU with a ``MemoryError`` before ``solve_bvp``
#: can return ``success=False``, taking the converged part of the sweep with
#: it.
#:
#: Raising this buys resolution everywhere except at the crossing, at about
#: 0.85 GB per 1000 nodes, and diverges from the reference implementation.
DEFAULT_MAX_NODES = 10_000 // N_TOTAL


@dataclass(frozen=True, eq=False)
class SweepResult:
    """Outcome of a cell-voltage sweep."""

    voltages: np.ndarray            #: [V] cell voltages, in sweep order
    current_densities: np.ndarray   #: [A/cm^2] current density at each voltage
    solutions: list                 #: solve_bvp solution object per voltage
    params: Params                  #: parameter set the sweep was run with

    @property
    def n_voltages(self) -> int:
        return len(self.voltages)

    @property
    def power_densities(self) -> np.ndarray:
        """[W/cm^2] cell power density at each voltage."""
        return self.voltages * self.current_densities

    @property
    def layer_boundaries(self) -> np.ndarray:
        """[m] cumulative positions of the six layer interfaces."""
        return self.params.Lsum

    @property
    def converged(self) -> bool:
        return all(sol.success for sol in self.solutions)


def solve(voltages: Sequence[float] | np.ndarray | None = None,
          tol: float = 1e-4,
          n_per_region: int = 11,
          max_nodes: int = DEFAULT_MAX_NODES,
          verbose: int = 0,
          params: Params | None = None) -> SweepResult:
    """Solve the model over a sweep of cell voltages.

    Each voltage starts from the previous converged solution (continuation),
    so the sweep should run from high voltage (low current) downwards.

    Parameters
    ----------
    voltages
        Cell voltages [V]; defaults to the parameter set's own sweep.
    tol
        Tolerance for ``scipy.integrate.solve_bvp``.
    n_per_region
        Initial mesh points per region.
    max_nodes
        Ceiling on adaptive mesh refinement. Defaults to MATLAB ``bvp4c``'s
        ``floor(10000/n)``, the ceiling the reference implementation ran on;
        see ``DEFAULT_MAX_NODES`` before raising it.
    verbose
        Passed through to ``solve_bvp`` (0, 1 or 2).
    params
        Parameter set; defaults to ``Params()``.
    """
    params = params if params is not None else Params()
    voltages = np.asarray(params.U_list if voltages is None else voltages, dtype=float)

    s_mesh = np.linspace(0.0, 1.0, n_per_region)
    mesh, guess = s_mesh, build_initial_guess(params, voltages[0], s_mesh)

    # j_e at the right-hand end of the CGDL is the total cell current
    current_index = stacked_index(State.J_E, Region.CGDL)
    ode = partial(full_ode, params=params)

    currents: list[float] = []
    solutions: list = []
    for position, U_cell in enumerate(voltages):
        bc = partial(boundary_conditions, U_cell=U_cell, params=params)
        try:
            sol = solve_bvp(ode, bc, mesh, guess, tol=tol,
                            max_nodes=max_nodes, verbose=verbose)
        except MemoryError:
            # SuperLU could not factorize the collocation Jacobian; the mesh
            # has run away. See DEFAULT_MAX_NODES.
            remaining = voltages[position:]
            warnings.warn(
                f"ran out of memory solving U={U_cell:.3f} V with "
                f"max_nodes={max_nodes}; stopping the sweep with "
                f"{len(remaining)} of {len(voltages)} voltages unsolved: "
                f"{np.array2string(remaining, precision=3)}")
            voltages = voltages[:position]
            break

        if not sol.success:
            # bvp4c warns and carries on in exactly this situation, and the
            # reference curve below about 0.75 V is made of such points, so
            # the sweep continues too.
            warnings.warn(f"BVP did not converge for U={U_cell:.3f} V: {sol.message}")

        if not np.all(np.isfinite(sol.y)):
            # Continuation is genuinely poisoned: a non-finite solution cannot
            # seed the next voltage, so every later point would fail too.
            remaining = voltages[position:]
            warnings.warn(
                f"solution for U={U_cell:.3f} V is not finite; stopping the "
                f"sweep with {len(remaining)} of {len(voltages)} voltages "
                f"unsolved: {np.array2string(remaining, precision=3)}")
            voltages = voltages[:position]
            break

        currents.append(sol.y[current_index, -1] / 1e4)  # [A/m^2] -> [A/cm^2]
        solutions.append(sol)
        mesh, guess = sol.x, sol.y  # continuation into the next voltage

    return SweepResult(voltages=voltages,
                       current_densities=np.asarray(currents),
                       solutions=solutions,
                       params=params)
