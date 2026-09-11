"""
1D PEM fuel cell model -- steady-state, non-isothermal, two-phase,
macro-homogeneous MEA model.

Starting point: the reference MATLAB implementation MMM1D.m
(Vetter & Schumacher, 2019, Comp. Phys. Comm. 234:223-234;
v2 by Herrendörfer & Schumacher).

--------------------------------------------------------------------------
NOTE ON THE SOLVER TRANSLATION (read before use)
--------------------------------------------------------------------------
The original MATLAB code uses bvp4c with a MULTI-REGION boundary value
problem: the mesh has duplicated nodes at the 4 internal interfaces,
and odefun/bcfun receive a subdomain index (1..5), with bcfun receiving
ya/yb as 16x5 matrices (one pair of endpoints per region).

scipy.integrate.solve_bvp does NOT natively support multiple regions
(it requires a strictly increasing mesh, with no duplicated nodes). To
preserve exactly the same mathematical structure as the original
problem, the 5 regions are here "stacked" into a single system of
5*16=80 first-order ODEs, solved simultaneously over a single
normalized independent variable s in [0,1] (one copy of s per region,
all sharing the same mesh in s). The chain rule converts d/dx -> d/ds:

    x_region_d(s) = Lsum[d-1] + s*L[d-1]      =>      dY/ds = L[d-1] * dY/dx

The boundary conditions (bcfun) couple the blocks EXACTLY as in the
original MATLAB code (potential/flux continuity at the interfaces,
Dirichlet conditions at the gas channels). This is mathematically
equivalent to the original MATLAB multi-region problem, just
reformulated to fit the scipy API.
"""
import numpy as np
from scipy.integrate import solve_bvp

from .params import Params
from .pc_s import pc_s

NEQ = 8        # number of 2nd-order ODEs (potential-type state variables)
NSTATE = 16    # NEQ*2 (potential + flux, first-order form)
NREGION = 5    # AGDL, ACL, PEM, CCL, CGDL
NTOT = NSTATE * NREGION  # 80

REGION_NAMES = ["AGDL", "ACL", "PEM", "CCL", "CGDL"]

# Names of the 8 potential/state variables, in the order in which they
# appear within each 16-row block (potential, flux, potential, flux, ...)
VAR_NAMES = ["phi_e", "phi_p", "T", "lambda", "w_H2O", "w_O2", "P_liq/s", "P_gas"]

# Active-domain matrix per variable x region (identical to the one in
# the original MMM1D_postprocessing.m)
DOMAINS = np.array([
    [1, 1, 0, 1, 1],  # phi_e
    [0, 1, 1, 1, 0],  # phi_p
    [1, 1, 1, 1, 1],  # T
    [0, 1, 1, 1, 0],  # lambda
    [1, 1, 0, 1, 1],  # w_H2O
    [0, 0, 0, 1, 1],  # w_O2
    [0, 0, 0, 1, 1],  # s (P_liq/rho_u_liq)
    [1, 1, 0, 1, 1],  # P_gas (rho_u_gas)
], dtype=bool)


# =============================================================================
# ODEFUN -- one function per subdomain (equivalent to the "case 1..5" blocks
# in the original MATLAB code)
# =============================================================================

def _unpack(block):
    """block: array (16, m). Returns the 16 state variables as vectors."""
    return (block[0], block[1], block[2], block[3], block[4], block[5],
            block[6], block[7], block[8], block[9], block[10], block[11],
            block[12], block[13], block[14], block[15])


def _agdl(block, p):
    """case 1: AGDL (anode gas diffusion layer)."""
    phi_e, j_e, phi_p, j_p, T, j_T, lam, j_lam, w_H2O, j_H2O, w_O2, j_O2, \
        P_liq, rho_u_liq, P_gas, rho_u_gas = _unpack(block)
    z = np.zeros_like(phi_e)
    dphi_e = dj_e = dphi_p = dj_p = dT = dj_T = dlam = dj_lam = z
    dw_H2O = dj_H2O = dw_O2 = dj_O2 = dP_liq = drho_u_liq = dP_gas = drho_u_gas = z

    w_H2 = 1 - w_H2O
    Mn = 1.0 / (w_H2O / p.M_H2O + w_H2 / p.M_H2)
    C = P_gas / (p.R * T)
    rho_gas = Mn * C
    u_gas = rho_u_gas / rho_gas
    dP_gas = -u_gas / (p.kappa_GDL / p.mu_gas)
    dphi_e = -j_e / p.sigma_e_GDL
    dT = -j_T / p.k_GDL
    s = np.zeros_like(T)
    dw_H2O = -(j_H2O - rho_gas * w_H2O * u_gas) / (
        rho_gas * p.D_H2O_A(p.eps_p_GDL, p.tau_GDL, s, T, P_gas))
    dj_T = -j_e * dphi_e

    return np.array([dphi_e, dj_e, dphi_p, dj_p, dT, dj_T, dlam, dj_lam,
                      dw_H2O, dj_H2O, dw_O2, dj_O2, dP_liq, drho_u_liq,
                      dP_gas, drho_u_gas])


def _acl(block, p):
    """case 2: ACL (anode catalyst layer, HOR)."""
    phi_e, j_e, phi_p, j_p, T, j_T, lam, j_lam, w_H2O, j_H2O, w_O2, j_O2, \
        P_liq, rho_u_liq, P_gas, rho_u_gas = _unpack(block)
    z = np.zeros_like(phi_e)
    dphi_e = dj_e = dphi_p = dj_p = dT = dj_T = dlam = dj_lam = z
    dw_H2O = dj_H2O = dw_O2 = dj_O2 = dP_liq = drho_u_liq = dP_gas = drho_u_gas = z

    w_H2 = 1 - w_H2O
    Mn = 1.0 / (w_H2O / p.M_H2O + w_H2 / p.M_H2)
    x_H2O = w_H2O / p.M_H2O * Mn
    x_H2 = 1 - x_H2O
    C = P_gas / (p.R * T)
    rho_gas = Mn * C
    x_sat = p.P_sat(T) / P_gas
    lambda_eq = p.sorption(x_H2O / x_sat)

    L_ACL = p.L[1]  # ACL thickness (L(2) in 1-based MATLAB indexing)
    S_ad = p.k_ad(lam, lambda_eq, T) / (L_ACL * p.V_m) * (lambda_eq - lam)
    P_H2 = x_H2 * P_gas
    eta = phi_e - phi_p + T * p.DeltaS_HOR / (2 * p.F) + p.R * T / (2 * p.F) * np.log(P_H2 / p.P_ref)
    i = p.BV(p.i_0_HOR(T), p.a_ACL, T, p.beta_HOR, eta)
    S_F = i / (2 * p.F)

    u_gas = rho_u_gas / rho_gas
    dP_gas = -u_gas / (p.kappa_CL / p.mu_gas)
    dphi_e = -j_e / p.sigma_e_CL
    dphi_p = -j_p / p.sigma_p(p.eps_i_CL, lam, T)
    dT = -j_T / p.k_CL
    dlam = (-j_lam + p.xi(lam) / p.F * j_p) * p.V_m / p.D_lambda(p.eps_i_CL, lam, T)
    s = np.zeros_like(T)
    dw_H2O = -(j_H2O - rho_gas * w_H2O * u_gas) / (
        rho_gas * p.D_H2O_A(p.eps_p_CL, p.tau_CL, s, T, P_gas))
    dj_e = -i
    dj_p = i
    dj_T = -j_e * dphi_e - j_p * dphi_p + i * eta - S_F * T * p.DeltaS_HOR + p.H_ad * S_ad
    dj_lam = S_ad
    sumS_gas = -p.M_H2O * S_ad - p.M_H2 * S_F
    dj_H2O = -S_ad * p.M_H2O
    drho_u_gas = sumS_gas

    return np.array([dphi_e, dj_e, dphi_p, dj_p, dT, dj_T, dlam, dj_lam,
                      dw_H2O, dj_H2O, dw_O2, dj_O2, dP_liq, drho_u_liq,
                      dP_gas, drho_u_gas])


def _pem(block, p):
    """case 3: PEM (membrane)."""
    phi_e, j_e, phi_p, j_p, T, j_T, lam, j_lam, w_H2O, j_H2O, w_O2, j_O2, \
        P_liq, rho_u_liq, P_gas, rho_u_gas = _unpack(block)
    z = np.zeros_like(phi_e)
    dphi_e = dj_e = dphi_p = dj_p = dT = dj_T = dlam = dj_lam = z
    dw_H2O = dj_H2O = dw_O2 = dj_O2 = dP_liq = drho_u_liq = dP_gas = drho_u_gas = z

    dphi_p = -j_p / p.sigma_p(1.0, lam, T)
    dT = -j_T / p.k_PEM
    dlam = (-j_lam + p.xi(lam) / p.F * j_p) * p.V_m / p.D_lambda(1.0, lam, T)
    dj_T = -j_p * dphi_p

    return np.array([dphi_e, dj_e, dphi_p, dj_p, dT, dj_T, dlam, dj_lam,
                      dw_H2O, dj_H2O, dw_O2, dj_O2, dP_liq, drho_u_liq,
                      dP_gas, drho_u_gas])


def _ccl(block, p):
    """case 4: CCL (cathode catalyst layer, ORR).

    NOTE: exactly as in the original MMM1D.m, dP_gas and dP_liq in this
    layer use ``kappa_GDL`` (not ``kappa_CL``) in Darcy's law. This is
    preserved *as-is* from the original source code (the lines
    "dP_gas=-u_gas./(kappa_GDL/mu_gas)" and the following dP_liq line,
    inside case 4/CCL). It is worth confirming this choice against the
    reference paper before using the results for the thesis -- it
    could be intentional (CL and GDL with similar permeabilities in
    the parameter set used) or a copy-paste slip from case 5/CGDL.
    """
    phi_e, j_e, phi_p, j_p, T, j_T, lam, j_lam, w_H2O, j_H2O, w_O2, j_O2, \
        P_liq, rho_u_liq, P_gas, rho_u_gas = _unpack(block)
    z = np.zeros_like(phi_e)
    dphi_e = dj_e = dphi_p = dj_p = dT = dj_T = dlam = dj_lam = z
    dw_H2O = dj_H2O = dw_O2 = dj_O2 = dP_liq = drho_u_liq = dP_gas = drho_u_gas = z

    w_N2 = 1 - w_H2O - w_O2
    Mn = 1.0 / (w_H2O / p.M_H2O + w_O2 / p.M_O2 + w_N2 / p.M_N2)
    x_H2O = w_H2O / p.M_H2O * Mn
    x_O2 = w_O2 / p.M_O2 * Mn
    C = P_gas / (p.R * T)
    rho_gas = Mn * C
    p_c = P_liq - P_gas
    s = pc_s(p_c, p.s_im)
    x_sat = p.P_sat(T) / P_gas
    S_ec = np.zeros_like(s)  # gamma_ec(...) is discarded in the original (see pc_s module docstring)
    lambda_eq = p.sorption(x_H2O / x_sat)

    L_CCL = p.L[3]  # CCL thickness (L(4) in 1-based MATLAB indexing)
    S_ad = p.k_ad(lam, lambda_eq, T) / (L_CCL * p.V_m) * (lambda_eq - lam)
    P_O2 = x_O2 * P_gas
    eta = -(p.DeltaH - T * p.DeltaS_ORR) / (2 * p.F) + p.R * T / (4 * p.F) * np.log(P_O2 / p.P_ref) - (phi_e - phi_p)
    i = p.BV(p.i_0_ORR(T, P_O2), p.a_CCL, T, p.beta_ORR, eta)
    S_F = i / (2 * p.F)

    u_gas = rho_u_gas / rho_gas
    u_liq = rho_u_liq / p.rho_liq
    dP_gas = -u_gas / (p.kappa_GDL / p.mu_gas)          # (sic, see docstring above)
    dP_liq = -u_liq / (p.kappa_GDL * p.kappa_rel_liq(s) / p.mu(T))  # (sic, see docstring above)
    dphi_e = -j_e / p.sigma_e_CL
    dphi_p = -j_p / p.sigma_p(p.eps_i_CL, lam, T)
    dT = -j_T / p.k_CL
    dlam = (-j_lam + p.xi(lam) / p.F * j_p) * p.V_m / p.D_lambda(p.eps_i_CL, lam, T)
    dw_H2O = -(j_H2O - rho_gas * w_H2O * u_gas) / (
        rho_gas * p.D_H2O_C(p.eps_p_CL, p.tau_CL, s, T, P_gas))
    dw_O2 = -(j_O2 - rho_gas * w_O2 * u_gas) / (
        rho_gas * p.D_O2(p.eps_p_CL, p.tau_CL, s, T, P_gas))
    dj_e = i
    dj_p = -i
    dj_T = -j_e * dphi_e - j_p * dphi_p + i * eta - S_F * T * p.DeltaS_ORR + p.H_ad * S_ad + p.H_ec * S_ec
    dj_lam = S_F + S_ad
    sumS_gas = -p.M_H2O * (S_ec + S_ad) - p.M_O2 * S_F / 2
    dj_H2O = -p.M_H2O * (S_ec + S_ad)
    dj_O2 = -p.M_O2 * S_F / 2
    drho_u_liq = p.M_H2O * S_ec
    drho_u_gas = sumS_gas

    return np.array([dphi_e, dj_e, dphi_p, dj_p, dT, dj_T, dlam, dj_lam,
                      dw_H2O, dj_H2O, dw_O2, dj_O2, dP_liq, drho_u_liq,
                      dP_gas, drho_u_gas])


def _cgdl(block, p):
    """case 5: CGDL (cathode gas diffusion layer)."""
    phi_e, j_e, phi_p, j_p, T, j_T, lam, j_lam, w_H2O, j_H2O, w_O2, j_O2, \
        P_liq, rho_u_liq, P_gas, rho_u_gas = _unpack(block)
    z = np.zeros_like(phi_e)
    dphi_e = dj_e = dphi_p = dj_p = dT = dj_T = dlam = dj_lam = z
    dw_H2O = dj_H2O = dw_O2 = dj_O2 = dP_liq = drho_u_liq = dP_gas = drho_u_gas = z

    w_N2 = 1 - w_H2O - w_O2
    Mn = 1.0 / (w_H2O / p.M_H2O + w_O2 / p.M_O2 + w_N2 / p.M_N2)
    C = P_gas / (p.R * T)
    rho_gas = Mn * C

    u_gas = rho_u_gas / rho_gas
    u_liq = rho_u_liq / p.rho_liq
    p_c = P_liq - P_gas
    s = pc_s(p_c, p.s_im)
    dP_gas = -u_gas / (p.kappa_GDL / p.mu_gas)
    dP_liq = -u_liq / (p.kappa_GDL * p.kappa_rel_liq(s) / p.mu(T))
    S_ec = np.zeros_like(s)  # gamma_ec(...) is discarded in the original
    dphi_e = -j_e / p.sigma_e_GDL
    dT = -j_T / p.k_GDL
    dw_H2O = -(j_H2O - rho_gas * w_H2O * u_gas) / (
        rho_gas * p.D_H2O_C(p.eps_p_GDL, p.tau_GDL, s, T, P_gas))
    dw_O2 = -(j_O2 - rho_gas * w_O2 * u_gas) / (
        rho_gas * p.D_O2(p.eps_p_GDL, p.tau_GDL, s, T, P_gas))
    dj_T = -j_e * dphi_e + p.H_ec * S_ec
    sumS_gas = -p.M_H2O * S_ec
    dj_H2O = -p.M_H2O * S_ec
    drho_u_liq = p.M_H2O * S_ec
    drho_u_gas = sumS_gas

    return np.array([dphi_e, dj_e, dphi_p, dj_p, dT, dj_T, dlam, dj_lam,
                      dw_H2O, dj_H2O, dw_O2, dj_O2, dP_liq, drho_u_liq,
                      dP_gas, drho_u_gas])


_REGION_FUNCS = [_agdl, _acl, _pem, _ccl, _cgdl]  # index 0 -> region 1 (AGDL), etc.


def full_ode(s, Y, p):
    """Stacked system of all 5 regions (80 first-order ODEs) as a function of s in [0,1].

    Y: array (80, m). Returns dY/ds, same shape.
    """
    dYds = np.empty_like(Y)
    for d in range(1, NREGION + 1):
        block = Y[(d - 1) * NSTATE: d * NSTATE, :]
        dblock_dx = _REGION_FUNCS[d - 1](block, p)
        dYds[(d - 1) * NSTATE: d * NSTATE, :] = p.L[d - 1] * dblock_dx
    return dYds


# =============================================================================
# BOUNDARY AND INTERFACE CONDITIONS (equivalent to bcfun)
# =============================================================================

def _yidx(k, d):
    """0-based index into the 80-entry stacked vector for variable k
    (1..16, MATLAB-style) of region d (1..5, MATLAB-style)."""
    return (d - 1) * NSTATE + (k - 1)


def bcfun(Ya, Yb, Uval, p):
    """Line-by-line replica of bcfun(ya, yb, U) from the original MMM1D.m.

    Ya, Yb: vectors (80,) holding the values at s=0 and s=1 of each of
    the 5 stacked regions (block d occupies Ya[(d-1)*16 : d*16]).
    """
    def Y(y, k, d):
        return y[_yidx(k, d)]

    def r(i):
        return i - 1  # 1-based MATLAB index -> 0-based Python index

    res = Ya.copy()  # "res = ya(:)" -- homogeneous condition by default

    # ELECTRONS
    res[r(0 * NEQ + 1)] = Y(Ya, 1, 1)
    res[r(0 * NEQ + 2)] = Y(Ya, 2, 2) - Y(Yb, 2, 1)
    res[r(2 * NEQ + 1)] = Y(Ya, 1, 2) - Y(Yb, 1, 1)
    res[r(2 * NEQ + 2)] = Y(Yb, 2, 2)
    res[r(6 * NEQ + 1)] = Y(Ya, 1, 5) - Y(Yb, 1, 4)
    res[r(6 * NEQ + 2)] = Y(Ya, 2, 4)
    res[r(8 * NEQ + 1)] = Y(Yb, 1, 5) - Uval
    res[r(8 * NEQ + 2)] = Y(Ya, 2, 5) - Y(Yb, 2, 4)

    # PROTONS
    res[r(2 * NEQ + 3)] = Y(Ya, 4, 4) - Y(Yb, 4, 3)
    res[r(2 * NEQ + 4)] = Y(Ya, 4, 2)
    res[r(4 * NEQ + 3)] = Y(Ya, 3, 3) - Y(Yb, 3, 2)
    res[r(4 * NEQ + 4)] = Y(Ya, 4, 3) - Y(Yb, 4, 2)
    res[r(6 * NEQ + 3)] = Y(Ya, 3, 4) - Y(Yb, 3, 3)
    res[r(6 * NEQ + 4)] = Y(Yb, 4, 4)

    # TEMPERATURE
    for d in (2, 3, 4, 5):
        res[r(2 * (d - 1) * NEQ + 5)] = Y(Ya, 5, d) - Y(Yb, 5, d - 1)
        res[r(2 * (d - 1) * NEQ + 6)] = Y(Ya, 6, d) - Y(Yb, 6, d - 1)
    res[r(0 * NEQ + 5)] = Y(Ya, 5, 1) - p.T_A
    res[r(0 * NEQ + 6)] = Y(Yb, 5, 5) - p.T_C

    # DISSOLVED WATER
    res[r(2 * NEQ + 7)] = Y(Ya, 8, 4) - Y(Yb, 8, 3)
    res[r(2 * NEQ + 8)] = Y(Ya, 8, 2)
    res[r(4 * NEQ + 7)] = Y(Ya, 7, 3) - Y(Yb, 7, 2)
    res[r(4 * NEQ + 8)] = Y(Ya, 8, 3) - Y(Yb, 8, 2)
    res[r(6 * NEQ + 7)] = Y(Ya, 7, 4) - Y(Yb, 7, 3)
    res[r(6 * NEQ + 8)] = Y(Yb, 8, 4)

    # WATER VAPOR
    res[r(0 * NEQ + 9)] = Y(Ya, 9, 1) - p.w_H2O_A
    res[r(0 * NEQ + 10)] = Y(Ya, 10, 2) - Y(Yb, 10, 1)
    res[r(2 * NEQ + 9)] = Y(Ya, 9, 2) - Y(Yb, 9, 1)
    res[r(2 * NEQ + 10)] = Y(Yb, 10, 2)
    res[r(6 * NEQ + 9)] = Y(Ya, 9, 5) - Y(Yb, 9, 4)
    res[r(6 * NEQ + 10)] = Y(Ya, 10, 4)
    res[r(8 * NEQ + 9)] = Y(Yb, 9, 5) - p.w_H2O_C
    res[r(8 * NEQ + 10)] = Y(Ya, 10, 5) - Y(Yb, 10, 4)

    # OXYGEN
    res[r(6 * NEQ + 11)] = Y(Ya, 11, 5) - Y(Yb, 11, 4)
    res[r(6 * NEQ + 12)] = Y(Ya, 12, 4)
    res[r(8 * NEQ + 11)] = Y(Yb, 11, 5) - p.w_O2_C
    res[r(8 * NEQ + 12)] = Y(Ya, 12, 5) - Y(Yb, 12, 4)

    # LIQUID WATER
    res[r(6 * NEQ + 13)] = Y(Ya, 13, 5) - Y(Yb, 13, 4)
    res[r(6 * NEQ + 14)] = Y(Ya, 14, 4)
    res[r(8 * NEQ + 13)] = Y(Yb, 13, 5) - p.P_liq_C
    res[r(8 * NEQ + 14)] = Y(Ya, 14, 5) - Y(Yb, 14, 4)

    # GAS
    res[r(0 * NEQ + 15)] = Y(Ya, 15, 1) - p.P_A
    res[r(0 * NEQ + 16)] = Y(Ya, 16, 2) - Y(Yb, 16, 1)
    res[r(2 * NEQ + 15)] = Y(Ya, 15, 2) - Y(Yb, 15, 1)
    res[r(2 * NEQ + 16)] = Y(Yb, 16, 2)
    res[r(6 * NEQ + 15)] = Y(Ya, 15, 5) - Y(Yb, 15, 4)
    res[r(6 * NEQ + 16)] = Y(Ya, 16, 4)
    res[r(8 * NEQ + 15)] = Y(Yb, 15, 5) - p.P_C
    res[r(8 * NEQ + 16)] = Y(Ya, 16, 5) - Y(Yb, 16, 4)

    return res


# =============================================================================
# INITIAL GUESS (equivalent to yinit)
# =============================================================================

def _yinit_region(d, p, U1):
    phi_e = U1 if d > 3 else 0.0
    phi_p = 0.0
    T = (p.T_C + p.T_A) / 2
    lam = p.sorption(1.0) if (1 < d < 5) else 0.0
    w_H2O = p.w_H2O_A if d < 3 else (p.w_H2O_C if d > 3 else 0.0)
    w_O2 = p.w_O2_C if d > 3 else 0.0
    P_liq = p.P_liq_C if d > 3 else 0.0
    P_gas = p.P_A if d < 3 else (p.P_C if d > 3 else 0.0)
    return np.array([phi_e, 0, phi_p, 0, T, 0, lam, 0, w_H2O, 0, w_O2, 0, P_liq, 0, P_gas, 0])


def build_initial_guess(p, U1, s_mesh):
    m = len(s_mesh)
    Y0 = np.zeros((NTOT, m))
    for d in range(1, NREGION + 1):
        y0d = _yinit_region(d, p, U1)
        Y0[(d - 1) * NSTATE: d * NSTATE, :] = np.tile(y0d.reshape(-1, 1), (1, m))
    return Y0


# =============================================================================
# MAIN SOLVE LOOP (equivalent to the body of MMM1D.m)
# =============================================================================

class MMM1DResult:
    """Output container, equivalent to [I,U,SOL,x,Lsum,Np,Neq,domains]."""

    def __init__(self, I, U, SOL, Lsum, Np, Neq, domains, params):
        self.I = I
        self.U = U
        self.SOL = SOL          # list of scipy solution objects (one per voltage)
        self.Lsum = Lsum
        self.Np = Np
        self.Neq = Neq
        self.domains = domains
        self.params = params


def solve(voltages=None, tol=1e-4, n_per_region=11, max_nodes=200000, verbose=0):
    """Solves the model for a sweep of cell voltages.

    Equivalent to ``[I,U,SOL,x,Lsum,Np,Neq,domains] = MMM1D()``.

    Parameters
    ----------
    voltages : array_like, optional
        Cell voltages [V] to sweep over (default: 1.15:-0.05:1.00, as in
        the original).
    tol : float
        Tolerance for ``scipy.integrate.solve_bvp`` (RelTol was 1e-4 in
        the original MATLAB code).
    n_per_region : int
        Number of initial mesh points per region (the original uses ~11).
    max_nodes : int
        Maximum number of nodes the adaptive mesh refinement can reach
        (NMax in MATLAB).
    verbose : int
        Verbosity passed through to solve_bvp (0, 1 or 2).
    """
    p = Params()
    if voltages is None:
        voltages = p.U_list
    voltages = np.asarray(voltages, dtype=float)

    s_mesh = np.linspace(0.0, 1.0, n_per_region)
    Y0 = build_initial_guess(p, voltages[0], s_mesh)

    I_list = []
    SOL = []
    x0, y0 = s_mesh, Y0
    idx_current = _yidx(2, 5)  # j_e at the end of region 5 (CGDL) = total current

    for Uval in voltages:
        fun = lambda s, Y, U=Uval: full_ode(s, Y, p)
        bc = lambda ya, yb, U=Uval: bcfun(ya, yb, U, p)
        sol = solve_bvp(fun, bc, x0, y0, tol=tol, max_nodes=max_nodes, verbose=verbose)
        if not sol.success:
            import warnings
            warnings.warn(f"BVP did not converge for U={Uval:.3f} V: {sol.message}")
        I_val = sol.y[idx_current, -1] / 1e4  # [A/cm^2]
        I_list.append(I_val)
        SOL.append(sol)
        x0, y0 = sol.x, sol.y  # continuation: next voltage starts from the previous solution

    return MMM1DResult(
        I=np.array(I_list), U=voltages, SOL=SOL, Lsum=p.Lsum,
        Np=len(voltages), Neq=NEQ, domains=DOMAINS, params=p,
    )
