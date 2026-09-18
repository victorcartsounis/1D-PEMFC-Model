"""Physical parameters and constitutive relations for the 1D PEM fuel cell model.

Parameter values and constitutive relations follow:

    R. Vetter and J. O. Schumacher, "Free open reference implementation of a
    two-phase PEM fuel cell model", Computer Physics Communications 234 (2019)
    223-234. https://doi.org/10.1016/j.cpc.2018.07.023

``Params`` is frozen: the physical constants of a running model cannot be
mutated by accident, and a modified parameter set is made with
``dataclasses.replace``.

Attribute names deliberately keep the paper's notation (``M_H2O``,
``sigma_e_CL``, ``eps_p_GDL``) so that the code can be read side by side with
the publication. Method names are spelled out in full, since a relation is
easier to recognise by name than by symbol at the call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ArrayLike = float | np.ndarray


def _default_layer_thicknesses() -> np.ndarray:
    return np.array([160, 10, 25, 10, 160]) * 1e-6


def _default_voltage_sweep() -> np.ndarray:
    """[V] 1.15 V down to 0.40 V in 50 mV steps.

    The whole polarization curve, so that a default run draws all three of its
    regions: activation near open circuit, the near-linear ohmic middle, and
    the mass-transport-limited plateau past about 0.50 V. Stopping at 1.00 V,
    as this used to, shows only the first.

    50 mV is the step, not the resolution: each voltage is solved by
    continuation from the previous one, and widening the step makes the
    starting guess worse until the solver stops converging at all.

    Below about 0.75 V ``solve_bvp`` reports that it could not meet its
    tolerance, for the reason set out in ``model.DEFAULT_MAX_NODES``, and the
    sweep carries on from the solution anyway -- as the reference
    implementation does. Judge those points by the mesh-convergence check in
    :mod:`pemfc_1d.metrics` rather than by the tolerance flag.
    """
    return np.arange(1.15, 0.40 - 1e-9, -0.05)


@dataclass(frozen=True, eq=False)
class Params:
    """Constants, operating conditions and constitutive relations."""

    # ---------------- CONSTANTS ----------------
    M_H2O: float = 18e-3        # [kg/mol] molar mass of water
    M_H2: float = 2e-3          # [kg/mol] molar mass of hydrogen
    M_N2: float = 28e-3         # [kg/mol] molar mass of nitrogen
    M_O2: float = 32e-3         # [kg/mol] molar mass of oxygen
    T_0: float = 273.15         # [K] zero degrees Celsius
    R: float = 8.31446          # [J*mol/K] universal gas constant
    F: float = 96485.333        # [C/mol] Faraday constant
    P_ref: float = 101325       # [Pa] reference pressure

    # ---------------- OPERATING CONDITIONS ----------------
    P_A: float = 1.5e5          # [Pa] total pressure in the anode gas channel
    P_C: float = 1.5e5          # [Pa] total pressure in the cathode gas channel
    RH_A: float = 0.9           # [-] relative humidity in the anode channel
    RH_C: float = 0.9           # [-] relative humidity in the cathode channel
    s_C: float = 0.12           # [-] liquid saturation at the cathode GDL/GC interface
    alpha_O2: float = 0.21      # [-] O2 mole fraction in the dry oxidant gas
    T_A_celsius: float = 70.0   # [degC] anode bipolar plate/channel temperature
    T_C_celsius: float = 70.0   # [degC] cathode bipolar plate/channel temperature
    T_ref_celsius: float = 80.0 # [degC] reference temperature of the correlations

    # ---------------- ELECTROCHEMICAL PARAMETERS ----------------
    beta_HOR: float = 0.5           # [-] HOR symmetry factor
    beta_ORR: float = 0.5           # [-] ORR symmetry factor
    DeltaH: float = -285.83e3       # [J/mol] enthalpy of formation of liquid water
    DeltaS_HOR: float = 0.104       # [J/(mol*K)] HOR reaction entropy
    DeltaS_ORR: float = -163.3      # [J/(mol*K)] ORR reaction entropy

    # ---------------- MATERIAL PARAMETERS ----------------
    L: np.ndarray = field(default_factory=_default_layer_thicknesses)  # [m] layer thicknesses
    a_ACL: float = 1e7          # [1/m] active area density of the ACL
    a_CCL: float = 3e7          # [1/m] active area density of the CCL
    H_ec: float = 42e3          # [J/mol] molar enthalpy of evaporation/condensation
    k_GDL: float = 1.6          # [W/(m*K)] GDL thermal conductivity
    k_CL: float = 0.27          # [W/(m*K)] CL thermal conductivity
    k_PEM: float = 0.3          # [W/(m*K)] PEM thermal conductivity
    V_m: float = 1.02 / 1.97e3  # [m^3/mol] molar volume of dry membrane
    rho_liq: float = 0.978e3    # [kg/m^3] liquid water density
    mu_gas: float = 2.1e-5      # [Pa*s] gas dynamic viscosity
    eps_i_CL: float = 0.3       # [-] ionomer volume fraction in the dry CL
    eps_p_GDL: float = 0.76     # [-] GDL porosity
    eps_p_CL: float = 0.4       # [-] CL porosity
    kappa_GDL: float = 6.15e-12 # [m^2] GDL absolute permeability
    kappa_CL: float = 1e-13     # [m^2] CL absolute permeability
    sigma_e_GDL: float = 1250   # [S/m] GDL electrical conductivity
    sigma_e_CL: float = 350     # [S/m] CL electrical conductivity
    tau_GDL: float = 1.6        # [-] GDL pore tortuosity
    tau_CL: float = 1.6         # [-] CL pore tortuosity

    U_list: np.ndarray = field(default_factory=_default_voltage_sweep)  # [V] default sweep

    # ---------------- DERIVED (set in __post_init__) ----------------
    T_ref: float = field(init=False)       # [K] reference temperature
    T_A: float = field(init=False)         # [K] anode plate temperature
    T_C: float = field(init=False)         # [K] cathode plate temperature
    H_ad: float = field(init=False)        # [J/mol] enthalpy of absorption/desorption
    s_im: float = field(init=False)        # [-] immobile liquid water saturation
    V_w: float = field(init=False)         # [m^3/mol] molar volume of liquid water
    Lsum: np.ndarray = field(init=False)   # [m] cumulative layer interface positions
    Nd: int = field(init=False)            # number of layers
    w_H2O_A: float = field(init=False)     # [-] anode channel vapour mass fraction
    w_H2O_C: float = field(init=False)     # [-] cathode channel vapour mass fraction
    w_O2_C: float = field(init=False)      # [-] cathode channel oxygen mass fraction
    P_liq_C: float = field(init=False)     # [Pa] cathode channel liquid pressure

    def __post_init__(self) -> None:
        # frozen dataclasses block normal assignment; this is the documented
        # way to fill fields that are computed from the others
        def derive(name: str, value: object) -> None:
            object.__setattr__(self, name, value)

        derive("T_ref", self.T_0 + self.T_ref_celsius)
        derive("T_A", self.T_0 + self.T_A_celsius)
        derive("T_C", self.T_0 + self.T_C_celsius)
        derive("H_ad", self.H_ec)
        derive("s_im", self.s_C)
        derive("V_w", self.M_H2O / self.rho_liq)
        derive("Lsum", np.concatenate(([0.0], np.cumsum(self.L))))
        derive("Nd", len(self.L))

        x_H2O_A = self.RH_A * self.P_sat(self.T_A) / self.P_A
        derive("w_H2O_A", 1.0 / (self.M_H2 / self.M_H2O * (1.0 / x_H2O_A - 1.0) + 1.0))

        x_H2O_C = self.RH_C * self.P_sat(self.T_C) / self.P_C
        x_O2_C = self.alpha_O2 * (1.0 - x_H2O_C)
        derive("w_H2O_C", x_H2O_C * self.M_H2O / (
            self.M_N2 - x_O2_C * (self.M_N2 - self.M_O2) - x_H2O_C * (self.M_N2 - self.M_H2O)))
        derive("w_O2_C", x_O2_C * self.M_O2 / (x_H2O_C * self.M_H2O) * self.w_H2O_C)
        derive("P_liq_C", self.capillary_pressure(self.s_C) + self.P_C)

    # =====================================================================
    # WATER CONSTITUTIVE RELATIONSHIPS
    # =====================================================================
    def P_sat(self, T: ArrayLike) -> ArrayLike:
        """[Pa] water vapour saturation pressure."""
        return np.exp(23.1963 - 3816.44 / (T - 46.13))

    def mu_liq(self, T: ArrayLike) -> ArrayLike:
        """[Pa*s] liquid water dynamic viscosity."""
        return 1e-3 * np.exp(-3.63148 + 542.05 / (T - 144.15))

    # =====================================================================
    # MODEL PARAMETERIZATION
    # =====================================================================
    def arrhenius(self, E_activation: float, T: ArrayLike) -> ArrayLike:
        """[-] Arrhenius correction from the reference temperature to ``T``."""
        return np.exp(E_activation / self.R * (1.0 / self.T_ref - 1.0 / T))

    def i_0_HOR(self, T: ArrayLike) -> ArrayLike:
        """[A/m^2] HOR exchange current density."""
        return 0.27e4 * self.arrhenius(16e3, T)

    def i_0_ORR(self, T: ArrayLike, P_O2: ArrayLike) -> ArrayLike:
        """[A/m^2] ORR exchange current density."""
        return 2.47e-4 * (P_O2 / self.P_ref) ** 0.54 * self.arrhenius(67e3, T)

    def butler_volmer(self, i_0: ArrayLike, a: float, T: ArrayLike,
                      beta: float, eta: ArrayLike) -> ArrayLike:
        """[A/m^3] volumetric current density from the Butler-Volmer equation."""
        return i_0 * a * (
            np.exp(beta * 2 * self.F / (self.R * T) * eta)
            - np.exp(-(1 - beta) * 2 * self.F / (self.R * T) * eta)
        )

    def _diffusivity_scaling(self, eps_p: float, tau: float, s: ArrayLike,
                             T: ArrayLike, P: ArrayLike) -> ArrayLike:
        """[-] porosity, saturation, temperature and pressure scaling shared by
        every gas-phase diffusivity.

        Argument order deliberately matches ``D_O2`` / ``D_H2O_A`` /
        ``D_H2O_C`` so that the four signatures cannot drift apart.
        """
        return eps_p / tau ** 2 * (1 - s) ** 3 * (T / self.T_ref) ** 1.5 * (self.P_ref / P)

    def D_O2(self, eps_p: float, tau: float, s: ArrayLike,
             T: ArrayLike, P: ArrayLike) -> ArrayLike:
        """[m^2/s] oxygen diffusion coefficient."""
        return 0.28e-4 * self._diffusivity_scaling(eps_p, tau, s, T, P)

    def D_H2O_A(self, eps_p: float, tau: float, s: ArrayLike,
                T: ArrayLike, P: ArrayLike) -> ArrayLike:
        """[m^2/s] water vapour diffusion coefficient on the anode side."""
        return 1.24e-4 * self._diffusivity_scaling(eps_p, tau, s, T, P)

    def D_H2O_C(self, eps_p: float, tau: float, s: ArrayLike,
                T: ArrayLike, P: ArrayLike) -> ArrayLike:
        """[m^2/s] water vapour diffusion coefficient on the cathode side."""
        return 0.36e-4 * self._diffusivity_scaling(eps_p, tau, s, T, P)

    def water_volume_fraction(self, lam: ArrayLike) -> ArrayLike:
        """[-] water volume fraction in the ionomer."""
        return lam * self.V_w / (self.V_m + lam * self.V_w)

    # =====================================================================
    # MATERIAL CONSTITUTIVE RELATIONSHIPS
    # =====================================================================
    def k_ad(self, lam: ArrayLike, lambda_eq: ArrayLike, T: ArrayLike) -> ArrayLike:
        """[m/s] mass transfer coefficient for vapour sorption in Nafion."""
        return (np.where(lam < lambda_eq, 3.53e-5, 1.42e-4)
                * self.water_volume_fraction(lam) * self.arrhenius(20e3, T))

    def reduced_saturation(self, s: ArrayLike) -> ArrayLike:
        """[-] liquid water saturation reduced by the immobile fraction."""
        return (s - self.s_im) / (1 - self.s_im)

    def gamma_ec(self, x_H2O: ArrayLike, x_sat: ArrayLike,
                 s: ArrayLike, T: ArrayLike) -> ArrayLike:
        """[1/s] evaporation/condensation rate.

        Not currently used: phase change is disabled in this version of the
        model (see the "Current simplifications" note in ``model.py``). Kept
        here for the planned two-phase water transport work.
        """
        reduced = self.reduced_saturation(s)
        return (
            2e6
            * np.where(x_H2O < x_sat, 5e-4 * reduced, 6e-3 * (1 - reduced))
            * np.sqrt(self.R * T / (2 * np.pi * self.M_H2O))
        )

    def sigma_p(self, eps_i: float, lam: ArrayLike, T: ArrayLike) -> ArrayLike:
        """[S/m] Nafion proton conductivity."""
        return (eps_i ** 1.5 * 116
                * np.maximum(0.0, self.water_volume_fraction(lam) - 0.06) ** 1.5
                * self.arrhenius(15e3, T))

    def sorption(self, RH: ArrayLike) -> ArrayLike:
        """[-] Nafion vapour sorption isotherm."""
        return 0.043 + 17.81 * RH - 39.85 * RH ** 2 + 36.0 * RH ** 3

    def D_lambda(self, eps_i: float, lam: ArrayLike, T: ArrayLike) -> ArrayLike:
        """[m^2/s] water diffusion coefficient in Nafion."""
        return (
            eps_i ** 1.5
            * (3.842 * lam ** 3 - 32.03 * lam ** 2 + 67.74 * lam)
            / (lam ** 3 - 2.115 * lam ** 2 - 33.013 * lam + 103.37)
            * 1e-10
            * self.arrhenius(20e3, T)
        )

    def electro_osmotic_drag(self, lam: ArrayLike) -> ArrayLike:
        """[-] Nafion electro-osmotic drag coefficient."""
        return 2.5 * lam / 22

    def capillary_pressure(self, s: ArrayLike) -> ArrayLike:
        """[Pa] GDL capillary pressure-saturation relation."""
        return (-0.00011 * np.exp(-44.02 * (s - 0.496))
                + 278.3 * np.exp(8.103 * (s - 0.496)) - 191.8)

    def kappa_rel_liq(self, s: ArrayLike) -> ArrayLike:
        """[-] relative permeability of the liquid phase."""
        return np.maximum(1e-6, s)
