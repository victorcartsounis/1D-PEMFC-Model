"""
Physical parameters and constitutive relations for the 1D PEM fuel cell model.

Direct translation of the CONSTANTS, OPERATING CONDITIONS,
ELECTROCHEMICAL PARAMETERS, MATERIAL PARAMETERS, WATER CONSTITUTIVE
RELATIONSHIPS, MODEL PARAMETERIZATION and MATERIAL CONSTITUTIVE
RELATIONSHIPS sections of the original reference MATLAB implementation
(Vetter & Schumacher, 2019, Comp. Phys. Comm. 234:223-234).

All functions accept and return np.ndarray (or scalars), operating
element-wise exactly like the vectorized anonymous functions of the
original MATLAB code (.* -> *, ./ -> /, .^ -> **).
"""
import numpy as np


class Params:
    """Container for all constants, operating conditions and
    constitutive functions of the model. A single instance is passed
    to the physics functions of each subdomain."""

    def __init__(self):
        # ---------------------------------------------------------------
        # CONSTANTS
        # ---------------------------------------------------------------
        self.M_H2O = 18e-3   # [kg/mol] molar mass of water
        self.M_H2 = 2e-3     # [kg/mol] molar mass of hydrogen
        self.M_N2 = 28e-3    # [kg/mol] molar mass of nitrogen
        self.M_O2 = 32e-3    # [kg/mol] molar mass of oxygen
        self.T_0 = 273.15    # [K] zero degrees Celsius
        self.R = 8.31446     # [J*mol/K] universal gas constant
        self.F = 96485.333   # [C/mol] Faraday constant
        self.P_ref = 101325  # [Pa] reference pressure
        self.T_ref = self.T_0 + 80  # [K] reference temperature

        # ---------------------------------------------------------------
        # OPERATING CONDITIONS
        # ---------------------------------------------------------------
        # [V] cell voltage sweep (equivalent to 1.15:-0.05:1.00)
        self.U_list = np.arange(1.15, 1.00 - 1e-9, -0.05)
        self.P_A = 1.5e5     # [Pa] total pressure in the anode gas channel
        self.P_C = 1.5e5     # [Pa] total pressure in the cathode gas channel
        self.RH_A = 0.9      # [-] relative humidity in the anode channel
        self.RH_C = 0.9      # [-] relative humidity in the cathode channel
        self.s_C = 0.12      # [-] liquid water saturation at the cathode GDL/GC interface
        self.T_A = self.T_0 + 70  # [K] anode bipolar plate/channel temperature
        self.T_C = self.T_0 + 70  # [K] cathode bipolar plate/channel temperature
        self.alpha_O2 = 0.21  # [-] O2 mole fraction in the dry oxidant gas

        # ---------------------------------------------------------------
        # ELECTROCHEMICAL PARAMETERS
        # ---------------------------------------------------------------
        self.beta_HOR = 0.5   # [-] HOR symmetry factor
        self.beta_ORR = 0.5   # [-] ORR symmetry factor
        self.DeltaH = -285.83e3       # [J/mol] enthalpy of formation of liquid water
        self.DeltaS_HOR = 0.104       # [J/(mol*K)] HOR reaction entropy
        self.DeltaS_ORR = -163.3      # [J/(mol*K)] ORR reaction entropy

        # ---------------------------------------------------------------
        # MATERIAL PARAMETERS
        # ---------------------------------------------------------------
        self.L = np.array([160, 10, 25, 10, 160]) * 1e-6  # [m] thickness of the 5 MEA layers
        self.a_ACL = 1e7      # [1/m] electrochemically active area density of the ACL
        self.a_CCL = 3e7      # [1/m] electrochemically active area density of the CCL
        self.H_ec = 42e3      # [J/mol] molar enthalpy of evaporation/condensation
        self.H_ad = self.H_ec  # [J/mol] molar enthalpy of absorption/desorption
        self.k_GDL = 1.6      # [W/(m*K)] GDL thermal conductivity
        self.k_CL = 0.27      # [W/(m*K)] CL thermal conductivity
        self.k_PEM = 0.3      # [W/(m*K)] PEM thermal conductivity
        self.s_im = self.s_C  # [-] immobile liquid water saturation
        self.V_m = 1.02 / 1.97e3  # [m^3/mol] molar volume of dry membrane
        self.rho_liq = 0.978e3    # [kg/m^3] liquid water density
        self.V_w = self.M_H2O / self.rho_liq  # [m^3/mol] molar volume of liquid water
        self.mu_gas = 2.1e-5   # [Pa*s] gas dynamic viscosity
        self.eps_i_CL = 0.3    # [-] ionomer volume fraction in the dry CL
        self.eps_p_GDL = 0.76  # [-] GDL porosity
        self.eps_p_CL = 0.4    # [-] CL porosity
        self.kappa_GDL = 6.15e-12  # [m^2] GDL absolute permeability
        self.kappa_CL = 1e-13      # [m^2] CL absolute permeability
        self.sigma_e_GDL = 1250    # [S/m] GDL electrical conductivity
        self.sigma_e_CL = 350      # [S/m] CL electrical conductivity
        self.tau_GDL = 1.6     # [-] GDL pore tortuosity
        self.tau_CL = 1.6      # [-] CL pore tortuosity

        # Cumulative mesh of layer interface positions
        self.Lsum = np.concatenate(([0.0], np.cumsum(self.L)))
        self.Nd = len(self.L)  # number of domains (5)

        # ---------------------------------------------------------------
        # Quantities derived from the gas-channel boundary conditions
        # (depend only on T_A, T_C, RH_A, RH_C, P_A, P_C, alpha_O2)
        # ---------------------------------------------------------------
        x_H2O_A = self.RH_A * self.P_sat(self.T_A) / self.P_A
        self.w_H2O_A = 1.0 / (self.M_H2 / self.M_H2O * (1.0 / x_H2O_A - 1.0) + 1.0)
        x_H2O_C = self.RH_C * self.P_sat(self.T_C) / self.P_C
        x_O2_C = self.alpha_O2 * (1.0 - x_H2O_C)
        self.w_H2O_C = x_H2O_C * self.M_H2O / (
            self.M_N2 - x_O2_C * (self.M_N2 - self.M_O2) - x_H2O_C * (self.M_N2 - self.M_H2O)
        )
        self.w_O2_C = x_O2_C * self.M_O2 / (x_H2O_C * self.M_H2O) * self.w_H2O_C
        self.P_liq_C = self.pc(self.s_C) + self.P_C

    # =====================================================================
    # WATER CONSTITUTIVE RELATIONSHIPS
    # =====================================================================
    def P_sat(self, T):
        """[Pa] water vapor saturation pressure."""
        return np.exp(23.1963 - 3816.44 / (T - 46.13))

    def mu(self, T):
        """[Pa*s] liquid water dynamic viscosity."""
        return 1e-3 * np.exp(-3.63148 + 542.05 / (T - 144.15))

    # =====================================================================
    # MODEL PARAMETERIZATION
    # =====================================================================
    def A(self, E, T):
        """[-] Arrhenius correction."""
        return np.exp(E / self.R * (1.0 / self.T_ref - 1.0 / T))

    def i_0_HOR(self, T):
        """[A/m^2] HOR exchange current density."""
        return 0.27e4 * self.A(16e3, T)

    def i_0_ORR(self, T, P_O2):
        """[A/m^2] ORR exchange current density."""
        return 2.47e-4 * (P_O2 / self.P_ref) ** 0.54 * self.A(67e3, T)

    def BV(self, i_0, a, T, beta, eta):
        """[A/m^3] Butler-Volmer equation."""
        return i_0 * a * (
            np.exp(beta * 2 * self.F / (self.R * T) * eta)
            - np.exp(-(1 - beta) * 2 * self.F / (self.R * T) * eta)
        )

    def D(self, eps_p, tau, s, P, T):
        """[-] scaling factor for gas-phase diffusivities."""
        return eps_p / tau ** 2 * (1 - s) ** 3 * (T / self.T_ref) ** 1.5 * (self.P_ref / P)

    def D_O2(self, eps_p, tau, s, T, P):
        """[m^2/s] oxygen diffusion coefficient."""
        return 0.28e-4 * self.D(eps_p, tau, s, P, T)

    def D_H2O_A(self, eps_p, tau, s, T, P):
        """[m^2/s] water vapor diffusion coefficient on the anode side."""
        return 1.24e-4 * self.D(eps_p, tau, s, P, T)

    def D_H2O_C(self, eps_p, tau, s, T, P):
        """[m^2/s] water vapor diffusion coefficient on the cathode side."""
        return 0.36e-4 * self.D(eps_p, tau, s, P, T)

    def f(self, lam):
        """[-] water volume fraction in the ionomer."""
        return lam * self.V_w / (self.V_m + lam * self.V_w)

    # =====================================================================
    # MATERIAL CONSTITUTIVE RELATIONSHIPS
    # =====================================================================
    def k_ad(self, lam, lambda_eq, T):
        """[m/s] mass transfer coefficient for vapor sorption in Nafion."""
        return np.where(lam < lambda_eq, 3.53e-5, 1.42e-4) * self.f(lam) * self.A(20e3, T)

    def s_red(self, s):
        """[-] reduced liquid water saturation."""
        return (s - self.s_im) / (1 - self.s_im)

    def gamma_ec(self, x_H2O, x_sat, s, T):
        """[1/s] evaporation/condensation rate (kept for completeness;
        its use is discarded in the original model, see the note in
        model.py)."""
        return (
            2e6
            * np.where(x_H2O < x_sat, 5e-4 * self.s_red(s), 6e-3 * (1 - self.s_red(s)))
            * np.sqrt(self.R * T / (2 * np.pi * self.M_H2O))
        )

    def sigma_p(self, eps_i, lam, T):
        """[S/m] Nafion proton conductivity."""
        return eps_i ** 1.5 * 116 * np.maximum(0.0, self.f(lam) - 0.06) ** 1.5 * self.A(15e3, T)

    def sorption(self, RH):
        """[-] Nafion vapor sorption isotherm."""
        return 0.043 + 17.81 * RH - 39.85 * RH ** 2 + 36.0 * RH ** 3

    def D_lambda(self, eps_i, lam, T):
        """[m^2/s] water diffusion coefficient in Nafion."""
        return (
            eps_i ** 1.5
            * (3.842 * lam ** 3 - 32.03 * lam ** 2 + 67.74 * lam)
            / (lam ** 3 - 2.115 * lam ** 2 - 33.013 * lam + 103.37)
            * 1e-10
            * self.A(20e3, T)
        )

    def xi(self, lam):
        """[-] Nafion electro-osmotic drag coefficient."""
        return 2.5 * lam / 22

    def pc(self, s):
        """[Pa] GDL capillary pressure-saturation relation."""
        return -0.00011 * np.exp(-44.02 * (s - 0.496)) + 278.3 * np.exp(8.103 * (s - 0.496)) - 191.8

    def kappa_rel_liq(self, s):
        """[-] relative permeability of the liquid phase."""
        return np.maximum(1e-6, s)
