"""Layout of the stacked state vector.

The boundary value problem is solved for all five MEA layers at once, as a
single flat vector of ``N_REGIONS * N_STATE`` entries. This module gives that
layout names: which physical quantity lives in which row of a region block,
and which regions each quantity is physically defined on.

The enums are ``IntEnum`` subclasses, so members index numpy arrays directly
(``block[State.W_O2]``) while still printing as names when debugging.
"""
from __future__ import annotations

from enum import IntEnum


class Region(IntEnum):
    """The five MEA layers, ordered anode to cathode."""

    AGDL = 0   # anode gas diffusion layer
    ACL = 1    # anode catalyst layer
    PEM = 2    # polymer electrolyte membrane
    CCL = 3    # cathode catalyst layer
    CGDL = 4   # cathode gas diffusion layer


class State(IntEnum):
    """Rows of one region block: each quantity followed by its flux."""

    PHI_E = 0          # electron potential
    J_E = 1            # electron flux
    PHI_P = 2          # proton potential
    J_P = 3            # proton flux
    T = 4              # temperature
    J_T = 5            # heat flux
    LAMBDA = 6         # dissolved water content
    J_LAMBDA = 7       # dissolved water flux
    W_H2O = 8          # water vapour mass fraction
    J_H2O = 9          # water vapour flux
    W_O2 = 10          # oxygen mass fraction
    J_O2 = 11          # oxygen flux
    P_LIQ = 12         # liquid water pressure
    RHO_U_LIQ = 13     # liquid water mass flux
    P_GAS = 14         # gas pressure
    RHO_U_GAS = 15     # gas mass flux


class Quantity(IntEnum):
    """The eight second-order quantities, each owning a ``State`` pair."""

    PHI_E = 0
    PHI_P = 1
    T = 2
    LAMBDA = 3
    W_H2O = 4
    W_O2 = 5
    SATURATION = 6
    P_GAS = 7

    @property
    def value_row(self) -> State:
        """Row holding the quantity itself."""
        return State(2 * self)

    @property
    def flux_row(self) -> State:
        """Row holding the quantity's flux."""
        return State(2 * self + 1)

    @property
    def rows(self) -> slice:
        """Both rows, as a slice into a region block."""
        return slice(self.value_row, self.flux_row + 1)


N_REGIONS = len(Region)
N_STATE = len(State)
N_QUANTITIES = len(Quantity)
N_TOTAL = N_REGIONS * N_STATE

#: Regions on which each quantity is physically defined. Outside these the
#: quantity has no meaning and post-processing masks it with NaN.
ACTIVE_REGIONS: dict[Quantity, tuple[Region, ...]] = {
    Quantity.PHI_E: (Region.AGDL, Region.ACL, Region.CCL, Region.CGDL),
    Quantity.PHI_P: (Region.ACL, Region.PEM, Region.CCL),
    Quantity.T: tuple(Region),
    Quantity.LAMBDA: (Region.ACL, Region.PEM, Region.CCL),
    Quantity.W_H2O: (Region.AGDL, Region.ACL, Region.CCL, Region.CGDL),
    Quantity.W_O2: (Region.CCL, Region.CGDL),
    Quantity.SATURATION: (Region.CCL, Region.CGDL),
    Quantity.P_GAS: (Region.AGDL, Region.ACL, Region.CCL, Region.CGDL),
}


def block_rows(region: Region) -> slice:
    """Rows of the stacked vector belonging to ``region``."""
    return slice(region * N_STATE, (region + 1) * N_STATE)


def stacked_index(state: State, region: Region) -> int:
    """Index of a single ``state`` of ``region`` in the flat stacked vector."""
    return int(region) * N_STATE + int(state)
