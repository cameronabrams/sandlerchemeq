from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import pint

from sandlerprops.compound import Compound
from sandlerprops.properties import PropertiesDatabase, get_database
from sandlermisc import ureg

@dataclass
class Component(Compound):
    """
    A chemical component with thermochemical and system properties
    based on the **Compound** class of `sandlerprops`.
    """

    T: pint.Quantity = field(default_factory=lambda: 298.15 * ureg.K)
    """ Temperature """
    P: pint.Quantity = field(default_factory=lambda: 1.0 * ureg.bar)
    """ Pressure """
    Tref: pint.Quantity = field(default_factory=lambda: 298.15 * ureg.K)
    """ Reference temperature """

    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self.T, 'units'):
            self.T = self.T * ureg.K
        if not hasattr(self.P, 'units'):
            self.P = self.P * ureg.bar
        if not hasattr(self.Tref, 'units'):
            self.Tref = self.Tref * ureg.K

    @classmethod
    def from_compound(cls, compound: Compound, **kwargs) -> Component:
        """
        Create a Component instance from a Compound instance

        Parameters
        ----------
        compound : Compound
            An instance of the Compound class

        Returns
        -------
        component : Component
            An instance of the Component class
        """
        data = asdict(compound)
        data.update(kwargs)
        return cls(**data)

    def Cp_polynomial_as_tex(self):
        """
        Returns a LaTeX-formatted string for the compound's heat capacity polynomial

        Returns
        -------
        retstr : str
            LaTeX formatted heat capacity polynomial string
        """
        return Cp_as_tex(self.Cp)

    def CpInt_polynomial_as_tex(self):
        """
        Returns a LaTeX-formatted string for the integral of the compound's heat capacity polynomial

        Returns
        -------
        retstr : str
            LaTeX formatted integral of heat capacity polynomial string
        """
        Cp = self.Cp
        retstr = (f"{Cp[0]:.3f}($T_2-T_1$) + "
                  r'$\frac{1}{2}$('+format_sig(Cp[1], sig=4)+r') ($T_2^2-T_1^2$) + '
                  r'$\frac{1}{3}$('+format_sig(Cp[2], sig=4)+r') ($T_2^3-T_1^3$) + '
                  r'$\frac{1}{4}$('+format_sig(Cp[3], sig=4)+r') ($T_2^4-T_1^4$)')
        return(retstr)

    @property
    def _cpI(self):
        """∫Cp dT from Tref to T (J/mol)"""
        cp = self.Cp
        T_K, Tref_K = self.T.m_as('K'), self.Tref.m_as('K')
        val = sum([cp[i]/(i+1)*(T_K**(i+1)-Tref_K**(i+1)) for i in range(len(cp))])
        return val * ureg.J / ureg.mol

    @property
    def _cpTI(self):
        """∫(Cp/T) dT from Tref to T (J/(mol·K))"""
        cp = self.Cp
        T_K, Tref_K = self.T.m_as('K'), self.Tref.m_as('K')
        val = cp[0]*np.log(T_K/Tref_K)+sum([cp[i]/i*(T_K**i-Tref_K**i) for i in range(1,len(cp))])
        return val * ureg.J / (ureg.mol * ureg.K)

    @property
    def dGf_T(self) -> pint.Quantity:
        """
        Ideal-gas Gibbs energy of formation at temperature T
        """
        T_rat = (self.T / self.Tref).m_as('')   # dimensionless ratio as plain float
        return (self.dGf * T_rat
                + self.dHf * (1 - T_rat)
                + self._cpI
                - self.T * self._cpTI).to('J/mol')

    def __eq__(self, other: Compound):
        """
        Compounds are equal if their empirical formulas are identical
        """
        return self.Formula == other.Formula

    def __hash__(self):
        """
        Hash based on object id
        """
        return id(self)

    def __str__(self):
        return self.Formula.split('^')[0] + ('' if self.charge==0 else r'^{'+f'{self.charge:+}'+r'}')

    def countAtoms(self, a: str) -> int:
        """
        Returns the count of atom a in the compound

        Parameters
        ----------
        a : str
            atom name

        Returns
        -------
        count : int
            Number of atoms of type a in the compound
        """
        return self.atomdict.get(a, 0)
