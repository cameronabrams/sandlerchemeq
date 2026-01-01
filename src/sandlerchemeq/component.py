from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

from sandlermisc.texutils import format_sig, Cp_as_tex
from sandlerprops.compound import Compound
from sandlerprops.properties import PropertiesDatabase, get_database

@dataclass
class Component(Compound):
    """ 
    A chemical component with thermochemical and system properties
    based on the **Compound** class of `sandlerprops`.
    """

    T: float = 298.15
    """ Temperature in K """
    P: float = 1.0
    """ Pressure in bar """
    Tref: float = 298.15
    """ Reference temperature in K """

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
    def dGf_T(self):
        """
        Ideal-gas Gibbs energy of formation at temperature T
        """
        go = self.dGf
        ho = self.dHf
        Tref = self.Tref
        T = self.T # current temperature of component/system
        dGft = go * T / Tref + ho * (1 - T / Tref) + self._cpI - T * self._cpTI
        if isinstance(dGft, np.float64):
            return dGft.item()
        return dGft

    @property
    def _cpI(self):
        cp = self.Cp
        TL = (self.Tref, self.T)
        return sum([cp[i]/(i+1)*(TL[1]**(i+1)-TL[0]**(i+1)) for i in range(len(cp))])
    @property
    def _cpTI(self):
        cp = self.Cp
        TL = (self.Tref, self.T)
        return cp[0]*np.log(TL[1]/TL[0])+sum([cp[i]/i*(TL[1]**i-TL[0]**i) for i in range(1,len(cp))])

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

    def as_tex(self):
        """
        Returns a LaTeX-formatted string of the compound's empirical formula
        using the mhchem package's \\ce{} command.
        
        Returns
        -------
        retstr : str
            LaTeX-formatted empirical formula string
        """
        return r'\ce{'+str(self)+r'}'

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

