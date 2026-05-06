# Author: Cameron F. Abrams, <cfa22@drexel.edu>

"""
Chemical equilibrium system solver
"""

import roman
import pint

import numpy as np

from dataclasses import dataclass, field
from scipy.optimize import fsolve

from sandlermisc import R, ureg
from sandlerprops.properties import get_database

from .reaction import Reaction
from .component import Component

@dataclass
class ChemEqSystem:
    """
    Chemical equilibrium system solver using either explicit reactions
    with equilibrium constants or implicit Lagrange multiplier method.
    """
    Pstdst: pint.Quantity = field(default=1.0 * ureg.bar, init=False, repr=False)
    """ Standard state pressure """
    T0: pint.Quantity = field(default=298.15 * ureg.K, init=False, repr=False)
    """ Standard state temperature """
    P: pint.Quantity = field(default_factory=lambda: 1.0 * ureg.bar)
    """ System pressure """
    T: pint.Quantity = field(default_factory=lambda: 298.15 * ureg.K)
    """ System temperature """

    components: list[Component] = field(default_factory=list)
    """ List of all components in the system """
    N0: np.ndarray = field(default_factory=lambda: np.array([]))
    """ Initial moles of each component """
    reactions: list[Reaction] = field(default_factory=list)
    """ List of explicit reactions in the system """

    N: np.ndarray = field(default_factory=lambda: np.array([]))
    """ Moles of each component at equilibrium """
    ys: np.ndarray = field(default_factory=lambda: np.array([]))
    """ Mole fractions of each component at equilibrium """

    def __post_init__(self):
        if not hasattr(self.T, 'units'):
            self.T = self.T * ureg.K
        if not hasattr(self.P, 'units'):
            self.P = self.P * ureg.bar
        self.C = len(self.components)
        self.R = R                              # pint.Quantity  J/(mol·K)
        self.RT = (self.R * self.T).to('J/mol') # pint.Quantity  J/mol
        self.M = len(self.reactions)
        for c in self.components:
            c.T = self.T
            c.P = self.P
            c.Tref = self.T0
        if self.M > 0:
            ''' Explicit reactions are specified; will use equilibrium constants
                and extents of reaction to solve '''
            self.nu = np.zeros((self.M, self.C))
            dGr_vals, dHr_vals = [], []
            self.dCp = np.array([])
            for i, r in enumerate(self.reactions):
                dGr_vals.append(r.stoProps['dGf'].m_as('J/mol') if hasattr(r.stoProps['dGf'], 'm_as') else r.stoProps['dGf'])
                dHr_vals.append(r.stoProps['dHf'].m_as('J/mol') if hasattr(r.stoProps['dHf'], 'm_as') else r.stoProps['dHf'])
                self.dCp = np.append(self.dCp, r.stoProps['Cp'])
                for c in self.components:
                    if c in r.components:
                        ci = self.components.index(c)
                        nu = r.nu[r.components.index(c)]
                        self.nu[i][ci] = nu
            self.dGr = np.array(dGr_vals) * ureg.J / ureg.mol  # pint.Quantity
            self.dHr = np.array(dHr_vals) * ureg.J / ureg.mol  # pint.Quantity

            # van't Hoff: extract SI magnitudes for numerical computation
            R_val   = self.R.m_as('J/(mol*K)')
            T_val   = self.T.m_as('K')
            T0_val  = self.T0.m_as('K')
            dGr_val = self.dGr.m_as('J/mol')
            dHr_val = self.dHr.m_as('J/mol')

            self.Ka0 = np.exp(-dGr_val / (R_val * T0_val))
            # full van't Hoff equation to get Ka at T
            arg1 = self.dCp[0]/R_val * np.log(T_val/T0_val)
            arg2 = self.dCp[1]/(2*R_val) * (T_val - T0_val)
            arg3 = self.dCp[2]/(6*R_val) * (T_val**2 - T0_val**2)
            arg4 = self.dCp[3]/(12*R_val) * (T_val**3 - T0_val**3)
            bigterm1 = arg1 + arg2 + arg3 + arg4
            rtdiff = 1/R_val*(1/T_val - 1/T0_val)
            term5 = -dHr_val
            term6 = self.dCp[0] * T0_val
            term7 = self.dCp[1] / 2 * T0_val**2
            term8 = self.dCp[2] / 3 * T0_val**3
            term9 = self.dCp[3] / 4 * T0_val**4
            bigterm2 = rtdiff * (term5 + term6 + term7 + term8 + term9)
            logratio = bigterm1 + bigterm2
            self.KaT = self.Ka0 * np.exp(logratio)
            self.KaT_simplified = self.Ka0 * np.exp(rtdiff * term5)
            self.Xeq = np.zeros(self.M)
            # store intermediate van't Hoff terms for reporting later if needed
            self.vantHoff_terms = {}
            for t in ['arg1','arg2','arg3','arg4',
                      'rtdiff','term5','term6','term7','term8','term9', 'logratio', 'bigterm1', 'bigterm2']:
                self.vantHoff_terms[t] = eval(t)

    def solve_implicit(self, Xinit=None, ideal=True, simplified=False):
        """
        Implicit solution of M equations using equilibrium constants. Solutions
        are stored in attributes **Xeq**, **N**, and **ys**.

        Parameters
        ----------
        Xinit : list, optional
            Initial guess for extent of reaction (default is zeros)
        ideal : bool, optional
            Whether to assume ideal behavior (default is True)
        simplified : bool, optional
            Whether to use simplified van't Hoff equation (default is False).
            Use ``False`` (full van't Hoff with Cp correction) to match the
            Lagrange solver.  ``True`` omits the heat-capacity correction and
            is provided for educational comparison only.
        """
        if self.M == 0:
            raise ValueError('No reactions specified for implicit solution.')
        if Xinit is None:
            Xinit = np.zeros(self.M)
        P_ratio = (self.P / self.Pstdst).m_as('')   # dimensionless plain float
        def _NX(X):
            """ Numbers of moles from extent of reaction """
            return self.N0 + np.dot(X, self.nu)
        def _YX(X):
            """ Mole fractions from numbers of moles """
            n = _NX(X)
            return n / sum(n)
        def f_func(X):
            """
            enforces equality of given and apparent equilibrium constants
            by solving for extents of reaction

            Parameters
            ----------
            X : np.ndarray
                extent of reaction guesses

            Returns
            -------
            np.ndarray
                residuals of equilibrium constant equations
            """
            y = _YX(X)
            phi = np.ones(self.C)
            KaT = self.KaT_simplified if simplified else self.KaT
            if not ideal:
                pass
                # to do -- fugacity coefficient calculation
            Ka_app = [np.prod(y**nu_j)*np.prod(phi**nu_j)*(P_ratio)**sum(nu_j) for nu_j in self.nu]
            return np.array([(kk-ka)/(kk+ka) for kk,ka in zip(Ka_app, KaT)])
        self.Xeq = fsolve(f_func, Xinit)
        self.N = _NX(self.Xeq)
        self.ys = _YX(self.Xeq)

    def solve_lagrange(self, ideal: bool = True, zInit: np.ndarray = []):
        """
        Implicit solution of chemical equilibrium system using
        Lagrange multipliers. Solutions are stored in attributes
        **N** and **ys**.

        Parameters
        ----------
        ideal : bool, optional
            Whether to assume ideal behavior (default is True) (NOT USED)
        zInit : np.ndarray, optional
            Initial guess for mole numbers and Lagrange multipliers (default is [])
        """
        atomset = set()
        for c in self.components:
            atomset.update(c.atomset)
        self.atomlist = list(atomset)
        self.E = len(self.atomlist)
        self.A = np.zeros(self.E)
        for i, c in enumerate(self.components):
            for k in range(self.E):
                self.A[k] += self.N0[i] * c.countAtoms(self.atomlist[k])
        RT_val  = self.RT.m_as('J/mol')          # plain float for fsolve
        P_ratio = (self.P / self.Pstdst).m_as('') # plain float
        def f_func(z):
            F = np.zeros(self.C + self.E)
            N = 0.0
            for i in range(self.C):
                N += z[i]
            phi = np.ones(self.C)
            for i in range(self.C):
                dGfoT_val = self.components[i].dGf_T.m_as('J/mol')
                F[i] = dGfoT_val/RT_val + np.log(z[i]/N * phi[i] * P_ratio)
                for k in range(self.E):
                    F[i] += z[self.C+k]/RT_val * self.components[i].countAtoms(self.atomlist[k])
                    F[k+self.C] += z[i] * self.components[i].countAtoms(self.atomlist[k])
            for k in range(self.E):
                F[k+self.C] -= self.A[k]
            return F
        zGuess = zInit
        if len(zGuess) == 0:
            zGuess = np.array([0.1]*self.C + [1000.]*self.E)
        z = fsolve(f_func, zGuess)
        self.N = z[:self.C]
        self.ys = self.N / sum(self.N)

    def report(self) -> str:
        """
        Generates a textual report of the chemical equilibrium system.

        Returns
        -------
        str
            Textual report of reactions and mole fractions at equilibrium
        """
        result = ''
        if len(self.reactions)>0:
            for i,(r,k,x) in enumerate(zip(self.reactions,self.KaT,self.Xeq)):
                result += f'Reaction {roman.toRoman(i+1):>4s}:  '
                result += str(r)
                result += f'  |  Ka({self.T:.2f~P})={k:.5e} => Xeq={x:.5e}'
                result += '\n'
        for i,(c,N,y) in enumerate(zip(self.components,self.N,self.ys)):
            result += f'N_{{{str(c)}}}={N:.4f} y_{{{str(c)}}}={y:.4f}'
            result += '\n'
        return result
