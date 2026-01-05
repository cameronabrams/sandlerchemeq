# Author: Cameron F. Abrams, <cfa22@drexel.edu>

"""
Chemical equilibrium system solver
"""

import roman

import numpy as np

from dataclasses import dataclass, field
from scipy.optimize import fsolve

from sandlermisc.gas_constant import GasConstant
from sandlerprops.properties import get_database

from .reaction import Reaction
from .component import Component

@dataclass
class ChemEqSystem:
    """
    Chemical equilibrium system solver using either explicit reactions
    with equilibrium constants or implicit Lagrange multiplier method.
    """
    Pstdst = 1.0 # bar
    """ Standard state pressure in bar """
    T0 = 298.15 # K
    """ Standard state temperature in K """
    P: float = 1.0
    """ System pressure in bar """
    T: float = 298.15
    """ System temperature in K """

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
        self.C = len(self.components)
        self.R = GasConstant() # J/mol.K
        self.RT = self.R * self.T
        self.M = len(self.reactions)
        for c in self.components:
            c.T = self.T
            c.P = self.P
            c.Tref = self.T0
        if self.M > 0:
            ''' Explicit reactions are specified; will use equilibrium constants
                and extents of reaction to solve '''
            self.nu = []
            self.dGr = np.array([])
            self.dHr = np.array([])
            self.dCp = np.array([])
            self.nu = np.zeros((self.M, self.C))
            for i, r in enumerate(self.reactions):
                self.dGr = np.append(self.dGr, r.stoProps['dGf'])
                self.dHr = np.append(self.dHr, r.stoProps['dHf'])
                self.dCp = np.append(self.dCp, r.stoProps['Cp'])
                for c in self.components:
                    if c in r.components:
                        ci = self.components.index(c)
                        nu = r.nu[r.components.index(c)]
                        self.nu[i][ci] = nu
            self.Ka0 = np.exp(-self.dGr/(self.R*self.T0))
            # use full van't hoff equation to get Ka at T
            arg1 = self.dCp[0]/self.R * np.log(self.T/self.T0)
            arg2 = self.dCp[1]/(2*self.R) * (self.T - self.T0)
            arg3 = self.dCp[2]/(6*self.R) * (self.T**2 - self.T0**2)
            arg4 = self.dCp[3]/(12*self.R) * (self.T**3 - self.T0**3)
            bigterm1 = arg1 + arg2 + arg3 + arg4
            rtdiff = 1/self.R*(1/self.T - 1/self.T0)
            term5 = -self.dHr
            term6 = self.dCp[0] * self.T0
            term7 = self.dCp[1] / 2 * self.T0**2
            term8 = self.dCp[2] / 3 * self.T0**3
            term9 = self.dCp[3] / 4 * self.T0**4
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

    def solve_implicit(self, Xinit=[], ideal=True, simplified=True):
        """
        Implicit solution of M equations using equilibrium constants. Solutions
        are stored in attributes **Xeq**, **N**, and **ys**.
        
        Parameters
        ----------
        Xinit : list, optional
            Initial guess for extent of reaction (default is [])
        ideal : bool, optional
            Whether to assume ideal behavior (default is True)
        simplified : bool, optional
            Whether to use simplified van't Hoff equation (default is True)
        """
        if self.M == 0:
            raise ValueError('No reactions specified for implicit solution.')
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
            Ka_app = [np.prod(y**nu_j)*np.prod(phi**nu_j)*(self.P/self.Pstdst)**sum(nu_j) for nu_j in self.nu]
            # print(y,Ka_app)
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
            # compute total moles N by summing over mole numbers; 
            for k in range(self.E):
                # compute constant number of atom-moles, A[]
                self.A[k] += self.N0[i] * c.countAtoms(self.atomlist[k])
        def f_func(z):
            F = np.zeros(self.C + self.E)
            N = 0.0
            for i in range(self.C):
                # compute total moles N by summing over mole numbers; 
                N += z[i]
            # stub:  phi values are all ones
            phi = np.ones(self.C)
            for i in range(self.C):
                # Computed Gibbs energy for each molecular species...
                dGfoT = self.components[i].dGf_T
                F[i] = dGfoT/self.RT + np.log(z[i]/N * phi[i] * self.P/self.Pstdst)
                for k in range(self.E):
                    # sum up Lagrange multiplier terms from each atom-balance
                    F[i] += z[self.C+k]/self.RT * self.components[i].countAtoms(self.atomlist[k])
                    # sum up each atom balance
                    F[k+self.C] += z[i] * self.components[i].countAtoms(self.atomlist[k])
            for k in range(self.E):
                # close each atom balance
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
            result += f'  |  Ka({self.T:.2f} K)={k:.5e} => Xeq={x:.5e}' 
            result += '\n'
    for i,(c,N,y) in enumerate(zip(self.components,self.N,self.ys)):
        result += f'N_{{{str(c)}}}={N:.4f} y_{{{str(c)}}}={y:.4f}' 
        result += '\n'
    return result