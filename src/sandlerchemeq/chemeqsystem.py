# Author: Cameron F. Abrams, <cfa22@drexel.edu>

"""
Chemical equilibrium system solver
"""

import roman

import numpy as np

from dataclasses import dataclass, field
from scipy.optimize import fsolve

from sandlermisc.texutils import table_as_tex, format_sig
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

    Components: list[Component] = field(default_factory=list)
    """ List of all components in the system """
    N0: np.ndarray = field(default_factory=lambda: np.array([]))
    """ Initial moles of each component """
    Reactions: list[Reaction] = field(default_factory=list)
    """ List of explicit reactions in the system """

    N: np.ndarray = field(default_factory=lambda: np.array([]))
    """ Moles of each component at equilibrium """
    ys: np.ndarray = field(default_factory=lambda: np.array([]))
    """ Mole fractions of each component at equilibrium """

    def __post_init__(self):
        self.C = len(self.Components)
        self.R = GasConstant() # J/mol.K
        self.RT = self.R * self.T
        self.M = len(self.Reactions)
        for c in self.Components:
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
            for i, r in enumerate(self.Reactions):
                self.dGr = np.append(self.dGr, r.stoProps['dGf'])
                self.dHr = np.append(self.dHr, r.stoProps['dHf'])
                self.dCp = np.append(self.dCp, r.stoProps['Cp'])
                for c in self.Components:
                    if c in r.components:
                        ci = self.Components.index(c)
                        nu = r.nu[r.components.index(c)]
                        self.nu[i][ci] = nu
            self.Ka0 = np.exp(-self.dGr/(self.R*self.T0))
            # use full van't hoff equation to get Ka at T
            arg1 = self.dCp[0]/self.R * np.log(self.T/self.T0)
            arg2 = self.dCp[1]/(2*self.R) * (self.T - self.T0)
            arg3 = self.dCp[2]/(6*self.R) * (self.T**2 - self.T0**2)
            arg4 = self.dCp[3]/(12*self.R) * (self.T**3 - self.T0**3)
            rtdiff = 1/self.R*(1/self.T - 1/self.T0)
            term5 = -self.dHr
            term6 = self.dCp[0] * self.T0
            term7 = self.dCp[1] / 2 * self.T0**2
            term8 = self.dCp[2] / 3 * self.T0**3
            term9 = self.dCp[3] / 4 * self.T0**4
            arg5 = rtdiff * (term5 + term6 + term7 + term8 + term9)
            logratio = arg1 + arg2 + arg3 + arg4 + arg5
            self.KaT = self.Ka0 * np.exp(logratio)
            self.Xeq = np.zeros(self.M)

    def solve_implicit(self, Xinit=[], ideal=True):
        """
        Implicit solution of M equations using equilibrium constants. Solutions
        are stored in attributes **Xeq**, **N**, and **ys**.
        
        Parameters
        ----------
        Xinit : list, optional
            Initial guess for extent of reaction (default is [])
        ideal : bool, optional
            Whether to assume ideal behavior (default is True)
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
            if not ideal:
                pass
                # to do -- fugacity coefficient calculation
            Ka_app = [np.prod(y**nu_j)*np.prod(phi**nu_j)*(self.P/self.Pstdst)**sum(nu_j) for nu_j in self.nu]
            # print(y,Ka_app)
            return np.array([(kk-ka)/(kk+ka) for kk,ka in zip(Ka_app,self.KaT)])
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
        for c in self.Components:
            atomset.update(c.atomset)
        self.atomlist = list(atomset)
        self.E = len(self.atomlist)
        self.A = np.zeros(self.E)
        for i, c in enumerate(self.Components):
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
                dGfoT = self.Components[i].dGf_T
                F[i] = dGfoT/self.RT + np.log(z[i]/N * phi[i] * self.P/self.Pstdst)
                for k in range(self.E):
                    # sum up Lagrange multiplier terms from each atom-balance
                    F[i] += z[self.C+k]/self.RT * self.Components[i].countAtoms(self.atomlist[k])
                    # sum up each atom balance
                    F[k+self.C] += z[i] * self.Components[i].countAtoms(self.atomlist[k])
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

    def texgen_kacalculations(self, simplified=True, sig=5) -> str:
        """ 
        Generates LaTeX string for equilibrium constant calculations.
        
        Parameters
        ----------
        simplified : bool, optional
            whether to use simplified van't Hoff equation (default is True)
        sig : int, optional
            number of significant figures for formatting (default is 5)
            
        Returns
        -------
        str
            LaTeX formatted string of equilibrium constant calculations
        """
        ka_calcslines = []
        for i in range(self.M):
            rxn_super = ''
            if self.M > 1:
                ka_calcslines.append(rf'\underline{{Reaction {roman.toRoman(i+1)}}}\\')
                rxn_super = f'^{{{roman.toRoman(i+1)}}}'
            ka_calcslines.append(r'\begin{align*}')
            ka_calcslines.append(r'K_a'+rxn_super+r'(T_0) & = \exp\left[\frac{-\gr'+rxn_super+r'}{RT_0}\right] ')
            ka_calcslines.append(r' = \exp\left[\frac{' + f'-({self.dGr[i]:.0f})' + r'}{' + f'({self.R})({self.T0})' + r'}\right]')
            ka_calcslines.append(r' = ' + f'{format_sig(self.Ka0[i],sig)}' + r'\\')
            if simplified:
                ka_calcslines.append(r'K_a'+rxn_super+r'(T) & = K_a'+rxn_super+r'(T_0) \exp\left[\frac{-\hr'+rxn_super+r'}{R}\left(\frac{1}{T} - \frac{1}{T_0}\right)\right]\\')
                ka_calcslines.append((r'& = ' + f'({format_sig(self.Ka0[i],sig)})' + r'\exp\left[\frac{' + f'-({self.dHr[i]:.0f})' + r'}{'
                                    f'({self.R})' + r'}\left(\frac{1}{' + f'{self.T}' + r'} - \frac{1}{' + f'{self.T0}' + r'}\right)\right]\\'))
            ka_calcslines.append(r'& = ' + f'{format_sig(self.KaT[i],sig)}')
            ka_calcslines.append(r'\end{align*}')
        return '\n'.join(ka_calcslines)

    def report(self) -> str:
        """
        Generates a textual report of the chemical equilibrium system.
        
        Returns
        -------
        str
            Textual report of reactions and mole fractions at equilibrium
        """
        result = ''
        if len(self.Reactions)>0:
            for i,(r,k,x) in enumerate(zip(self.Reactions,self.KaT,self.Xeq)):
                result += f'Reaction {roman.toRoman(i+1):>4s}:  '
                result += str(r)
                result += f'  |  Ka({self.T:.2f} K)={k:.5e} => Xeq={x:.5e}' 
                result += '\n'
        for i,(c,N,y) in enumerate(zip(self.Components,self.N,self.ys)):
            result += f'N_{{{str(c)}}}={N:.4f} y_{{{str(c)}}}={y:.4f}' 
            result += '\n'
        return result
    
    def stoichiometrictable_as_tex(self, float_format='{:.3f}'):
        """
        Generates LaTeX formatted stoichiometric table.
        
        Parameters
        ----------
        float_format : str, optional
            format string for floating point numbers (default is '{:.3f}')
        
        Returns
        -------
        str
            LaTeX formatted stoichiometric table
        """
        total_row = ['Totals:']
        col1 = [c.as_tex() for c in self.compounds]
        fstr = float_format
        col2 = [r'\('+fstr.format(n)+r'\)' for n in self.N0] 
        total_row += [r'\('+fstr.format(self.N0.sum())+r'\)']
        tabledict = {'Species': col1, r'$N_{0,i}$': col2}
        tabledict[f'$N_{{i}}$'] = []
        Nus = [self.nu[j].sum() for j in range(self.M)]
        Nis = []
        for i,c in enumerate(self.compounds):
            Ni = fstr.format(self.N0[i])
            for j,r in enumerate(self.Reactions):
                nu_sign = [' + ' if self.nu[j][i]>=0 else ' - ' for i in range(self.C)]
                Xj = f'X_{{{roman.toRoman(j+1)}}}' if self.M>1 else 'X'
                Ni += f'{nu_sign[i]}\\left({np.fabs(self.nu[j][i]):.2f}\\right){Xj}'
            Nis.append(Ni)
            tabledict[f'$N_{{i}}$'].append(r'\('+Ni+r'\)')
        Ni_total = fstr.format(self.N0.sum())
        for j in range(self.M):
            nu_sign = ' + ' if Nus[j]>=0 else ' - '
            Xj = f'X_{{{roman.toRoman(j+1)}}}' if self.M>1 else 'X'
            Ni_total += f'{nu_sign}\\left({np.fabs(Nus[j]):.2f}\\right){Xj}'
        total_row += [r'\('+Ni_total+r'\)']
        tabledict[f'$y_{{i}} = N_{{i}} / \\sum_{{i}} N_{{i}}$'] = []
        for i in range(self.C):
            yi = r'\left[' + Nis[i] + r'\right] / \left[' + Ni_total + r'\right]'
            tabledict[f'$y_{{i}} = N_{{i}} / \\sum_{{i}} N_{{i}}$'].append(r'\('+yi+r'\)')
        total_row += [r'\(1.0\)']
        return table_as_tex(tabledict, float_format=float_format, total_row=total_row, index=False)
    
    def thermochemicaltable_as_tex(self, float_format=r'\({:,.0f}\)'):
        """
        Generates LaTeX formatted thermochemical data table.
        
        Parameters
        ----------
        float_format : str, optional
            format string for floating point numbers (default is {:,.0f}')
        
        Returns
        -------
        str
            LaTeX formatted thermochemical data table
        """
        return table_as_tex({
            'Species':[c.as_tex() for c in self.compounds],
            r'$\hf$ (J/mol)':[c.thermoChemicalData['H'] for c in self.compounds],
            r'$\gf$ (J/mol)':[c.thermoChemicalData['G'] for c in self.compounds]},
            drop_zeros=[False,True,True], float_format=float_format.format)
    