# Author: Cameron F. Abrams, <cfa22@drexel.edu>
"""
Reaction class for chemical reactions
"""

import numpy as np
import fractions as fr
from dataclasses import dataclass, field
from scipy.linalg import null_space
from .component import Component

@dataclass
class Reaction:
    """
    A chemical reaction consisting of reactant and product Components.
    Automatically balances the reaction upon initialization.
    """
    components: list[Component] = field(default_factory=list[Component])
    """ List of all components in the reaction (reactants + products) """
    nu: np.ndarray = field(default_factory=lambda: np.array([]))
    """ Stoichiometric coefficients (negative for reactants, positive for products) """
    stoProps: dict = field(default_factory=dict)
    """ Dictionary of stoichiometric property sums """

    def __post_init__(self):
        if len(self.components) > 0:
            self._balance()
            self._computeStoSums()
        else:
            print('Empty Reaction created.')

    def _balance(self):
        ''' Uses nullspace of (element)x(count-in-molecule) matrix to balance reaction '''
        self.nComponents = len(self.components)
        atomset = set()
        for c in self.components:
            atomset.update(c.atomset)
        self.atomList = list(atomset)
        self.nAtoms = len(self.atomList)
        # make element x count-in-compound matrix
        mat = np.zeros((self.nComponents, self.nAtoms))
        for i, c in enumerate(self.components):
            for j, a in enumerate(self.atomList):
                mat[i][j] = c.countAtoms(a)

        # find its nullspace vector
        ns = null_space(mat.T)
        ns *= np.sign(ns[0, 0])
        # set nu; scale so lowest value is 1 making all integers
        self.nu = -np.array([a[0] for a in ns/min([np.abs(x) for x in ns])])
        # split into reactants and products
        self.nReactants = sum(1 for n in self.nu if n < 0)
        self.nProducts = self.nComponents - self.nReactants
        self.R = self.components[:self.nReactants]
        self.P = self.components[self.nReactants:]

    def stosum(self, propname: str) -> float | np.ndarray:
        """ compute stoichiometric sum of property named propname """
        s = 0
        for i, c in enumerate(self.components):
            s += getattr(c, propname) * self.nu[i]
        return s

    def _computeStoSums(self):
        propNames = ['dGf', 'dHf', 'Cp']
        for p in propNames:
            self.stoProps[p] = self.stosum(p)

    def __str__(self):
        ''' spoof nu if reaction is not yet balanced '''
        nuR=['*']*self.nReactants if len(self.nu)==0 else self.nu[:self.nReactants]
        nuP=['*']*self.nProducts  if len(self.nu)==0 else self.nu[self.nReactants:]
        reactant_string = '  +  '.join([f'{n:.0f} {str(s)}' for n,s in zip(-nuR,self.R)])
        product_string = '  +  '.join([f'{n:.0f} {str(s)}' for n,s in zip(nuP,self.P)])
        return reactant_string+'   <->   '+product_string
    
    def show(self):
        retstr=str(self)+'\n'
        infolist=[]
        for p,v in self.stoProps.items():
            if hasattr(v, '__iter__'):
                infolist.append(f'Δ{p} = ['+', '.join([f'{vv:.4e}' for vv in v])+']')
            else:
                infolist.append(f'Δ{p} = {v:.2f}')
        return retstr+' '.join(infolist)

 



