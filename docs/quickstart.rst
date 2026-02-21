.. _quickstart:

Quick Start Guide
=================

This guide walks through typical use of **sandlerchemeq** for both solution
methods: Lagrange multipliers and explicit equilibrium constants.

All temperatures are in Kelvin (K), pressures in bar, and molar amounts are
dimensionless mole numbers.  Thermodynamic quantities (Gibbs energies,
enthalpies) are expressed as :class:`pint.Quantity` objects in J/mol.

Building Blocks
---------------

Components
~~~~~~~~~~

A :class:`~sandlerchemeq.component.Component` is a species in the equilibrium
system.  It wraps a :class:`~sandlerprops.compound.Compound` from the
``sandlerprops`` database and adds temperature- and pressure-dependent
thermodynamic properties.

.. code-block:: python

   from sandlerchemeq import Component
   from sandlerprops.properties import get_database

   db = get_database()
   ammonia = Component.from_compound(db.get_compound('ammonia'), T=400.0, P=100.0)

   print(ammonia.dGf)       # standard-state ΔGf at 298.15 K (pint.Quantity, J/mol)
   print(ammonia.dGf_T)     # ideal-gas ΔGf at T=400 K (pint.Quantity, J/mol)

Reactions
~~~~~~~~~

A :class:`~sandlerchemeq.reaction.Reaction` is automatically balanced from an
ordered list of components (reactants first, products last).  The null-space of
the element-count matrix determines the stoichiometric coefficients.

.. code-block:: python

   from sandlerchemeq import Component, Reaction
   from sandlerprops.properties import get_database

   db = get_database()
   T, P = 400.0, 100.0
   hydrogen  = Component.from_compound(db.get_compound('hydrogen (equilib)'), T=T, P=P)
   nitrogen  = Component.from_compound(db.get_compound('nitrogen'),           T=T, P=P)
   ammonia   = Component.from_compound(db.get_compound('ammonia'),            T=T, P=P)

   rxn = Reaction(components=[hydrogen, nitrogen, ammonia])
   print(rxn)         # 3 H2  +  1 N2   <->   2 NH3
   print(rxn.nu)      # [-3. -1.  2.]
   print(rxn.stoProps['dGf'])   # ΔGf of reaction (J/mol)

Lagrange Multiplier Method
--------------------------

The Lagrange multiplier method minimises the total Gibbs energy subject to
atom-balance constraints.  You only need to list the species you expect to be
present — no explicit reactions are required.  This makes it more robust for
complex multi-reaction systems.

.. code-block:: python

   from sandlerchemeq import Component, ChemEqSystem
   from sandlerprops.properties import get_database
   import numpy as np

   db = get_database()
   T, P = 400.0, 100.0   # K, bar

   comp_names = ['hydrogen (equilib)', 'nitrogen', 'ammonia']
   components = [Component.from_compound(db.get_compound(c), T=T, P=P) for c in comp_names]
   N0 = np.array([3.0, 1.0, 0.0])   # initial moles

   system = ChemEqSystem(components=components, N0=N0, T=T, P=P)
   system.solve_lagrange()
   print(system.report())

Output::

   N_{H2}=0.1113 y_{H2}=0.0536
   N_{N2}=0.0371 y_{N2}=0.0179
   N_{NH3}=1.9258 y_{NH3}=0.9285

Explicit Reaction Method
------------------------

When you have a specific reaction and want to track the extent of reaction
:math:`\xi`, use :meth:`~sandlerchemeq.chemeqsystem.ChemEqSystem.solve_implicit`.
The equilibrium constant at temperature *T* is computed via the full van't Hoff
equation (or the simplified form with ``simplified=True``).

A reasonable initial guess ``Xinit`` is needed; the method uses
:func:`scipy.optimize.fsolve` internally.

.. code-block:: python

   from sandlerchemeq import Component, Reaction, ChemEqSystem
   from sandlerprops.properties import get_database
   import numpy as np

   db = get_database()
   T, P = 400.0, 100.0

   comp_names = ['hydrogen (equilib)', 'nitrogen', 'ammonia']
   components = [Component.from_compound(db.get_compound(c), T=T, P=P) for c in comp_names]
   rxn = Reaction(components=components)
   N0 = np.array([3.0, 1.0, 0.0])

   system = ChemEqSystem(components=components, reactions=[rxn], N0=N0, T=T, P=P)
   system.solve_implicit(Xinit=[0.95], simplified=False)
   print(system.report())

Output::

   Reaction    I:  3 H2  +  1 N2   <->   2 NH3  |  Ka(400.00 K)=3.12266e+01 => Xeq=9.62910e-01
   N_{H2}=0.1113 y_{H2}=0.0536
   N_{N2}=0.0371 y_{N2}=0.0179
   N_{NH3}=1.9258 y_{NH3}=0.9285

.. note::
   The Lagrange and explicit reaction methods give identical answers when the
   full van't Hoff equation is used (``simplified=False``).  The simplified
   form neglects the heat-capacity correction and is a good approximation only
   near 298.15 K.

Effect of Temperature and Pressure
-----------------------------------

The Haber–Bosch synthesis of ammonia illustrates Le Chatelier's principle.
Higher temperature and lower pressure both suppress conversion:

.. code-block:: python

   for T in [400.0, 600.0]:
       for P in [100.0, 50.0]:
           components = [Component.from_compound(db.get_compound(c), T=T, P=P)
                         for c in comp_names]
           system = ChemEqSystem(components=components, N0=N0, T=T, P=P)
           system.solve_lagrange()
           y_NH3 = system.ys[comp_names.index('ammonia')]
           print(f"T={T:.0f} K, P={P:.0f} bar:  y_NH3={y_NH3:.4f}")
