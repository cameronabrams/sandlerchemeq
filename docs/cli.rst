.. _cli:

Command-Line Interface
======================

``sandlerchemeq`` provides a command-line tool for solving chemical equilibrium
systems without writing Python code.

.. code-block:: bash

   sandlerchemeq [<global-options>] <command> [<command-options>]

Global Options
--------------

.. code-block:: text

   -h, --help
       Show help message and exit.

   -b, --banner / --no-banner
       Toggle the banner message (default: off).

   --logging-level {info,debug,warning}
       Verbosity of diagnostic messages (default: debug).

   -l LOG, --log LOG
       Write diagnostic log to this file.

Commands
--------

``solve``
~~~~~~~~~

Solve a chemical equilibrium system.

.. code-block:: bash

   sandlerchemeq solve [options]

**Required options:**

``-T TEMPERATURE`` / ``--temperature TEMPERATURE``
    System temperature in Kelvin.

``-P PRESSURE`` / ``--pressure PRESSURE``
    System pressure in bar.

``-n0 N [N ...]`` / ``--initial-moles N [N ...]``
    Initial molar amounts for each component (same order as components).

**Mutually exclusive — specify one:**

``-c NAME [NAME ...]`` / ``--components NAME [NAME ...]``
    List of component names (database names from sandlerprops).
    Uses the **Lagrange multiplier** method.

``-r RXNSTR [RXNSTR ...]`` / ``--reactions RXNSTR [RXNSTR ...]``
    One or more reactions, each as a comma-delimited string of component names
    (no spaces).  Components are inferred from the reaction strings.
    Uses the **explicit reaction** method.

**Optional:**

``-xinit X [X ...]`` / ``--X-init X [X ...]``
    Initial guess for extents of reaction (one per reaction).
    Required when ``--reactions`` is used for good convergence.

Examples
--------

Lagrange Method
~~~~~~~~~~~~~~~

No explicit reactions — just list the species and initial amounts:

.. code-block:: bash

   sandlerchemeq solve -c "hydrogen (equilib)" nitrogen ammonia -T 400 -P 100 -n0 3. 1. 0.

Output::

   N_{H2}=0.1113 y_{H2}=0.0536
   N_{N2}=0.0371 y_{N2}=0.0179
   N_{NH3}=1.9258 y_{NH3}=0.9285

.. code-block:: bash

   sandlerchemeq solve -c "hydrogen (equilib)" nitrogen ammonia -T 600 -P 100 -n0 3. 1. 0.

Output::

   N_{H2}=1.2110 y_{H2}=0.4314
   N_{N2}=0.4037 y_{N2}=0.1438
   N_{NH3}=1.1926 y_{NH3}=0.4248

Explicit Reaction Method
~~~~~~~~~~~~~~~~~~~~~~~~

Specify reactions as comma-delimited component lists (reactants first,
products last):

.. code-block:: bash

   sandlerchemeq solve -r "hydrogen (equilib),nitrogen,ammonia" -T 400 -P 100 -n0 3. 1. 0. -xinit 0.95

Output::

   Reaction    I:  3 H2  +  1 N2   <->   2 NH3  |  Ka(400.00 K)=3.12266e+01 => Xeq=9.62910e-01
   N_{H2}=0.1113 y_{H2}=0.0536
   N_{N2}=0.0371 y_{N2}=0.0179
   N_{NH3}=1.9258 y_{NH3}=0.9285

.. tip::
   Compound names containing spaces must be quoted on the command line.
   For reactions, names within the comma-delimited string must not contain
   commas.  The compound name ``hydrogen (equilib)`` is the standard-state
   H\ :sub:`2` entry (ΔGf = ΔHf = 0) in the sandlerprops database.
