.. sandlerchemeq documentation master file

sandlerchemeq
=============

.. image:: https://img.shields.io/pypi/v/sandlerchemeq.svg
   :target: https://pypi.org/project/sandlerchemeq/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/sandlerchemeq.svg
   :target: https://pypi.org/project/sandlerchemeq/
   :alt: Python versions

Chemical equilibrium calculations via Gibbs energy minimization
---------------------------------------------------------------

**sandlerchemeq** implements computational tools for chemical equilibrium
calculations based on *Chemical, Biochemical, and Engineering Thermodynamics*
(5th edition) by Stan Sandler (Wiley, USA).

.. warning::
   This package should be used for **educational purposes only**.

Two solution methods are provided:

* **Lagrange multiplier method** — minimize the total Gibbs energy subject to
  atom-balance constraints.  No explicit reactions are required; only an
  exhaustive list of possible species and their initial molar amounts.

* **Explicit reaction method** — solve for extents of reaction using
  equilibrium constants computed via the van't Hoff equation.

Quick Start
-----------

Installation::

   pip install sandlerchemeq

Lagrange equilibrium (no reactions needed)::

   sandlerchemeq solve -c hydrogen nitrogen ammonia -T 400 -P 100 -n0 3. 1. 0.

Explicit reaction equilibrium::

   sandlerchemeq solve -r hydrogen,nitrogen,ammonia -T 400 -P 100 -n0 3. 1. 0. -xinit 0.95

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   cli

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/API.rst

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   contributing
   changelog

License
=======

This project is licensed under the MIT License — see the LICENSE file for details.

Citation
========

If you use this package for academic work, please cite:

* Sandler, S. (2017). *Chemical, Biochemical, and Engineering Thermodynamics* (5th ed.). Wiley.

Contact
=======

Cameron F. Abrams — cfa22@drexel.edu

GitHub: https://github.com/cameronabrams/sandlerchemeq
