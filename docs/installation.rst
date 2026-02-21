.. _installation:

Installation
============

Requirements
------------

sandlerchemeq requires:

* Python 3.7 or later
* NumPy
* SciPy
* pint
* roman
* sandlermisc
* sandlerprops

Install from PyPI
-----------------

.. code-block:: bash

   pip install sandlerchemeq

This will automatically install all required dependencies.

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/cameronabrams/sandlerchemeq.git
   cd sandlerchemeq
   pip install -e .

Verify Installation
-------------------

.. code-block:: bash

   sandlerchemeq --help

Alternatively, from Python:

.. code-block:: python

   import sandlerchemeq
   from sandlerchemeq import Component, Reaction, ChemEqSystem
