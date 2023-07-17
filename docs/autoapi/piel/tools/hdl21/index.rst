:py:mod:`piel.tools.hdl21`
==========================

.. py:module:: piel.tools.hdl21


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   circuit/index.rst
   simulator/index.rst
   units/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.hdl21.convert_numeric_to_prefix



.. py:function:: convert_numeric_to_prefix(value: float)

   This function converts a numeric value to a number under a SPICE unit closest to the base prefix. This allows us to connect a particular number real output, into a term that can be used in a SPICE netlist.
