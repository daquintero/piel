:py:mod:`piel.tools.hdl21.units`
================================

.. py:module:: piel.tools.hdl21.units

.. autoapi-nested-parse::

   These are the corresponding prefixes from `hdl21`:

   f = FEMTO = Prefix.FEMTO
   p = PICO = Prefix.PICO
   n = NANO = Prefix.NANO
   µ = u = MICRO = Prefix.MICRO # Note both `u` and `µ` are valid
   m = MILLI = Prefix.MILLI
   K = KILO = Prefix.KILO
   M = MEGA = Prefix.MEGA
   G = GIGA = Prefix.GIGA
   T = TERA = Prefix.TERA
   P = PETA = Prefix.PETA
   UNIT = Prefix.UNIT



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.hdl21.units.convert_numeric_to_prefix



.. py:function:: convert_numeric_to_prefix(value: float)

   This function converts a numeric value to a number under a SPICE unit closest to the base prefix. This allows us to connect a particular number real output, into a term that can be used in a SPICE netlist.
