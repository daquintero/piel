:py:mod:`piel.models.physical.electronic.spice.capacitor`
=========================================================

.. py:module:: piel.models.physical.electronic.spice.capacitor

.. autoapi-nested-parse::

   The way each of these models should work is that they use the settings from the `gdsfactory` component,
   to create a parametric SPICE directive.

   These functions map a particular model, with an instance representation that corresponds to the given netlist
   connectivity, and returns a PySpice representation of the circuit. This function will be called after parsing the
   circuit netlist accordingly, and creating a mapping from the instance definitions to the fundamental components.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.physical.electronic.spice.capacitor.add_basic_capacitor



.. py:function:: add_basic_capacitor(settings) -> str

   This function takes in the settings from a gdsfactory component, some connectivity node translated directly from
   the gdsfactory netlist.

   See Mike Smith “WinSpice3 User’s Manual” 25 October, 1999

   SPICE capacitor model:

   .. code-block::

       CXXXXXXX N+ N- VALUE <IC=INCOND>

   Where the parameters are:

   .. code-block::

       N+ = the positive terminal
       N- = the negative terminal
       VALUE = capacitance in farads
       <IC=INCOND> = starting voltage in a simulation
