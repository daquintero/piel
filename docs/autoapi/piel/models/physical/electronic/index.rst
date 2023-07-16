:py:mod:`piel.models.physical.electronic`
=========================================

.. py:module:: piel.models.physical.electronic


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   capacitor/index.rst
   defaults/index.rst
   resistor/index.rst
   straight/index.rst
   taper/index.rst
   via_stack/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.physical.electronic.add_basic_capacitor
   piel.models.physical.electronic.get_default_models



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



.. py:function:: get_default_models(custom_defaults: dict | None = None) -> dict

   Returns the default models dictionary.

   :param custom_defaults: Custom defaults dictionary.
   :type custom_defaults: dict

   :returns: Default models dictionary.
   :rtype: dict
