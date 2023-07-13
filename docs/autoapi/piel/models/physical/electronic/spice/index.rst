:py:mod:`piel.models.physical.electronic.spice`
===============================================

.. py:module:: piel.models.physical.electronic.spice


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   capacitor/index.rst
   defaults/index.rst
   resistor/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.physical.electronic.spice.add_basic_capacitor
   piel.models.physical.electronic.spice.add_basic_resistor
   piel.models.physical.electronic.spice.get_default_models



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



.. py:function:: add_basic_resistor(circuit: PySpice.Spice.Netlist.Circuit, instance_id: int, input_node: str, output_node: str)

   SPICE Resistor Structure

   See Mike Smith “WinSpice3 User’s Manual” 25 October, 1999

   .. code-block::

       RXXXXXXX N1 N2 <VALUE> <MNAME> <L=LENGTH> <W=WIDTH> <TEMP=T>

   Where the terminals are:

   .. code-block::

       N1 = the first terminal
       N2 = the second terminal
       <VALUE> = resistance in ohms.
       <MNAME> = name of the model used (useful for semiconductor resistors)
       <L=LENGTH> = length of the resistor (useful for semiconductor resistors)
       <W=WIDTH> = width of the resistor (useful for semiconductor resistors)
       <TEMP=T> = temperature of the resistor in Kelvin (useful in noise analysis and
       semiconductor resistors)

   An example is:

   .. code-block::

       RHOT n1 n2 10k TEMP=500


.. py:function:: get_default_models(custom_defaults: dict | None = None) -> dict

   Returns the default models dictionary.

   :param custom_defaults: Custom defaults dictionary.
   :type custom_defaults: dict

   :returns: Default models dictionary.
   :rtype: dict
