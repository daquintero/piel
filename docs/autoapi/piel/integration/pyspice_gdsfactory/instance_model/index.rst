:py:mod:`piel.integration.pyspice_gdsfactory.instance_model`
============================================================

.. py:module:: piel.integration.pyspice_gdsfactory.instance_model


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.pyspice_gdsfactory.instance_model.instance_to_pyspice



.. py:function:: instance_to_pyspice(component_model)

   This function maps a particular model, with an instance representation that corresponds to the given netlist
   connectivity, and returns a PySpice representation of the circuit. This function will be called after parsing the
   circuit netlist accordingly, and creating a mapping from the instance definitions to the fundamental components.

   :param component_model: Function that represents a SPICE component with the given parameters.
   :type component_model: func
