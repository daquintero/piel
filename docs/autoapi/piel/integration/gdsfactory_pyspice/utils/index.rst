:py:mod:`piel.integration.gdsfactory_pyspice.utils`
===================================================

.. py:module:: piel.integration.gdsfactory_pyspice.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.gdsfactory_pyspice.utils.rename_gdsfactory_connections_to_spice
   piel.integration.gdsfactory_pyspice.utils.convert_tuples_to_strings



.. py:function:: rename_gdsfactory_connections_to_spice(connections: dict)

   We convert the connection connectivity of the gdsfactory netlist into names that can be integrated into a SPICE
   netlist. It iterates on each key value pair, and replaces each comma with an underscore.

   # TODO docs


.. py:function:: convert_tuples_to_strings(tuple_list)
