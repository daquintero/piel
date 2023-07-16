:py:mod:`piel.integration.gdsfactory_hdl21.conversion`
======================================================

.. py:module:: piel.integration.gdsfactory_hdl21.conversion

.. autoapi-nested-parse::

   `sax` has very good GDSFactory integration functions, so there is a question on whether implementing our own circuit
   construction, and SPICE netlist parser from it, accordingly. We need in some form to connect electrical models to our
   parsed netlist, in order to apply SPICE passive values, and create connectivity for each particular device. Ideally,
   this would be done from the component instance as that way the component model can be integrated with its geometrical
   parameters, but does not have to be done necessarily. This comes down to implementing a backend function to compile
   SAX compiled circuit.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.gdsfactory_hdl21.conversion.convert_connections_to_tuples
   piel.integration.gdsfactory_hdl21.conversion.gdsfactory_netlist_with_hdl21_models



.. py:function:: convert_connections_to_tuples(connections: dict)

   Convert from:

   .. code-block::

       {
       'straight_1,e1': 'taper_1,e2',
       'straight_1,e2': 'taper_2,e2',
       'taper_1,e1': 'via_stack_1,e3',
       'taper_2,e1': 'via_stack_2,e1'
       }

   to:

   .. code-block::

       [(('straight_1', 'e1'), ('taper_1', 'e2')), (('straight_1', 'e2'), ('taper_2', 'e2')), (('taper_1', 'e1'),
       ('via_stack_1', 'e3')), (('taper_2', 'e1'), ('via_stack_2', 'e1'))]


.. py:function:: gdsfactory_netlist_with_hdl21_models(gdsfactory_netlist: dict, models=None)

   This function allows us to map the ``hdl21`` models dictionary in a `sax`-like implementation to the ``GDSFactory`` netlist. This allows us to iterate over each instance in the netlist and construct a circuit after this function.]

   Example usage:

   .. code-block::

       >>> import gdsfactory as gf
       >>> from piel.integration.gdsfactory_hdl21.conversion import gdsfactory_netlist_with_hdl21_models
       >>> from piel.models.physical.electronic import get_default_models
       >>> gdsfactory_netlist_with_hdl21_models(gdsfactory_netlist=gf.components.mzi2x2_2x2_phase_shifter().get_netlist(exclude_port_types="optical"), models=get_default_models())

   :param gdsfactory_netlist: The netlist from ``GDSFactory`` to map to the ``hdl21`` models dictionary.
   :param models: The ``hdl21`` models dictionary to map to the ``GDSFactory`` netlist.

   :returns: The ``GDSFactory`` netlist with the ``hdl21`` models dictionary.
