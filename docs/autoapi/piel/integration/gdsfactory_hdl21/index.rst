:py:mod:`piel.integration.gdsfactory_hdl21`
===========================================

.. py:module:: piel.integration.gdsfactory_hdl21


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   conversion/index.rst
   core/index.rst
   utils/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.gdsfactory_hdl21.gdsfactory_netlist_to_spice_netlist
   piel.integration.gdsfactory_hdl21.construct_hdl21_module
   piel.integration.gdsfactory_hdl21.convert_connections_to_tuples
   piel.integration.gdsfactory_hdl21.gdsfactory_netlist_with_hdl21_generators



.. py:function:: gdsfactory_netlist_to_spice_netlist(gdsfactory_netlist: dict, generators: dict, **kwargs) -> hdl21.Module

   This function converts a GDSFactory electrical netlist into a standard SPICE netlist. It follows the same
   principle as the `sax` circuit composition.

   Each GDSFactory netlist has a set of instances, each with a corresponding model, and each instance with a given
   set of geometrical settings that can be applied to each particular model. We know the type of SPICE model from
   the instance model we provides.

   We know that the gdsfactory has a set of instances, and we can map unique models via sax through our own
   composition circuit. Write the SPICE component based on the model into a total circuit representation in string
   from the reshaped gdsfactory dictionary into our own structure.

   :param gdsfactory_netlist: GDSFactory netlist
   :param generators: Dictionary of Generators

   :returns: hdl21 module or raw SPICE string


.. py:function:: construct_hdl21_module(spice_netlist: dict, **kwargs) -> hdl21.Module

   This function converts a gdsfactory-spice converted netlist using the component models into a SPICE circuit.

   Part of the complexity of this function is the multiport nature of some components and models, and assigning the
   parameters accordingly into the SPICE function. This is because not every SPICE component will be bi-port,
   and many will have multi-ports and parameters accordingly. Each model can implement the composition into a
   SPICE circuit, but they depend on a set of parameters that must be set from the instance. Another aspect is
   that we may want to assign the component ID according to the type of component. However, we can also assign the
   ID based on the individual instance in the circuit, which is also a reasonable approximation. However,
   it could be said, that the ideal implementation would be for each component model provided to return the SPICE
   instance including connectivity except for the ID.

   # TODO implement validators


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


.. py:function:: gdsfactory_netlist_with_hdl21_generators(gdsfactory_netlist: dict, generators=None)

   This function allows us to map the ``hdl21`` models dictionary in a `sax`-like implementation to the ``GDSFactory`` netlist. This allows us to iterate over each instance in the netlist and construct a circuit after this function.]

   Example usage:

   .. code-block::

       >>> import gdsfactory as gf
       >>> from piel.integration.gdsfactory_hdl21.conversion import gdsfactory_netlist_with_hdl21_generators
       >>> from piel.models.physical.electronic import get_default_models
       >>> gdsfactory_netlist_with_hdl21_generators(gdsfactory_netlist=gf.components.mzi2x2_2x2_phase_shifter().get_netlist(exclude_port_types="optical"),generators=get_default_models())

   :param gdsfactory_netlist: The netlist from ``GDSFactory`` to map to the ``hdl21`` models dictionary.
   :param generators: The ``hdl21`` models dictionary to map to the ``GDSFactory`` netlist.

   :returns: The ``GDSFactory`` netlist with the ``hdl21`` models dictionary.
