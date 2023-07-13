:py:mod:`piel.integration.gdsfactory_pyspice`
=============================================

.. py:module:: piel.integration.gdsfactory_pyspice


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

   piel.integration.gdsfactory_pyspice.gdsfactory_netlist_to_pyspice
   piel.integration.gdsfactory_pyspice.spice_netlist_to_pyspice_circuit
   piel.integration.gdsfactory_pyspice.gdsfactory_netlist_to_spice_netlist



.. py:function:: gdsfactory_netlist_to_pyspice(gdsfactory_netlist: dict, return_raw_spice: bool = False)

   This function converts a GDSFactory electrical netlist into a standard PySpice configuration. It follows the same
   principle as the `sax` circuit composition. It returns a PySpice circuit and can return it in raw_spice form if
   necessary.

   Each GDSFactory netlist has a set of instances, each with a corresponding model, and each instance with a given
   set of geometrical settings that can be applied to each particular model. We know the type of SPICE model from
   the instance model we provides.

   We know that the gdsfactory has a set of instances, and we can map unique models via sax through our own
   composition circuit. Write the SPICE component based on the model into a total circuit representation in string
   from the reshaped gdsfactory dictionary into our own structure.


.. py:function:: spice_netlist_to_pyspice_circuit(spice_netlist: dict)

   This function converts a SPICE netlist into a PySpice circuit.

   # TODO implement validators


.. py:function:: gdsfactory_netlist_to_spice_netlist(gdsfactory_netlist: dict, models=None)

   This function maps the connections of a netlist to a node that can be used in a SPICE netlist. SPICE netlists are
   in the form of:

   .. code-block::

       RXXXXXXX N1 N2 <VALUE> <MNAME> <L=LENGTH> <W=WIDTH> <TEMP=T>

   This means that every instance, is an electrical type, and we define the two particular nodes in which it is
   connected. This means we need to convert the gdsfactory dictionary netlist into a form that allows us to map the
   connectivity for every instance. Then we can define that as a line of the SPICE netlist with a particular
   electrical model. For passives this works fine when it's a two port network such as sources, or electrical
   elements. However, non-passive elements like transistors have three ports or more which are provided in an ordered form.

   This means that the order of translations is as follows:

   .. code-block::

       1. Extract all instances and required models from the netlist
       2. Verify that the models have been provided. Each model describes the type of component this is, how many ports it requires and so on.
       3. Map the connections to each instance port as part of the instance dictionary.

   We should get as an output a dictionary in the structure:

   .. code-block::

       {
           instance_1: {
               ...
               "connections": [('straight_1,e1', 'taper_1,e2'),
                               ('straight_1,e2', 'taper_2,e2')],
               'spice_nets': {'e1': 'straight_1__e1___taper_1__e2',
                       'e2': 'straight_1__e2___taper_2__e2'},
               'spice_model': <function piel.models.physical.electronic.spice.resistor.basic_resistor()>},
           }
           ...
       }
