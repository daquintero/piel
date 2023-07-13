:py:mod:`piel.integration.gdsfactory_pyspice.conversion`
========================================================

.. py:module:: piel.integration.gdsfactory_pyspice.conversion

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

   piel.integration.gdsfactory_pyspice.conversion.gdsfactory_netlist_to_spice_netlist



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
