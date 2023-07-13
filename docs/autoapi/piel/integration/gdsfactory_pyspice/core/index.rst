:py:mod:`piel.integration.gdsfactory_pyspice.core`
==================================================

.. py:module:: piel.integration.gdsfactory_pyspice.core

.. autoapi-nested-parse::

   Most of the ``pyspice``-``gdsfactory`` integration functions will be contributed directly to `gdsfactory`. However,
   some `translation language` inherent to the ``piel`` implementation of these tools is included here.

   Note that to be able to construct a full circuit model of the netlist tools provided, it is necessary to create
   individual circuit models of the devices that we will interconnect, and then map them to a larger netlist. This means
   that it is necessary to create specific SPICE models for each particular component, say in an electrical netlist.

   This functions convert a GDSFactory netlist, with a set of component models, into `PySPICE` that accounts for the
   instance properties, which can then be connected into a VLSIR compatible `Netlist` implementation.

   Eventually we will implement RCX where we can extract the netlist with parasitics directly from the layout,
   but for now this will be the implementation. The output structure of our SPICE should be compatible with the
   `Netlist` package BaseModel.

   We follow the principle in: https://eee.guc.edu.eg/Courses/Electronics/ELCT503%20Semiconductors/Lab/spicehowto.pdf

   .. code-block::

       Spice Simulation 1-1
       *** MODEL Descriptions ***
       .model nm NMOS level=2 VT0=0.7
       KP=80e-6 LAMBDA=0.01

       *** NETLIST Description ***
       M1 vdd ng 0 0 nm W=3u L=3u
       R1 in ng 50
       Vdd vdd 0 5
       Vin in 0 2.5

       *** SIMULATION Commands ***
       .op
       .end

   Note that the netlist device connectivity structure of most passive components is in the form:

   .. code-block::

       <DEVICE ID> <CONNECTION_0> <CONNECTION_1> <DEVICE_VALUE> <MORE_PARAMETERS>

   Our example GDSFactory netlist format is in the simplified form:

   .. code-block::

       {
           "connections": {
               "straight_1": {
                   "e1": "taper_1,e2",
                   "e2": "taper_2,e2"
               },
               "taper_1": {
                   "e1": "via_stack_1,e3"
               },
               "taper_2": {
                   "e1": "via_stack_2,e1"
               }
           },
           "instances": {
               "straight_1": {
                   "component": "straight",
                   "info": {
                       "length": 15.0,
                       "width": 0.5,
                       "cross_section": "strip_heater_metal",
                       "settings": {
                           "width": 0.5,
                           "layer": "WG",
                           "heater_width": 2.5,
                           "layer_heater": "HEATER"
                       }
                   }
               },
               "taper_1": {
                   "component": "taper",
                   "info": {
                       "length": 5.0,
                       "width1": 11.0,
                       "width2": 2.5
                   },
                   "settings": {
                       "cross_section": {
                           "layer": "HEATER",
                           "width": 2.5,
                           "offset": 0.0,
                           "taper_length": 10.0,
                           "gap": 5.0,
                           "min_length": 5.0,
                           "port_names": ["e1", "e2"]
                       }
                   }
               },
               "taper_2": {
                   "component": "taper",
                   "info": {
                       "length": 5.0,
                       "width1": 11.0,
                       "width2": 2.5
                   },
                   "settings": {
                       "cross_section": {
                           "layer": "HEATER",
                           "width": 2.5,
                           "offset": 0.0,
                           "taper_length": 10.0,
                           "gap": 5.0,
                           "min_length": 5.0,
                           "port_names": ["e1", "e2"]
                       }
                   }
               },
               "via_stack_1": {
                   "component": "via_stack",
                   "info": {
                       "size": [11.0, 11.0],
                       "layer": "M3"
                   },
                   "settings": {
                       "layers": ["HEATER", "M2", "M3"]
                   }
               },
               "via_stack_2": {
                   "component": "via_stack",
                   "info": {
                       "size": [11.0, 11.0],
                       "layer": "M3"
                   },
                   "settings": {
                       "layers": ["HEATER", "M2", "M3"]
                   }
               }
           },
           "placements": {
               "straight_1": {"x": 0.0, "y": 0.0, "rotation": 0, "mirror": 0},
               "taper_1": {"x": -5.0, "y": 0.0, "rotation": 0, "mirror": 0},
               "taper_2": {"x": 20.0, "y": 0.0, "rotation": 180, "mirror": 0},
               "via_stack_1": {"x": -10.5, "y": 0.0, "rotation": 0, "mirror": 0},
               "via_stack_2": {"x": 25.5, "y": 0.0, "rotation": 0, "mirror": 0}
           },
           "ports": {
               "e1": "taper_1,e2",
               "e2": "taper_2,e2"
           },
           "name": "straight_heater_metal_simple",
       }

   This is particularly useful when creating our components and connectivity, because what we can do is instantiate our
   devices with their corresponding values, and then create our connectivity accordingly. To do this properly from our
   GDSFactory netlist to PySpice, we can then extract the total SPICE circuit, and convert it to a VLSIR format using
   the `Netlist` module. The reason why we can't use the Netlist package from Dan Fritchman directly is that we need to
   apply a set of models that translate a particular component instantiation into an electrical model. Because we are
   not yet doing layout extraction as that requires EM solvers, we need to create some sort of SPICE level assignment
   based on the provided dictionary.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.gdsfactory_pyspice.core.gdsfactory_netlist_to_pyspice
   piel.integration.gdsfactory_pyspice.core.spice_netlist_to_pyspice_circuit



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
