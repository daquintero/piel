"""Most of the ``hdl21``-``gdsfactory`` integration functions will be contributed directly to `gdsfactory`. However,
some `translation language` inherent to the ``piel`` implementation of these tools is included here.

Note that to be able to construct a full circuit model of the netlist tools provided, it is necessary to create
individual circuit models of the devices that we will interconnect, and then map them to a larger netlist. This means
that it is necessary to create specific SPICE models for each particular component, say in an electrical netlist.

This functions convert a GDSFactory netlist, with a set of component models, into `hdl21` that accounts for the
instance properties, which can then be connected into a VLSIR compatible `Netlist` implementation.

Eventually we will implement RCX where we can extract the netlist with parasitics directly from the layout,
but for now this will be the implementation. The output structure of our SPICE should be compatible with the
`Netlist` package BaseModel.

We follow the principle in: https://eee.guc.edu.eg/Courses/Electronics/ELCT503%20Semiconductors/Lab/spicehowto.pdf

.. code-block:: spice

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

.. code-block:: spice

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
GDSFactory netlist to ``hdl21``, we can then extract the total SPICE circuit, and convert it to a VLSIR format using
the ``Netlist`` module. The reason why we can't use the Netlist package from Dan Fritchman directly is that we need to
apply a set of models that translate a particular component instantiation into an electrical model. Because we are
not yet doing layout extraction as that requires EM solvers, we need to create some sort of SPICE level assignment
based on the provided dictionary.

Note that ``hdl21`` already can implement the port connectivity directly from internal instances, and translate this
to our connectivity netlist. This means we only need to iterate to create our instances based on our models into a
``hdl21`` module, then we can easily assign the corresponding values. It is also possible to create the assigned
parameters as part of the ``hdl21`` component which would form part of our module. Because the gdsfactory names are
compatible with ``hdl21``, then it is fine to create the integration accordingly.

The algorithm can be to:

1. Parse the gdsfactory netlist, assign the electrical ports for the model. Extract all instances and
required models from the netlist.
2. Verify that the models have been provided. Each model describes the type of
component this is, how many ports it requires and so on. Create a ``hdl21`` top level module for every gdsfactory
netlist, this is reasonable as it is composed, and not a generator class. This generates a large amount of instantiated ``hdl21`` modules that are generated from `generators`.
3. Map the connections to each instance port as part of the instance dictionary. This parses the connectivity in the ``gdsfactory`` netlist and connects the ports accordingly.

The connections are a bit more complex. So each of our connections dictionary is in the form:

.. code-block::

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

We know what our top model ports are. We know our internal instance ports as well, and this will be provided by the
model too. For the sake of easiness, we can describe these as ``hdl21`` equivalent ``InOut`` or ``Port` `ports and
not have to deal with directionality. After instance declaration, and models for each of these components with the
corresponding port topology, it is then straightforward to parse the connectivity and implement the network,
and extract the SPICE. """
import hdl21 as h
from hdl21.elab.elaborators.conntypes import get_unconnected_instance_connections
from .conversion import convert_connections_to_tuples

__all__ = ["gdsfactory_netlist_to_spice_netlist", "construct_hdl21_module"]


def gdsfactory_netlist_to_spice_netlist(
    gdsfactory_netlist: dict,
    models: dict,
    return_raw_spice: bool = False,
):
    """
    This function converts a GDSFactory electrical netlist into a standard SPICE netlist. It follows the same
    principle as the `sax` circuit composition.

    Each GDSFactory netlist has a set of instances, each with a corresponding model, and each instance with a given
    set of geometrical settings that can be applied to each particular model. We know the type of SPICE model from
    the instance model we provides.

    We know that the gdsfactory has a set of instances, and we can map unique models via sax through our own
    composition circuit. Write the SPICE component based on the model into a total circuit representation in string
    from the reshaped gdsfactory dictionary into our own structure.
    """
    pass


def construct_hdl21_module(spice_netlist: dict, **kwargs) -> h.Module:
    """
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
    """
    circuit = h.Module(name=spice_netlist["name"])
    instance_id = 0
    # Declare all the instances
    for instance_name_i, instance_settings_i in spice_netlist["instances"].items():
        instance_i = instance_settings_i["hdl21_model"](
            name=instance_name_i, **instance_settings_i["settings"]
        )()
        circuit.add(val=instance_i, name=instance_name_i)
        instance_id += 1

    # Create top level ports
    for port_name_i, _ in spice_netlist["ports"].items():
        # TODO include directionality on port_settings so that it can be easily interconencted with hdl21
        circuit.ports[port_name_i] = h.Port()

    # Create the connectivity
    connections_list = convert_connections_to_tuples(spice_netlist["connections"])
    for connection_tuple in connections_list:
        # Connects the corresponding ports.
        first_instance = getattr(circuit, connection_tuple[0][0])
        second_instance = getattr(circuit, connection_tuple[1][0])
        second_port = getattr(second_instance, connection_tuple[1][1])
        first_instance.connect(connection_tuple[0][1], second_port)
        # circuit.instances[connection_tuple[0][0]].ports[
        #     connection_tuple[0][1]
        # ] = circuit.instances[connection_tuple[1][0]]().ports[
        #     connection_tuple[1][1]
        # ]

    # Check missing conns and create top-level ports for them if necessary:
    for instance_name_i, _ in circuit.instances.items():
        instance_i = getattr(circuit, instance_name_i)
        print(instance_i)
        print(circuit)
        print(instance_i.conns)
        print(get_unconnected_instance_connections(module=circuit, inst=instance_i))

    return h.elaborate(circuit)
