__all__ = ["rename_gdsfactory_connections_to_spice"]


def instance_to_pyspice(component_model):
    """
    This function maps a particular model, with an instance representation that corresponds to the given netlist
    connectivity, and returns a PySpice representation of the circuit. This function will be called after parsing the
    circuit netlist accordingly, and creating a mapping from the instance definitions to the fundamental components.

    Args:
        component_model(func): Function that represents a SPICE component with the given parameters.
    """
    pass


def rename_gdsfactory_connections_to_spice(connections: dict):
    """
    We convert the connection connectivity of the gdsfactory netlist into names that can be integrated into a SPICE
    netlist. It iterates on each key value pair, and replaces each comma with an underscore.

    # TODO docs
    """
    spice_connections = {}
    for key, value in connections.items():
        new_key = key.replace(",", "_")
        new_value = value.replace(",", "_")
        spice_connections[new_key] = new_value
    return spice_connections


def reshape_gdsfactory_netlist_to_spice_dictionary():
    """
    This function maps the connections of a netlist to a node that can be used in a SPICE netlist. SPICE netlists are
    in the form of:

    .. code-block::
        RXXXXXXX N1 N2 <VALUE> <MNAME> <L=LENGTH> <W=WIDTH> <TEMP=T>

    This means that every instance, is an electrical type, and we define the two particular nodes in which it is
    connected. This means we need to convert the gdsfactory dictionary netlist into a form that allows us to map the
    connectivity for every instance. Then we can define that as a line of the SPICE netlist with a particular
    electrical model. For passives this works fine when it's a two port network such as sources, or electrical
    elements. However, non-passive elements like transistors have three ports or more which are provided in an ordered form.
     The output format would ideally be in this form:

    .. code-block::
        {
            instance_someid: {
                "connections"
            }
        }
    """
