import sax

__all__ = ["rename_gdsfactory_connections_to_spice"]


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


def reshape_gdsfactory_netlist_to_spice_dictionary(
    gdsfactory_netlist: dict,
):
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

    This means that the order of translations is as follows:

    .. code-block::

        1. Extract all instances and required models from the netlist
        2. Verify that the models have been provided. Each model describes the type of component this is, how many ports it requires and so on.
        3. Map the connections to each instance port as part of the instance dictionary.
    """
    recursive_netlist = sax.netlist(gdsfactory_netlist)
    dependency_graph = sax.circuit.create_dag(recursive_netlist)
    return dependency_graph
