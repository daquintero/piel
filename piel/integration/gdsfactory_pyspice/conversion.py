import networkx as nx

# import sax
from sax.circuit import (
    create_dag,
    find_leaves,
    find_root,
    _ensure_recursive_netlist_dict,
    remove_unused_instances,
    _extract_instance_models,
    _validate_net,
)
from sax.netlist import RecursiveNetlist

from ...models.physical.electronic.spice import get_default_models

__all__ = [
    "rename_gdsfactory_connections_to_spice",
    "reshape_gdsfactory_netlist_to_spice_dictionary",
]


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
    models=None,
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

    We should get as an output a dictionary in the form:

    .. code-block::

        {
            instance_1: {
                "connections"
            }
        }
    """
    netlist = _ensure_recursive_netlist_dict(gdsfactory_netlist)

    # TODO: do the following two steps *after* recursive netlist parsing.
    netlist = remove_unused_instances(netlist)
    netlist, instance_models = _extract_instance_models(netlist)
    recursive_netlist: RecursiveNetlist = _validate_net(netlist)

    if models is None:
        models = get_default_models()

    # TODO we might need to implement undirected graphs for valid SPICE circuits, hack for now. Mainly for path measurements no?
    dependency_graph = create_dag(recursive_netlist)
    # required_models = find_leaves(dependency_graph)
    model_names = list(nx.topological_sort(dependency_graph))[::-1]
    new_models = {}
    # current_models = {}
    for model_name in model_names:
        if model_name in models:
            new_models[model_name] = models[model_name]
            continue

        flatnet = recursive_netlist.__root__[model_name]
        inst2model = {
            k: models[inst.component] for k, inst in flatnet.instances.items()
        }
        print(inst2model)
    print(model_names)
    print(find_leaves(dependency_graph))
    print(find_root(dependency_graph))
    return dependency_graph
