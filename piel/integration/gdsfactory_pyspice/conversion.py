import networkx as nx

# import sax
from sax.circuit import (
    create_dag,
    _ensure_recursive_netlist_dict,
    remove_unused_instances,
    _extract_instance_models,
    _validate_net,
    _validate_dag,
    _validate_models,
)
from sax.netlist import RecursiveNetlist

from ...models.physical.electronic.spice import get_default_models
from .utils import convert_tuples_to_strings

__all__ = [
    "reshape_gdsfactory_netlist_to_spice_dictionary",
]


def get_matching_connections(names: list, connections: dict):
    """
    # TODO docs
    """
    matching_connections = []
    for key, value in connections.items():
        for name in names:
            if name in key or name in value:
                matching_connections.append((key, value))
    return matching_connections


def get_matching_port_nets(names, connections):
    matching_tuples = get_matching_connections(names, connections)
    matching_strings = convert_tuples_to_strings(matching_tuples)
    return matching_strings


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
    spice_netlist = {}
    spice_netlist["instances"] = gdsfactory_netlist["instances"]
    if models is None:
        models = get_default_models()

    netlist = _ensure_recursive_netlist_dict(gdsfactory_netlist)

    # TODO: do the following two steps *after* recursive netlist parsing.
    netlist = remove_unused_instances(netlist)
    netlist, instance_models = _extract_instance_models(netlist)

    recnet: RecursiveNetlist = _validate_net(netlist)
    dependency_dag: nx.DiGraph = _validate_dag(
        create_dag(recnet, models)
    )  # directed acyclic graph
    models = _validate_models({**(models or {}), **instance_models}, dependency_dag)

    new_models = {}
    current_models = {}
    model_names = list(nx.topological_sort(dependency_dag))[::-1]
    for model_name in model_names:
        if model_name in models:
            new_models[model_name] = models[model_name]
            continue

        flatnet = recnet.__root__[model_name]
        # connections, ports, new_models = _make_singlemode_or_multimode(
        #     flatnet, modes, new_models
        # )
        current_models.update(new_models)
        new_models = {}
        inst2model = {
            k: models[inst.component] for k, inst in flatnet.instances.items()
        }

        # Iterate over every instance and append the corresponding required SPICE connectivity
        for instance_name_i, _ in list(spice_netlist["instances"].items()):
            print(instance_name_i)
            spice_netlist["instances"][instance_name_i][
                "connections"
            ] = get_matching_connections(
                names=[instance_name_i], connections=gdsfactory_netlist["connections"]
            )
            spice_netlist["instances"][instance_name_i][
                "nets"
            ] = get_matching_port_nets(
                names=[instance_name_i], connections=gdsfactory_netlist["connections"]
            )
            spice_netlist["instances"][instance_name_i]["spice_model"] = inst2model[
                instance_name_i
            ]
    # print(model_names)
    # print(find_leaves(dependency_dag))
    # print(find_root(dependency_dag))
    return dependency_dag, flatnet, spice_netlist
