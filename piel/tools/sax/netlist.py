"""
This module aims to extend sax from standard netlist operations to include more complex operations that enable connectivity.
"""
import sax

__all__ = ["get_netlist_instances_by_prefix", "get_component_instances"]


def get_netlist_instances_by_prefix(
        recursive_netlist: dict,
        instance_prefix: str,
) -> str:
    """
    Returns a list of all instances with a given prefix in a recursive netlist.

    Args:
        recursive_netlist: The recursive netlist to search.
        instance_prefix: The prefix to search for.

    Returns:
        A list of all instances with the given prefix.
    """
    # TODO interim function until Floris merges.
    recursive_netlist = sax.netlist(recursive_netlist)
    recursive_netlist_root = recursive_netlist.dict()["__root__"]
    result = []
    for key in recursive_netlist_root.keys():
        if key.startswith(instance_prefix):
            result.append(key)

    if len(result) == 1:
        return result[0]
    elif len(result) == 0:
        raise ValueError("No instances with prefix: " + instance_prefix + " found. These are the available instances: " + str(recursive_netlist_root.keys()))
    else:
        print(len(result))
        raise ValueError("More than one instance with prefix: " + instance_prefix + "found. These are the matched "
                                                                                    "instances: " + str(result))


def get_component_instances(
        recursive_netlist: dict,
        top_level_prefix: str,
        component_name_prefix: str,
):
    """
    Returns a dictionary of all instances of a given component in a recursive netlist.

    Args:
        recursive_netlist: The recursive netlist to search.
        top_level_prefix: The prefix of the top level instance.
        component_name_prefix: The name of the component to search for.

    Returns:
        A dictionary of all instances of the given component.
    """
    # TODO interim function until Floris merges.
    recursive_netlist_sax = sax.netlist(recursive_netlist)
    instance_names = []
    recursive_netlist_root = recursive_netlist_sax.dict()["__root__"]
    top_level_prefix = get_netlist_instances_by_prefix(recursive_netlist, instance_prefix=top_level_prefix) # Should only be one in a netlist-to-digraph. Can always be very specified.
    for key in recursive_netlist_root[top_level_prefix]["instances"]:
        if recursive_netlist_root[top_level_prefix]["instances"][key]["component"].startswith(component_name_prefix):
            # Note priority encoding on match.
            instance_names.append(key)
    return {component_name_prefix: instance_names}
