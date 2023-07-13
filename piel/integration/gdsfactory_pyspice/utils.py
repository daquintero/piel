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


def convert_tuples_to_strings(tuple_list):
    result = {}
    for tpl in tuple_list:
        combined_string = "___".join(tpl).replace(",", "__")
        key, value = tpl
        result[key.split(",")[1]] = combined_string
    return result
