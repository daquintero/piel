"""
This module provides functions to generate a raw netlist semi-compatible with gdsfactory from a hdl21 module object.
"""

from ...types import AnalogueModule

ParsedProtoVLSIR = dict


def _parse_module_to_proto_dict(module: AnalogueModule) -> ParsedProtoVLSIR:
    """
    Parse a hdl21 module object into a dictionary with the same structure as the proto VLSIR format.
    """
    import hdl21 as h

    def parse_value(lines, index):
        value = {}
        while index < len(lines):
            line = lines[index].strip()
            if line == "}":
                return value, index
            elif line.endswith("{"):
                key = line[:-1].strip()
                sub_value, new_index = parse_value(lines, index + 1)
                if key not in value:
                    value[key] = []
                value[key].append(sub_value)
                index = new_index
            else:
                key, val = line.split(":", 1)
                value[key.strip()] = val.strip().strip('"')
            index += 1
        return value, index

    raw_proto_str = str(h.to_proto(module))
    lines = raw_proto_str.split("\n")
    result = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.endswith("{"):
            key = line[:-1].strip()
            sub_value, new_index = parse_value(lines, index + 1)
            if key not in result:
                result[key] = []
            result[key].append(sub_value)
            index = new_index
        else:
            index += 1

    return result


def _parse_connections(proto_dict: ParsedProtoVLSIR) -> dict:
    """
    Extract the connections from the proto_dict and return a dictionary with the connections.
    """
    connections = {}

    # Extract the instances and their connections
    for module in proto_dict.get("modules", []):
        for instance in module.get("instances", []):
            instance_name = instance["name"]
            for connection in instance.get("connections", []):
                portname = connection["portname"]
                target = connection["target"][0]

                # Handle different structures of target
                if "sig" in target:
                    target_signal = target["sig"]
                elif "slice" in target:
                    target_signal = target["slice"][0]["signal"]
                else:
                    # Log a warning or provide a default value for unknown structures
                    target_signal = "unknown_signal"
                    print(
                        f"Warning: Unknown target structure in connection for instance '{instance_name}', port '{portname}'"
                    )

                connection_key = f"{instance_name},{portname}"
                # Find the target instance and port
                target_instance_port = _find_target_instance_port(
                    proto_dict, target_signal, instance_name
                )
                if target_instance_port:
                    connections[connection_key] = target_instance_port

    return connections


def _find_target_instance_port(
    proto_dict: ParsedProtoVLSIR, target_signal, current_instance_name
):
    """
    Find the target instance and port of the target signal in the proto_dict.
    """
    # Search in the same module
    for module in proto_dict.get("modules", []):
        for instance in module.get("instances", []):
            if instance["name"] == current_instance_name:
                continue
            for connection in instance.get("connections", []):
                target = connection["target"][0]

                # Handle different structures of target
                if "sig" in target:
                    signal = target["sig"]
                elif "slice" in target:
                    signal = target["slice"][0]["signal"]
                else:
                    signal = None
                    print(
                        f"Warning: Unknown target structure for instance '{instance['name']}', port '{connection['portname']}'"
                    )

                if signal == target_signal:
                    return f"{instance['name']},{connection['portname']}"

    # Search in external modules
    for ext_module in proto_dict.get("ext_modules", []):
        for port in ext_module.get("ports", []):
            if port["signal"] == target_signal:
                for instance in module.get("instances", []):
                    if instance["name"] == current_instance_name:
                        continue
                    for connection in instance.get("connections", []):
                        target = connection["target"][0]

                        if "sig" in target:
                            signal = target["sig"]
                        elif "slice" in target:
                            signal = target["slice"][0]["signal"]
                        else:
                            signal = None
                            print(
                                f"Warning: Unknown target structure for instance '{instance['name']}', port '{connection['portname']}'"
                            )

                        if signal == target_signal:
                            return f"{instance['name']},{connection['portname']}"

    return None


def _generate_top_level_connections(proto_dict: ParsedProtoVLSIR):
    """
    Generate the top-level connections from the proto_dict.
    """
    top_level_connections = {}

    # Iterate over the top-level module ports
    for module in proto_dict.get("modules", []):
        for port in module.get("ports", []):
            port_signal = port["signal"]
            connection = _find_port_connection(proto_dict, port_signal)
            if connection:
                top_level_connections[port_signal] = connection

    return top_level_connections


def _find_port_connection(proto_dict: ParsedProtoVLSIR, port_signal):
    """
    Find the connection of the port signal in the proto_dict.
    """
    # Search within the module instances
    for module in proto_dict.get("modules", []):
        for instance in module.get("instances", []):
            instance_name = instance["name"]
            for connection in instance.get("connections", []):
                target = connection["target"][0]

                # Handle different structures of target
                if "sig" in target:
                    signal = target["sig"]
                elif "slice" in target:
                    signal = target["slice"][0]["signal"]
                else:
                    signal = None
                    print(
                        f"Warning: Unknown target structure in connection for instance '{instance_name}', port '{connection['portname']}'"
                    )

                if signal == port_signal:
                    return f"{instance_name},{connection['portname']}"
    return None


def _extract_instance_parameters(proto_dict: ParsedProtoVLSIR):
    """
    Extract the instance parameters from the proto_dict.
    """
    instance_parameters = {}

    for module in proto_dict.get("modules", []):
        for instance in module.get("instances", []):
            instance_name = instance["name"]
            instance_info = {
                "component": _extract_component_name(instance),
                "info": {},
                "settings": {},
            }

            # Extract parameters into the settings
            for parameter in instance.get("parameters", []):
                param_name = parameter["name"]
                param_value = _extract_parameter_value(parameter["value"])
                instance_info["settings"][param_name] = param_value

            # Extract connections and add to settings
            instance_info["settings"]["ports"] = {}
            for connection in instance.get("connections", []):
                portname = connection["portname"]
                target = connection["target"][0]

                if "sig" in target:
                    target_signal = target["sig"]
                elif "slice" in target:
                    target_signal = target["slice"][0]["signal"]
                else:
                    # Handle cases where 'target' does not have 'sig' or 'slice'
                    target_signal = "unknown_signal"

                instance_info["settings"]["ports"][portname] = target_signal

            instance_parameters[instance_name] = instance_info

    return instance_parameters


def _extract_component_name(instance):
    """
    Extract the component name from the instance.
    """
    external_modules = instance.get("module", [])
    if external_modules:
        name = external_modules[0].get("external", [{}])[0].get("name", "")
        return f"{name}"
    return "unknown_component"


def _extract_parameter_value(value):
    """
    Extract the parameter value from the value dictionary.
    """
    if value and "literal" in value[0]:
        return value[0]["literal"]
    elif value and "prefixed" in value[0]:
        prefix = value[0]["prefixed"][0].get("prefix", "")
        int64_value = value[0]["prefixed"][0].get("int64_value", "")
        return f"{prefix}_{int64_value}"
    return None


def _generate_raw_netlist_dict_from_proto_dict(proto_dict: ParsedProtoVLSIR):
    """
    Generate a raw netlist dictionary from the proto_dict.
    """
    raw_netlist_dict = {"name": "", "instances": {}, "connections": {}, "ports": {}}

    # Extract the top-level module name
    if proto_dict.get("modules"):
        raw_netlist_dict["name"] = proto_dict["modules"][0].get("name", "")

    # Generate instances information
    raw_netlist_dict["instances"] = _extract_instance_parameters(proto_dict)

    # Generate connections
    raw_netlist_dict["connections"] = _parse_connections(proto_dict)

    # Generate top-level connections
    raw_netlist_dict["ports"] = _generate_top_level_connections(proto_dict)

    return raw_netlist_dict


def generate_raw_netlist_dict_from_module(module: AnalogueModule):
    """
    Generate a raw netlist dictionary from a hdl21 module object.
    This just gives us a raw structure of the hdl21 modules, we cannot use this json equivalently to a gdsfactory netlist.
    """
    proto_dict = _parse_module_to_proto_dict(module)
    return _generate_raw_netlist_dict_from_proto_dict(proto_dict)


def generate_raw_yaml_from_module(module: AnalogueModule):
    """
    Generate a raw netlist yaml from a hdl21 module object which could be manually edited for specific instances
    related to the corresponding SPICE.
    """
    import yaml

    raw_netlist = generate_raw_netlist_dict_from_module(module)
    return yaml.dump(raw_netlist, default_flow_style=False)
