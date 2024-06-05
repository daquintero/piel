from collections.abc import Callable
from difflib import get_close_matches

import hdl21 as h
import sky130

from gplugins.schematic_editor import SchematicEditor
from .netlist import (
    _generate_raw_netlist_dict_from_proto_dict,
    _parse_module_to_proto_dict,
)

custom_mapping_dict = {
    "sky130_fd_pr__nfet_01v8": "sky130_fd_pr__rf_nfet_01v8_aM02W1p65L0p15",
    "sky130_fd_pr__pfet_01v8": "sky130_fd_pr__rf_pfet_01v8_mcM04W3p00L0p15",
}


def find_most_relevant_gds(
    component_name, component_dict=sky130.cells, custom_mapping=None
):
    if custom_mapping is None:
        custom_mapping = custom_mapping_dict

    if component_name in custom_mapping.keys():
        print(f"Mapping for {component_name}: {custom_mapping[component_name]}")
        return custom_mapping[component_name]

    all_components = [
        name for name in component_dict.keys() if "rf_test_coil" not in name
    ]
    closest_matches = get_close_matches(component_name, all_components, n=1, cutoff=0.1)
    print(f"Closest matches for {component_name}: {closest_matches}")
    return closest_matches[0] if closest_matches else component_name


def filter_port(port):
    """
    Filter the port name to match spice declaration to gds port name, specifically focused on the SKY130nm technology.
    """
    if port == "d":
        return "DRAIN"
    elif port == "g":
        return "GATE"
    elif port == "s":
        return "SOURCE"
    else:
        return port


def hdl21_module_to_schematic_editor(
    module: h.module,
    yaml_schematic_file_name: str,
    spice_gds_mapping_method: Callable | None = find_most_relevant_gds,
    port_filter_method: Callable = filter_port,
) -> SchematicEditor:
    """
    Constructs a SchematicEditor instance from a hdl21 module object.

    Args:
        module (h.module): The hdl21 module object.
        yaml_schematic_file_name (str): The yaml schematic file name.
        spice_gds_mapping_method (Callable): The method to map the spice instance name to the component name.
        port_filter_method (Callable): The method to filter the port name.
    """
    proto_dict = _parse_module_to_proto_dict(module)
    raw_netlist_dict = _generate_raw_netlist_dict_from_proto_dict(proto_dict)

    # This just gives us a raw structure of the hdl21 modules.
    se = SchematicEditor(yaml_schematic_file_name)
    print(raw_netlist_dict["instances"])
    for instance_name_i, instance_i in raw_netlist_dict["instances"].items():
        # Maps the spice instance name to the component name.
        # TODO implement setting mapping and custom name mapping
        if spice_gds_mapping_method is None:
            gds_component_name_i = instance_i["component"]
        else:
            gds_component_name_i = spice_gds_mapping_method(
                instance_i["component"], sky130.cells
            )
        se.add_instance(
            instance_name=instance_name_i,
            component=sky130.cells[gds_component_name_i](),
        )

    for connection_source_i, connection_target_i in raw_netlist_dict[
        "connections"
    ].items():
        source_instance, source_port = connection_source_i.split(",")
        target_instance, target_port = connection_target_i.split(",")
        source_port = port_filter_method(source_port)
        target_port = port_filter_method(target_port)
        se.add_net(source_instance, source_port, target_instance, target_port)

    return se
