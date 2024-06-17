"""
This file contains the design flow from going from a photonic component into an analogue.
"""
import sys
import hdl21 as h
from piel.types import CircuitComponent, PathTypes
from piel.integration import (
    gdsfactory_netlist_with_hdl21_generators,
    construct_hdl21_module,
)


def extract_component_spice_from_netlist(
    component: CircuitComponent, output_path: PathTypes = sys.stdout, fmt: str = "spice"
):
    """
    This function extracts the SPICE netlist from a component definition and writes it to a file. The function uses
    the HDL21 library to generate the SPICE netlist from the component's netlist. The netlist is then written to a
    file in the specified format.

    Args:
        component (CircuitComponent): The component for which to extract the SPICE netlist.
        output_path (str): The path to the output file where the SPICE netlist will be written.
        fmt (str, optional): The format in which the netlist will be written. Defaults to "spice".

    Returns:
        None
    """
    # Get the netlist of the component
    component_netlist = component.get_netlist(
        allow_multiple=True, exclude_port_types="optical"
    )

    #
    spice_component_netlist = gdsfactory_netlist_with_hdl21_generators(
        component_netlist
    )

    hdl21_module = construct_hdl21_module(spice_netlist=spice_component_netlist)

    h.netlist(
        hdl21_module,
        output_path,
        fmt="spice",
    )
