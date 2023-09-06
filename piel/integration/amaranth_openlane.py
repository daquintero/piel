"""
This file enhances some functions that easily translates between an `amaranth` function to implement a `openlane` flow.
"""
import amaranth as am
from typing import Optional, Literal
import types
from ..tools.amaranth import (
    construct_amaranth_module_from_truth_table,
    generate_verilog_from_amaranth,
    verify_truth_table,
)
from ..file_system import return_path, create_new_directory
from ..project_structure import create_empty_piel_project
from ..tools.openlane.v1 import write_configuration_openlane_v1
from ..tools.openlane.v2 import run_openlane_flow
from ..tools.openlane.defaults import test_basic_open_lane_configuration_v1
from ..config import piel_path_types

__all__ = ["layout_amaranth_truth_table_through_openlane"]


def layout_amaranth_truth_table_through_openlane(
    amaranth_module: am.Module,
    inputs_name_list: list[str],
    outputs_name_list: list[str],
    parent_directory: piel_path_types,
    target_directory_name: Optional[str] = None,
    openlane_version: Literal["v1", "v2"] = "v1",
):
    """
    This function implements an amaranth truth-table module through the openlane flow. There are several ways to
    implement a module. Fundamentally, this requires the verilog files to be generated from the openlane-module in a
    particular directory. For the particular directory provided, this function will generate the verilog files in the
    corresponding directory. It can also generate the ``openlane`` configuration files for this particular location.

    This function does a few things:

    1. Starts off from a ``amaranth`` module class.
    2. Determines the output directory in which to generate the files, and creates one accordingly if it does not exist.
    3. Generates the verilog files from the ``amaranth`` module class.
    4. Generates the ``openlane`` configuration files for this particular location.
    5. Implements the ``openlane`` flow for this particular location to generate a chip.

    Args:
        amaranth_module (amaranth.Module): Amaranth module class.
        inputs_name_list (list[str]): List of input names.
        outputs_name_list (list[str]): List of output names.
        parent_directory (piel_path_types): Parent directory PATH.
        target_directory_name (Optional[str]): Target directory name. If none is provided, it will default to the name of the amaranth elaboratable class.
        openlane_version (Literal["v1", "v2"]): OpenLane version. Defaults to ``v1``.

    Returns:
        None
    """
    # Determines the output directory in which to generate the files, and creates one accordingly if it does not exist.
    if isinstance(parent_directory, types.ModuleType):
        parent_directory = return_path(parent_directory)
        design_directory = parent_directory
        src_folder = parent_directory / "src"
    else:
        parent_directory = return_path(parent_directory)
        if not parent_directory.exists():
            create_new_directory(parent_directory)
        # Creates a corresponding `piel` project accordingly.
        target_directory_name = (
            target_directory_name
            if target_directory_name is not None
            else amaranth_module.__class__.__name__
        )
        create_empty_piel_project(
            project_name=target_directory_name,
            parent_directory=parent_directory,
        )
        design_directory = (
            parent_directory / target_directory_name / target_directory_name
        )
        src_folder = design_directory / "src"

    # Generates the verilog files from the ``amaranth`` elaboratable class.
    ports_list = inputs_name_list + outputs_name_list
    generate_verilog_from_amaranth(
        amaranth_module=amaranth_module,
        ports_list=ports_list,
        target_file_name="our_truth_table_module.v",
        target_directory=src_folder,
    )
    # Generates the ``openlane`` configuration files for this particular location.
    our_amaranth_openlane_config = test_basic_open_lane_configuration_v1

    # Runs the `openlane` flow
    if openlane_version == "v1":
        write_configuration_openlane_v1(
            configuration=our_amaranth_openlane_config,
            design_directory=design_directory,
        )
        # TODO Copy to the openlane root directory.

    elif openlane_version == "v2":
        run_openlane_flow(
            configuration=our_amaranth_openlane_config,
            design_directory=design_directory,
        )
