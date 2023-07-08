"""
These set of functions aim to provide functionality to automate interacting with OpenLanes v1 design into Python environment, whilst `OpenLanes2` is under development.
"""

import os
import pathlib
import json
from piel.parametric import multi_parameter_sweep
from piel.file_system import (
    copy_source_folder,
    permit_script_execution,
    read_json,
    return_path,
    run_script,
    write_script,
)

__all__ = [
    "check_config_json_exists_openlane_v1",
    "check_design_exists_openlane_v1",
    "configure_and_run_design_openlane_v1",
    "configure_parametric_designs_openlane_v1",
    "configure_flow_script_openlane_v1",
    "create_parametric_designs_openlane_v1",
    "get_design_directory_from_root_openlane_v1",
    "get_latest_version_root_openlane_v1",
    "read_configuration_openlane_v1",
    "write_configuration_openlane_v1",
]


def check_config_json_exists_openlane_v1(
    design_name: str,
    root_directory: str | pathlib.Path | None = None,
) -> bool:
    """
    Checks if a design has a `config.json` file.

    Args:
        design_name(str): Name of the design.

    Returns:
        config_json_exists(bool): True if `config.json` exists.
    """
    config_json_exists = False
    design_directory = get_design_directory_from_root_openlane_v1(
        design_name=design_name, root_directory=root_directory
    )
    if (design_directory / "config.json").exists():
        config_json_exists = True
    return config_json_exists


def check_design_exists_openlane_v1(
    design_name: str,
    root_directory: str | pathlib.Path | None = None,
) -> bool:
    """
    Checks if a design exists in the OpenLane v1 design folder.

    Lists all designs inside the Openlane V1 design root.

    Args:
        design_name(str): Name of the design.

    Returns:
        design_exists(bool): True if design exists.
    """
    if root_directory is None:
        root_directory = get_latest_version_root_openlane_v1()

    design_exists = False
    openlane_v1_design_directory = root_directory / "designs"
    all_openlane_v1_designs = list(openlane_v1_design_directory.iterdir())
    if (openlane_v1_design_directory / design_name) in all_openlane_v1_designs:
        design_exists = True
    return design_exists


def configure_and_run_design_openlane_v1(
    design_name: str,
    configuration: dict | None = None,
    root_directory: str | pathlib.Path | None = None,
) -> None:
    """
    Configures and runs an OpenLane v1 design.

    This function does the following:
    1. Check that the design_directory provided is under $OPENLANE_ROOT/<latestversion>/designs
    2. Check if `config.json` has already been provided for this design. If a configuration dictionary is inputted into the function parameters, then it overwrites the default `config.json`.
    3. Create a script directory, a script is written and permissions are provided for it to be executable.
    4. Permit and execute the `openlane_flow.sh` script in the `scripts` directory.

    Args:
        design_name(str): Name of the design.
        configuration(dict | None): Configuration dictionary.
        root_directory(str | pathlib.Path): Design directory.

    Returns:
        None
    """

    design_directory = get_design_directory_from_root_openlane_v1(
        design_name=design_name, root_directory=root_directory
    )

    # Check configuration
    config_json_exists = check_config_json_exists_openlane_v1(design_name)

    if config_json_exists:
        pass
    else:
        if configuration is None:
            raise ValueError(
                "Configuration dictionary is None. Please provide a configuration dictionary."
            )
        else:
            write_configuration_openlane_v1(
                configuration=configuration, design_directory=design_directory
            )

    # Create script directory
    configure_flow_script_openlane_v1(design_name=design_name)

    # Execute script
    openlane_flow_script_path = design_directory / "scripts" / "openlane_flow.sh"
    permit_script_execution(openlane_flow_script_path)
    run_script(openlane_flow_script_path)


def configure_flow_script_openlane_v1(
    design_name: str,
    root_directory: str | pathlib.Path | None = None,
) -> None:
    """
    Configures the OpenLane v1 flow script after checking that the design directory exists.

    Args:
        design_directory(str | pathlib.Path | None): Design directory. Defaults to latest OpenLane root.

    Returns:
        None
    """
    design_directory = get_design_directory_from_root_openlane_v1(
        design_name=design_name, root_directory=root_directory
    )
    if check_design_exists_openlane_v1(design_name):
        commands_list = [
            "#!/bin/sh",
            "cd " + str(root_directory),
            "./flow.tcl -design " + design_name,
        ]
        script = " \n".join(commands_list)
        write_script(
            directory_path=design_directory / "scripts",
            script=script,
            script_name="openlane_flow.sh",
        )
    else:
        raise ValueError(
            "Design: "
            + str(design_name)
            + " not found in "
            + os.environ["OPENLANE_ROOT"]
        )


def configure_parametric_designs_openlane_v1(
    design_name: str,
    parameter_sweep_dictionary: dict,
    add_id: bool = True,
) -> list:
    """
    For a given `source_design_directory`, this function reads in the config.json file and returns a set of parametric sweeps that gets used when creating a set of parametric designs.

    Args:
        add_id(bool): Add an ID to the design name. Defaults to True.
        parameter_sweep_dictionary(dict): Dictionary of parameters to sweep.
        source_design_directory(str | pathlib.Path): Source design directory.

    Returns:
        configuration_sweep(list): List of configurations to sweep.
    """
    source_configuration = read_configuration_openlane_v1(design_name=design_name)
    configuration_sweep = multi_parameter_sweep(
        base_design_configuration=source_configuration,
        parameter_sweep_dictionary=parameter_sweep_dictionary,
    )
    if add_id:
        i = 0
        for configuration_i in configuration_sweep:
            # Checks the unique ID of the configuration
            configuration_id = id(configuration_i)
            # Adds the ID to the configuration list
            configuration_sweep[i]["id"] = configuration_id
            i += 1
    return configuration_sweep


def create_parametric_designs_openlane_v1(
    design_name: str,
    parameter_sweep_dictionary: dict,
    target_directory: str | pathlib.Path | None = None,
) -> None:
    """
    Takes a OpenLane v1 source directory and creates a parametric combination of these designs.

    Args:
        design_name(str): Name of the design.
        parameter_sweep_dictionary(dict): Dictionary of parameters to sweep.
        target_directory(str | pathlib.Path | None): Optional target directory.

    Returns:
        None
    """
    source_design_directory = get_design_directory_from_root_openlane_v1(
        design_name=design_name
    )
    source_design_name = design_name

    if target_directory is None:
        target_directory = get_latest_version_root_openlane_v1() / "designs"

    parameter_sweep_configuration_list = configure_parametric_designs_openlane_v1(
        add_id=True,
        design_name=design_name,
        parameter_sweep_dictionary=parameter_sweep_dictionary,
    )

    for configuration_i in parameter_sweep_configuration_list:
        # Create a target directory with the name of the design and the configuration ID
        target_directory_i = (
            target_directory / source_design_name + "_" + str(configuration_i["id"])
        )
        # Copy the source design directory to the target directory
        copy_source_folder(
            source_directory=source_design_directory,
            target_directory=target_directory_i,
        )


def get_latest_version_root_openlane_v1() -> pathlib.Path:
    """
    Gets the latest version root of OpenLane v1.
    """
    openlane_tool_directory = pathlib.Path(os.environ["OPENLANE_ROOT"])
    latest_openlane_version = list(openlane_tool_directory.iterdir())
    openlane_v1_design_directory = openlane_tool_directory / latest_openlane_version[-1]
    return openlane_v1_design_directory


def get_design_directory_from_root_openlane_v1(
    design_name: str,
    root_directory: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """
    Gets the design directory from the root directory.

    Args:
        design_name(str): Name of the design.
        root_directory(str | pathlib.Path): Design directory.

    Returns:
        design_directory(pathlib.Path): Design directory.
    """
    if root_directory is None:
        root_directory = get_latest_version_root_openlane_v1()

    root_directory = return_path(root_directory)
    design_exists = check_design_exists_openlane_v1(design_name)
    if design_exists:
        pass
    else:
        raise ValueError(
            "Design: "
            + design_name
            + " is not found in "
            + str(root_directory / "designs")
        )
    design_directory = root_directory / "designs" / design_name
    return design_directory


def read_configuration_openlane_v1(
    design_name: str,
    root_directory: str | pathlib.Path | None = None,
) -> dict:
    """
    Reads a `config.json` from a design directory.

    Args:
        design_name(str): Design name.
        root_directory(str | pathlib.Path): Design directory.

    Returns:
        configuration(dict): Configuration dictionary.
    """
    config_json_exists = check_config_json_exists_openlane_v1(design_name)
    if config_json_exists:
        design_directory = get_design_directory_from_root_openlane_v1(
            design_name=design_name, root_directory=root_directory
        )
        configuration = read_json(design_directory / "config.json")
        return configuration
    else:
        raise ValueError(
            "Configuration file for design: "
            + design_name
            + " is not found in "
            + str(root_directory / "designs" / design_name)
        )


def write_configuration_openlane_v1(
    configuration: dict,
    design_directory: str | pathlib.Path,
) -> None:
    """
    Writes a `config.json` onto a `design_directory`

    Args:
        configuration(dict): OpenLane configuration dictionary.
        design_directory(str): Design directory PATH.

    Returns:
        None
    """
    with open(str((design_directory / "config.json").resolve()), "w") as write_file:
        json.dump(configuration, write_file, indent=4)
