import os
import pathlib
import json
from ..parametric import multi_parameter_sweep
from ..file_system import copy_source_folder, return_path, write_script


def configure_openlane_v1_flow_script(
    design_directory: str | pathlib.Path,
    design_name: str,
) -> None:
    """
    Configures the OpenLane v1 flow script after checking that the design directory exists.

    Args:
        design_directory(str | pathlib.Path): Design directory.

    Returns:
        None
    """
    design_directory = return_path(design_directory)
    if check_design_exists_openlane_v1(design_name):
        commands_list = [
            "cd $OPENLANE_ROOT",
            "./flow.tcl -design " + design_name,
        ]
        script = ";\n".join(commands_list)
        write_script(
            design_directory=design_directory / "scripts",
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


def check_design_exists_openlane_v1(
    design_name: str,
) -> bool:
    """
    Checks if a design exists in the OpenLane v1 design folder.

    Lists all designs inside the Openlane V1 design root.

    Args:
        design_name(str): Name of the design.

    Returns:
        design_exists(bool): True if design exists.
    """
    design_exists = False
    openlane_v1_design_directory = pathlib.Path(os.environ["OPENLANE_ROOT"]) / "designs"
    all_openlane_v1_designs = list(openlane_v1_design_directory.iterdir())
    if design_name in all_openlane_v1_designs:
        design_exists = True
    return design_exists


def configure_parametric_designs(
    parameter_sweep_dictionary: dict,
    source_design_directory: str | pathlib.Path,
) -> list:
    """
    For a given `source_design_directory`, this function reads in the config.json file and returns a set of parametric sweeps that gets used when creating a set of parametric designs.

    Args:
        parameter_sweep_dictionary(dict): Dictionary of parameters to sweep.
        source_design_directory(str | pathlib.Path): Source design directory.

    Returns:
        configuration_sweep(list): List of configurations to sweep.
    """
    source_design_directory = return_path(source_design_directory)
    source_design_configuration_path = source_design_directory / "config.json"
    with open(source_design_configuration_path, "r") as config_json:
        source_configuration = json.load(config_json)
    configuration_sweep = multi_parameter_sweep(
        base_design_configuration=source_configuration,
        parameter_sweep_dictionary=parameter_sweep_dictionary,
    )
    return configuration_sweep


def create_parametric_designs(
    parameter_sweep_dictionary: dict,
    source_design_directory: str | pathlib.Path,
    target_directory: str | pathlib.Path,
) -> None:
    """
    Takes a OpenLane v1 source directory and creates a parametric combination of these designs.

    Args:
        parameter_sweep_dictionary(dict): Dictionary of parameters to sweep.
        source_design_directory(str): Source design directory.
        target_directory(str): Target directory.

    Returns:
        None
    """
    source_design_directory = return_path(source_design_directory)
    source_design_name = source_design_directory.parent.name
    target_directory = return_path(target_directory)
    parameter_sweep_configuration_list = configure_parametric_designs(
        parameter_sweep_dictionary=parameter_sweep_dictionary,
        source_design_directory=source_design_directory,
    )

    for configuration_i in parameter_sweep_configuration_list:
        configuration_id = id(configuration_i)
        configuration_i["parametric_id"] = configuration_id
        # TODO improve this for relevant parametric variation naming
        target_directory_i = (
            target_directory / source_design_name + "_" + str(configuration_id)
        )
        copy_source_folder(
            source_directory=source_design_directory,
            target_directory=target_directory_i,
        )


def write_openlane_configuration(
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


def configure_and_run_openlane_v1_design(
    design_directory: str | pathlib.Path,
    configuration: dict | None = None,
) -> None:
    """
    Configures and runs an OpenLane v1 design.

    This function does the following:
    1. Checks that the design_directory provided is under $OPENLANE_ROOT
    2. Checks if `config.json` has already been provided for this design. If a configuration dictionary is inputted into the function parameters, then it overwrites the default `config.json`.
    3. Creates a script directory, a script is written and permissions are provided for it to be executable.
    4. It executes the `openlane_flow.sh` script in the `scripts` directory.

    Args:
        design_directory(str | pathlib.Path): Design directory.
        configuration(dict | None): Configuration dictionary.

    Returns:
        None
    """
    design_directory = return_path(design_directory)
    pass


__all__ = [
    "check_design_exists_openlane_v1",
    "configure_openlane_v1_flow_script",
    "configure_parametric_designs",
    "create_parametric_designs",
    "write_openlane_configuration",
]
