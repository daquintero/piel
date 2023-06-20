import pathlib
import json
from ..parametric import multi_parameter_sweep
from ..file_system import copy_source_folder, return_path


def write_openlane_configuration(
    configuration: dict,
    design_directory: str | pathlib.Path,
) -> None:
    """
    Writes a config.json onto a design_directory

    Args:
        configuration(dict): OpenLane configuration dictionary.
        design_directory(str): Design directory PATH.

    Returns:
        None
    """
    with open(str((design_directory / "config.json").resolve()), "w") as write_file:
        json.dump(configuration, write_file, indent=4)


def configure_parametric_designs(
    parameter_sweep_dictionary: dict,
    source_design_directory: str | pathlib.Path,
) -> list:
    """
    For a given source_design_directory, this function reads in the config.json file and returns a set of parametric sweeps that gets used when creating a set of parametric designs.

    Args:
        parameter_sweep_dictionary(dict): Dictionary of parameters to sweep.
        source_design_directory(str): Source design directory.

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


__all__ = ["write_openlane_configuration"]
