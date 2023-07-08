"""
This file allows us to automate several aspects of creating a fully compatible project structure.
"""
from .config import piel_path_types
from .file_system import return_path, write_script, read_json, check_path_exists

__all__ = [
    "read_configuration",
    "create_setup_py_from_config_json",
]


def read_configuration(design_directory: piel_path_types) -> dict:
    """
    This function reads the configuration file found in the design directory.

    Args:
        design_directory(piel_path_types): Design directory PATH.

    Returns:
        config_dictionary(dict): Configuration dictionary.
    """
    design_directory = return_path(design_directory)
    config_path = design_directory / "config.json"
    check_path_exists(config_path, raise_errors=True)
    config_dictionary = read_json(config_path)
    return config_dictionary


def create_setup_py_from_config_json(design_directory: piel_path_types) -> None:
    """
    This function creates a setup.py file from the config.json file found in the design directory.

    Args:
        design_directory(piel_path_types): Design directory PATH or module name.

    Returns:
        None
    """
    design_directory = return_path(design_directory)
    config_json_path = design_directory / "config.json"
    config_dictionary = read_json(config_json_path)
    commands_list = [
        "#!/usr/bin/env python",
        "from distutils.core import setup \n",
        "setup(name=" + config_dictionary["NAME"] + ",",
        "\tversion=" + str(config_dictionary["VERSION"]) + ",",
        "\tdescription=" + config_dictionary["DESCRIPTION"] + ","
        "\tauthor=" + "Dario Quintero" + ",",
        "\tauthor_email=" + "darioaquintero@gmail.com" + ",",
        "\turl=" + "https://github.com/daquintero/piel" + ",",
        "\tpackages=" + str(["models", "tb"]) + ",",
        ")",
    ]
    script = " \n".join(commands_list)
    write_script(directory_path=design_directory, script=script, script_name="setup.py")
