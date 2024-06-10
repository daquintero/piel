"""
This file allows us to automate several aspects of creating a fully compatible project structure.
"""
import subprocess
import types
from typing import Literal, Optional

from .types import PathTypes
from .file_system import (
    return_path,
    read_json,
    check_path_exists,
    create_new_directory,
    write_file,
)


def create_setup_py(
    design_directory: PathTypes,
    project_name: Optional[str] = None,
    from_config_json: bool = True,
) -> None:
    """
    This function creates a setup.py file from the config.json file found in the design directory.

    Args:
        design_directory(PathTypes): Design directory PATH or module name.

    Returns:
        None
    """
    design_directory = return_path(design_directory)
    if from_config_json:
        config_json_path = design_directory / "config.json"
        config_dictionary = read_json(config_json_path)
    else:
        config_dictionary = {
            "NAME": project_name
            if project_name is not None
            else "example_piel_project",
            "VERSION": '"0.0.1"',
            "DESCRIPTION": '"Example empty piel project."\n',
        }
    commands_list = [
        "#!/usr/bin/env python",
        "from distutils.core import setup \n",
        'setup(name="' + config_dictionary["NAME"] + '",',
        "\tversion=" + str(config_dictionary["VERSION"]) + ",",
        "\tdescription=" + config_dictionary["DESCRIPTION"] + ","
        "\tauthor=" + '"Dario Quintero"' + ",",
        "\tauthor_email=" + '"darioaquintero@gmail.com' + '",',
        "\turl=" + '"https://github.com/daquintero/piel' + '",',
        "\tpackages="
        + str(
            [project_name] if project_name is not None else ['"example_emtpy_project"']
        )
        + ",",
        ")",
    ]
    script = " \n".join(commands_list)
    write_file(directory_path=design_directory, file_text=script, file_name="setup.py")


def create_empty_piel_project(project_name: str, parent_directory: PathTypes) -> None:
    """
    This function creates an empty piel-structure project in the target directory. Structuring your files in this way
    enables the co-design and use of the tools supported by piel whilst maintaining the design flow ordered,
    clean and extensible. You can read more about it in the documentation TODO add link.

    TODO just make this a cookiecutter. TO BE DEPRECATED whenever I get round to that.

    Args:
        project_name(str): Name of the project.
        parent_directory(PathTypes): Parent directory of the project.

    Returns:
        None
    """
    target_directory = return_path(parent_directory) / project_name

    # Create the main directory
    create_new_directory(target_directory)
    create_new_directory(target_directory / "docs")  # Documentation files

    # Create project structure
    module_directory = target_directory / project_name
    create_new_directory(module_directory)
    create_new_directory(module_directory / "io")  # IO files
    create_new_directory(
        module_directory / "analogue"
    )  # analogue `gdsfactory` layout files

    #### COMPONENTS ####
    create_new_directory(module_directory / "components")  # Custom components
    create_new_directory(
        module_directory / "components" / "analogue"
    )  # Custom components
    create_new_directory(
        module_directory / "components" / "photonics"
    )  # Custom components
    create_new_directory(
        module_directory / "components" / "digital"
    )  # Custom components

    #### MODELS ####
    create_new_directory(
        module_directory / "models"
    )  # Custom Python models for digital, analog and photonic
    create_new_directory(module_directory / "models" / "analogue")
    create_new_directory(module_directory / "models" / "frequency")
    create_new_directory(module_directory / "models" / "logic")
    create_new_directory(module_directory / "models" / "physical")
    create_new_directory(module_directory / "models" / "transient")

    create_new_directory(
        module_directory / "photonic"
    )  # photonic `gdsfactory` layout files
    create_new_directory(module_directory / "runs")  # OpenLane v1 flow
    create_new_directory(module_directory / "scripts")  # Python scripts
    create_new_directory(module_directory / "sdc")  # SDC files
    create_new_directory(module_directory / "src")  # Digital source files
    create_new_directory(module_directory / "tb")  # Digital testbench files
    create_new_directory(module_directory / "tb" / "out")  # Digital testbench files

    ##### Create __init__.py files
    write_file(
        directory_path=module_directory, file_text="", file_name="__init__.py"
    )  # Top level
    write_file(
        directory_path=module_directory / "analogue",
        file_text="",
        file_name="__init__.py",
    )

    ### COMPONENTS ###
    write_file(
        directory_path=module_directory / "components",
        file_text="",
        file_name="__init__.py",
    )  # Models
    write_file(
        directory_path=module_directory / "components" / "photonics",
        file_text="",
        file_name="__init__.py",
    )  # Models
    write_file(
        directory_path=module_directory / "components" / "analogue",
        file_text="",
        file_name="__init__.py",
    )  # Models
    write_file(
        directory_path=module_directory / "components" / "digital",
        file_text="",
        file_name="__init__.py",
    )  # Models

    ### MODELS ###
    write_file(
        directory_path=module_directory / "models",
        file_text="",
        file_name="__init__.py",
    )  # Models
    write_file(
        directory_path=module_directory / "models" / "analogue",
        file_text="",
        file_name="__init__.py",
    )  # Models
    write_file(
        directory_path=module_directory / "models" / "frequency",
        file_text="",
        file_name="__init__.py",
    )  # Models
    write_file(
        directory_path=module_directory / "models" / "logic",
        file_text="",
        file_name="__init__.py",
    )  # Models
    write_file(
        directory_path=module_directory / "models" / "physical",
        file_text="",
        file_name="__init__.py",
    )  # Models
    write_file(
        directory_path=module_directory / "models" / "transient",
        file_text="",
        file_name="__init__.py",
    )  # Models

    write_file(
        directory_path=module_directory / "photonic",
        file_text="",
        file_name="__init__.py",
    )
    write_file(
        directory_path=module_directory / "tb", file_text="", file_name="__init__.py"
    )

    ##### Create setup.py
    create_setup_py(target_directory, project_name=project_name, from_config_json=False)

    ##### README.md
    write_file(
        directory_path=target_directory / "docs",
        file_text=project_name,
        file_name="README.md",
    )

    # TODO suitable .gitignore


def get_module_folder_type_location(
    module: types.ModuleType,
    folder_type: Literal["digital_source", "digital_testbench"],
):
    """
    This is an easy helper function that saves a particular file in the corresponding location of a `piel` project structure.

    TODO DOCS
    """
    module_path = return_path(module)
    folder_path = module_path
    if folder_type == "digital_source":
        folder_path = module_path / "src"
    elif folder_type == "digital_testbench":
        folder_path = module_path / "tb"
    return folder_path


def pip_install_local_module(module_path: PathTypes):
    """
    This function installs a local module in editable mode.

    Args:
        module_path(PathTypes): Path to the module to be installed.

    Returns:
        None
    """
    module_path = return_path(module_path)
    try:
        subprocess.check_call(["pip", "install", "-e", str(module_path)])
        print(f"Local module at '{module_path}' installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install local module at '{module_path}'.")


def read_configuration(design_directory: PathTypes) -> dict:
    """
    This function reads the configuration file found in the design directory.

    Args:
        design_directory(PathTypes): Design directory PATH.

    Returns:
        config_dictionary(dict): Configuration dictionary.
    """
    design_directory = return_path(design_directory)
    config_path = design_directory / "config.json"
    check_path_exists(config_path, raise_errors=True)
    config_dictionary = read_json(config_path)
    return config_dictionary
