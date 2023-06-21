import os
import pathlib
import json
from ..file_system import (
    return_path,
    write_script,
    run_script,
    permit_script_execution,
)


def configure_flow_script_openlane_v1(
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
    if root_directory is None:
        root_directory = get_latest_version_root_openlane_v1()

    config_json_exists = False
    openlane_v1_design_directory = root_directory / "designs"
    design_directory = openlane_v1_design_directory / design_name
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
    if design_name in all_openlane_v1_designs:
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
    if root_directory is None:
        root_directory = get_latest_version_root_openlane_v1()

    root_directory = return_path(root_directory)
    design_exists = check_design_exists_openlane_v1(design_name)
    design_directory = root_directory / "designs" / design_name

    # Check design existence
    if design_exists:
        pass
    else:
        raise ValueError(
            "Design: "
            + design_name
            + " is not found in "
            + str(root_directory / "designs")
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
            write_openlane_configuration(
                configuration=configuration, design_directory=design_directory
            )

    # Create script directory
    configure_flow_script_openlane_v1(design_directory=design_directory)

    # Execute script
    openlane_flow_script_path = design_directory / "scripts" / "openlane_flow.sh"
    permit_script_execution(openlane_flow_script_path)
    run_script(openlane_flow_script_path)


def get_latest_version_root_openlane_v1() -> pathlib.Path:
    """
    Gets the latest version root of OpenLane v1.
    """
    openlane_tool_directory = pathlib.Path(os.environ["OPENLANE_ROOT"])
    latest_openlane_version = list(openlane_tool_directory.iterdir())
    openlane_v1_design_directory = openlane_tool_directory / latest_openlane_version[-1]
    return openlane_v1_design_directory


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


__all__ = [
    "check_config_json_exists_openlane_v1",
    "check_design_exists_openlane_v1",
    "configure_and_run_design_openlane_v1",
    "configure_flow_script_openlane_v1",
    "get_latest_version_root_openlane_v1",
    "write_openlane_configuration",
]
