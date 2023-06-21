import subprocess

import openlane
import os
import pathlib
import shutil
import stat
from typing import Literal


def check_directory_exists(directory_path: str | pathlib.Path) -> bool:
    """
    Checks if a directory exists.

    Args:
        directory_path(str | pathlib.Path): Input path.

    Returns:
        directory_exists(bool): True if directory exists.
    """
    directory_exists = False
    directory_path = return_path(directory_path)
    if directory_path.exists():
        directory_exists = True
    else:
        pass
    return directory_exists


def check_example_design(design_name: str | pathlib.Path = "simple_design") -> bool:
    """
    We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.

    Args:
        design_name(str): Name of the design to check.

    Returns:
        None
    """
    design_folder = (
        pathlib.Path(os.environ["DESIGNS"]) / design_name
    )  # TODO verify this copying operation
    return design_folder.exists()


def copy_source_folder(
    source_directory: str | pathlib.Path, target_directory: str | pathlib.Path
) -> None:
    """
    Copies the files from a source_directory to a target_directory

    Args:
        source_directory(str): Source directory.
        target_directory(str): Target directory.

    Returns:
        None
    """
    source_directory = return_path(source_directory)
    target_directory = return_path(target_directory)
    if target_directory.exists():
        answer = input("Confirm deletion of: " + str(target_directory.resolve()))
        if answer.upper() in ["Y", "YES"]:
            shutil.rmtree(target_directory)
        elif answer.upper() in ["N", "NO"]:
            print(
                "Copying files now from: "
                + str(source_directory.resolve())
                + " to "
                + str(target_directory.resolve())
            )

    shutil.copytree(
        source_directory,
        target_directory,
        symlinks=False,
        ignore=None,
        copy_function=shutil.copy2,
        ignore_dangling_symlinks=False,
        dirs_exist_ok=False,
    )


def create_new_directory(
    directory_path: str | pathlib.Path,
) -> None:
    """
    Creates a new directory.

    If the parents of the target_directory do not exist, they will be created too.

    Args:
        directory_path(str | pathlib.Path): Input path.

    Returns:
        None
    """
    directory_path = return_path(directory_path)

    # Check permissions of the parent to be able to create the directory
    parent_directory = directory_path.parent
    parent_directory_permissions = oct(parent_directory.stat().st_mode)

    # If permissions are not read, write and execute for all, we change them
    if parent_directory_permissions != "0o777":
        permit_directory_all(parent_directory)

    # Create the directory
    directory_path.mkdir(parents=True)


def permit_script_execution(script_path: str | pathlib.Path) -> None:
    """
    Permits the execution of a script.

    Args:
        script_path(str): Script path.

    Returns:
        None
    """
    script = return_path(script_path)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def permit_directory_all(directory_path: str | pathlib.Path) -> None:
    """
    Permits a directory to be read, written and executed. Use with care as it can be a source for security issues.

    Args:
        directory_path(str | pathlib.Path): Input path.

    Returns:
        None
    """
    directory_path = return_path(directory_path)
    try:
        directory_path.chmod(0o777)
    except PermissionError:
        print(
            "Could not change permissions of directory: "
            + str(directory_path.resolve())
            + " to 777. Your Python executable might not have the required permissions."
        )


def return_path(input_path: str | pathlib.Path) -> pathlib.Path:
    """
    Returns a pathlib.Path to be able to perform operations accordingly internally.

    This allows us to maintain compatibility between POSIX and Windows systems.

    Args:
        input_path(str): Input path.

    Returns:
        pathlib.Path: Pathlib path.
    """
    if type(input_path) == str:
        output_path = pathlib.Path(input_path)
    elif isinstance(input_path, pathlib.Path):
        output_path = input_path
    else:
        raise ValueError(
            "input_path: " + str(input_path) + " is of type: " + str(type(input_path))
        )
    return output_path


def run_script(script_path: str | pathlib.Path) -> None:
    """
    Runs a script on the filesystem `script_path`.

    Args:
        script_path(str): Script path.

    Returns:
        None
    """
    script = return_path(script_path)
    subprocess.run(script, shell=True, check=True, capture_output=True)


def setup_example_design(
    project_source: Literal["piel", "openlane"] = "piel",
    example_name: str = "simple_design",
) -> None:
    """
    We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.

    Args:
        project_source(str): Source of the project.
        example_name(str): Name of the example design.

    Returns:
        None
    """
    if project_source == "piel":
        example_design_folder = (
            os.environ["PIEL_PACKAGE_DIRECTORY"] + "/docs/examples/" + example_name
        )
    elif project_source == "openlane":
        example_design_folder = (
            pathlib.Path(openlane.__file__).parent.resolve() / example_name
        )
    design_folder = os.environ["DESIGNS"] + "/" + example_name
    copy_source_folder(
        source_directory=example_design_folder, target_directory=design_folder
    )


def write_script(
    directory_path: str | pathlib.Path,
    script: str,
    script_name: str,
) -> None:
    """
    Records a `script_name` in the `scripts` project directory.

    Args:
        directory_path(str): Design directory.
        script(str): Script to write.
        script_name(str): Name of the script.

    Returns:
        None
    """
    directory_path = return_path(directory_path)

    directory_exists = check_directory_exists(directory_path)

    if directory_exists:
        pass
    else:
        try:
            create_new_directory(directory_path)
        except PermissionError:
            print(
                "Could not create directory: "
                + str(directory_path.resolve())
                + ". Your Python executable might not have the required permissions. Restructure your project directory so Python does not have to change permissions."
            )

    file = open(str(directory_path / script_name), "w")
    file.write(script)
    file.close()


__all__ = [
    "check_example_design",
    "copy_source_folder",
    "check_example_design",
    "permit_script_execution",
    "setup_example_design",
    "return_path",
    "write_script",
]
