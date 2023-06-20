import openlane
import os
import pathlib
import shutil
from typing import Literal


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
    elif type(input_path) == pathlib.Path:
        output_path = input_path
    return output_path


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
    design_directory: str | pathlib.Path,
    script: str,
    script_name: str,
) -> None:
    """
    Records a `script_name` in the `scripts` project directory.

    Args:
        design_directory(str): Design directory.
        script(str): Script to write.
        script_name(str): Name of the script.

    Returns:
        None
    """
    design_directory = return_path(design_directory)
    file = open(str(design_directory / script_name), "w")
    file.write(script)
    file.close()


__all__ = [
    "check_example_design",
    "copy_source_folder",
    "check_example_design",
    "setup_example_design",
    "return_path",
    "write_script",
]
