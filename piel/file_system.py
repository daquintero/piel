import os
import shutil
import pathlib
import openlane
from typing import Literal

def copy_source_folder(source_directory: str, target_directory: str):
    if os.path.exists(target_directory):
        answer = input("Confirm deletion of: " + target_directory)
        if answer.upper() in ["Y", "YES"]:
            shutil.rmtree(target_directory)
        elif answer.upper() in["N", "NO"]:
            print("Copying files now from: " + source_directory + " to " + target_directory)

    shutil.copytree(
        source_directory,
        target_directory,
        symlinks=False,
        ignore=None,
        copy_function=shutil.copy2,
        ignore_dangling_symlinks=False,
        dirs_exist_ok=False,
    )


def setup_example_design(
    project_source: Literal["piel", "openlane"] = "piel",
    example_name:str="simple_design"
):
    """
    We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.
    """
    if project_source == "piel":
        example_design_folder = os.environ["PIEL_PACKAGE_DIRECTORY"] + "/docs/examples/" + example_name
    elif project_source == "openlane":
        example_design_folder = pathlib.Path(openlane.__file__).parent.resolve() / example_name
    design_folder = os.environ["DESIGNS"] + "/" + example_name
    copy_source_folder(
        source_directory=example_design_folder,
        target_directory=design_folder
    )

def check_example_design(
    example_name:str="simple_design"
):
    """
    We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.
    """
    design_folder = os.environ["DESIGNS"] + "/" + example_name # TODO verify this copying operation
    return os.path.exists(design_folder)

__all__ = [
    "copy_source_folder",
    "setup_example_design",
    "check_example_design"
]
