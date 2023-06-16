import os
import shutil

def copy_source_folder(source_directory: str, target_directory: str):
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
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
    example_name:str="simple_design"
):
    """
    We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.
    """
    example_design_folder = os.environ["PIEL_PACKAGE_DIRECTORY"] + "/docs/examples/" + example_name
    design_folder = os.environ["DESIGNS"]
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
    return pathlib.Path.exists(design_folder)

__all__ = [
    "copy_source_folder",
    "setup_example_design",
    "check_example_design"
]
