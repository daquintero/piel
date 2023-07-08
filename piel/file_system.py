import glob
import json
import openlane
import os
import pathlib
import shutil
import stat
import subprocess
import types
from typing import Literal
from .config import piel_path_types


__all__ = [
    "check_path_exists",
    "check_example_design",
    "copy_source_folder",
    "create_new_directory",
    "delete_path",
    "delete_path_list_in_directory",
    "get_files_recursively_in_directory",
    "permit_directory_all",
    "permit_script_execution",
    "setup_example_design",
    "read_json",
    "return_path",
    "run_script",
    "write_script",
]


def check_path_exists(
    path: piel_path_types,
    raise_errors: bool = False,
) -> bool:
    """
    Checks if a directory exists.

    Args:
        path(piel_path_types): Input path.

    Returns:
        directory_exists(bool): True if directory exists.
    """
    directory_exists = False
    path = return_path(path)
    if path.exists():
        directory_exists = True
    else:
        if raise_errors:
            raise ValueError("Path: " + str(path) + " does not exist.")
    return directory_exists


def check_example_design(
    design_name: str = "simple_design",
    designs_directory: piel_path_types | None = None,
) -> bool:
    """
    We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.

    Args:
        design_name(str): Name of the design to check.
        designs_directory(piel_path_types): Directory that contains the DESIGNS environment flag.
        # TODO

    Returns:
        None
    """
    if designs_directory is None:
        designs_directory = pathlib.Path(os.environ["DESIGNS"])

    design_folder = (
        designs_directory / design_name
    )  # TODO verify this copying operation
    return design_folder.exists()


def copy_source_folder(
    source_directory: piel_path_types,
    target_directory: piel_path_types,
) -> None:
    """
    Copies the files from a source_directory to a target_directory

    Args:
        source_directory(piel_path_types): Source directory.
        target_directory(piel_path_types): Target directory.

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


def convert_list_to_path_list(
    input_list: list[piel_path_types],
) -> list[pathlib.Path]:
    """
    Converts a list of strings or pathlib.Path to a list of pathlib.Path.

    Args:
        input_list(list[piel_path_types]): Input list.

    Returns:
        output_list(list[pathlib.Path]): Output list.
    """
    output_list = []
    for item in input_list:
        item = return_path(item)
        output_list.append(item)
    return output_list


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


def delete_path(path: str | pathlib.Path) -> None:
    """
    Deletes a path.

    Args:
        path(str | pathlib.Path): Input path.

    Returns:
        None
    """
    path = return_path(path)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def delete_path_list_in_directory(
    directory_path: piel_path_types,
    path_list: list,
    ignore_confirmation: bool = False,
    validate_individual: bool = False,
) -> None:
    """
    Deletes a list of files in a directory.

    Args:
        directory_path(piel_path_types): Input path.
        path_list(list): List of files.
        ignore_confirmation(bool): Ignore confirmation. Default: False.
        validate_individual(bool): Validate individual files. Default: False.

    Returns:
        None
    """
    directory_path = return_path(directory_path)
    path_list = convert_list_to_path_list(path_list)
    if validate_individual:
        if ignore_confirmation:
            for path in path_list:
                if path.exists():
                    delete_path(path)
        else:
            for path in path_list:
                if path.exists():
                    answer = input("Confirm deletion of: " + str(path))
                    if answer.upper() in ["Y", "YES"]:
                        delete_path(path)
                    elif answer.upper() in ["N", "NO"]:
                        print("Skipping deletion of: " + str(path))
    else:
        if ignore_confirmation:
            for path in path_list:
                if path.exists():
                    delete_path(path)
        else:
            answer = input("Confirm deletion of: " + str(path_list))
            if answer.upper() in ["Y", "YES"]:
                for path in path_list:
                    if path.exists():
                        delete_path(path)
            elif answer.upper() in ["N", "NO"]:
                print("Skipping deletion of: " + str(path_list))


def get_files_recursively_in_directory(
    path: piel_path_types,
    extension: str = "*",
):
    """
    Returns a list of files in a directory.

    Args:
        path(piel_path_types): Input path.
        extension(str): File extension.

    Returns:
        file_list(list): List of files.
    """
    path = return_path(path)
    file_list = []
    for x in os.walk(str(path.resolve())):
        for file_path in glob.glob(os.path.join(x[0], f"*.{extension}")):
            file_list.append(file_path)
    return file_list


def permit_script_execution(script_path: piel_path_types) -> None:
    """
    Permits the execution of a script.

    Args:
        script_path(piel_path_types): Script path.

    Returns:
        None
    """
    script = return_path(script_path)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def permit_directory_all(directory_path: piel_path_types) -> None:
    """
    Permits a directory to be read, written and executed. Use with care as it can be a source for security issues.

    Args:
        directory_path(piel_path_types): Input path.

    Returns:
        None
    """
    directory_path = return_path(directory_path)
    try:
        directory_path.chmod(0o777)
    except PermissionError:
        print(
            UserWarning(
                "Could not change permissions of directory: "
                + str(directory_path.resolve())
                + " to 777. Your Python executable might not have the required permissions. Restructure your project directory so Python does not have to change permissions."
            )
        )


def read_json(path: piel_path_types) -> dict:
    """
    Reads a JSON file.

    Args:
        path(piel_path_types): Input path.

    Returns:
        json_data(dict): JSON data.
    """
    path = return_path(path)
    with open(path, "r") as json_file:
        json_data = json.load(json_file)
    return json_data


def return_path(input_path: piel_path_types) -> pathlib.Path:
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
    elif isinstance(input_path, types.ModuleType):
        output_path = pathlib.Path(input_path.__file__).parent
    else:
        raise ValueError(
            "input_path: " + str(input_path) + " is of type: " + str(type(input_path))
        )
    return output_path


def run_script(script_path: piel_path_types) -> None:
    """
    Runs a script on the filesystem `script_path`.

    Args:
        script_path(piel_path_types): Script path.

    Returns:
        None
    """
    script = return_path(script_path)
    subprocess.run(str(script.resolve()), check=True, capture_output=True)


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
            os.environ["PIEL_PACKAGE_DIRECTORY"]
            + "/docs/examples/designs/"
            + example_name
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
    directory_path: piel_path_types,
    script: str,
    script_name: str,
) -> None:
    """
    Records a `script_name` in the `scripts` project directory.

    Args:
        directory_path(piel_path_types): Design directory.
        script(str): Script to write.
        script_name(str): Name of the script.

    Returns:
        None
    """
    directory_path = return_path(directory_path)

    directory_exists = check_path_exists(directory_path)

    if directory_exists:
        pass
    else:
        try:
            create_new_directory(directory_path)
        except PermissionError:
            print(
                UserWarning(
                    "Could not create directory: "
                    + str(directory_path.resolve())
                    + ". Your Python executable might not have the required permissions. Restructure your project directory so Python does not have to change permissions."
                )
            )

    file = open(str(directory_path / script_name), "w")
    file.write(script)
    file.close()
