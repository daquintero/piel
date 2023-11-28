import pathlib
import piel
import subprocess
import os

__all__ = [
    "append_to_bashrc_if_does_not_exist",
    "default_openlane2_directory",
    "echo_and_run_subprocess",
    "echo_and_check_subprocess",
    "get_python_install_directory",
    "get_piel_home_directory",
]

default_openlane2_directory = (pathlib.Path.home() / ".piel" / "openlane2")


def append_to_bashrc_if_does_not_exist(line: str):
    """
    Appends a line to .bashrc if it does not exist.

    Args:
        line:

    Returns:

    """
    bashrc_path = os.path.join(os.path.expanduser('~'), '.bashrc')

    # Check if the line already exists in .bashrc
    with open(bashrc_path, 'r') as file:
        if line in file.read():
            print("Line: `" + line + "` already exists in .bashrc")
            return

    # Append the line to .bashrc
    with open(bashrc_path, 'a') as file:
        file.write("\n" + line)
        print("Line: " + line + " appended to .bashrc")


def echo_and_run_subprocess(command: list, **kwargs):
    """
    Runs a subprocess and prints the command.

    Args:
        command:
        **kwargs:

    Returns:

    """
    concatenated_command = " ".join(command)
    print("Running: " + concatenated_command)
    return subprocess.run(command, cwd=get_python_install_directory(), **kwargs)


def echo_and_check_subprocess(command: list, **kwargs):
    """
    Runs a subprocess and prints the command. Raises an exception if the subprocess fails.

    Args:
        command:
        **kwargs:

    Returns:

    """
    concatenated_command = " ".join(command)
    print("Running: " + concatenated_command)
    # Check cwd not in kwargs, if not, add it
    if "cwd" not in kwargs:
        kwargs["cwd"] = get_python_install_directory()
    return subprocess.check_call(command, **kwargs)


def get_python_install_directory():
    """
    Gets the piel installation directory.

    Returns:
        pathlib.Path: The piel installation directory.

    """
    return pathlib.Path(piel.__file__).parent.parent.absolute()


def get_piel_home_directory():
    """
    Gets the piel home directory.

    Returns:
        pathlib.Path: The piel home directory.

    """
    return pathlib.Path.home() / ".piel"
