import click
import os
import pathlib
import platform
import piel

from .environment import environment
from ..utils import (
    echo_and_check_subprocess,
    append_to_bashrc_if_does_not_exist,
    default_openlane2_directory,
    get_python_install_directory,
)
from ...file_system import create_new_directory

__all__ = [
    "install_nix",
    "install_openlane",
    "activate_piel_nix",
]


def install_and_configure_nix():
    """
    Downloads and installs the nix package manager. Instructions taken from https://nixos.wiki/wiki/Nix_Installation_Guide
    TODO update with nix-eda instructions

    Returns:

    """
    if platform.system() == "Windows":
        """Not Supported"""
        raise NotImplementedError(
            "This installation method is not supported on Windows."
        )
    elif platform.system() == "Darwin":
        """Not Supported"""
        raise NotImplementedError(
            "This installation method is not supported on Windows."
        )
    elif platform.system() == "Linux":
        # TODO sort this out for non Debian
        try:
            # sudo apt-get install -y curl
            echo_and_check_subprocess(["sudo", "apt-get", "install", "-y", "curl"])
            # sudo install -d -m755 -o $(id -u) -g $(id -g) /nix
            echo_and_check_subprocess(
                [
                    "sudo",
                    "install",
                    "-d",
                    "-m755",
                    "-o",
                    str(os.getuid()),  # Equivalent to $(id -u) in shell
                    "-g",
                    str(os.getgid()),  # Equivalent to $(id -g) in shel
                    "/nix",
                ]
            )
            # curl -L https://nixos.org/nix/install | sh
            echo_and_check_subprocess(
                ["curl -L https://nixos.org/nix/install | sh"],
                shell=True,
            )
            # . $HOME/.nix-profile/etc/profile.d/nix.sh
            nix_environment_script = ". $HOME/.nix-profile/etc/profile.d/nix.sh"
            echo_and_check_subprocess(
                [nix_environment_script],
                shell=True,
            )
            append_to_bashrc_if_does_not_exist(nix_environment_script)
            # nix-env -f "<nixpkgs>" -iA cachix
            try:
                # nix-channel --update
                echo_and_check_subprocess(["nix-channel", "--update"])
                echo_and_check_subprocess(
                    ["nix-env", "-f", "<nixpkgs>", "-iA", "cachix"]
                )
            except FileNotFoundError:
                raise Exception(
                    "You have to reinstall your bash shell in order to continue. Then run this script again."
                )

        except:  # NOQA: E722
            raise Exception(
                "Failed to install nix. Read the instructions in the corresponding nix function."
            )

    return 0


def update_openlane_directory(
    directory: piel.PathTypes = default_openlane2_directory,
    branch: str = "main",
):
    """
    Updates the openlane directory.
    TODO update with nix-eda instructions

    Returns:

    """
    openlane2_directory = piel.return_path(directory)
    if openlane2_directory.exists():
        echo_and_check_subprocess(["git", "checkout", branch], cwd=openlane2_directory)
        echo_and_check_subprocess(
            ["git", "pull", "origin", "main"], cwd=openlane2_directory
        )


@environment.command(
    name="activate-piel-nix", help="Activates the specific piel nix environment."
)
def activate_piel_nix(openlane2_directory: pathlib.Path = default_openlane2_directory):
    """
    Enters the custom piel nix environment with all the supported tools installed and configured packages.
    Runs the nix-shell command on the piel/environment/nix/ directory.
    TODO update with nix-eda instructions
    """
    if platform.system() == "Windows":
        """Not Supported"""
        raise NotImplementedError(
            "This installation method is not supported on Windows."
        )
    elif platform.system() == "Darwin":
        """Not Supported"""
        raise NotImplementedError(
            "This installation method is not supported on Windows."
        )
    elif platform.system() == "Linux":
        # cachix use openlane
        # create_and_activate_venv()  # Currently unused, TODO future poetry integration
        nix_shell_directory = (
            get_python_install_directory() / "environment" / "nix" / "nix-eda"
        )
        # nix shell .#{ngspice,xschem,verilator,yosys}
        echo_and_check_subprocess(
            [
                "nix",
                "shell",
                ".#{ngspice,xschem,verilator,yosys}",
            ],
            cwd=nix_shell_directory,
        )
        pass


@environment.command(name="install-nix", help="Installs the nix package manager.")
def install_nix():
    """Installs the nix package manager."""
    return install_and_configure_nix()


@environment.command(
    name="install-openlane",
    help="Installs all the openlane configuration and packages reproducibly.",
)
def install_openlane(openlane2_directory: pathlib.Path = default_openlane2_directory):
    """CLI that installs both the openlane2 python interface and the OpenROAD binaries."""
    if platform.system() == "Windows":
        """Not Supported"""
        raise NotImplementedError(
            "This installation method is not supported on Windows."
        )
    elif platform.system() == "Darwin":
        """Not Supported"""
        raise NotImplementedError(
            "This installation method is not supported on Windows."
        )
    elif platform.system() == "Linux":
        # cachix use openlane
        echo_and_check_subprocess(["cachix", "use", "openlane"])
        # git clone https://github.com/efabless/openlane2.git into ~/.piel
        directory_exists = create_new_directory(openlane2_directory)
        if not directory_exists:
            try:
                update_openlane_directory(openlane2_directory)
                print(
                    Warning(
                        "The openlane2 directory already exists. Updating it instead from the main branch."
                    )
                )
                return 0
            except:  # NOQA: E722
                raise FileExistsError(
                    "The openlane2 directory already exists. Please delete it and try again."
                )
        echo_and_check_subprocess(
            [
                "git",
                "clone",
                "https://github.com/efabless/openlane2.git",
                str(openlane2_directory),
            ]
        )
        return 0


@environment.command(name="update-openlane", help="Updates the openlane directory.")
@click.option(
    "-d",
    "--directory",
    default=str(default_openlane2_directory),
    help="The openlane2 directory.",
)
@click.option(
    "-b",
    "--branch",
    default="main",
    help="The branch to checkout.",
)
def update_openlane_directory_command(
    directory: str = str(default_openlane2_directory),
    branch: str = "main",
):
    """
    Updates the openlane directory. Checks out the main branch.

    Returns:

    """
    if directory is None:
        directory = default_openlane2_directory
    directory = piel.return_path(directory)
    update_openlane_directory(directory=directory, branch=branch)
    return 0
