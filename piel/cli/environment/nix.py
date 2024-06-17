import platform

from .environment import environment
from ..utils import (
    echo_and_check_subprocess,
    append_to_bashrc_if_does_not_exist,
)

__all__ = [
    "install_nix",
    "activate",
]

import subprocess
import os


# TODO move to file system utils.
def echo_and_check_shell_subprocess(command, **kwargs):
    """
    Runs a subprocess and prints the command. Raises an exception if the subprocess fails.

    Args:
        command (list or str): The command and its arguments as a list or a single string.
        **kwargs: Additional keyword arguments to pass to subprocess.check_call.

    Returns:
        None
    """
    if isinstance(command, list):
        concatenated_command = " ".join(command)
    else:
        concatenated_command = command

    print("Running: " + concatenated_command)
    # Pass the command as a single string to be interpreted by the shell
    return subprocess.run(concatenated_command, shell=True, **kwargs)

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


@environment.command(
    name="activate", help="Activates the specific piel nix environment."
)
def activate():
    """
    Enters the custom piel nix environment with all the supported tools installed and configured packages.
    Runs the nix-shell command on the piel/environment/nix/ directory.
    """
    if platform.system() == "Windows":
        raise NotImplementedError("This installation method is not supported on Windows.")
    elif platform.system() == "Darwin":
        raise NotImplementedError("This installation method is not supported on macOS.")
    elif platform.system() == "Linux":
        # Setup the nix shell environment
        nix_command = (
            "nix shell "
            "github:efabless/nix-eda#{ngspice,xschem,verilator,yosys} "
            "github:efabless/openlane2 "
            "nixpkgs#verilog "
            "nixpkgs#gtkwave"
        )
        print("Please run this in your shell:")
        print(nix_command)


@environment.command(name="install-nix", help="Installs the nix package manager.")
def install_nix():
    """Installs the nix package manager."""
    return install_and_configure_nix()
