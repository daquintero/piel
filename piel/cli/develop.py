import click
import subprocess
import platform
from ..file_system import return_path
from .utils import echo_and_check_subprocess, get_python_install_directory
from .main import main


@click.group(name="develop")
def develop():
    """Development related commands."""
    pass


@develop.command(name="build-docs", help="Builds the sphinx documentation.")
def build_documentation(args=None):
    """Verifies and builds the documentation."""
    # Runs the documentation build from the poetry environment
    # TODO fix this so it runs from poetry.
    echo_and_check_subprocess(["python", "-m", "sphinx", "docs/", "_docs/"])
    return 0


@develop.command(
    name="generate-poetry2nix-flake", help="Generates the poetry2nix flake."
)
def generate_poetry2nix_flake(args=None):
    """
    Generates the poetry2nix flakes file. Requires nix to be installed.

    Returns:

    """
    # nix flake init --template github:nix-community/poetry2nix  --experimental-features 'nix-command flakes'
    try:
        echo_and_check_subprocess(
            [
                "nix",
                "flake",
                "init",
                "--template",
                "github:nix-community/poetry2nix",
                "--experimental-features",
                "nix-command flakes",
            ]
        )
    except subprocess.CalledProcessError:
        # Check if flake.nix exists
        if return_path("flake.nix").exists():
            # mv flake.nix environment/nix/flake.nix
            echo_and_check_subprocess(["mv", "flake.nix", "environment/nix/flake.nix"])
        else:
            raise Exception(
                "Failed to generate the poetry2nix flake and none was found in the root directory.s"
            )


@develop.command(
    name="build-piel-cachix", help="Activates the specific piel nix environment."
)
def build_piel_cachix_command(args=None):
    """
    Enters the custom piel nix environment with all the supported tools installed and configured packages.
    Runs the nix-shell command on the piel/environment/nix/ directory.
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
        nix_shell_directory = get_python_install_directory() / "environment" / "nix"
        # nix develop --extra-experimental-features nix-command --extra-experimental-features flakes
        echo_and_check_subprocess(
            [
                "nix",
                "build",
                "--extra-experimental-features",
                "nix-command",
                "--extra-experimental-features",
                "flakes",
                "--show-trace",
            ],
            cwd=nix_shell_directory,
        )
        pass


main.add_command(develop)
