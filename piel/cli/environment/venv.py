from ..utils import echo_and_check_subprocess, get_piel_home_directory
from .environment import environment

__all__ = ["create_and_activate_venv"]


def create_and_activate_venv() -> None:
    """
    Creates and activates the piel virtual environment.

    Returns:
        None: None.
    """
    venv_path = get_piel_home_directory() / ".venv"
    activate_script_path = venv_path / "bin" / "activate"
    echo_and_check_subprocess(["python", "-m", "venv", str(venv_path)])
    echo_and_check_subprocess(["bash", str(activate_script_path)])


@environment.command(
    name="create-piel-venv",
    help="Creates the piel virtual environment shared by all the tools.",
)
def create_and_activate_venv_command():
    """Installs the nix package manager."""
    return create_and_activate_venv()
