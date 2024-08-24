import click
from .utils import get_python_install_directory
from .main import main


@click.command(name="get-install-directory", help="Gets the piel base directory.")
def get_piel_install_directory():
    """Gets the piel installation directory."""
    print(get_python_install_directory())
    return 0


main.add_command(get_piel_install_directory)
