import click
import sys
from .main import main


@click.group(name="run")
def run():
    """Running related commands."""
    pass


@run.command(name="python", help="Runs a python shell.")
def run_python():
    """Runs a python shell where piel is installed."""
    return sys.exec_prefix


main.add_command(run)
