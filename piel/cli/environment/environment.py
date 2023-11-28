import click
from ..main import main

__all__ = ["environment"]


@click.group(name="environment")
def environment():
    """Environment related commands."""
    pass


main.add_command(environment)
