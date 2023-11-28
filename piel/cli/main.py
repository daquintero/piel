import click

__all__ = ["main"]


@click.group()
def main(args=None):
    """CLI Interface for piel There are available many helper commands to help you set up your
    environment and design your projects."""
