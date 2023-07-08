"""
These functions provide easy tools for easily migrating between OpenLane v1 and v2 based designs.
"""
import pathlib
from .v1 import get_design_directory_from_root_openlane_v1
from piel.file_system import return_path

__all__ = [
    "get_design_from_openlane_migration",
]


def get_design_from_openlane_migration(
    v1: bool = True,
    design_name_v1: str | None = None,
    design_directory: str | pathlib.Path | None = None,
    root_directory_v1: str | pathlib.Path | None = None,
) -> (str, pathlib.Path):
    """
    This function provides the integration mechanism for easily migrating the interconnection with other toolsets from an OpenLane v1 design to an OpenLane v2 design.

    This function checks if the inputs are to be treated as v1 inputs. If so, and a `design_name` is provided then it will set the `design_directory` to the corresponding `design_name` directory in the corresponding `root_directory_v1 / designs`. If no `root_directory` is provided then it returns `$OPENLANE_ROOT/"<latest>"/. If a `design_directory` is provided then this will always take precedence even with a `v1` flag.

    Args:
        v1(bool): If True, it will migrate from v1 to v2.
        design_name_v1(str): Design name of the v1 design that can be found within `$OPENLANE_ROOT/"<latest>"/designs`.
        design_directory(str): Design directory PATH. Optional path for v2-based designs.
        root_directory_v1(str): Root directory of OpenLane v1. If set to None it will return `$OPENLANE_ROOT/"<latest>"`

    Returns:
        None
    """
    if design_directory is not None:
        design_directory = return_path(design_directory)
        design_name = design_directory.name
        return design_name, design_directory
    elif v1:
        design_directory = get_design_directory_from_root_openlane_v1(
            design_name=design_name_v1, root_directory=root_directory_v1
        )
        design_name = design_name_v1
        return design_name, design_directory
    else:
        raise ValueError(
            "You must provide either a design_directory or a design_name_v1"
        )
