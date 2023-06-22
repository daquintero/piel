"""
These functions provide easy tools for easily migrating between OpenLane v1 and v2 based designs.
"""
import pathlib


def openlane_migration(
    v1: bool = True,
    design_name_v1: str | None = None,
    design_directory: str | pathlib.Path | None = None,
    root_directory_v1: str | pathlib.Path | None = None,
) -> None:
    """
    This function provides the integration mechanism for easily migrating the interconnection with other toolsets from an OpenLane v1 design to an OpenLane v2 design.

    Args:
        v1(bool): If True, it will migrate from v1 to v2.
        design_name_v1(str): Design name of the v1 design that can be found within `$OPENLANE_ROOT/"<latest>"/designs`.
        design_directory(str): Design directory PATH. Optional path for v2-based designs.
        root_directory_v1(str): Root directory of OpenLane v1. If set to None it will return `$OPENLANE_ROOT/"<latest>"/designs
    """
    pass
