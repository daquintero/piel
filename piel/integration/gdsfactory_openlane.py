"""
There are a number of ways to generate gdsfactory integration.

It is worth noting that GDSFactory has already the following PDKs installed:
* SKY130nm https://gdsfactory.github.io/skywater130/
* GF180nm https://gdsfactory.github.io/gf180/
"""
import gdsfactory as gf
from ..config import piel_path_types
from ..file_system import check_path_exists
from piel.tools.openlane.migrate import get_design_from_openlane_migration
from piel.tools.openlane import find_design_run

__all__ = ["create_gdsfactory_component_from_openlane"]


def create_gdsfactory_component_from_openlane(
    design_name_v1: str | None = None,
    design_directory: piel_path_types | None = None,
    run_name: str | None = None,
    v1: bool = True,
) -> gf.Component:
    """
    This function cretes a gdsfactory layout component that can be included in the network codesign of the device, or that can be used for interconnection codesign.

    It will look into the latest design run and extract the final OpenLane-generated GDS. You do not have to have run this with OpenLane2 as it just looks at the latest run.

    Args:
        design_name_v1(str): Design name of the v1 design that can be found within `$OPENLANE_ROOT/"<latest>"/designs`.
        design_directory(piel_path_types): Design directory PATH.
        run_name(str): Name of the run to extract the GDS from. If None, it will look at the latest run.
        v1(bool): If True, it will import the design from the OpenLane v1 configuration.

    Returns:
        component(gf.Component): GDSFactory component.
    """
    design_name, design_directory = get_design_from_openlane_migration(
        v1=v1, design_name_v1=design_name_v1, design_directory=design_directory
    )
    latest_design_run_directory = find_design_run(design_directory, run_name=run_name)
    final_gds_run = (
        latest_design_run_directory
        / "results"
        / "final"
        / "gds"
        / (design_name + ".gds")
    )
    check_path_exists(final_gds_run, raise_errors=True)
    component = gf.import_gds(final_gds_run, name=design_name)
    return component
