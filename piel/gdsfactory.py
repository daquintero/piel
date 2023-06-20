"""
There are a number of ways to generate gdsfactory integration.

It is worth noting that GDSFactory has already the following PDKs installed:
* SKY130nm https://gdsfactory.github.io/skywater130/
* GF180nm https://gdsfactory.github.io/gf180/

"""
import gdsfactory as gf
import pathlib
from .openlane.utils import find_design_run
from .file_system import return_path


def create_gdsfactory_component_from_openlane(
    design_directory: str | pathlib.Path, run_name: str | None = None
) -> gf.Component:
    """
    This function cretes a gdsfactory layout component that can be included in the network codesign of the device, or that can be used for interconnection codesign.

    It will look into the latest design run and extract the final OpenLane-generated GDS. You do not have to have run this with OpenLane2 as it just looks at the latest run.

    Args:
        design_directory(str): Design directory PATH.
        run_name(str): Name of the run to extract the GDS from. If None, it will look at the latest run.

    Returns:
        component(gf.Component): GDSFactory component.
    """
    design_directory = return_path(design_directory)
    design_name = design_directory.parent.name
    latest_design_run_directory = find_design_run(design_directory, run_name=run_name)
    final_gds_run = (
        latest_design_run_directory / "results" / "final" / "gds" / design_name + ".gds"
    )
    component = gf.import_gds(final_gds_run, name=design_name)
    return component


__all__ = ["create_gdsfactory_component_from_openlane"]
