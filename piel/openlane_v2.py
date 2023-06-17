import json
import openlane
from .defaults import test_spm_open_lane_configuration


def run_openlane_flow(
    configuration: dict | None,
    design_directory: str = "/foss/designs/spm",
) -> None:
    """
    Runs the OpenLane flow.

    Args:
        configuration(dict): OpenLane configuration dictionary. If none is present it will default to the config.json file on the design_directory.
        design_directory(str): Design directory PATH.

    Returns:
        None
    """
    if configuration is None:
        # Get extract configuration file from config.json on directory
        configuration = json.load(design_directory + "/config.json")

    Classic = openlane.Flow.get("Classic")

    flow = Classic(
        configuration,
        design_dir=design_directory,
    )

    flow.start()


__all__ = ["run_openlane_flow"]
