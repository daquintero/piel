import openlane
from piel.config import piel_path_types
from piel.file_system import return_path, read_json
from piel.defaults import test_spm_open_lane_configuration

__all__ = ["run_openlane_flow"]


def run_openlane_flow(
    configuration: dict | None = test_spm_open_lane_configuration,
    design_directory: piel_path_types = "/foss/designs/spm",
) -> None:
    """
    Runs the OpenLane flow.

    Args:
        configuration(dict): OpenLane configuration dictionary. If none is present it will default to the config.json file on the design_directory.
        design_directory(piel_path_types): Design directory PATH.

    Returns:
        None
    """
    design_directory = return_path(design_directory)
    if configuration is None:
        # Get extract configuration file from config.json on directory
        config_json_filepath = design_directory / "config.json"
        configuration = read_json(config_json_filepath)

    Classic = openlane.Flow.get("Classic")

    flow = Classic(
        config=configuration,
        design_dir=str(design_directory.resolve()),
    )

    flow.start()
