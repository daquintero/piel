from openlane.flows import Flow
from piel.config import piel_path_types
from piel.file_system import return_path, read_json
from .defaults import test_spm_open_lane_configuration

__all__ = ["run_openlane_flow"]


def run_openlane_flow(
    configuration: dict | None = None,
    design_directory: piel_path_types = ".",
    only_generate_flow_setup: bool = False,
):
    """
    Runs the OpenLane v2 flow.

    Args:
        configuration(dict): OpenLane configuration dictionary. If none is present it will default to the config.json file on the design_directory.
        design_directory(piel_path_types): Design directory PATH.
        TODO ADD DOCS HERE

    Returns:
        TODO UPDATE
    """
    design_directory = return_path(design_directory)
    if configuration is None:
        # Get extract configuration file from config.json on directory
        config_json_filepath = design_directory / "config.json"
        configuration = read_json(str(config_json_filepath.resolve()))

    Classic = Flow.factory.get("Classic")

    flow = Classic(
        config=configuration,
        design_dir=str(design_directory.resolve()),
    )
    if only_generate_flow_setup:
        return flow
    else:
        flow.start()
