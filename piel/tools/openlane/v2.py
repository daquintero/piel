from openlane.flows import Flow
from piel.config import piel_path_types
from piel.file_system import return_path, read_json

__all__ = ["run_openlane_flow"]


def run_openlane_flow(
    configuration: dict | None = None,
    design_directory: piel_path_types = ".",
    parallel_asynchronous_run: bool = False,
    only_generate_flow_setup: bool = False,
):
    """
    Runs the OpenLane v2 flow.

    Args:
        configuration(dict): OpenLane configuration dictionary. If none is present it will default to the config.json file on the design_directory.
        design_directory(piel_path_types): Design directory PATH.
        parallel_asynchronous_run(bool): Run the flow in parallel.
        only_generate_flow_setup(bool): Only generate the flow setup.

    Returns:

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
        if parallel_asynchronous_run:
            # TODO implement
            flow.start()
        else:
            flow.start()
