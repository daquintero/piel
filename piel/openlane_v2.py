import openlane
from .defaults import test_spm_open_lane_configuration


def run_openlane(
    design_directory: str = ".", configuration: dict = test_spm_open_lane_configuration
):
    Classic = openlane.Flow.get("Classic")

    flow = Classic(
        configuration,
        design_dir=design_directory,
    )

    flow.start()


__all__ = ["run_openlane"]
