from ..types import VNASParameterMeasurement
from ...types import PathTypes
from ...file_system import return_path


def compose_vna_s_parameter_measurement(
    instance_directory: PathTypes, **kwargs
) -> VNASParameterMeasurement:
    """
    There should only be one .s2p s-parameter file in this directory. If there are more than one, it will read the first one.
    This function will iterate through the instance directory and find the .s2p file. It will return a measurement accordingly.
    """
    instance_directory = return_path(instance_directory)
    # This is the file that we are looking for
    s2p_file = None
    for file_i in instance_directory.iterdir():
        if file_i.suffix == ".s2p":
            s2p_file = file_i
            break
    if s2p_file is None:
        raise FileNotFoundError(
            f"Could not find the .s2p file in the directory {instance_directory}"
        )
    return VNASParameterMeasurement(
        parent_directory=instance_directory, spectrum_file=s2p_file, **kwargs
    )
