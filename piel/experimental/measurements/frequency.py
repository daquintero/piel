from piel.types.experimental import VNASParameterMeasurement
from piel.types import PathTypes, VNAPowerSweepMeasurement
from piel.file_system import return_path


def compose_vna_s_parameter_measurement(
    instance_directory: PathTypes, skip_missing: bool = False, **kwargs
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
        error = FileNotFoundError(
            f"Could not find the .s2p file in the directory {instance_directory}"
        )
        if skip_missing:
            print(error)
            return VNASParameterMeasurement()
        else:
            raise error

    return VNASParameterMeasurement(
        parent_directory=instance_directory, spectrum_file=s2p_file, **kwargs
    )


def compose_vna_power_sweep_measurement(
    instance_directory: PathTypes, skip_missing: bool = False, **kwargs
) -> VNAPowerSweepMeasurement:
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
        error = FileNotFoundError(
            f"Could not find the .s2p file in the directory {instance_directory}"
        )
        if skip_missing:
            print(error)
            return VNAPowerSweepMeasurement()
        else:
            raise error

    return VNAPowerSweepMeasurement(
        parent_directory=instance_directory, spectrum_file=s2p_file, **kwargs
    )
