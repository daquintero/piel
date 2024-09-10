from piel.types.experimental import PropagationDelayMeasurement
from ...types import PathTypes
from ...file_system import return_path


def compose_propagation_delay_measurement(
    instance_directory: PathTypes,
    dut_file_prefix: str = "Ch1",
    reference_file_prefix: str = "Ch2",
    measurement_file_prefix: str = "",
    skip_missing: bool = False,
    **kwargs,
) -> PropagationDelayMeasurement:
    """
    This function will iterate through the instance directory and find the files that correspond to the propagation delay measurement.
    The files are expected to be in the form of:
    - {dut_file_prefix}_waveform.csv
    - {reference_file_prefix}_waveform.csv
    - {measurement_file_prefix}_measurements.csv
    """
    instance_directory = return_path(instance_directory)
    dut_file = None
    reference_file = None
    measurements_file = None
    for file_i in instance_directory.iterdir():
        if (dut_file_prefix in file_i.name) and (file_i.suffix == ".csv"):
            dut_file = file_i
        elif (reference_file_prefix in file_i.name) and file_i.suffix == ".csv":
            reference_file = file_i
        elif (measurement_file_prefix in file_i.name) and file_i.suffix == ".csv":
            measurements_file = file_i
    if (dut_file is None) or (reference_file is None) or (measurements_file is None):
        missing_error = FileNotFoundError(
            f"Could not find the required files in the directory {instance_directory}"
        )
        if skip_missing:
            print(missing_error)
            return PropagationDelayMeasurement()
        else:
            raise missing_error
    return PropagationDelayMeasurement(
        parent_directory=instance_directory,
        dut_waveform_file=dut_file,
        reference_waveform_file=reference_file,
        measurements_file=measurements_file,
        **kwargs,
    )
