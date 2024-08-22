from ..types import OscilloscopeMeasurement
from ...types import PathTypes
from ...file_system import return_path, get_files_recursively_in_directory


def compose_oscilloscope_measurement(
    instance_directory: PathTypes,
    skip_missing: bool = False,
    **kwargs,
) -> OscilloscopeMeasurement:
    """
    This function composes an OscilloscopeMeasurement from a given directory. The OscilloscopeMeasurement.waveform_file_list
    will be a collection of files that end with a suffix ``Ch*.csv``. This function will compose the list of files in the order
    of the channel number. The OscilloscopeMeasurement.measurements_file will be a file that ends without a suffix ``Ch*.csv``.
    """
    instance_directory = return_path(instance_directory)
    waveform_file_list = []
    measurements_file = ""

    # List all files in the directory
    all_files = get_files_recursively_in_directory(
        extension="csv", path=instance_directory
    )

    # Separate waveform files (with Ch*.csv) from the measurements file
    for file in all_files:
        file = return_path(file)
        if "Ch" in file.stem:
            waveform_file_list.append(file)
        else:
            if measurements_file == "":
                measurements_file = file
            elif not skip_missing:
                raise ValueError(
                    "Multiple measurement files found without a channel suffix."
                )

    # Sort waveform files by channel number (assumes format ChX.csv where X is the channel number)
    waveform_file_list.sort(key=lambda f: int(f.stem.split("Ch")[-1]))

    # Handle missing measurement file if required
    if not measurements_file and not skip_missing:
        raise FileNotFoundError("No measurements file found without a channel suffix.")

    return OscilloscopeMeasurement(
        parent_directory=instance_directory,
        waveform_file_list=waveform_file_list,
        measurements_file=measurements_file,
        **kwargs,
    )
