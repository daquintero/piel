from unittest.mock import MagicMock
from pathlib import Path


# Helper function to mock Path.iterdir() return value
def mock_iterdir(mock_files):
    def _mock_iterdir():
        for file in mock_files:
            yield str(file)

    return _mock_iterdir


def mock_path(name, suffix=".csv", exists=True):
    """Helper function to create a mocked Path object."""
    mock_path = MagicMock(spec=Path)
    mock_path.name = name
    mock_path.suffix = suffix
    mock_path.exists.return_value = exists
    return mock_path


def test_compose_propagation_delay_measurement_success():
    mock_files = [
        mock_path("Ch1_waveform.csv"),
        mock_path("Ch2_waveform.csv"),
        mock_path("measurements.csv"),
    ]
    # mock_directory = MagicMock(spec=Path)
    # mock_directory.iterdir = mock_iterdir(mock_files)
    #
    # measurement = compose_propagation_delay_measurement(mock_directory)
    # assert isinstance(measurement, PropagationDelayMeasurement)
    # assert measurement.dut_waveform_file.name == "Ch1_waveform.csv"
    # assert measurement.reference_waveform_file.name == "Ch2_waveform.csv"
    # assert measurement.measurements_file.name == "measurements.csv"


def test_compose_propagation_delay_measurement_skip_missing():
    mock_files = [
        mock_path("Ch1_waveform.csv"),
        # Missing Ch2_waveform.csv
        mock_path("measurements.csv"),
    ]
    # mock_directory = MagicMock(spec=Path)
    # # Mock the iterdir() method to return the file list
    # mock_directory.iterdir.return_value = mock_files
    # mock_directory.glob.return_value = mock_files
    #
    # measurement = compose_propagation_delay_measurement(mock_directory, skip_missing=True)
    # assert isinstance(measurement, PropagationDelayMeasurement)
    # assert measurement.dut_waveform_file is not None
    # assert measurement.reference_waveform_file is ""
    # assert measurement.measurements_file is not None


def test_compose_propagation_delay_measurement_custom_prefix():
    mock_files = [
        mock_path("CustomDut_waveform.csv"),
        mock_path("CustomRef_waveform.csv"),
        mock_path("CustomMeasure_measurements.csv"),
    ]
    # mock_directory = MagicMock(spec=Path)
    # # mock_directory.iterdir = mock_iterdir(mock_files)
    #
    # measurement = compose_propagation_delay_measurement(
    #     mock_directory,
    #     dut_file_prefix="CustomDut",
    #     reference_file_prefix="CustomRef",
    #     measurement_file_prefix="CustomMeasure"
    # )
    # assert isinstance(measurement, PropagationDelayMeasurement)
    # assert measurement.dut_waveform_file.name == "CustomDut_waveform.csv"
    # assert measurement.reference_waveform_file.name == "CustomRef_waveform.csv"
    # assert measurement.measurements_file.name == "CustomMeasure_measurements.csv"
