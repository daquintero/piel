import pandas as pd

import piel.units
from piel.types.experimental import (
    PropagationDelayMeasurement,
    PropagationDelayMeasurementCollection,
    PropagationDelayMeasurementDataCollection,
    PropagationDelayMeasurementData,
    OscilloscopeMeasurement,
    OscilloscopeMeasurementDataCollection,
    OscilloscopeMeasurementData,
)
from piel.types import (
    DataTimeSignalData,
    MultiDataTimeSignal,
    PathTypes,
    ScalarMetrics,
    ScalarMetricCollection,
)
from piel.file_system import return_path


def extract_measurement_to_dataframe(file: PathTypes) -> pd.DataFrame:
    """
    Extracts the measurement files from a csv file and returns it as a pandas dataframe.

    Parameters
    ----------
    file : PathTypes
        The path to the csv file.

    Returns
    -------
    pd.DataFrame
        The measurement files as a pandas dataframe.
    """
    # TODO write here functionality to validate the file exists and it is a csv file in a particular structure compatible with a measurement.
    # TODO sort out actual measurement information
    dataframe = pd.read_csv(
        file,
        names=tuple(
            [
                "value",
                "mean",
                "min",
                "max",
                "standard_deviation",
                "count",
                "name1",
                "name2",
                "name3",
            ]
        ),
    )

    dataframe["count"] = piel.units.prefix2int(dataframe["count"].values[0])

    # Merge the name columns into a single column called 'name'
    dataframe["name"] = dataframe[["name1", "name2", "name3"]].apply(
        lambda x: " ".join(x.dropna().astype(str)), axis=1
    )

    # Drop the original name columns
    dataframe = dataframe.drop(columns=["name1", "name2", "name3"])

    # Convert all spaces to underscores in the 'name' column
    dataframe["name"] = dataframe["name"].str.replace(" ", "_")
    dataframe["name"] = dataframe["name"].str.replace("(", "_")
    dataframe["name"] = dataframe["name"].str.replace(")", "_")

    # Handle duplicate names by adding a prefix
    dataframe["name"] = dataframe["name"].apply(lambda x: x.lower())
    name_counts = dataframe["name"].value_counts()
    duplicates = name_counts[name_counts > 1].index

    for dup in duplicates:
        duplicate_indices = dataframe[dataframe["name"] == dup].index
        for i, idx in enumerate(duplicate_indices, 1):
            dataframe.at[idx, "name"] = f"{dup}_{i}"

    return dataframe


def extract_waveform_to_dataframe(file: PathTypes) -> pd.DataFrame:
    """
    Extracts the waveform files from a csv file and returns it as a pandas dataframe.

    Parameters
    ----------
    file : PathTypes
        The path to the csv file.

    Returns
    -------
    pd.DataFrame
        The waveform files as a pandas dataframe.
    """
    # TODO write here functionality to validate the file exists and it is a csv file in a particular structure
    return pd.read_csv(file, header=0, names=["time_s", "voltage_V"], usecols=[3, 4])


def extract_to_data_time_signal(
    file: PathTypes,
) -> DataTimeSignalData:
    """
    Extracts the waveform files from a csv file and returns it as a DataTimeSignal that can be used to analyse the signal with other methods.

    Parameters
    ----------
    file : PathTypes
        The path to the csv file.

    Returns
    -------
    DataTimeSignalData
        The waveform files as a DataTimeSignal.
    """
    dataframe = extract_waveform_to_dataframe(file)
    data_time_signal = DataTimeSignalData(
        time_s=dataframe.time_s.values,
        data=dataframe.voltage_V.values,
        data_name="voltage_V",
    )
    return data_time_signal


def extract_propagation_delay_data_from_measurement(
    propagation_delay_measurement: PropagationDelayMeasurement,
) -> PropagationDelayMeasurementData:
    propagation_delay_measurement_i = propagation_delay_measurement
    data_i = dict()
    data_i["name"] = propagation_delay_measurement.name
    if hasattr(propagation_delay_measurement_i, "measurements_file"):
        file = propagation_delay_measurement_i.measurements_file
        file = return_path(file)
        if not file.exists():
            # Try appending to parent directory if file does not exist
            file = (
                propagation_delay_measurement_i.parent_directory
                / propagation_delay_measurement_i.measurements_file
            )
        data_i["measurements"] = extract_to_signal_measurement(file)

    if hasattr(propagation_delay_measurement_i, "reference_waveform_file"):
        file = propagation_delay_measurement_i.reference_waveform_file
        file = return_path(file)
        if not file.exists():
            # Try appending to parent directory if file does not exist
            file = (
                propagation_delay_measurement_i.parent_directory
                / propagation_delay_measurement_i.reference_waveform_file
            )
        data_i["reference_waveform"] = extract_to_data_time_signal(file)

    if hasattr(propagation_delay_measurement_i, "dut_waveform_file"):
        file = propagation_delay_measurement_i.dut_waveform_file
        file = return_path(file)
        if not file.exists():
            # Try appending to parent directory if file does not exist
            file = (
                propagation_delay_measurement_i.parent_directory
                / propagation_delay_measurement_i.dut_waveform_file
            )
        data_i["dut_waveform"] = extract_to_data_time_signal(file)

    return PropagationDelayMeasurementData(**data_i)


def extract_propagation_delay_measurement_sweep_data(
    propagation_delay_measurement_sweep: PropagationDelayMeasurementCollection,
) -> PropagationDelayMeasurementDataCollection:
    """
    This function is used to extract the relevant measurement files amd relate them to the sweep parameter. Because
    this function extracts multi-index files then we use xarray to analyze this files more clearly. It aims to extract all
    the files in the sweep file collection.
    """
    measurement_sweep_data = list()
    for (
        propagation_delay_measurement_i
    ) in propagation_delay_measurement_sweep.collection:
        measurement_data_i = extract_propagation_delay_data_from_measurement(
            propagation_delay_measurement_i
        )
        measurement_sweep_data.append(measurement_data_i)

    measurement_data_collection = PropagationDelayMeasurementDataCollection(
        name=propagation_delay_measurement_sweep.name,
        collection=measurement_sweep_data,
    )
    return measurement_data_collection


def extract_oscilloscope_data_from_measurement(
    oscilloscope_measurement: OscilloscopeMeasurement,
) -> OscilloscopeMeasurementData:
    data_i = dict()
    data_i["name"] = oscilloscope_measurement.name

    # Extracting measurements from the measurements file if it exists
    if hasattr(oscilloscope_measurement, "measurements_file"):
        file = oscilloscope_measurement.measurements_file
        file = return_path(file)
        if not file.exists():
            # Try appending to parent directory if file does not exist
            file = oscilloscope_measurement.parent_directory / file
        if file.exists():
            data_i["measurements"] = extract_to_signal_measurement(file)
        else:
            raise FileNotFoundError(f"Measurements file {file} does not exist.")
    else:
        data_i["measurements"] = None  # Or some default value if needed

    # Extracting data from waveform files
    waveform_data_list = []
    for waveform_file in oscilloscope_measurement.waveform_file_list:
        file = return_path(waveform_file)
        if not file.exists():
            # Try appending to parent directory if file does not exist
            file = oscilloscope_measurement.parent_directory / waveform_file
        if file.exists():
            waveform_data = extract_to_data_time_signal(file)
            waveform_data_list.append(waveform_data)
        else:
            raise FileNotFoundError(f"Waveform file {file} does not exist.")

    data_i["waveform_list"] = waveform_data_list

    return OscilloscopeMeasurementData(**data_i)


def extract_oscilloscope_measurement_data_collection(
    oscilloscope_measurement_data: OscilloscopeMeasurementData,
) -> OscilloscopeMeasurementDataCollection:
    # TODO write this.
    """
    This function is used to extract the relevant measurement files amd relate them to the sweep parameter. Because
    this function extracts multi-index files then we use xarray to analyze this files more clearly. It aims to extract all
    the files in the sweep file collection.
    """
    measurement_data_collection_raw = list()
    for oscilloscope_measurement_i in oscilloscope_measurement_data.collection:
        measurement_data_i = extract_oscilloscope_data_from_measurement(
            oscilloscope_measurement_i
        )
        measurement_data_collection_raw.append(measurement_data_i)

    measurement_data_collection = OscilloscopeMeasurementDataCollection(
        name=oscilloscope_measurement_data.name,
        collection=measurement_data_collection_raw,
    )
    return measurement_data_collection


def extract_to_signal_measurement(file: PathTypes, **kwargs) -> ScalarMetricCollection:
    """
    Extracts the measurement files from a csv file and returns it as a SignalMeasurement that can be used to analyse the signal.

    Parameters
    ----------
        file : PathTypes

    Returns
    -------
        SignalMetricsMeasurementCollection : dict[str, SignalMetricsData]
    """
    dataframe = extract_measurement_to_dataframe(file)
    metrics_list = list()
    for _, row in dataframe.iterrows():
        metrics_i = ScalarMetrics(**row)
        metrics_list.append(metrics_i)

    return ScalarMetricCollection(metrics=metrics_list, **kwargs)


def combine_channel_data(
    channel_file: list[PathTypes],
) -> MultiDataTimeSignal:
    """
    Extracts the waveform files from a list of csv files and returns it as a MultiDataTimeSignal that can be used to analyse the signals together.

    Parameters
    ----------
    channel_file : list[PathTypes]
        The list of paths to the csv files.

    Returns
    -------
    MultiDataTimeSignal
        The waveform files as a MultiDataTimeSignal.
    """
    multi_channel_data_time_signals = list()

    for file in channel_file:
        data_time_signal_i = extract_to_data_time_signal(file)
        multi_channel_data_time_signals.append(data_time_signal_i)

    return multi_channel_data_time_signals
