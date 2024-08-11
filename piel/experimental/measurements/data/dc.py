import pandas as pd
from ....types import PathTypes, QuantityTypesDC
from ....file_system import return_path
from ....models.physical.electrical import (
    construct_dc_signal,
    construct_current_dc_signal,
    construct_voltage_dc_signal,
)
from ...types import (
    SourcemeterSweepMeasurementData,
    DCSweepMeasurementData,
    DCSweepMeasurementDataCollection,
    MultimeterSweepVoltageMeasurementData,
    SourcemeterVoltageCurrentSignalNamePair,
)


def construct_sourcemeter_sweep_signal_from_csv(
    file_path: PathTypes,
    voltage_signal_name: str,
    current_signal_name: str,
    **kwargs,
) -> SourcemeterSweepMeasurementData:
    file = return_path(file_path)
    dataframe = pd.read_csv(file)
    signal = construct_sourcemeter_sweep_signal_from_dataframe(
        dataframe=dataframe,
        voltage_signal_name=voltage_signal_name,
        current_signal_name=current_signal_name,
        **kwargs,
    )
    return signal


def construct_sourcemeter_sweep_signal_from_dataframe(
    dataframe: pd.DataFrame,
    voltage_signal_name: str,
    current_signal_name: str,
    signal_kwargs: dict = None,
    **kwargs,
) -> SourcemeterSweepMeasurementData:
    if signal_kwargs is None:
        signal_kwargs = {}

    voltage_signal_data = dataframe[voltage_signal_name].values
    current_signal_data = dataframe[current_signal_name].values

    signal = construct_dc_signal(
        voltage_signal_name=voltage_signal_name,
        voltage_signal_values=voltage_signal_data,
        current_signal_name=current_signal_name,
        current_signal_values=current_signal_data,
        **signal_kwargs,
    )

    return SourcemeterSweepMeasurementData(signal=signal, **kwargs)


def construct_multimeter_sweep_signal_from_csv(
    file_path: PathTypes,
    signal_name: str,
    signal_type: QuantityTypesDC = "voltage",
    **kwargs,
) -> MultimeterSweepVoltageMeasurementData:
    """
    Construct a multimeter sweep signal from a CSV file.

    Parameters
    ----------

    file_path : PathTypes
        The path to the CSV file.
    signal_name : str
        The name of the signal.
    signal_type : QuantityTypesDC
        The type of signal.
    **kwargs

    Returns
    -------

    MultimeterSweepVoltageMeasurementData
        The multimeter sweep signal
    """
    file = return_path(file_path)
    dataframe = pd.read_csv(file)

    if signal_type == "voltage":
        signal = construct_voltage_dc_signal(
            name=signal_name, values=dataframe[signal_name].values
        )
    elif signal_type == "current":
        signal = construct_current_dc_signal(
            name=signal_name, values=dataframe[signal_name].values ** kwargs
        )
    else:
        raise ValueError(f"Unimplemented signal type: {signal_type}")

    return signal


def construct_multimeter_sweep_signal_from_dataframe(
    dataframe: pd.DataFrame,
    signal_name: str,
    signal_kwargs: dict = None,
    **kwargs,
) -> MultimeterSweepVoltageMeasurementData:
    """
    Construct a multimeter sweep signal from a dataframe.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe containing the multimeter sweep signal data.
    signal_name : str
        The name of the signal.
    signal_kwargs : dict
        Additional keyword arguments.
    **kwargs

    Returns
    -------

    MultimeterSweepVoltageMeasurementData
        The multimeter sweep signal
    """
    if signal_kwargs is None:
        signal_kwargs = {}

    signal_data = dataframe[signal_name].values

    signal = construct_voltage_dc_signal(name=signal_name, values=signal_data)

    return MultimeterSweepVoltageMeasurementData(signal=signal, **kwargs)


def extract_signal_data_from_dataframe(
    dataframe: pd.DataFrame,
    sourcemeter_voltage_current_signal_name_pairs: list[
        SourcemeterVoltageCurrentSignalNamePair
    ],
    multimeter_signals: list[str],
    **kwargs,
) -> DCSweepMeasurementData:
    """
    Extract DC sweep data from a dataframe.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe containing the DC sweep data.
    sourcemeter_voltage_current_signal_name_pairs : list[SourcemeterVoltageCurrentSignalNamePair]
        The pairs of sourcemeter voltage and current signal names.
    multimeter_signals : list[str]
        The multimeter signals.
    **kwargs
        Additional keyword arguments.

    Returns
    -------

    DCSweepMeasurementData
        The DC sweep data.
    """
    # Iterate through the sourcemeter signals and create the sourcemeter sweep signals
    sourcemeter_sweep_signals = []

    for (
        sourcemeter_voltage_signal_pair
    ) in sourcemeter_voltage_current_signal_name_pairs:
        voltage_signal_name_i = sourcemeter_voltage_signal_pair[0]
        current_signal_name_i = sourcemeter_voltage_signal_pair[1]
        sourcemeter_sweep_signal = construct_sourcemeter_sweep_signal_from_dataframe(
            dataframe=dataframe,
            voltage_signal_name=voltage_signal_name_i,
            current_signal_name=current_signal_name_i,
        )
        sourcemeter_sweep_signals.append(sourcemeter_sweep_signal)

    # Iterate through the multimeter signals and create the multimeter sweep signals
    multimeter_sweep_signals = []

    for multimeter_signal in multimeter_signals:
        multimeter_sweep_signal = construct_multimeter_sweep_signal_from_dataframe(
            dataframe=dataframe, signal_name=multimeter_signal
        )
        multimeter_sweep_signals.append(multimeter_sweep_signal)

    return DCSweepMeasurementData(
        inputs=sourcemeter_sweep_signals,
        outputs=multimeter_sweep_signals,
        **kwargs,
    )


def extract_signal_data_from_csv(
    file_path: PathTypes,
    sourcemeter_voltage_current_signal_name_pairs: list[
        SourcemeterVoltageCurrentSignalNamePair
    ],
    multimeter_signals: list[str],
    **kwargs,
) -> DCSweepMeasurementData:
    """
    Extract DC sweep data from a CSV file.

    Parameters
    ----------
    file_path : PathTypes
        The path to the CSV file.
    sourcemeter_voltage_current_signal_name_pairs : list[SourcemeterVoltageCurrentSignalNamePair]
        The pairs of sourcemeter voltage and current signal names.
    multimeter_signals : list[str]
        The multimeter signals.
    **kwargs
        Additional keyword arguments.

    Returns
    -------

    DCSweepMeasurementData
        The DC sweep data.
    """
    file = return_path(file_path)
    dataframe = pd.read_csv(file)
    return extract_signal_data_from_dataframe(
        dataframe=dataframe,
        sourcemeter_voltage_current_signal_name_pairs=sourcemeter_voltage_current_signal_name_pairs,
        multimeter_signals=multimeter_signals,
        **kwargs,
    )


def extract_dc_sweeps_from_operating_point_csv(
    file_path: PathTypes,
    sourcemeter_voltage_current_signal_name_pairs: list[
        SourcemeterVoltageCurrentSignalNamePair
    ],
    multimeter_signals: list[str],
    unique_operating_point_columns: list[str],
    **kwargs,
) -> DCSweepMeasurementDataCollection:
    """
    Extract DC sweep data from a full operating point CSV file. The operating point CSV file contains the DC sweep data
    for multiple operating points. The unique operating point columns are used to extract the unique operating points
    from the CSV file. The DC sweep data is then extracted for each unique operating point. The DC sweep data is
    returned as a DCMeasurementDataCollection. The DCMeasurementDataCollection is a list of DCMeasurementDataTypes.

    Parameters
    ----------

    file_path : PathTypes
        The path to the operating point CSV file.
    sourcemeter_voltage_current_signal_name_pairs : list[SourcemeterVoltageCurrentSignalNamePair]
        The pairs of sourcemeter voltage and current signal names.
    multimeter_signals : list[str]
        The multimeter signals.
    unique_operating_point_columns : list[str]
        The unique operating point columns.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    DCMeasurementDataCollection
        The DC sweep data collection.
    """
    file = return_path(file_path)
    dataframe = pd.read_csv(file)

    # Extract the unique operating points
    unique_operating_points = dataframe[
        unique_operating_point_columns
    ].drop_duplicates()

    # Iterate through the unique operating points and extract the DC sweep data
    dc_sweep_data = []

    for _, operating_point in unique_operating_points.iterrows():
        operating_point_data = dataframe[
            (dataframe[unique_operating_point_columns] == operating_point).all(axis=1)
        ]

        dc_sweep = extract_signal_data_from_dataframe(
            dataframe=operating_point_data,
            sourcemeter_voltage_current_signal_name_pairs=sourcemeter_voltage_current_signal_name_pairs,
            multimeter_signals=multimeter_signals,
            **kwargs,
        )

        dc_sweep_data.append(dc_sweep)

    return DCSweepMeasurementDataCollection(collection=dc_sweep_data)
