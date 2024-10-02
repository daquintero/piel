import pandas as pd
from piel.types import PathTypes, V, A, Unit
from piel.file_system import return_path
from piel.models.physical.electrical import (
    construct_dc_signal,
    construct_current_dc_signal,
    construct_voltage_dc_signal,
)
from piel.types import (
    SignalDCCollection,
    DCSweepMeasurementDataCollection,
    SignalDC,
    VoltageCurrentSignalNamePair,
    Experiment,
    ExperimentData,
)
from piel.analysis.signals.dc import compile_dc_min_max_metrics_from_dc_collection


def construct_sourcemeter_sweep_signal_from_csv(
    file_path: PathTypes,
    voltage_signal_name: str,
    current_signal_name: str,
    **kwargs,
) -> SignalDC:
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
) -> SignalDC:
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

    return signal


def construct_multimeter_sweep_signal_from_csv(
    file_path: PathTypes,
    signal_name: str,
    unit: Unit = V,
    **kwargs,
) -> SignalDC:
    """
    Construct a multimeter sweep signal from a CSV file.

    Parameters
    ----------

    file_path : PathTypes
        The path to the CSV file.
    signal_name : str
        The name of the signal.
    unit: Unit
        Determines type of signal.
    **kwargs

    Returns
    -------

    SignalDC
        The multimeter sweep signal
    """
    file = return_path(file_path)
    dataframe = pd.read_csv(file)

    if unit is V:
        signal = construct_voltage_dc_signal(
            name=signal_name, values=dataframe[signal_name].values
        )
    elif unit is A:
        signal = construct_current_dc_signal(
            name=signal_name, values=dataframe[signal_name].values ** kwargs
        )
    else:
        raise ValueError(f"Unimplemented signal unit: {unit}")

    return signal


def construct_multimeter_sweep_signal_from_dataframe(
    dataframe: pd.DataFrame,
    signal_name: str,
    signal_kwargs: dict = None,
    **kwargs,
) -> SignalDC:
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

    SignalDC
        The multimeter sweep signal
    """
    if signal_kwargs is None:
        signal_kwargs = {}

    signal_data = dataframe[signal_name].values

    signal = construct_voltage_dc_signal(name=signal_name, values=signal_data)

    return signal


def extract_signal_data_from_dataframe(
    dataframe: pd.DataFrame,
    input_signal_name_list: list[VoltageCurrentSignalNamePair],
    output_signal_name_list: list[str],
    power_signal_name_list: list[VoltageCurrentSignalNamePair],
    **kwargs,
) -> SignalDCCollection:
    """
    Extract DC sweep data from a dataframe.

    Parameters
    ----------

    dataframe : pd.DataFrame
        The dataframe containing the DC sweep data.
    input_signal_name_list : list[VoltageCurrentSignalNamePair]
        The pairs of sourcemeter voltage and current signal names.
    output_signal_name_list : list[str]
        The multimeter signals.
    power_signal_name_list : list[VoltageCurrentSignalNamePair]
        The pairs of sourcemeter voltage and current signal names.
    **kwargs
        Additional keyword arguments.

    Returns
    -------

    SignalDCCollection
        The DC sweep data.
    """
    # Iterate through the sourcemeter signals and create the sourcemeter sweep signals
    input_sweep_signals = []

    for sourcemeter_voltage_signal_pair in input_signal_name_list:
        voltage_signal_name_i = sourcemeter_voltage_signal_pair[0]
        current_signal_name_i = sourcemeter_voltage_signal_pair[1]
        sourcemeter_sweep_signal = construct_sourcemeter_sweep_signal_from_dataframe(
            dataframe=dataframe,
            voltage_signal_name=voltage_signal_name_i,
            current_signal_name=current_signal_name_i,
        )
        input_sweep_signals.append(sourcemeter_sweep_signal)

    power_sweep_signals = []

    for sourcemeter_voltage_signal_pair in power_signal_name_list:
        voltage_signal_name_i = sourcemeter_voltage_signal_pair[0]
        current_signal_name_i = sourcemeter_voltage_signal_pair[1]
        sourcemeter_sweep_signal = construct_sourcemeter_sweep_signal_from_dataframe(
            dataframe=dataframe,
            voltage_signal_name=voltage_signal_name_i,
            current_signal_name=current_signal_name_i,
        )
        power_sweep_signals.append(sourcemeter_sweep_signal)

    # Iterate through the multimeter signals and create the multimeter sweep signals
    output_sweep_signals = []

    for multimeter_signal in output_signal_name_list:
        multimeter_sweep_signal = construct_multimeter_sweep_signal_from_dataframe(
            dataframe=dataframe, signal_name=multimeter_signal
        )
        output_sweep_signals.append(multimeter_sweep_signal)

    return SignalDCCollection(
        inputs=input_sweep_signals,
        outputs=output_sweep_signals,
        power=power_sweep_signals,
        **kwargs,
    )


def extract_signal_data_from_csv(
    file_path: PathTypes,
    input_signal_name_list: list[VoltageCurrentSignalNamePair],
    output_signal_name_list: list[str],
    power_signal_name_list: list[VoltageCurrentSignalNamePair],
    **kwargs,
) -> SignalDCCollection:
    """
    Extract DC sweep data from a CSV file.

    Parameters
    ----------
    file_path : PathTypes
        The path to the CSV file.
    input_signal_name_list : list[VoltageCurrentSignalNamePair]
        The pairs of sourcemeter voltage and current signal names.
    output_signal_name_list : list[str]
        The multimeter signals.
    power_signal_name_list : list[VoltageCurrentSignalNamePair]
        The pairs of sourcemeter voltage and current signal names relating to power lines.
    **kwargs
        Additional keyword arguments.

    Returns
    -------

    SignalDCCollection
        The DC sweep data.
    """
    file = return_path(file_path)
    dataframe = pd.read_csv(file)
    return extract_signal_data_from_dataframe(
        dataframe=dataframe,
        input_signal_name_list=input_signal_name_list,
        output_signal_name_list=output_signal_name_list,
        power_signal_name_list=power_signal_name_list,
        **kwargs,
    )


def extract_dc_sweeps_from_operating_point_csv(
    file_path: PathTypes,
    input_signal_name_list: list[VoltageCurrentSignalNamePair],
    output_signal_name_list: list[str],
    power_signal_name_list: list[VoltageCurrentSignalNamePair],
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
    input_signal_name_list : list[VoltageCurrentSignalNamePair]
        The pairs of sourcemeter voltage and current signal names.
    output_signal_name_list : list[str]
        The multimeter signals.
    power_signal_name_list : list[VoltageCurrentSignalNamePair]
        The pairs of sourcemeter voltage and current signal names relating to power lines.
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
            input_signal_name_list=input_signal_name_list,
            output_signal_name_list=output_signal_name_list,
            power_signal_name_list=power_signal_name_list,
            **kwargs,
        )

        dc_sweep_data.append(dc_sweep)

    return DCSweepMeasurementDataCollection(collection=dc_sweep_data)


def extract_dc_sweep_experiment_data_from_csv(
    file_path: PathTypes,
    input_signal_name_list: list[VoltageCurrentSignalNamePair],
    output_signal_name_list: list[str],
    power_signal_name_list: list[VoltageCurrentSignalNamePair],
    unique_operating_point_columns: list[str],
    **kwargs,
) -> ExperimentData:
    """
    Extract DC sweep data experiment data from a full operating point CSV file. The operating point CSV file contains the DC sweep data
    for multiple operating points. The unique operating point columns are used to extract the unique operating points
    from the CSV file. The DC sweep data is then extracted for each unique operating point. The DC sweep data is returned as a ExperimentData with the unique_operating_point_columns as part of the parameter_list definition, and the sweep data as part of the collection DCSweepMeasurementDataCollection.

    Parameters
    ----------

    file_path : PathTypes
        The path to the operating point CSV file.
    input_signal_name_list : list[VoltageCurrentSignalNamePair]
        The pairs of sourcemeter voltage and current signal names.
    output_signal_name_list : list[str]
        The multimeter signals.
    power_signal_name_list : list[VoltageCurrentSignalNamePair]
        The pairs of sourcemeter voltage and current signal names of the power lines.
    unique_operating_point_columns : list[str]
        The unique operating point columns.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    ExperimentData
      A collection of experiment and metadata to represent a DC sweep analysis.
    """
    file = return_path(file_path)
    dataframe = pd.read_csv(file)

    # Extract the unique operating points
    unique_operating_points = dataframe[
        unique_operating_point_columns
    ].drop_duplicates()

    parameters_list = unique_operating_points.to_dict(orient="records")

    data_collection = extract_dc_sweeps_from_operating_point_csv(
        file_path=file_path,
        input_signal_name_list=input_signal_name_list,
        output_signal_name_list=output_signal_name_list,
        power_signal_name_list=power_signal_name_list,
        unique_operating_point_columns=unique_operating_point_columns,
        **kwargs,
    )

    # Create metadata containers for automatic plotting/analysis.
    experiment = Experiment(
        parameters_list=parameters_list,
    )

    # Final output
    experiment_data = ExperimentData(experiment=experiment, data=data_collection)

    return experiment_data


def extract_dc_metrics_from_experiment_data(
    experiment_data: ExperimentData,
    parameter_column: str = "driver_b_v_set",
    label_column_name="ID",
    **kwargs,
):
    experiment_data_metrics = compile_dc_min_max_metrics_from_dc_collection(
        [collection for collection in experiment_data.data.collection],
        label_list=[
            v_dd for v_dd in experiment_data.experiment.parameters[parameter_column]
        ],
        label_column_name=label_column_name,
        **kwargs,
    )
    return experiment_data_metrics
