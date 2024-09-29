from piel import return_path
from piel.types.experimental import (
    VNASParameterMeasurementData,
    VNASParameterMeasurement,
)
from piel.types import (
    FrequencyTransmissionState,
    FrequencyTransmissionCollection,
    NumericalTypes,
    PathTypes,
    SDict,
    VNAPowerSweepMeasurement,
    VNAPowerSweepMeasurementData,
    FrequencyTransmissionArrayState,
)
import logging

logger = logging.getLogger(__name__)


def extract_s_parameter_data_from_vna_measurement(
    measurement: VNASParameterMeasurement, **kwargs
) -> VNASParameterMeasurementData:
    # TODO implement skrf requirement message
    import skrf

    network = skrf.Network(name=measurement.name, file=measurement.spectrum_file)
    return VNASParameterMeasurementData(
        name=measurement.name,
        network=network,
        **kwargs,
    )


def extract_power_sweep_data_from_vna_measurement(
    measurement: VNAPowerSweepMeasurement, **kwargs
) -> VNAPowerSweepMeasurementData:
    logger.debug("extracting frequecny array state")
    frequency_array_state = extract_power_sweep_s2p_to_frequency_array_state(
        file_path=measurement.spectrum_file,
    )
    logger.debug(f"frequecny array state: {frequency_array_state}")
    return VNAPowerSweepMeasurementData(
        name=measurement.name, network=frequency_array_state
    )


# TODO move everything down here to another file.


def extract_power_sweep_s2p_to_dataframe(
    file_path: PathTypes, input_frequency_Hz: float = 0, **kwargs
):
    """
    Extracts numerical data from an S2P (Touchstone) file and returns it as a pandas DataFrame.

    This function reads an S2P file, parses its numerical data, and organizes it into a structured
    pandas DataFrame. It skips comment lines and ensures that each data line contains the expected
    number of columns. If discrepancies are found, warnings are printed, and those lines are skipped.

    Parameters:
    -----------
    file_path : PathTypes
        The path to the S2P file to be processed. Can be a string or a Path-like object.

    input_frequency_Hz : float, optional (default=0)
        The input frequency in Hertz to be added as a column in the resulting DataFrame.

    **kwargs :
        Additional keyword arguments to pass to the pandas DataFrame constructor.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the extracted data with the following columns:
        - `p_in_dbm` : Input power in dBm.
        - `s_11_db` : S-parameter S11 in dB.
        - `s_11_deg` : S-parameter S11 in degrees.
        - `s_21_db` : S-parameter S21 in dB.
        - `s_21_deg` : S-parameter S21 in degrees.
        - `s_12_db` : S-parameter S12 in dB.
        - `s_12_deg` : S-parameter S12 in degrees.
        - `s_22_db` : S-parameter S22 in dB.
        - `s_22_deg` : S-parameter S22 in degrees.
        - `input_frequency_Hz` : The input frequency provided as a parameter.

    Example:
    --------
    >>> df = extract_power_sweep_s2p_to_dataframe('path_to_file.s2p', input_frequency_Hz=1e9)
    >>> print(df.head())
       p_in_dbm  s_11_db  s_11_deg  s_21_db  s_21_deg  s_12_db  s_12_deg  s_22_db  s_22_deg  input_frequency_Hz
    0   -10.0000 -8.311036  90.38824 -11.35558  137.4781 -55.67513   54.62733  -8.564775 -164.7370         1000000000.0
    1    -9.9977 -8.307557  90.38396 -11.34543  137.4497 -55.04230   53.47807  -8.555398 -164.7173         1000000000.0
    2    -9.9953 -8.309752  90.35067 -11.35137  137.4250 -55.01111    48.11482  -8.562661 -164.6533         1000000000.0
    3    -9.9930 -8.310988  90.35760 -11.34326  137.3693 -56.74514    39.34027  -8.559170 -164.7386         1000000000.0

    Notes:
    ------
    - Lines in the S2P file starting with '!' or '#' are treated as comments or headers and are skipped.
    - Each valid data line is expected to have exactly 9 numerical values corresponding to the defined columns.
    - If a line does not have 9 values or contains non-numeric data, a warning or error is printed, and the line is skipped.
    """
    import pandas as pd

    # Define the column names
    column_names = [
        "p_in_dbm",
        "s_11_db",
        "s_11_deg",
        "s_21_db",
        "s_21_deg",
        "s_12_db",
        "s_12_deg",
        "s_22_db",
        "s_22_deg",
    ]

    # Initialize a list to store data rows
    data_rows = []

    # Open and read the file
    with open(file_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            # Skip lines starting with '!' or '#'
            if line.startswith(("!", "#")):
                continue
            # Split the line into parts based on whitespace
            parts = line.split()
            if len(parts) != 9:
                # Optionally handle or log unexpected number of columns
                print(
                    f"Warning: Unexpected number of columns ({len(parts)}) in line {line_number}: {line}"
                )
                continue
            # Convert all parts to float
            try:
                float_parts = [float(part) for part in parts]
                data_rows.append(float_parts)
            except ValueError as e:
                # Optionally handle or log conversion errors
                print(f"Error converting line {line_number} to floats: {line}\n{e}")
                continue

    # Create a DataFrame
    df = pd.DataFrame(data_rows, columns=column_names, **kwargs)
    df["input_frequency_Hz"] = input_frequency_Hz

    return df


def convert_power_sweep_s2p_to_frequency_array_state(
    dataframe,
) -> FrequencyTransmissionArrayState:
    """
    Converts a pandas DataFrame containing S2P power sweep data into a FrequencyTransmissionArrayState object.

    This function takes a DataFrame structured with specific S-parameter columns and transforms it into
    a FrequencyTransmissionArrayState instance, facilitating further analysis or processing within
    the application's context.

    Parameters:
    -----------
    df : pd.DataFrame
        A pandas DataFrame containing S2P power sweep data. The DataFrame is expected to have columns
        corresponding to the following parameters:
        - `p_in_dbm`
        - `s_11_db`
        - `s_11_deg`
        - `s_21_db`
        - `s_21_deg`
        - `s_12_db`
        - `s_12_deg`
        - `s_22_db`
        - `s_22_deg`
        - `input_frequency_Hz`

    Returns:
    --------
    FrequencyTransmissionArrayState
        An instance of FrequencyTransmissionArrayState populated with the data from the DataFrame.

    Example:
    --------
    >>> df = extract_power_sweep_s2p_to_dataframe('path_to_file.s2p', input_frequency_Hz=1e9)
    >>> state = convert_power_sweep_s2p_to_frequency_array_state(df)
    >>> print(state)
    FrequencyTransmissionArrayState(p_in_dbm=[-10.0, -9.9977, ...], s_11_db=[-8.311036, -8.307557, ...], ...)

    Notes:
    ------
    - The FrequencyTransmissionArrayState class must be defined elsewhere in your codebase.
    - Ensure that the DataFrame `df` contains all the required columns before invoking this function.
    """
    s2p_dict = dataframe.to_dict(orient="list")
    logger.debug(f"s2p dict {str(s2p_dict.keys())}")
    logger.debug("extracting here")
    frequency_transmission_array_state = FrequencyTransmissionArrayState(**s2p_dict)
    logger.debug("extracted")
    logger.debug(
        f"frequency transmission array state {frequency_transmission_array_state}"
    )
    return frequency_transmission_array_state


def extract_power_sweep_s2p_to_frequency_array_state(
    file_path: PathTypes, input_frequency_Hz: float = 0, **kwargs
) -> FrequencyTransmissionArrayState:
    """
    Extracts power sweep data from an S2P file and converts it into a FrequencyTransmissionArrayState object.

    This function combines the functionalities of extracting data from an S2P file into a pandas DataFrame
    and then converting that DataFrame into a FrequencyTransmissionArrayState instance. It serves as a
    convenient single-step process for obtaining structured transmission state data from an S2P file.

    Parameters:
    -----------
    file_path : PathTypes
        The path to the S2P file to be processed. Can be a string or a Path-like object.

    input_frequency_Hz : float, optional (default=0)
        The input frequency in Hertz to be added to the DataFrame before conversion.

    **kwargs :
        Additional keyword arguments to pass to the `extract_power_sweep_s2p_to_dataframe` function.

    Returns:
    --------
    FrequencyTransmissionArrayState
        An instance of FrequencyTransmissionArrayState populated with the extracted and converted data.

    Example:
    --------
    >>> state = extract_power_sweep_s2p_to_frequency_array_state('path_to_file.s2p', input_frequency_Hz=1e9)
    >>> print(state)
    FrequencyTransmissionArrayState(p_in_dbm=[-10.0, -9.9977, ...], s_11_db=[-8.311036, -8.307557, ...], ...)

    Notes:
    ------
    - This function internally calls `extract_power_sweep_s2p_to_dataframe` and
      `convert_power_sweep_s2p_to_frequency_array_state`.
    - Ensure that the FrequencyTransmissionArrayState class is properly defined and accessible in your environment.
    """
    df = extract_power_sweep_s2p_to_dataframe(file_path, input_frequency_Hz, **kwargs)
    return convert_power_sweep_s2p_to_frequency_array_state(df)


def convert_row_to_sdict(row) -> SDict:
    """
    Converts a single DataFrame row containing S-parameter data into an SDict.

    Parameters:
    -----------
    row : pd.Series
        A pandas Series containing S-parameter data with the following indices:
        - p_in_dbm
        - s_11_db
        - s_11_deg
        - s_21_db
        - s_21_deg
        - s_12_db
        - s_12_deg
        - s_22_db
        - s_22_deg

    Returns:
    --------
    SDict
        A dictionary mapping PortCombination tuples to complex S-parameter arrays.

    Example:
    --------
    >>> sdict = convert_row_to_sdict(df.iloc[0])
    >>> print(sdict)
    {
        ('in0', 'in0'): DeviceArray(-0.03295842+0.0313j, dtype=float32),
        ('in0', 'out0'): DeviceArray(-0.03361994+0.0325j, dtype=float32),
        ('out0', 'in0'): DeviceArray(0.03118884+0.0477j, dtype=float32),
        ('out0', 'out0'): DeviceArray(-0.03206138-0.0143j, dtype=float32)
    }
    """
    import jax.numpy as jnp

    # Define port names
    port_map = {
        "s_11": ("in0", "in0"),
        "s_21": ("in0", "out0"),
        "s_12": ("out0", "in0"),
        "s_22": ("out0", "out0"),
    }

    sdict: SDict = {}

    for s_param, ports in port_map.items():
        db_col = f"{s_param}_db"
        deg_col = f"{s_param}_deg"

        # Convert dB to linear magnitude
        magnitude_linear = 10 ** (row[db_col] / 20)

        # Convert degrees to radians
        phase_rad = jnp.deg2rad(jnp.array(row[deg_col]))

        # Compute complex S-parameter
        complex_s = magnitude_linear * (jnp.cos(phase_rad) + 1j * jnp.sin(phase_rad))

        sdict[ports] = complex_s

    return sdict


def extract_power_sweep_s2p_to_frequency_transmission_collection(
    file_path: PathTypes,
    name: str = "Power Sweep",
    input_frequency_Hz: NumericalTypes = 0,  # Default frequency set to 0 GHz
    **kwargs,
) -> FrequencyTransmissionCollection:
    """
    Extracts power sweep data from an S2P file and returns a FrequencyTransmissionCollection.
    TODO improve performance as this is pretty slow.

    Parameters:
    -----------
    file_path : PathTypes
        The path to the S2P file to be processed.

    name : str, optional
        The name of the FrequencyTransmissionCollection. Default is "Power Sweep".

    input_frequency_Hz : NumericalTypes, optional
        The input frequency in Hertz. Default is 1 GHz.
        This parameter is used for all entries in the collection.
        If your S2P file contains frequency information, you should modify this function to extract it accordingly.

    Returns:
    --------
    FrequencyTransmissionCollection
        A collection containing FrequencyTransmissionState instances for each power sweep point.

    Example:
    --------
    >>> ft_collection = extract_power_sweep_s2p_to_frequency_transmission_collection('path_to_file.s2p',input_frequency_Hz=2e9)
    >>> print(ft_collection)
    FrequencyTransmissionCollection(
        name='Power Sweep',
        collection=[
            FrequencyTransmissionState(input_frequency_Hz=2e9, input_power_dBm=-10.0, transmission=SDict(...)),
            FrequencyTransmissionState(input_frequency_Hz=2e9, input_power_dBm=-9.99765625, transmission=SDict(...)),
            ...
        ]
    )
    """
    file_path = return_path(file_path)

    import time

    pre_df = time.time()
    # Extract DataFrame from S2P file
    df = extract_power_sweep_s2p_to_dataframe(file_path=file_path, **kwargs)
    post_df = time.time()
    print(f"Dataframe Import time: {post_df - pre_df}")

    # Initialize the collection list
    collection: list[FrequencyTransmissionState] = []

    pre_instance = time.time()
    # Iterate over each row in the DataFrame to create FrequencyTransmissionState instances
    for index, row in df.iterrows():
        # Convert the current row to SDict
        sdict = convert_row_to_sdict(row)

        # Create FrequencyTransmissionState
        ft_state = FrequencyTransmissionState(
            input_frequency_Hz=input_frequency_Hz,
            input_power_dBm=row["p_in_dbm"],
            transmission=sdict,  # SDict specific to this row
        )
        collection.append(ft_state)
    post_instance = time.time()
    print(f"Instance creation time: {post_instance - pre_instance}. TODO improve.")

    # Create FrequencyTransmissionCollection
    ft_collection = FrequencyTransmissionCollection(name=name, collection=collection)

    return ft_collection
