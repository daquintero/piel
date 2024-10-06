from piel.types.experimental import (
    VNASParameterMeasurementData,
    VNASParameterMeasurement,
)
from piel.types import (
    PathTypes,
    VNAPowerSweepMeasurement,
    VNAPowerSweepMeasurementData,
    NetworkTransmission,
    Phasor,
    PathTransmission,
    dBm,
    degree,
)
import numpy as np
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
    logger.debug("Extracting frequency array state")
    frequency_array_state = extract_power_sweep_s2p_to_network_transmission(
        file_path=measurement.spectrum_file
    )
    logger.debug(f"Frequency array state: {frequency_array_state}")
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


def convert_power_sweep_s2p_to_network_transmission(
    dataframe,
) -> NetworkTransmission:
    """
    Converts a pandas DataFrame containing S2P power sweep data into a NetworkTransmission object.

    The DataFrame is expected to have the following columns:
        - 'input_frequency_Hz': Frequency in Hz.
        - 'p_in_dbm': Input power in dBm.
        - S-parameter magnitude and phase columns for each S-parameter, e.g., 's_11_db', 's_11_deg', etc.

    Assumptions:
        - All rows correspond to the same input frequency. If multiple frequencies are present, the first one is used.
        - Each row represents a different input power level.
    """
    # Extract the unique input frequencies
    input_frequencies = dataframe["input_frequency_Hz"].unique()
    if len(input_frequencies) != 1:
        logger.warning("Multiple input frequencies found; using the first one.")
    input_frequency_Hz = input_frequencies[0]
    logger.debug(f"Using input frequency: {input_frequency_Hz} Hz")

    # Number of data points (input power levels)
    num_points = len(dataframe)
    logger.debug(f"Number of data points (input power levels): {num_points}")

    # Extract input power in dBm as a list
    p_in_dbm = dataframe["p_in_dbm"].tolist()
    logger.debug(f"Input power (dBm): {p_in_dbm}")

    # Create Phasor for input
    phasor = Phasor(
        magnitude=p_in_dbm,
        phase=[0.0] * num_points,  # No phase information for input power
        frequency=[input_frequency_Hz]
        * num_points,  # Replicate frequency to match length
        magnitude_unit=dBm,
        phase_unit=degree,
    )
    logger.debug(f"Input Phasor created: {phasor}")

    # Define S-parameters and corresponding port mappings
    s_params = ["s_11", "s_21", "s_12", "s_22"]
    port_mappings = {
        "s_11": ("in0", "in0"),
        "s_21": ("in0", "out0"),
        "s_12": ("out0", "in0"),
        "s_22": ("out0", "out0"),
    }

    # Initialize network transmissions list
    network: list[PathTransmission] = []

    for s_param in s_params:
        db_col = f"{s_param}_db"
        deg_col = f"{s_param}_deg"

        # Check if the required columns exist in the DataFrame
        if db_col not in dataframe.columns or deg_col not in dataframe.columns:
            logger.error(f"Missing columns for {s_param}: {db_col}, {deg_col}")
            raise ValueError(
                f"DataFrame must contain columns '{db_col}' and '{deg_col}' for S-parameter '{s_param}'."
            )

        # Extract magnitude in dB and phase in degrees
        s_db = dataframe[db_col].tolist()
        s_deg = dataframe[deg_col].tolist()
        logger.debug(
            f"S-parameter {s_param}: magnitude (dB)={s_db}, phase (deg)={s_deg}"
        )

        # Convert dB to linear magnitude
        magnitude_linear = 10 ** (np.array(s_db) / 20)
        magnitude_linear = magnitude_linear.tolist()
        logger.debug(f"S-parameter {s_param}: linear magnitude={magnitude_linear}")

        # Ensure phase is in degrees and convert if necessary
        # Assuming 's_deg' is already in degrees. If in radians, convert using np.rad2deg.
        phase_deg = s_deg  # Modify if conversion is needed

        # Create Phasor for transmission
        transmission_phasor = Phasor(
            magnitude=s_db,
            phase=phase_deg,
            frequency=[input_frequency_Hz]
            * num_points,  # Replicate frequency to match length
            phase_unit=degree,
        )
        logger.debug(
            f"S-parameter {s_param} Transmission Phasor created: {transmission_phasor}"
        )

        # Retrieve port mapping
        connection = port_mappings[s_param]

        # Create PathTransmission instance
        path_transmission = PathTransmission(
            connection=connection,  # Assuming connection is a tuple of (input_port, output_port)
            transmission=transmission_phasor,
        )
        logger.debug(f"PathTransmission for {s_param} created: {path_transmission}")

        # Append to network list
        network.append(path_transmission)

    # Create NetworkTransmission instance
    network_transmission = NetworkTransmission(
        input=phasor,
        network=network,
    )
    logger.debug(f"NetworkTransmission created: {network_transmission}")

    return network_transmission


def extract_power_sweep_s2p_to_network_transmission(
    file_path: PathTypes, input_frequency_Hz: float = 0, **kwargs
) -> NetworkTransmission:
    """
    Extracts power sweep data from an S2P file and converts it into a NetworkTransmission object.

    This function combines the functionalities of extracting data from an S2P file into a pandas DataFrame
    and then converting that DataFrame into a NetworkTransmission instance. It serves as a
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
    NetworkTransmission
        An instance of NetworkTransmission populated with the extracted and converted data.

    Example:
    --------
    >>> state = extract_power_sweep_s2p_to_network_transmission('path_to_file.s2p',input_frequency_Hz=1e9)
    >>> print(state)
    NetworkTransmission(p_in_dbm=[-10.0, -9.9977, ...], s_11_db=[-8.311036, -8.307557, ...], ...)

    Notes:
    ------
    - This function internally calls `extract_power_sweep_s2p_to_dataframe` and
      `convert_power_sweep_s2p_to_frequency_array_state`.
    - Ensure that the NetworkTransmission class is properly defined and accessible in your environment.
    """
    df = extract_power_sweep_s2p_to_dataframe(file_path, input_frequency_Hz, **kwargs)
    return convert_power_sweep_s2p_to_network_transmission(df)


def convert_row_to_sdict(row):
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

    sdict = {}

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
