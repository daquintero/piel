import pandas as pd
import logging
from piel.types import Phasor, ScalarMetric, FrequencyMetric, dBm

# Configure logger
logger = logging.getLogger(__name__)


def max_power_s21_frequency_metric_from_dataframe(
    dataframe: pd.DataFrame,
) -> FrequencyMetric:
    """
    Extracts the maximum s_21_magnitude_dBm from the DataFrame, composes the corresponding input input,
    and returns both as a Phasor and ScalarMetric.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing the required columns.

    Returns:
        Tuple[Phasor, ScalarMetric]: A tuple containing the input Phasor and the output ScalarMetric.
    """
    required_columns = [
        "magnitude_dBm",
        "phase_degree",
        "frequency_Hz",
        "s_21_magnitude_dBm",
        "s_21_phase_degree",
        "s_21_frequency_Hz",
    ]

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(
            f"The DataFrame is missing required columns: {missing_columns}"
        )

    # Find the row with the maximum s_21_magnitude_dBm
    max_s21_row = dataframe.loc[dataframe["s_21_magnitude_dBm"].idxmax()]
    logger.debug(f"Row with maximum s_21_magnitude_dBm: \n{max_s21_row}")

    # Extract input input components
    input_magnitude_dBm = max_s21_row["magnitude_dBm"]
    input_phase_degree = max_s21_row["phase_degree"]
    input_frequency_Hz = max_s21_row["frequency_Hz"]

    logger.debug(
        f"Input Phasor Components - Magnitude: {input_magnitude_dBm} dBm, "
        f"Phase: {input_phase_degree} degrees, Frequency: {input_frequency_Hz} Hz"
    )

    # Create Phasor instance for input (as scalar)
    input_phasor = Phasor(
        magnitude=input_magnitude_dBm,
        phase=input_phase_degree,
        frequency=input_frequency_Hz,
    )
    logger.debug(f"Input Phasor created: {input_phasor}")

    # Extract the maximum s_21_magnitude_dBm and its corresponding unit
    max_s21_magnitude_dBm = max_s21_row["s_21_magnitude_dBm"]
    logger.debug(f"Maximum s_21_magnitude_dBm: {max_s21_magnitude_dBm} dBm")

    # Create ScalarMetric instance for output
    scalar_metric = ScalarMetric(
        value=max_s21_magnitude_dBm,
        unit=dBm,
        description="Maximum s21 magnitude in dBm",
    )
    logger.debug(f"ScalarMetric created: {scalar_metric}")

    frequency_metric = FrequencyMetric(
        input=input_phasor,
        metric=scalar_metric,
    )

    return frequency_metric
