import numpy as np
from piel.types import ScalarMetrics, SignalDCCollection, W, SignalDC, SignalTraceDC, V
from ..utils import (
    get_trace_values_by_datum,
)
import logging


logger = logging.getLogger(__name__)


def calculate_power_signal_from_collection(
    collection: SignalDCCollection,
    lower_threshold_ratio: float = 0,
    upper_threshold_ratio: float = 1,
) -> ScalarMetrics:
    """
    Retrieves the minimum and maximum power values within a specified input voltage range,
    along with the corresponding V_in values where these extrema occur.

    Args:
        collection (SignalDCCollection): The collection of input, output, and power DC signals.
        lower_threshold_ratio (float, optional): The lower threshold as a fraction of V_in range (0-1). Defaults to 0.1.
        upper_threshold_ratio (float, optional): The upper threshold as a fraction of V_in range (0-1). Defaults to 0.9.

    Returns:
        ScalarMetrics
            - ScalarMetrics containing min and max power.
    """
    # Validate threshold ratios
    if not (0 <= lower_threshold_ratio < upper_threshold_ratio <= 1):
        raise ValueError("Threshold ratios must satisfy 0 <= lower < upper <= 1.")

    # Extract or compute power values
    power = None
    power_signal_dc = None
    power_unit = None
    power_values = None

    for signal_dc in collection.power:
        # Attempt to get direct power trace
        power = get_trace_values_by_datum(signal_dc, "watt")
        if power is not None:
            power_values = power
            # Identify the unit for power
            for trace in signal_dc.trace_list:
                if trace.unit.datum.lower() == "watt":
                    power_unit = trace.unit
                    break
            power_signal_dc = signal_dc
            break  # Power found, exit loop

        # If direct power not found, attempt to compute from voltage and current
        voltage = get_trace_values_by_datum(signal_dc, "voltage")
        current = get_trace_values_by_datum(signal_dc, "ampere")
        if voltage is not None and current is not None:
            logger.debug("Multiplying voltage and current values")
            power_values = voltage * current
            # Assume power unit is derived from voltage and current units
            voltage_unit = None
            current_unit = None
            for trace in signal_dc.trace_list:
                if trace.unit.datum.lower() == "voltage":
                    voltage_unit = trace.unit
                elif trace.unit.datum.lower() == "ampere":
                    current_unit = trace.unit
            if voltage_unit is not None and current_unit is not None:
                # Example: if voltage is in volts and current in amperes, power is in watts
                # Adjust accordingly based on actual unit implementation
                # power_unit = voltage_unit * current_unit # TODO implement this
                power_unit = W
            power_signal_trace = SignalTraceDC(
                unit=power_unit,
                values=power_values,
            )
            power_signal_dc = SignalDC(
                trace_list=[power_signal_trace],
            )
            break

    try:
        logger.debug(f"Voltage values: {voltage}")
        logger.debug(f"Current values: {current}")
        logger.debug(f"Power values: {power_values}")
    except Exception:
        pass

    if power_values is None or len(power_values) == 0:
        raise ValueError("Power trace not found or empty in the collection.")

    return power_signal_dc


def get_power_metrics(
    collection: SignalDCCollection,
    lower_threshold_ratio: float = 0,
    upper_threshold_ratio: float = 1,
) -> ScalarMetrics:
    """
    Retrieves the minimum and maximum power values within a specified input voltage range,
    along with the corresponding V_in values where these extrema occur.

    Args:
        collection (SignalDCCollection): The collection of input, output, and power DC signals.
        lower_threshold_ratio (float, optional): The lower threshold as a fraction of V_in range (0-1). Defaults to 0.1.
        upper_threshold_ratio (float, optional): The upper threshold as a fraction of V_in range (0-1). Defaults to 0.9.

    Returns:
        ScalarMetrics
            - ScalarMetrics containing min and max power.
    """
    # Validate threshold ratios
    power_signal_dc = calculate_power_signal_from_collection(
        collection, lower_threshold_ratio, upper_threshold_ratio
    )
    power_values = power_signal_dc.trace_list[0].values
    power_unit = power_signal_dc.trace_list[0].unit

    # Identify input voltage trace based on unit datum
    input_voltage = None
    for signal_dc in collection.inputs:
        input_voltage = get_trace_values_by_datum(signal_dc, "voltage")
        if input_voltage is not None:
            break

    if input_voltage is None or len(input_voltage) == 0:
        raise ValueError("Input voltage trace not found or empty.")

    if len(input_voltage) != len(power_values):
        raise ValueError("Input voltage and Power arrays must be of the same length.")

    # Define specified thresholds based on input voltage range
    V_in_min = np.min(input_voltage)
    V_in_max = np.max(input_voltage)
    lower_threshold = V_in_min + lower_threshold_ratio * (V_in_max - V_in_min)
    upper_threshold = V_in_min + upper_threshold_ratio * (V_in_max - V_in_min)

    # Select indices within the specified input voltage range
    vin_range_mask = (input_voltage >= lower_threshold) & (
        input_voltage <= upper_threshold
    )
    power_in_range = power_values[vin_range_mask]

    if len(power_in_range) == 0:
        raise ValueError(
            "No power values found within the specified input voltage range."
        )

    # Compute min and max power in the specified range
    min_power = np.min(power_in_range)
    max_power = np.max(power_in_range)
    mean_power = np.mean(power_in_range)
    std_power = np.std(power_in_range)
    count_power = len(power_in_range)
    # std_power = None

    # Identify the unit for power if not already identified
    if power_unit is None and power_signal_dc is not None:
        for trace in power_signal_dc.trace_list:
            if trace.unit.datum.lower() == "watt":
                power_unit = trace.unit
                break
    if power_unit is None:
        # Default to watts if unit not found
        power_unit = W

    # Create ScalarMetrics for power
    metrics = ScalarMetrics(
        value=mean_power,  # Not applicable
        mean=mean_power,  # Not applicable
        min=min_power,
        max=max_power,
        standard_deviation=std_power,  # Not applicable
        count=count_power,  # Not applicable
        unit=power_unit,
    )

    return metrics


def get_power_map_vin_metrics(
    collection: SignalDCCollection,
    lower_threshold_ratio: float = 0,
    upper_threshold_ratio: float = 1,
) -> ScalarMetrics:
    """
    Retrieves the mapped V_IN minimum and maximum power values within a specified input voltage range. Represents
    along with the corresponding V_in values where these power extrema occur.

    Args:
        collection (SignalDCCollection): The collection of input, output, and power DC signals.
        lower_threshold_ratio (float, optional): The lower threshold as a fraction of V_in range (0-1). Defaults to 0.1.
        upper_threshold_ratio (float, optional): The upper threshold as a fraction of V_in range (0-1). Defaults to 0.9.

    Returns:
        ScalarMetrics
            - ScalarMetrics containing min and max power.
    """
    # Validate threshold ratios
    power_signal_dc = calculate_power_signal_from_collection(
        collection, lower_threshold_ratio, upper_threshold_ratio
    )
    power_values = power_signal_dc.trace_list[0].values
    power_unit = power_signal_dc.trace_list[0].unit

    # Identify input voltage trace based on unit datum
    input_voltage = None
    for signal_dc in collection.inputs:
        input_voltage = get_trace_values_by_datum(signal_dc, "voltage")
        if input_voltage is not None:
            break

    if input_voltage is None or len(input_voltage) == 0:
        raise ValueError("Input voltage trace not found or empty.")

    if len(input_voltage) != len(power_values):
        raise ValueError("Input voltage and Power arrays must be of the same length.")

    # Define specified thresholds based on input voltage range
    V_in_min = np.min(input_voltage)
    V_in_max = np.max(input_voltage)
    lower_threshold = V_in_min + lower_threshold_ratio * (V_in_max - V_in_min)
    upper_threshold = V_in_min + upper_threshold_ratio * (V_in_max - V_in_min)

    # Select indices within the specified input voltage range
    vin_range_mask = (input_voltage >= lower_threshold) & (
        input_voltage <= upper_threshold
    )
    power_in_range = power_values[vin_range_mask]
    corresponding_vin = input_voltage[vin_range_mask]

    if len(power_in_range) == 0:
        raise ValueError(
            "No power values found within the specified input voltage range."
        )

    # Compute min and max power in the specified range
    min_power = np.min(power_in_range)
    max_power = np.max(power_in_range)
    # std_power = None

    # Find the V_in corresponding to min and max power
    min_power_indices = np.where(power_in_range == min_power)[0]
    # mean_power_indices = np.where(power_in_range == mean_power)[0]
    max_power_indices = np.where(power_in_range == max_power)[0]

    # In case of multiple occurrences, take the first one
    min_mapping_vin = corresponding_vin[min_power_indices[0]]
    # mean_mapping_vin = corresponding_vin[mean_power_indices[0]]
    max_mapping_vin = corresponding_vin[max_power_indices[0]]
    # vin_count = len(corresponding_vin)

    # Identify the unit for power if not already identified
    if power_unit is None and power_signal_dc is not None:
        for trace in power_signal_dc.trace_list:
            if trace.unit.datum.lower() == "watt":
                power_unit = trace.unit
                break
    if power_unit is None:
        # Default to watts if unit not found
        power_unit = W

    # Create ScalarMetrics for power
    metrics = ScalarMetrics(
        value=None,  # Not applicable
        mean=None,  # Not applicable
        min=min_mapping_vin,
        max=max_mapping_vin,
        standard_deviation=None,  # Not applicable
        count=None,  # Not applicable
        unit=V,
    )

    return metrics
