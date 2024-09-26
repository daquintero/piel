import numpy as np
from piel.types import ScalarMetrics, SignalDCCollection
from ..utils import (
    get_trace_values_by_datum,
)  # Ensure this utility is correctly implemented


def get_out_min_max(
    collection: SignalDCCollection,
    lower_threshold_ratio: float = 0.1,
    upper_threshold_ratio: float = 0.9,
) -> ScalarMetrics:
    """
    Retrieves the minimum and maximum output voltage values within a specified input voltage range.

    Args:
        collection (SignalDCCollection): The collection of input and output DC signals.
        lower_threshold_ratio (float, optional): The lower threshold as a fraction of V_in range (0-1). Defaults to 0.1.
        upper_threshold_ratio (float, optional): The upper threshold as a fraction of V_in range (0-1). Defaults to 0.9.

    Returns:
        ScalarMetrics: Metrics including min and max values of the output voltage in the specified input voltage range.
    """
    # Validate threshold ratios
    if not (0 <= lower_threshold_ratio < upper_threshold_ratio <= 1):
        raise ValueError("Threshold ratios must satisfy 0 <= lower < upper <= 1.")

    # Identify input voltage trace based on unit datum
    input_voltage = None
    for signal_dc in collection.inputs:
        input_voltage = get_trace_values_by_datum(signal_dc, "voltage")
        if input_voltage is not None:
            break

    if input_voltage is None or len(input_voltage) == 0:
        raise ValueError("Input voltage trace not found or empty.")

    # Identify output voltage trace based on unit datum
    output_voltage = None
    output_signal_dc = None
    for signal_dc in collection.outputs:
        output_voltage = get_trace_values_by_datum(signal_dc, "voltage")
        if output_voltage is not None:
            output_signal_dc = signal_dc
            break

    if output_voltage is None or len(output_voltage) == 0:
        raise ValueError("Output voltage trace not found or empty.")

    if len(input_voltage) != len(output_voltage):
        raise ValueError("Input and Output voltage arrays must be of the same length.")

    # Define specified thresholds based on input voltage range
    V_in_min = np.min(input_voltage)
    V_in_max = np.max(input_voltage)
    lower_threshold = V_in_min + lower_threshold_ratio * (V_in_max - V_in_min)
    upper_threshold = V_in_min + upper_threshold_ratio * (V_in_max - V_in_min)

    # Select indices within the specified input voltage range
    linear_region_mask = (input_voltage >= lower_threshold) & (
        input_voltage <= upper_threshold
    )
    linear_output_voltages = output_voltage[linear_region_mask]

    if len(linear_output_voltages) == 0:
        raise ValueError(
            "No output voltages found within the specified input voltage range."
        )

    # Identify the unit for output voltage
    output_unit = None
    for trace in output_signal_dc.trace_list:
        if trace.unit.datum.lower() == "voltage":
            output_unit = trace.unit
            break

    if output_unit is None:
        raise ValueError("Output voltage unit not found.")

    # Compute min and max output voltages in the linear region
    metrics = ScalarMetrics(
        value=None,  # Not applicable
        mean=None,  # Not applicable
        min=np.min(linear_output_voltages),
        max=np.max(linear_output_voltages),
        standard_deviation=None,  # Not applicable
        count=None,  # Not applicable
        unit=output_unit,
    )

    return metrics


def get_out_response_in_transition_range(
    collection: SignalDCCollection,
    lower_threshold_ratio: float = 0.1,
    upper_threshold_ratio: float = 0.9,
) -> ScalarMetrics:
    """
    Calculates the equivalent input voltage range (V_in) corresponding to specified thresholds of output voltage (V_out).

    Args:
        collection (SignalDCCollection): The collection of input and output DC signals.
        lower_threshold_ratio (float, optional): The lower threshold as a fraction of V_out's final value (0-1). Defaults to 0.1.
        upper_threshold_ratio (float, optional): The upper threshold as a fraction of V_out's final value (0-1). Defaults to 0.9.

    Returns:
        ScalarMetrics: Metrics including min and max V_in values corresponding to the specified V_out threshold range.
    """
    # Validate threshold ratios
    if not (0 <= lower_threshold_ratio < upper_threshold_ratio <= 1):
        raise ValueError("Threshold ratios must satisfy 0 <= lower < upper <= 1.")

    # Identify output voltage trace based on unit datum
    output_voltage = None
    for signal_dc in collection.outputs:
        output_voltage = get_trace_values_by_datum(signal_dc, "voltage")
        if output_voltage is not None:
            break

    if output_voltage is None or len(output_voltage) == 0:
        raise ValueError("Output voltage trace not found or empty.")

    # Identify input voltage trace based on unit datum
    input_voltage = None
    input_signal_dc = None
    for signal_dc in collection.inputs:
        input_voltage = get_trace_values_by_datum(signal_dc, "voltage")
        if input_voltage is not None:
            input_signal_dc = signal_dc
            break

    if input_voltage is None or len(input_voltage) == 0:
        raise ValueError("Input voltage trace not found or empty.")

    if len(input_voltage) != len(output_voltage):
        raise ValueError("Input and Output voltage arrays must be of the same length.")

    # Define specified thresholds based on output voltage's final value
    V_out_final = np.max(output_voltage)  # Assuming V_out approaches a final value
    lower_threshold = lower_threshold_ratio * V_out_final
    upper_threshold = upper_threshold_ratio * V_out_final

    # Select indices where output voltage is within the specified threshold range
    selected_indices = (output_voltage >= lower_threshold) & (
        output_voltage <= upper_threshold
    )
    corresponding_input_voltages = input_voltage[selected_indices]

    if len(corresponding_input_voltages) == 0:
        raise ValueError(
            "No input voltages found corresponding to the specified output voltage threshold range."
        )

    # Identify the unit for input voltage
    input_unit = None
    for trace in input_signal_dc.trace_list:
        if trace.unit.datum.lower() == "voltage":
            input_unit = trace.unit
            break

    if input_unit is None:
        raise ValueError("Input voltage unit not found.")

    # Compute min and max input voltages in the corresponding range
    metrics = ScalarMetrics(
        value=None,  # Not applicable
        mean=None,  # Not applicable
        min=np.min(corresponding_input_voltages),
        max=np.max(corresponding_input_voltages),
        standard_deviation=None,  # Not applicable
        count=None,  # Not applicable
        unit=input_unit,
    )

    return metrics
