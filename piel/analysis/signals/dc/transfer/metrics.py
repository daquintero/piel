import numpy as np
from piel.types import ScalarMetric, SignalDCCollection
from ..utils import (
    get_trace_values_by_datum,
)  # Ensure this utility is correctly implemented
from typing import Literal


def get_out_min_max(
    collection: SignalDCCollection,
    lower_threshold_ratio: float = 0.1,
    upper_threshold_ratio: float = 0.9,
    **kwargs,
) -> ScalarMetric:
    """
    Retrieves the minimum and maximum output voltage values within a specified input voltage range.

    Args:
        collection (SignalDCCollection): The collection of input and output DC signals.
        lower_threshold_ratio (float, optional): The lower threshold as a fraction of V_in range (0-1). Defaults to 0.1.
        upper_threshold_ratio (float, optional): The upper threshold as a fraction of V_in range (0-1). Defaults to 0.9.

    Returns:
        ScalarMetric: Metrics including min and max values of the output voltage in the specified input voltage range.
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
    metrics = ScalarMetric(
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
    collection: "SignalDCCollection",
    lower_threshold_ratio: float = 0.1,
    upper_threshold_ratio: float = 0.9,
    transition_type: Literal["analogue", "digital"] = "analogue",
    transition_direction: Literal["positive", "negative"] = "positive",
    **kwargs,
) -> "ScalarMetric":
    """
    Calculates the equivalent input voltage range (V_in) corresponding to specified thresholds of output voltage (V_out).

    Args:
        collection (SignalDCCollection): The collection of input and output DC signals.
        lower_threshold_ratio (float, optional): The lower threshold as a fraction of V_out's final value (0-1). Defaults to 0.1.
        upper_threshold_ratio (float, optional): The upper threshold as a fraction of V_out's final value (0-1). Defaults to 0.9.
        transition_type (Literal["analogue", "digital"], optional): Type of transition. Defaults to "analogue".
        transition_direction (Literal["positive", "negative"], optional): Direction of transition. Defaults to "positive".
        **kwargs: Additional keyword arguments.

    Returns:
        ScalarMetric: Metrics including min and max V_in values corresponding to the specified V_out threshold range.
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

    # Depending on transition direction, adjust the threshold comparison
    selected_indices = (output_voltage >= lower_threshold) & (
        output_voltage <= upper_threshold
    )

    corresponding_input_voltages = input_voltage[selected_indices]

    if len(corresponding_input_voltages) == 0:
        raise ValueError(
            "No input voltages found corresponding to the specified output voltage threshold range."
            f" Input_voltage: {input_voltage}, Output_voltage: {output_voltage}, "
            f"Selected_indices: {selected_indices}, Collection: {collection}"
        )

    # Identify the unit for input voltage
    input_unit = None
    for trace in input_signal_dc.trace_list:
        if trace.unit.datum.lower() == "voltage":
            input_unit = trace.unit
            break

    if input_unit is None:
        raise ValueError("Input voltage unit not found.")

    # Calculate metrics based on transition type
    if transition_type == "analogue":
        # Compute min and max input voltages in the corresponding range
        metrics = ScalarMetric(
            value=None,  # Not applicable
            mean=None,  # Not applicable
            min=np.min(corresponding_input_voltages),
            max=np.max(corresponding_input_voltages),
            standard_deviation=None,  # Not applicable
            count=None,  # Not applicable
            unit=input_unit,
        )
    elif transition_type == "digital":
        # For digital transitions, assuming binary states, find unique states within the range
        unique_voltages = np.unique(corresponding_input_voltages)
        if len(unique_voltages) < 2:
            raise ValueError(
                "Not enough unique input voltage levels found for a digital transition."
                f" Unique_voltages: {unique_voltages}"
            )
        elif len(unique_voltages) > 2:
            # If more than two unique levels, determine the two closest to the transition thresholds
            # Sort unique voltages based on proximity to V_out_final
            if transition_direction == "positive":
                unique_voltages_sorted = np.sort(unique_voltages)
            else:
                unique_voltages_sorted = np.sort(unique_voltages)[::-1]
            # Take the two extreme values as the digital levels
            digital_low = unique_voltages_sorted[0]
            digital_high = unique_voltages_sorted[1]
        else:
            digital_low, digital_high = unique_voltages

        metrics = ScalarMetric(
            value=None,  # Not applicable
            mean=None,  # Not applicable
            min=digital_low,
            max=digital_high,
            standard_deviation=None,  # Not applicable
            count=None,  # Not applicable
            unit=input_unit,
        )
    else:
        raise ValueError("transition_type must be either 'analogue' or 'digital'.")

    return metrics
