import pytest
from typing import List

# Import the functions to be tested
from piel.models.physical.electrical import (
    construct_voltage_dc_signal,
    construct_current_dc_signal,
    construct_dc_signal,
)

# Import the necessary classes and units
from piel.types import (
    SignalDC,
    SignalTraceDC,
    V,
    A,
)

# Sample data for testing
SAMPLE_VOLTAGE_VALUES: List[float] = [0.0, 1.1, 2.2, 3.3, 4.4]
SAMPLE_CURRENT_VALUES: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0]
EMPTY_VALUES: List[float] = []


def test_construct_voltage_dc_signal():
    """
    Test the construction of a voltage DC signal.
    """
    name = "Voltage Signal"
    values = SAMPLE_VOLTAGE_VALUES

    signal_dc = construct_voltage_dc_signal(name, values)

    # Assert that the returned object is an instance of SignalDC
    assert isinstance(
        signal_dc, SignalDC
    ), "Returned object is not a SignalDC instance."

    # Assert that there is exactly one trace in the trace list
    assert (
        len(signal_dc.trace_list) == 1
    ), "Voltage SignalDC should contain exactly one trace."

    trace = signal_dc.trace_list[0]

    # Assert that the trace is an instance of SignalTraceDC
    assert isinstance(trace, SignalTraceDC), "Trace is not a SignalTraceDC instance."

    # Assert that the trace has the correct name
    assert trace.name == name, f"Trace name should be '{name}'."

    # Assert that the trace has the correct values
    assert trace.values == values, "Trace values do not match the input values."

    # Assert that the trace has the correct unit (Voltage)
    assert trace.unit == V, "Trace unit is not Voltage (V)."


def test_construct_current_dc_signal():
    """
    Test the construction of a current DC signal.
    """
    name = "Current Signal"
    values = SAMPLE_CURRENT_VALUES

    signal_dc = construct_current_dc_signal(name, values)

    # Assert that the returned object is an instance of SignalDC
    assert isinstance(
        signal_dc, SignalDC
    ), "Returned object is not a SignalDC instance."

    # Assert that there is exactly one trace in the trace list
    assert (
        len(signal_dc.trace_list) == 1
    ), "Current SignalDC should contain exactly one trace."

    trace = signal_dc.trace_list[0]

    # Assert that the trace is an instance of SignalTraceDC
    assert isinstance(trace, SignalTraceDC), "Trace is not a SignalTraceDC instance."

    # Assert that the trace has the correct name
    assert trace.name == name, f"Trace name should be '{name}'."

    # Assert that the trace has the correct values
    assert trace.values == values, "Trace values do not match the input values."

    # Assert that the trace has the correct unit (Current)
    assert trace.unit == A, "Trace unit is not Current (A)."


def test_construct_dc_signal():
    """
    Test the construction of a combined DC signal with both voltage and current traces.
    """
    voltage_name = "Voltage Trace"
    voltage_values = SAMPLE_VOLTAGE_VALUES
    current_name = "Current Trace"
    current_values = SAMPLE_CURRENT_VALUES

    combined_signal_dc = construct_dc_signal(
        voltage_signal_name=voltage_name,
        voltage_signal_values=voltage_values,
        current_signal_name=current_name,
        current_signal_values=current_values,
    )

    # Assert that the returned object is an instance of SignalDC
    assert isinstance(
        combined_signal_dc, SignalDC
    ), "Returned object is not a SignalDC instance."

    # Assert that there are exactly two traces in the trace list
    assert (
        len(combined_signal_dc.trace_list) == 2
    ), "Combined SignalDC should contain exactly two traces."

    # Extract traces
    voltage_trace = next(
        (
            trace
            for trace in combined_signal_dc.trace_list
            if trace.name == voltage_name
        ),
        None,
    )
    current_trace = next(
        (
            trace
            for trace in combined_signal_dc.trace_list
            if trace.name == current_name
        ),
        None,
    )

    # Assert that both traces were found
    assert (
        voltage_trace is not None
    ), f"Voltage trace '{voltage_name}' not found in trace list."
    assert (
        current_trace is not None
    ), f"Current trace '{current_name}' not found in trace list."

    # Assert voltage trace details
    assert isinstance(
        voltage_trace, SignalTraceDC
    ), "Voltage trace is not a SignalTraceDC instance."
    assert (
        voltage_trace.values == voltage_values
    ), "Voltage trace values do not match input values."
    assert voltage_trace.unit == V, "Voltage trace unit is not Voltage (V)."

    # Assert current trace details
    assert isinstance(
        current_trace, SignalTraceDC
    ), "Current trace is not a SignalTraceDC instance."
    assert (
        current_trace.values == current_values
    ), "Current trace values do not match input values."
    assert current_trace.unit == A, "Current trace unit is not Current (A)."


def test_construct_voltage_dc_signal_empty_values():
    """
    Test constructing a voltage DC signal with empty values.
    """
    name = "Empty Voltage Signal"
    values = EMPTY_VALUES

    signal_dc = construct_voltage_dc_signal(name, values)

    # Assert that the trace list contains one trace
    assert (
        len(signal_dc.trace_list) == 1
    ), "SignalDC should contain exactly one trace even if values are empty."

    trace = signal_dc.trace_list[0]

    # Assert that the trace has empty values
    assert trace.values == values, "Trace values should be empty."


def test_construct_current_dc_signal_empty_values():
    """
    Test constructing a current DC signal with empty values.
    """
    name = "Empty Current Signal"
    values = EMPTY_VALUES

    signal_dc = construct_current_dc_signal(name, values)

    # Assert that the trace list contains one trace
    assert (
        len(signal_dc.trace_list) == 1
    ), "SignalDC should contain exactly one trace even if values are empty."

    trace = signal_dc.trace_list[0]

    # Assert that the trace has empty values
    assert trace.values == values, "Trace values should be empty."


def test_construct_dc_signal_empty_values():
    """
    Test constructing a combined DC signal with empty voltage and current values.
    """
    voltage_name = "Empty Voltage Trace"
    voltage_values = EMPTY_VALUES
    current_name = "Empty Current Trace"
    current_values = EMPTY_VALUES

    combined_signal_dc = construct_dc_signal(
        voltage_signal_name=voltage_name,
        voltage_signal_values=voltage_values,
        current_signal_name=current_name,
        current_signal_values=current_values,
    )

    # Assert that the trace list contains two traces
    assert (
        len(combined_signal_dc.trace_list) == 2
    ), "Combined SignalDC should contain exactly two traces."

    # Extract traces
    voltage_trace = next(
        (
            trace
            for trace in combined_signal_dc.trace_list
            if trace.name == voltage_name
        ),
        None,
    )
    current_trace = next(
        (
            trace
            for trace in combined_signal_dc.trace_list
            if trace.name == current_name
        ),
        None,
    )

    # Assert that both traces were found
    assert (
        voltage_trace is not None
    ), f"Voltage trace '{voltage_name}' not found in trace list."
    assert (
        current_trace is not None
    ), f"Current trace '{current_name}' not found in trace list."

    # Assert that both traces have empty values
    assert (
        voltage_trace.values == voltage_values
    ), "Voltage trace values should be empty."
    assert (
        current_trace.values == current_values
    ), "Current trace values should be empty."


# def test_construct_voltage_dc_signal_invalid_inputs():
#     """
#     Test constructing a voltage DC signal with invalid inputs.
#     """
#     name = "Invalid Voltage Signal"
#     invalid_values = "not a list"
#
#     with pytest.raises(TypeError):
#         construct_voltage_dc_signal(name, invalid_values)


# def test_construct_current_dc_signal_invalid_inputs():
#     """
#     Test constructing a current DC signal with invalid inputs.
#     """
#     name = "Invalid Current Signal"
#     invalid_values = {"key": "value"}
#
#     with pytest.raises(TypeError):
#         construct_current_dc_signal(name, invalid_values)


# def test_construct_dc_signal_invalid_inputs():
#     """
#     Test constructing a combined DC signal with invalid inputs.
#     """
#     voltage_name = "Invalid Voltage Trace"
#     voltage_values = "invalid"
#     current_name = "Invalid Current Trace"
#     current_values = None
#
#     with pytest.raises(TypeError):
#         construct_dc_signal(
#             voltage_signal_name=voltage_name,
#             voltage_signal_values=voltage_values,
#             current_signal_name=current_name,
#             current_signal_values=current_values,
#         )
