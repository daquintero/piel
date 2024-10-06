import pytest
import numpy as np

# Import the functions to be tested
from piel.analysis.signals.dc import (
    get_trace_values_by_datum,
    get_trace_values_by_unit,
)

# Import necessary classes and units
from piel.types import (
    SignalDC,
    SignalTraceDC,
    Unit,
    V,
    A,
    ratio,
)

# Sample Units
W = Unit(name="watt", datum="watt", base=1, label="Power W")
dB = Unit(name="Decibel", datum="dB", base=1, label="Ratio dB")

# Sample Data for Testing
VOLTAGE_VALUES = [0.0, 1.1, 2.2, 3.3, 4.4]
CURRENT_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]
RATIO_VALUES = [1, 2, 3, 4, 5]


def create_signal_dc(name: str, values: list, unit: Unit) -> SignalDC:
    """
    Helper function to create a SignalDC instance with a single trace.
    """
    trace = SignalTraceDC(name=name, values=values, unit=unit)
    return SignalDC(trace_list=[trace])


def test_get_trace_values_by_datum_voltage():
    """
    Test retrieving voltage trace values by datum.
    """
    signal_dc = create_signal_dc("Voltage Trace", VOLTAGE_VALUES, V)
    retrieved_values = get_trace_values_by_datum(signal_dc, "voltage")
    assert retrieved_values is not None, "Should retrieve voltage values."
    np.testing.assert_array_equal(retrieved_values, np.array(VOLTAGE_VALUES))


def test_get_trace_values_by_datum_current():
    """
    Test retrieving current trace values by datum.
    """
    signal_dc = create_signal_dc("Current Trace", CURRENT_VALUES, A)
    retrieved_values = get_trace_values_by_datum(signal_dc, "ampere")
    assert retrieved_values is not None, "Should retrieve current values."
    np.testing.assert_array_equal(retrieved_values, np.array(CURRENT_VALUES))


def test_get_trace_values_by_datum_ratio():
    """
    Test retrieving ratio trace values by datum.
    """
    signal_dc = create_signal_dc("Ratio Trace", RATIO_VALUES, ratio)
    retrieved_values = get_trace_values_by_datum(signal_dc, "1")
    assert retrieved_values is not None, "Should retrieve ratio values."
    np.testing.assert_array_equal(retrieved_values, np.array(RATIO_VALUES))


def test_get_trace_values_by_datum_case_insensitive():
    """
    Test that datum matching is case-insensitive.
    """
    signal_dc = create_signal_dc("Voltage Trace", VOLTAGE_VALUES, V)
    retrieved_values = get_trace_values_by_datum(signal_dc, "Voltage")
    assert (
        retrieved_values is not None
    ), "Should retrieve voltage values with case-insensitive datum."
    np.testing.assert_array_equal(retrieved_values, np.array(VOLTAGE_VALUES))


def test_get_trace_values_by_datum_not_found():
    """
    Test retrieving values with a datum that does not exist.
    """
    signal_dc = create_signal_dc("Voltage Trace", VOLTAGE_VALUES, V)
    retrieved_values = get_trace_values_by_datum(signal_dc, "current")
    assert retrieved_values is None, "Should return None when datum is not found."


def test_get_trace_values_by_unit_voltage():
    """
    Test retrieving voltage trace values by exact unit.
    """
    signal_dc = create_signal_dc("Voltage Trace", VOLTAGE_VALUES, V)
    retrieved_values = get_trace_values_by_unit(signal_dc, V)
    assert retrieved_values is not None, "Should retrieve voltage values by unit."
    np.testing.assert_array_equal(retrieved_values, np.array(VOLTAGE_VALUES))


def test_get_trace_values_by_unit_current():
    """
    Test retrieving current trace values by exact unit.
    """
    signal_dc = create_signal_dc("Current Trace", CURRENT_VALUES, A)
    retrieved_values = get_trace_values_by_unit(signal_dc, A)
    assert retrieved_values is not None, "Should retrieve current values by unit."
    np.testing.assert_array_equal(retrieved_values, np.array(CURRENT_VALUES))


def test_get_trace_values_by_unit_not_found():
    """
    Test retrieving values with a unit that does not exist.
    """
    signal_dc = create_signal_dc("Voltage Trace", VOLTAGE_VALUES, V)
    retrieved_values = get_trace_values_by_unit(
        signal_dc, W
    )  # Looking for Watt in Voltage Trace
    assert retrieved_values is None, "Should return None when unit is not found."


def test_get_trace_values_by_unit_case_insensitive():
    """
    Test that unit matching is case-insensitive.
    """
    signal_dc = create_signal_dc("Voltage Trace", VOLTAGE_VALUES, V)
    V_upper = Unit(name="Volt", datum="voltage", base=1, label="V")
    retrieved_values = get_trace_values_by_unit(signal_dc, V_upper)
    assert (
        retrieved_values is not None
    ), "Should retrieve voltage values with case-insensitive unit."
    np.testing.assert_array_equal(retrieved_values, np.array(VOLTAGE_VALUES))


def test_get_trace_values_by_unit_multiple_traces():
    """
    Test retrieving values when multiple traces exist.
    """
    voltage_signal = create_signal_dc("Voltage Trace", VOLTAGE_VALUES, V)
    current_signal = create_signal_dc("Current Trace", CURRENT_VALUES, A)
    combined_signal_dc = SignalDC(
        trace_list=voltage_signal.trace_list + current_signal.trace_list
    )

    retrieved_voltage = get_trace_values_by_unit(combined_signal_dc, V)
    assert (
        retrieved_voltage is not None
    ), "Should retrieve voltage values from combined traces."
    np.testing.assert_array_equal(retrieved_voltage, np.array(VOLTAGE_VALUES))

    retrieved_current = get_trace_values_by_unit(combined_signal_dc, A)
    assert (
        retrieved_current is not None
    ), "Should retrieve current values from combined traces."
    np.testing.assert_array_equal(retrieved_current, np.array(CURRENT_VALUES))


def test_get_trace_values_by_unit_duplicate_units():
    """
    Test retrieving values when multiple traces have the same unit.
    """
    voltage_signal1 = create_signal_dc("Voltage Trace 1", VOLTAGE_VALUES, V)
    voltage_signal2 = create_signal_dc("Voltage Trace 2", VOLTAGE_VALUES, V)
    combined_signal_dc = SignalDC(
        trace_list=voltage_signal1.trace_list + voltage_signal2.trace_list
    )

    retrieved_values = get_trace_values_by_unit(combined_signal_dc, V)
    assert (
        retrieved_values is not None
    ), "Should retrieve the first matching voltage trace."
    np.testing.assert_array_equal(
        retrieved_values, np.array(VOLTAGE_VALUES)
    )  # Assuming first trace is returned


def test_get_trace_values_by_unit_empty_trace_list():
    """
    Test retrieving values from a SignalDC with no traces.
    """
    empty_signal_dc = SignalDC(trace_list=[])
    retrieved_values = get_trace_values_by_unit(empty_signal_dc, V)
    assert retrieved_values is None, "Should return None when trace list is empty."


def test_get_trace_values_by_datum_empty_trace_list():
    """
    Test retrieving values by datum from a SignalDC with no traces.
    """
    empty_signal_dc = SignalDC(trace_list=[])
    retrieved_values = get_trace_values_by_datum(empty_signal_dc, "voltage")
    assert retrieved_values is None, "Should return None when trace list is empty."
