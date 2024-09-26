import numpy as np
from typing import Optional
from piel.types import SignalDC, Unit


def get_trace_values_by_datum(
    signal_dc: SignalDC, desired_datum: str
) -> Optional[np.ndarray]:
    """
    Retrieves the values of a trace from a SignalDC instance based on the unit's datum.

    Args:
        signal_dc (SignalDC): The SignalDC instance containing the traces.
        desired_datum (str): The datum type to filter traces (e.g., 'voltage', 'current').

    Returns:
        Optional[np.ndarray]: The numpy array of trace values if found, else None.
    """
    for trace in signal_dc.trace_list:
        if trace.unit.datum.lower() == desired_datum.lower():
            return np.array(trace.values)
    return None


def get_trace_values_by_unit(
    signal_dc: SignalDC, desired_unit: Unit
) -> Optional[np.ndarray]:
    """
    Retrieves the values of a trace from a SignalDC instance based on the exact unit.

    Args:
        signal_dc (SignalDC): The SignalDC instance containing the traces.
        desired_unit (Unit): The Unit instance to filter traces.

    Returns:
        Optional[np.ndarray]: The numpy array of trace values if found, else None.
    """
    for trace in signal_dc.trace_list:
        if trace.unit.name.lower() == desired_unit.name.lower():
            return np.array(trace.values)
    return None
