import numpy as np
from ..types import NumericalTypes


def Hz2s(hz: NumericalTypes) -> NumericalTypes:
    """
    Convert frequency in Hertz (Hz) to time period in seconds (s).
    :param hz: Frequency in Hz
    :return: Time period in seconds (s)
    """
    return np.reciprocal(hz)


def s2Hz(s: NumericalTypes) -> NumericalTypes:
    """
    Convert time period in seconds (s) to frequency in Hertz (Hz).
    :param s: Time period in seconds (s)
    :return: Frequency in Hertz (Hz)
    """
    return np.reciprocal(s)
