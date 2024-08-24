import numpy as np
from ..types import NumericalTypes


def dBm2watt(dBm: NumericalTypes) -> NumericalTypes:
    """
    Converts power from dBm (decibel-milliwatts) to Watts.

    The conversion is performed using the following equation:

    .. math::

        P_{\\text{Watt}} = 10^{\left(\\frac{P_{\\text{dBm}}}{10}\\right)} \\times 10^{-3}

    Args:
    - dBm (NumericalTypes): Power level in dBm (can be int, float, or numpy array).

    Returns:
    - NumericalTypes: Power level in Watts, in the same type as the input.
    """
    return 10 ** (dBm / 10) * 1e-3


def watt2dBm(watt: NumericalTypes) -> NumericalTypes:
    """
    Converts power from Watts to dBm (decibel-milliwatts).

    The conversion is performed using the following equation:

    .. math::
        P_{\text{dBm}} = 10 \times \log_{10}\left(\frac{P_{\text{Watt}}}{10^{-3}}\right)

    Args:
    - watt (NumericalTypes): Power level in Watts (can be int, float, or numpy array).

    Returns:
    - NumericalTypes: Power level in dBm, in the same type as the input.
    """
    return 10 * np.log10(watt / 1e-3)
