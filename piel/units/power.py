import numpy as np
from ..types import NumericalTypes


def dBm2watt(dBm: NumericalTypes) -> NumericalTypes:
    """
    Converts power from dBm (decibel-milliwatts) to Watts.

    The conversion is performed using the following equation:

    .. math::

        P_{\text{Watt}} = 10^{\left(\frac{P_{\text{dBm}}}{10}\right)} \times 10^{-3}

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


def watt2vrms(watt: NumericalTypes, impedance: NumericalTypes = 50.0) -> NumericalTypes:
    """
    Converts power in Watts to Vrms (root mean square voltage) for a given impedance.

    The conversion is performed using the following equation:

    .. math::

        V_{\text{rms}} = \sqrt{P_{\text{Watt}} \times Z}

    Args:
    - watt (NumericalTypes): Power level in Watts.
    - impedance (float): The network impedance in ohms. Defaults to 50.0 Ohms.

    Returns:
    - NumericalTypes: Vrms in the same type as the input.
    """
    return np.sqrt(watt * impedance)


def dBm2vrms(dBm: NumericalTypes, impedance: NumericalTypes = 50.0) -> NumericalTypes:
    """
    Converts power from dBm to Vrms (root mean square voltage) in a specified impedance network.

    The conversion is performed using the following steps:

    1. Convert dBm to Watts:
       .. math::

           P_{\text{Watt}} = 10^{\left(\frac{P_{\text{dBm}}}{10}\right)} \times 10^{-3}

    2. Convert Watts to Vrms:
       .. math::

           V_{\text{rms}} = \sqrt{P_{\text{Watt}} \times Z}

    Args:
    - dBm (NumericalTypes): Power level in dBm (can be int, float, or numpy array).
    - impedance (float): The network impedance in ohms. Default is 50 ohms.

    Returns:
    - NumericalTypes: Vrms value, in the same type as the input.
    """
    power_watt = dBm2watt(dBm)
    return watt2vrms(power_watt, impedance)


def vrms2vpp(vrms: NumericalTypes) -> NumericalTypes:
    """
    Converts Vrms (root mean square voltage) to Vpp (peak-to-peak voltage).

    The conversion is performed using the following equation:

    .. math::

        V_{\text{pp}} = V_{\text{rms}} \times \sqrt{2} \times 2

    Args:
    - vrms (NumericalTypes): Vrms value.

    Returns:
    - NumericalTypes: Vpp in the same type as the input.
    """
    return vrms * np.sqrt(2) * 2


def vrms2watt(vrms: NumericalTypes, impedance: NumericalTypes = 50.0) -> NumericalTypes:
    """
    Converts Vrms (root mean square voltage) to power in Watts for a given impedance.

    The conversion is performed using the following equation:

    .. math::
        P_{\text{Watt}} = \frac{V_{\text{rms}}^2}{Z}

    Args:
    - vrms (NumericalTypes): Vrms value.
    - impedance (float): The network impedance in ohms.

    Returns:
    - NumericalTypes: Power level in Watts, in the same type as the input.
    """
    return (vrms**2) / impedance


def vrms2dBm(vrms: NumericalTypes, impedance: NumericalTypes = 50.0) -> NumericalTypes:
    """
    Converts Vrms (root mean square voltage) to dBm in a specified impedance network.

    The conversion is performed using the following steps:

    1. Convert Vrms to Watts:
       .. math::
           P_{\text{Watt}} = \frac{V_{\text{rms}}^2}{Z}

    2. Convert Watts to dBm:
       .. math::
           P_{\text{dBm}} = 10 \times \log_{10}\left(\frac{P_{\text{Watt}}}{10^{-3}}\right)

    Args:
    - vrms (NumericalTypes): Vrms value.
    - impedance (float): The network impedance in ohms. Default is 50 ohms.

    Returns:
    - NumericalTypes: Power level in dBm, in the same type as the input.
    """
    power_watt = vrms2watt(vrms, impedance)
    return watt2dBm(power_watt)


def dBm2vpp(dBm: NumericalTypes, impedance: NumericalTypes = 50.0) -> NumericalTypes:
    """
    Converts power from dBm to Vpp (peak-to-peak voltage) in a specified impedance network.

    The conversion is performed using the following steps:

    1. Convert dBm to Watts:
       .. math::

           P_{\text{Watt}} = 10^{\left(\frac{P_{\text{dBm}}}{10}\right)} \times 10^{-3}

    2. Convert Watts to Vrms:
       .. math::

           V_{\text{rms}} = \sqrt{P_{\text{Watt}} \times Z}

    3. Convert Vrms to Vpp:
       .. math::

           V_{\text{pp}} = V_{\text{rms}} \times \sqrt{2} \times 2

    Args:
    - dBm (NumericalTypes): Power level in dBm (can be int, float, or numpy array).
    - impedance (float): The network impedance in ohms. Default is 50 ohms.

    Returns:
    - NumericalTypes: Vpp value, in the same type as the input.
    """
    vrms_value = dBm2vrms(dBm, impedance)
    return vrms2vpp(vrms_value)


def vpp2vrms(vpp: NumericalTypes) -> NumericalTypes:
    """
    Converts Vpp (peak-to-peak voltage) to Vrms (root mean square voltage).

    The conversion is performed using the following equation:

    .. math::
        V_{\text{rms}} = \frac{V_{\text{pp}}}{\sqrt{2} \times 2}

    Args:
    - vpp (NumericalTypes): Vpp value.

    Returns:
    - NumericalTypes: Vrms value, in the same type as the input.
    """
    return vpp / (np.sqrt(2) * 2)


def vpp2dBm(vpp: NumericalTypes, impedance: NumericalTypes = 50.0) -> NumericalTypes:
    """
    Converts Vpp (peak-to-peak voltage) to dBm in a specified impedance network.

    The conversion is performed using the following steps:

    1. Convert Vpp to Vrms:
       .. math::
           V_{\text{rms}} = \frac{V_{\text{pp}}}{\sqrt{2} \times 2}

    2. Convert Vrms to Watts:
       .. math::
           P_{\text{Watt}} = \frac{V_{\text{rms}}^2}{Z}

    3. Convert Watts to dBm:
       .. math::
           P_{\text{dBm}} = 10 \times \log_{10}\left(\frac{P_{\text{Watt}}}{10^{-3}}\right)

    Args:
    - vpp (NumericalTypes): Vpp value.
    - impedance (float): The network impedance in ohms. Default is 50 ohms.

    Returns:
    - NumericalTypes: Power level in dBm, in the same type as the input.
    """
    vrms_value = vpp2vrms(vpp)
    return vrms2dBm(vrms_value, impedance)
