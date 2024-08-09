from ....types import ArrayTypes, SignalInstanceDC, SignalDC


def construct_voltage_dc_signal(
    name: str,
    values: ArrayTypes,
) -> SignalDC:
    """
    Construct a DC signal instance for a voltage signal.

    Parameters
    ----------
    name : str
        The name of the signal.
    values : ArrayTypes
        The values of the signal.

    Returns
    -------
        SignalInstanceDC: A DC signal instance for a voltage signal.
    """
    voltage_signal = SignalInstanceDC(name=name, values=values, data_type="voltage")
    return SignalDC(signal_instances=[voltage_signal])


def construct_current_dc_signal(
    name: str,
    values: ArrayTypes,
) -> SignalDC:
    """
    Construct a DC signal instance for a current signal.

    Parameters
    ----------
    name : str
        The name of the signal.
    values : ArrayTypes
        The values of the signal.

    Returns
    -------
        SignalInstanceDC: A DC signal instance for a current signal.
    """
    current_signal = SignalInstanceDC(name=name, values=values, data_type="current")
    return SignalDC(signal_instances=[current_signal])


def construct_dc_signal(
    voltage_signal_name: str,
    voltage_signal_values: ArrayTypes,
    current_signal_name: str,
    current_signal_values: ArrayTypes,
) -> SignalDC:
    """
    Construct a DC signal with voltage and current signal instances.

    Parameters
    ----------
    voltage_signal_name : str
        The name of the voltage signal.
    voltage_signal_values : ArrayTypes
        The values of the voltage signal.
    current_signal_name : str
        The name of the current signal.
    current_signal_values : ArrayTypes
        The values of the current signal

    Returns
    -------
        SignalDC: A DC signal with voltage and current signal instances.
    """
    voltage_signal = construct_voltage_dc_signal(
        voltage_signal_name, voltage_signal_values
    )
    current_signal = construct_current_dc_signal(
        current_signal_name, current_signal_values
    )

    signals = voltage_signal.signal_instances + current_signal.signal_instances

    return SignalDC(signal_instances=signals)
