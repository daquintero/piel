from piel.types.connectivity.metrics import ComponentMetrics
from piel.types.metrics import ScalarMetric
from .amplifier import RFTwoPortAmplifier


class PowerAmplifierMetrics(ComponentMetrics):
    """
    A model representing the metrics for a high-voltage amplifier (HVA) or a Power Amplifier (PA).

    Attributes:
        footprint_mm2 ( ScalarMetric ):
            The physical footprint of the amplifier in square millimeters.
        bandwidth_Hz ( ScalarMetric ):
            The operational bandwidth of the amplifier in Hertz, given as a range (min, max).
        power_added_efficiency ( ScalarMetric ):
            The power added efficiency of the amplifier, given as a range (min, max).
        power_consumption_mW ( ScalarMetric ):
            The power consumption of the amplifier in milliwatts, given as a range (min, max).
        power_gain_dB ( ScalarMetric ):
            The power gain of the amplifier in decibels, given as a range (min, max).
        saturated_power_output_dBm ( ScalarMetric ):
            The saturated power output of the amplifier in dBm.
        supply_voltage_V ( ScalarMetric ):
            The supply voltage of the amplifier in volts.
        technology_nm ( ScalarMetric ):
            The technology node of the amplifier in nanometers.
        technology_material (Optional[str]):
            The material technology used in the amplifier.
    """

    footprint_mm2: ScalarMetric = ScalarMetric()
    """
    footprint_mm2 ( ScalarMetric ):
        The physical footprint of the amplifier in square millimeters.
    """

    bandwidth_Hz: ScalarMetric = ScalarMetric()
    """
    bandwidth_Hz ( ScalarMetric ):
        The operational bandwidth of the amplifier in Hertz, given as a range (min, max).
    """

    power_added_efficiency: ScalarMetric = ScalarMetric()
    """
    power_added_efficiency ( ScalarMetric ):
        The power added efficiency of the amplifier, given as a range (min, max).
    """

    power_consumption_mW: ScalarMetric = ScalarMetric()
    """
    power_consumption_mW ( ScalarMetric ):
        The power consumption of the amplifier in milliwatts, given as a range (min, max).
    """

    power_gain_dB: ScalarMetric = ScalarMetric()
    """
    power_gain_dB ( ScalarMetric ):
        The power gain of the amplifier in decibels, given as a range (min, max).
    """

    saturated_power_output_dBm: ScalarMetric = ScalarMetric()
    """
    saturated_power_output_dBm ( ScalarMetric ):
        The saturated power output of the amplifier in dBm.
    """

    supply_voltage_V: ScalarMetric = ScalarMetric()
    """
    supply_voltage_V ( ScalarMetric ):
        The supply voltage of the amplifier in volts.
    """

    technology_nm: ScalarMetric = ScalarMetric()
    """
    technology_nm ( ScalarMetric ):
        The technology node of the amplifier in nanometers.
    """

    technology_material: str = ""
    """
    technology_material (Optional[str]):
        The material technology used in the amplifier.
    """


class PowerAmplifier(RFTwoPortAmplifier):
    metrics: list[PowerAmplifierMetrics] = []
