from piel.types.connectivity.metrics import ComponentMetrics
from piel.types.metrics import ScalarMetric
from .amplifier import RFTwoPortAmplifier


class LNAMetrics(ComponentMetrics):
    """
    A model representing the metrics for a low-noise amplifier (LNA).

    Attributes:
        footprint_mm2 (  ScalarMetric ):
            The physical footprint of the amplifier in square millimeters.
        bandwidth_Hz (  ScalarMetric ):
            The operational bandwidth of the amplifier in Hertz, given as a range (min, max).
        noise_figure (  ScalarMetric ):
            The noise figure of the amplifier, given as a range (min, max).
        power_consumption_mW (  ScalarMetric ):
            The power consumption of the amplifier in milliwatts, given as a range (min, max).
        power_gain_dB (  ScalarMetric ):
            The power gain of the amplifier in decibels, given as a range (min, max).
        supply_voltage_V (  ScalarMetric ):
            The supply voltage of the amplifier in volts.
        technology_nm (  ScalarMetric ):
            The technology node of the amplifier in nanometers.
        technology_material (Optional[str]):
            The material technology used in the amplifier.
    """

    footprint_mm2: ScalarMetric = ScalarMetric()
    """
    footprint_mm2 (  ScalarMetric ):
        The physical footprint of the amplifier in square millimeters.
    """

    bandwidth_Hz: ScalarMetric = ScalarMetric()
    """
    bandwidth_Hz (  ScalarMetric ):
        The operational bandwidth of the amplifier in Hertz, given as a range (min, max).
    """

    noise_figure: ScalarMetric = ScalarMetric()
    """
    noise_figure (  ScalarMetric ):
        The noise figure of the amplifier, given as a range (min, max).
    """

    power_consumption_mW: ScalarMetric = ScalarMetric()
    """
    power_consumption_mW (  ScalarMetric ):
        The power consumption of the amplifier in milliwatts, given as a range (min, max).
    """

    power_gain_dB: ScalarMetric = ScalarMetric()
    """
    power_gain_dB (  ScalarMetric ):
        The power gain of the amplifier in decibels, given as a range (min, max).
    """

    supply_voltage_V: ScalarMetric = ScalarMetric()
    """
    supply_voltage_V (  ScalarMetric ):
        The supply voltage of the amplifier in volts.
    """

    technology_nm: ScalarMetric = ScalarMetric()
    """
    technology_nm (  ScalarMetric ):
        The technology node of the amplifier in nanometers.
    """

    technology_material: str = ""
    """
    technology_material (Optional[str]):
        The material technology used in the amplifier.
    """


class LowNoiseTwoPortAmplifier(RFTwoPortAmplifier):
    metrics: list[LNAMetrics] = []
