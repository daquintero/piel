from piel.types.connectivity.metrics import ComponentMetrics
from piel.types.metrics import ScalarMetrics
from .amplifier import RFTwoPortAmplifier


class LNAMetrics(ComponentMetrics):
    """
    A model representing the metrics for a low-noise amplifier (LNA).

    Attributes:
        footprint_mm2 (  ScalarMetrics ):
            The physical footprint of the amplifier in square millimeters.
        bandwidth_Hz (  ScalarMetrics ):
            The operational bandwidth of the amplifier in Hertz, given as a range (min, max).
        noise_figure (  ScalarMetrics ):
            The noise figure of the amplifier, given as a range (min, max).
        power_consumption_mW (  ScalarMetrics ):
            The power consumption of the amplifier in milliwatts, given as a range (min, max).
        power_gain_dB (  ScalarMetrics ):
            The power gain of the amplifier in decibels, given as a range (min, max).
        supply_voltage_V (  ScalarMetrics ):
            The supply voltage of the amplifier in volts.
        technology_nm (  ScalarMetrics ):
            The technology node of the amplifier in nanometers.
        technology_material (Optional[str]):
            The material technology used in the amplifier.
    """

    type: str = "LNAMetrics"

    footprint_mm2: ScalarMetrics = ScalarMetrics()
    """
    footprint_mm2 (  ScalarMetrics ):
        The physical footprint of the amplifier in square millimeters.
    """

    bandwidth_Hz: ScalarMetrics = ScalarMetrics()
    """
    bandwidth_Hz (  ScalarMetrics ):
        The operational bandwidth of the amplifier in Hertz, given as a range (min, max).
    """

    noise_figure: ScalarMetrics = ScalarMetrics()
    """
    noise_figure (  ScalarMetrics ):
        The noise figure of the amplifier, given as a range (min, max).
    """

    power_consumption_mW: ScalarMetrics = ScalarMetrics()
    """
    power_consumption_mW (  ScalarMetrics ):
        The power consumption of the amplifier in milliwatts, given as a range (min, max).
    """

    power_gain_dB: ScalarMetrics = ScalarMetrics()
    """
    power_gain_dB (  ScalarMetrics ):
        The power gain of the amplifier in decibels, given as a range (min, max).
    """

    supply_voltage_V: ScalarMetrics = ScalarMetrics()
    """
    supply_voltage_V (  ScalarMetrics ):
        The supply voltage of the amplifier in volts.
    """

    technology_nm: ScalarMetrics = ScalarMetrics()
    """
    technology_nm (  ScalarMetrics ):
        The technology node of the amplifier in nanometers.
    """

    technology_material: str = ""
    """
    technology_material (Optional[str]):
        The material technology used in the amplifier.
    """


class LowNoiseTwoPortAmplifier(RFTwoPortAmplifier):
    type: str = "LowNoiseTwoPortAmplifier"
    metrics: LNAMetrics = None
