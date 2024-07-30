from typing import Optional
from ..core import PielBaseModel, NumericalTypes, MinimumMaximumType
from .amplifier import RFTwoPortAmplifier


class HVAMetricsType(PielBaseModel):
    """
    A model representing the metrics for a high-voltage amplifier (HVA).

    Attributes:
        footprint_mm2 (Optional[NumericalTypes]):
            The physical footprint of the amplifier in square millimeters.
        bandwidth_Hz (MinimumMaximumType | None):
            The operational bandwidth of the amplifier in Hertz, given as a range (min, max).
        power_added_efficiency (MinimumMaximumType | None):
            The power added efficiency of the amplifier, given as a range (min, max).
        power_consumption_mW (MinimumMaximumType | None):
            The power consumption of the amplifier in milliwatts, given as a range (min, max).
        power_gain_dB (MinimumMaximumType | None):
            The power gain of the amplifier in decibels, given as a range (min, max).
        saturated_power_output_dBm (Optional[NumericalTypes]):
            The saturated power output of the amplifier in dBm.
        supply_voltage_V (Optional[NumericalTypes]):
            The supply voltage of the amplifier in volts.
        technology_nm (Optional[NumericalTypes]):
            The technology node of the amplifier in nanometers.
        technology_material (Optional[str]):
            The material technology used in the amplifier.
    """

    footprint_mm2: Optional[NumericalTypes] = None
    """
    footprint_mm2 (Optional[NumericalTypes]):
        The physical footprint of the amplifier in square millimeters.
    """

    bandwidth_Hz: MinimumMaximumType | None = None
    """
    bandwidth_Hz (MinimumMaximumType | None):
        The operational bandwidth of the amplifier in Hertz, given as a range (min, max).
    """

    power_added_efficiency: MinimumMaximumType | None = None
    """
    power_added_efficiency (MinimumMaximumType | None):
        The power added efficiency of the amplifier, given as a range (min, max).
    """

    power_consumption_mW: MinimumMaximumType | None = None
    """
    power_consumption_mW (MinimumMaximumType | None):
        The power consumption of the amplifier in milliwatts, given as a range (min, max).
    """

    power_gain_dB: MinimumMaximumType | None = None
    """
    power_gain_dB (MinimumMaximumType | None):
        The power gain of the amplifier in decibels, given as a range (min, max).
    """

    saturated_power_output_dBm: Optional[NumericalTypes] = None
    """
    saturated_power_output_dBm (Optional[NumericalTypes]):
        The saturated power output of the amplifier in dBm.
    """

    supply_voltage_V: Optional[NumericalTypes] = None
    """
    supply_voltage_V (Optional[NumericalTypes]):
        The supply voltage of the amplifier in volts.
    """

    technology_nm: Optional[NumericalTypes] = None
    """
    technology_nm (Optional[NumericalTypes]):
        The technology node of the amplifier in nanometers.
    """

    technology_material: Optional[str] = None
    """
    technology_material (Optional[str]):
        The material technology used in the amplifier.
    """


class HighVoltageTwoPortAmplifier(RFTwoPortAmplifier):
    metrics: HVAMetricsType = None
