"""
This module defines data models for low-noise amplifier (LNA) and high-voltage amplifier (HVA) metrics.
It provides structured types using pydantic for validation and includes type aliases for metric ranges.
"""
import gdsfactory as gf
from typing import Optional
from .core import PielBaseModel, NumericalTypes

# Type alias for a photonic circuit component in gdsfactory.
ElectronicCircuitComponent = gf.Component
"""
PhotonicCircuitComponent:
    A type representing a component in a photonic circuit, as defined in the gdsfactory framework.
    This type is used to handle and manipulate photonic components in circuit designs.
"""

# Type alias for representing minimum and maximum values as an optional tuple of floats.
MinimumMaximumType = tuple[Optional[float], Optional[float]]
"""
MinimumMaximumType:
    A tuple representing a range with minimum and maximum values.
    Each value in the tuple can be a float or None.
"""


class LNAMetricsType(PielBaseModel):
    """
    A model representing the metrics for a low-noise amplifier (LNA).

    Attributes:
        footprint_mm2 (Optional[NumericalTypes]):
            The physical footprint of the amplifier in square millimeters.
        bandwidth_Hz (MinimumMaximumType | None):
            The operational bandwidth of the amplifier in Hertz, given as a range (min, max).
        noise_figure (MinimumMaximumType | None):
            The noise figure of the amplifier, given as a range (min, max).
        power_consumption_mW (MinimumMaximumType | None):
            The power consumption of the amplifier in milliwatts, given as a range (min, max).
        power_gain_dB (MinimumMaximumType | None):
            The power gain of the amplifier in decibels, given as a range (min, max).
        supply_voltage_V (Optional[NumericalTypes]):
            The supply voltage of the amplifier in volts.
        technology_nm (Optional[NumericalTypes]):
            The technology node of the amplifier in nanometers.
        technology_material (Optional[str]):
            The material technology used in the amplifier.
    """

    footprint_mm2: Optional[NumericalTypes]
    """
    footprint_mm2 (Optional[NumericalTypes]):
        The physical footprint of the amplifier in square millimeters.
    """

    bandwidth_Hz: MinimumMaximumType | None
    """
    bandwidth_Hz (MinimumMaximumType | None):
        The operational bandwidth of the amplifier in Hertz, given as a range (min, max).
    """

    noise_figure: MinimumMaximumType | None
    """
    noise_figure (MinimumMaximumType | None):
        The noise figure of the amplifier, given as a range (min, max).
    """

    power_consumption_mW: MinimumMaximumType | None
    """
    power_consumption_mW (MinimumMaximumType | None):
        The power consumption of the amplifier in milliwatts, given as a range (min, max).
    """

    power_gain_dB: MinimumMaximumType | None
    """
    power_gain_dB (MinimumMaximumType | None):
        The power gain of the amplifier in decibels, given as a range (min, max).
    """

    supply_voltage_V: Optional[NumericalTypes]
    """
    supply_voltage_V (Optional[NumericalTypes]):
        The supply voltage of the amplifier in volts.
    """

    technology_nm: Optional[NumericalTypes]
    """
    technology_nm (Optional[NumericalTypes]):
        The technology node of the amplifier in nanometers.
    """

    technology_material: Optional[str]
    """
    technology_material (Optional[str]):
        The material technology used in the amplifier.
    """


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

    footprint_mm2: Optional[NumericalTypes]
    """
    footprint_mm2 (Optional[NumericalTypes]):
        The physical footprint of the amplifier in square millimeters.
    """

    bandwidth_Hz: MinimumMaximumType | None
    """
    bandwidth_Hz (MinimumMaximumType | None):
        The operational bandwidth of the amplifier in Hertz, given as a range (min, max).
    """

    power_added_efficiency: MinimumMaximumType | None
    """
    power_added_efficiency (MinimumMaximumType | None):
        The power added efficiency of the amplifier, given as a range (min, max).
    """

    power_consumption_mW: MinimumMaximumType | None
    """
    power_consumption_mW (MinimumMaximumType | None):
        The power consumption of the amplifier in milliwatts, given as a range (min, max).
    """

    power_gain_dB: MinimumMaximumType | None
    """
    power_gain_dB (MinimumMaximumType | None):
        The power gain of the amplifier in decibels, given as a range (min, max).
    """

    saturated_power_output_dBm: Optional[NumericalTypes]
    """
    saturated_power_output_dBm (Optional[NumericalTypes]):
        The saturated power output of the amplifier in dBm.
    """

    supply_voltage_V: Optional[NumericalTypes]
    """
    supply_voltage_V (Optional[NumericalTypes]):
        The supply voltage of the amplifier in volts.
    """

    technology_nm: Optional[NumericalTypes]
    """
    technology_nm (Optional[NumericalTypes]):
        The technology node of the amplifier in nanometers.
    """

    technology_material: Optional[str]
    """
    technology_material (Optional[str]):
        The material technology used in the amplifier.
    """
