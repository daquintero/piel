from piel.types import PielBaseModel
from typing import Optional

__all__ = [
    "LNAMetricsType",
]

MinimumMaximumType = tuple([Optional[float], Optional[float]])


class LNAMetricsType(PielBaseModel):
    """
    Low-noise amplifier metrics.
    """

    footprint_mm2: Optional[float]
    bandwidth_Hz: Optional[MinimumMaximumType]
    noise_figure: Optional[MinimumMaximumType]
    power_consumption_mW: Optional[MinimumMaximumType]
    power_gain_dB: Optional[MinimumMaximumType]
    supply_voltage_V: Optional[float]
    technology_nm: Optional[float]
    technology_material: Optional[str]


class HVAMetricsType(PielBaseModel):
    """
    High-voltage amplifier metrics.
    """

    footprint_mm2: Optional[float]
    bandwidth_Hz: Optional[MinimumMaximumType]
    power_added_efficiency: Optional[MinimumMaximumType]
    power_consumption_mW: Optional[MinimumMaximumType]
    power_gain_dB: Optional[MinimumMaximumType]
    saturated_power_output_dBm: Optional[float]
    supply_voltage_V: Optional[float]
    technology_nm: Optional[float]
    technology_material: Optional[str]
