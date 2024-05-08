import numpy as np
from .types import TemperatureRangeTypes, TemperatureRangeLimitType
from piel.types import ArrayTypes

__all__ = [
    "heat_transfer_1d_W",
]


def heat_transfer_1d_W(
    thermal_conductivity_fit,
    temperature_range_K: TemperatureRangeTypes,
    cross_sectional_area_m2: float,
    length_m: float,
    *args,
    **kwargs
) -> float:
    """
    Calculate the heat transfer in watts for a 1D system. The thermal conductivity is assumed to be a function of
    temperature.

    .. math::

        q = A \int_{T_1}^{T_2} k(T) dT

    Args:
        thermal_conductivity_fit:
        temperature_range_K:
        cross_sectional_area_m2:
        length_m:

    Returns:
        float: The heat transfer in watts for a 1D system.

    """
    if type(temperature_range_K) is tuple:
        # TODO how to compare this with the TemperatureRangeLimitType?
        temperature_range_K = np.linspace(
            temperature_range_K[0], temperature_range_K[1], num=1000, *args, **kwargs
        )
    elif isinstance(temperature_range_K, ArrayTypes):
        pass
    else:
        raise ValueError(
            "Invalid temperature_range_K type. Must be a TemperatureRangeType."
        )

    thermal_conductivity_integral_area = np.trapz(
        thermal_conductivity_fit, temperature_range_K
    )
    return cross_sectional_area_m2 * thermal_conductivity_integral_area / length_m
