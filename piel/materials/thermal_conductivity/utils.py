import numpy as np
from piel.types import ArrayTypes
from piel.models.physical.types import TemperatureRangeTypes
from piel.materials.thermal_conductivity.types import MaterialReferenceType
from piel.models.physical.types import TemperatureRangeLimitType
from piel.materials import thermal_conductivity

__all__ = ["get_thermal_conductivity_fit"]


def get_thermal_conductivity_fit(
    temperature_range_K: TemperatureRangeTypes,
    material: MaterialReferenceType,
    *args,
    **kwargs,
) -> ArrayTypes:
    """
    Get the thermal conductivity fit for a given material.

    Args:
        temperature_range_K:
        material:

    Returns:
    """
    material_name = material[0]

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

    if material_name == "copper":
        return thermal_conductivity.copper(
            temperature_range_K=temperature_range_K, material_reference=material
        )
    if material_name == "stainless_steel":
        return thermal_conductivity.stainless_steel(
            temperature_range_K=temperature_range_K, material_reference=material
        )
    if material_name == "aluminum":
        return thermal_conductivity.aluminum(
            temperature_range_K=temperature_range_K, material_reference=material
        )
    if material_name == "teflon":
        return thermal_conductivity.teflon(
            temperature_range_K=temperature_range_K, material_reference=material
        )
    else:
        raise ValueError(f"Material {material_name} not supported.")
