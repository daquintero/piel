from typing import Literal
import jax.numpy as jnp
from piel.types import ArrayTypes
from .types import MaterialReferenceType, MaterialReferencesTypes
from ...models.physical.types import TemperatureRangeTypes

__all__ = ["aluminum", "material_references"]

supported_specifications = [
    "1100",
]
material_references: MaterialReferencesTypes = [
    ("aluminum", specification_i) for specification_i in supported_specifications
]


def aluminum(
    temperature_range_K: TemperatureRangeTypes,
    material_reference: MaterialReferenceType,
    *args,
    **kwargs
) -> float:
    specification = material_reference[1]
    if specification == "1100":
        thermal_conductivity_fit = jnp.power(
            10,
            23.39172
            - 148.5733 * (jnp.log10(temperature_range_K))
            + 422.1917 * (jnp.log10(temperature_range_K) ** 2)
            - 653.6664 * (jnp.log10(temperature_range_K) ** 3)
            + 607.0402 * (jnp.log10(temperature_range_K) ** 4)
            - 346.152 * (jnp.log10(temperature_range_K) ** 5)
            + 118.4276 * (jnp.log10(temperature_range_K) ** 6)
            - 22.2781 * (jnp.log10(temperature_range_K) ** 7)
            + 1.770187 * (jnp.log10(temperature_range_K) ** 8),
        )
    else:
        raise ValueError("Invalid specification: " + specification)

    return thermal_conductivity_fit
