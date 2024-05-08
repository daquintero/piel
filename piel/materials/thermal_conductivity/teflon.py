import jax.numpy as jnp
from .types import MaterialReferenceType, MaterialReferencesTypes
from ...models.physical.types import TemperatureRangeTypes

__all__ = ["teflon", "material_references"]

material_references: MaterialReferencesTypes = [
    ("teflon", None),
]


def teflon(temperature_range_K: TemperatureRangeTypes, *args, **kwargs):
    """
    Trade Names for FEP resins include DuPont Teflon™, Daikin Neoflon™, Dyneon Hostaflon™, NiFlon, Sinoflon.
    Source: https://trc.nist.gov/cryogenics/materials/Teflon/Teflon_rev.htm

    Args:
        temperature_range_K:

    Returns:

    """
    thermal_conductivity_fit = 10 ** (
        +2.7380
        - 30.677 * jnp.log10(temperature_range_K)
        + 89.430 * (jnp.log10(temperature_range_K) ** 2)
        - 136.99 * (jnp.log10(temperature_range_K) ** 3)
        + 124.69 * (jnp.log10(temperature_range_K) ** 4)
        - 69.556 * (jnp.log10(temperature_range_K) ** 5)
        + 23.320 * (jnp.log10(temperature_range_K) ** 6)
        - 4.3135 * (jnp.log10(temperature_range_K) ** 7)
        + 0.33829 * (jnp.log10(temperature_range_K) ** 8)
    )
    return thermal_conductivity_fit
