import jax.numpy as jnp
from piel.types.materials import MaterialReferencesTypes, MaterialReferenceType
from piel.types.physical import TemperatureRangeTypes

__all__ = ["stainless_steel", "material_references"]

supported_specifications = ["304", "310", "316"]
material_references: MaterialReferencesTypes = [
    ("stainless_steel", specification_i) for specification_i in supported_specifications
]


def stainless_steel(
    temperature_range_K: TemperatureRangeTypes,
    material_reference: MaterialReferenceType,
    *args,
    **kwargs
):
    try:
        specification = material_reference[1]
        material_sub_name = specification[1]
    except IndexError as e:
        raise ValueError(
            "Invalid specification for stainless steel: "
            + specification
            + ". Valid options are: "
            + str(supported_specifications)
        ) from e

    if material_sub_name == "304":
        # https://trc.nist.gov/cryogenics/materials/304Stainless/304Stainless_rev.htm
        thermal_conductivity_fit = 10 ** (
            -1.4087
            + 1.3982 * jnp.log10(temperature_range_K)
            + 0.2543 * (jnp.log10(temperature_range_K) ** 2)
            - 0.6260 * (jnp.log10(temperature_range_K) ** 3)
            + 0.2334 * (jnp.log10(temperature_range_K) ** 4)
            + 0.4256 * (jnp.log10(temperature_range_K) ** 5)
            - 0.4658 * (jnp.log10(temperature_range_K) ** 6)
            + 0.1650 * (jnp.log10(temperature_range_K) ** 7)
            - 0.0199 * (jnp.log10(temperature_range_K) ** 8)
        )
    elif material_sub_name == "310":
        # https://trc.nist.gov/cryogenics/materials/310%20Stainless/310Stainless_rev.htm
        thermal_conductivity_fit = 10 ** (
            -0.81907
            - 2.1967 * jnp.log10(temperature_range_K)
            + 9.1059 * (jnp.log10(temperature_range_K) ** 2)
            - 13.078 * (jnp.log10(temperature_range_K) ** 3)
            + 10.853 * (jnp.log10(temperature_range_K) ** 4)
            - 5.1269 * (jnp.log10(temperature_range_K) ** 5)
            + 1.2583 * (jnp.log10(temperature_range_K) ** 6)
            - 0.12395 * (jnp.log10(temperature_range_K) ** 7)
        )
    elif material_sub_name == "316":
        # https://trc.nist.gov/cryogenics/materials/310%20Stainless/310Stainless_rev.htm
        # Assuming the formula is similar to 304 since the same values are used in your example.
        # If there's a different formula for 316, it should replace the coefficients here.
        thermal_conductivity_fit = 10 ** (
            -1.4087
            + 1.3982 * jnp.log10(temperature_range_K)
            + 0.2543 * (jnp.log10(temperature_range_K) ** 2)
            - 0.6260 * (jnp.log10(temperature_range_K) ** 3)
            + 0.2334 * (jnp.log10(temperature_range_K) ** 4)
            + 0.4256 * (jnp.log10(temperature_range_K) ** 5)
            - 0.4658 * (jnp.log10(temperature_range_K) ** 6)
            + 0.1650 * (jnp.log10(temperature_range_K) ** 7)
            - 0.0199 * (jnp.log10(temperature_range_K) ** 8)
        )
    else:
        raise ValueError("Invalid material sub name: " + material_sub_name)

    return thermal_conductivity_fit
