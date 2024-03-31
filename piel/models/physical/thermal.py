import jax.numpy as jnp

__all__ = [
    "heat_transfer_1d_W",
]


def heat_transfer_1d_W(
    thermal_conductivity_fit,
    temperature_range_K: list[float, float],
    cross_sectional_area_m2: float,
    length_m: float
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
    thermal_conductivity_integral_area = jnp.trapz(
        thermal_conductivity_fit, temperature_range_K
    )
    return cross_sectional_area_m2 * thermal_conductivity_integral_area / length_m
