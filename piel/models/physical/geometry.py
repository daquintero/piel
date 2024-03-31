import jax.numpy as jnp

__all__ = ["awg_to_cross_sectional_area_m2",
           "calculate_cross_sectional_area_m2", ]


def calculate_cross_sectional_area_m2(
    diameter_m: float,
) -> float:
    """
    Calculates the cross-sectional area of a circle in meters squared.

    Args:
        diameter_m (float): Diameter of the circle in meters.

    Returns:
        float: Cross sectional area in meters squared.
    """
    return jnp.pi * (diameter_m ** 2) / 4


def awg_to_cross_sectional_area_m2(
    awg: int,
) -> float:
    """
    Converts an AWG value to the cross-sectional area in meters squared.

    Args:
        awg (int): The AWG value to convert.

    Returns:
        float: The cross-sectional area in meters squared.
    """
    return jnp.pi * (0.127 * 92 ** ((36 - awg) / 39) ** 2) / 4

# old ((0.127) * (92 ** ((36 - self.core_diameter_awg) / 39))) * 1e-3
