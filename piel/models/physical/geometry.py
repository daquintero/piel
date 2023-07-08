from ...config import nso

__all__ = ["calculate_cross_sectional_area_m2"]


def calculate_cross_sectional_area_m2(
    diameter_m: float,
) -> float:
    """
    Calculates the cross sectional area of a circle in meters squared.

    Args:
        diameter_m (float): Diameter of the circle in meters.

    Returns:
        float: Cross sectional area in meters squared.
    """
    return nso.pi * (diameter_m**2) / 4
