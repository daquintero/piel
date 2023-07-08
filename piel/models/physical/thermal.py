from ...config import nso

__all__ = [
    "heat_transfer_1d_W",
]


def heat_transfer_1d_W(
    thermal_conductivity_fit, temperature_range_K, cross_sectional_area_m2, length_m
) -> float:
    thermal_conductivity_integral_area = nso.trapz(
        thermal_conductivity_fit, temperature_range_K
    )
    return cross_sectional_area_m2 * thermal_conductivity_integral_area / length_m
