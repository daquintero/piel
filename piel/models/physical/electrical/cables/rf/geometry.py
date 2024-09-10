from typing import Optional, Literal
from piel.models.physical.geometry import (
    calculate_cross_sectional_area_m2,
    awg_to_cross_sectional_area_m2,
)
from piel.types.electrical.cables import (
    CoaxialCableGeometryType,
)


def calculate_coaxial_cable_geometry(
    length_m: float = 1,
    sheath_top_diameter_m: float = 1.651e-3,
    sheath_bottom_diameter_m: float = 1.468e-3,
    core_diameter_dimension: Literal["awg", "metric"] = "metric",
    core_diameter_awg: Optional[float] = None,
    core_diameter_m: float = 2e-3,
    **kwargs,
) -> CoaxialCableGeometryType:
    """
    Calculate the geometry of a coaxial cable. Defaults are based on the parameters of a TODO

    Args:
        length_m: Length of the cable in meters.
        sheath_top_diameter_m: Diameter of the top of the sheath in meters.
        sheath_bottom_diameter_m: Diameter of the bottom of the sheath in meters.
        core_diameter_dimension: Dimension of the core diameter.
        core_diameter_awg: Core diameter in AWG.
        core_diameter_m: Core diameter in meters.
        **kwargs:

    Returns:
        CoaxialCableGeometryType: The geometry of the coaxial cable.
    """

    if core_diameter_dimension == "awg":
        core_diameter_m = awg_to_cross_sectional_area_m2(core_diameter_awg)

    core_cross_sectional_area_m2 = calculate_cross_sectional_area_m2(
        diameter_m=core_diameter_m
    )
    sheath_cross_sectional_area_m2 = calculate_cross_sectional_area_m2(
        diameter_m=sheath_top_diameter_m
    ) - calculate_cross_sectional_area_m2(diameter_m=sheath_bottom_diameter_m)
    total_cross_sectional_area_m2 = (
        core_cross_sectional_area_m2 + sheath_cross_sectional_area_m2
    )  # TODO dielectric

    return CoaxialCableGeometryType(
        length_m=length_m,
        core_cross_sectional_area_m2=core_cross_sectional_area_m2,
        sheath_cross_sectional_area_m2=sheath_cross_sectional_area_m2,
        total_cross_sectional_area_m2=total_cross_sectional_area_m2,
        **kwargs,
    )
