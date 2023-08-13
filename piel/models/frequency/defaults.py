from typing import Literal
from .photonic.straight_waveguide import (
    waveguide,
    lossless_straight,
    ideal_active_waveguide,
)
from .photonic.mmi2x2 import mmi2x2_50_50

__all__ = [
    "get_default_models",
]

# Default model dictionary library that can be overwritten for specific modelling applications.
__default_models_dictionary__ = {
    "bend_euler": waveguide,
    "mmi2x2": mmi2x2_50_50,
    "straight": waveguide,
    "straight_heater_metal_simple": ideal_active_waveguide,
}

__default_quantum_models_dictionary__ = {
    "bend_euler": lossless_straight,
    "mmi2x2": mmi2x2_50_50,
    "straight": lossless_straight,
    "straight_heater_metal_simple": ideal_active_waveguide,
}


def get_default_models(
    custom_defaults: dict | None = None,
    type: Literal["classical", "quantum"] = "classical",
) -> dict:
    """
    Returns the default models dictionary.

    Args:
        custom_defaults (dict): Custom defaults dictionary.
        type (Literal["default", "quantum"]): Type of default models dictionary to return.

    Returns:
        dict: Default models dictionary.
    """
    if custom_defaults is not None:
        return custom_defaults
    else:
        if type == "classical":
            return __default_models_dictionary__
        elif type == "quantum":
            return __default_quantum_models_dictionary__  #
        else:
            raise ValueError(f"Type {type} not recognised.")
