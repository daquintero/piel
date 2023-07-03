from .straight_waveguide import waveguide
from .mmi2x2 import mmi2x2_50_50


# Default model dictionary library that can be overwritten for specific modelling applications.
__default_models_dictionary__ = {
    "bend_euler": waveguide,
    "mmi2x2": mmi2x2_50_50,
    "straight": waveguide,
}


def get_default_models() -> dict:
    """
    Returns the default models dictionary.

    Returns:
        dict: Default models dictionary.
    """
    return __default_models_dictionary__


__all__ = [
    "get_default_models",
]
