# TODO finish
from .photonic import straight_waveguide
from .photonic import mmi1x2
from .photonic import mmi2x2
from .photonic import coupler_simple

# Default model dictionary library that can be overwritten for specific modelling applications.
__default_models_dictionary__ = {
    "coupler": coupler_simple,
    "mmi1x2_50_50": mmi1x2.mmi1x2_50_50,
    "mmi2x2_50_50": mmi2x2.mmi2x2_50_50,
    "ideal_active_waveguide": straight_waveguide.ideal_active_waveguide,
    "simple_straight": straight_waveguide.simple_straight,
    "waveguide": straight_waveguide.waveguide,
}


def get_all_models() -> dict:
    """
    Returns the default models dictionary.

    Returns:
        dict: Default models dictionary.
    """
    return __default_models_dictionary__


__all__ = [
    "get_all_models",
]
