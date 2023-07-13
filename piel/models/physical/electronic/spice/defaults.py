# from .capacitor import basic_capacitor
from .resistor import add_basic_resistor

__default_models_dictionary__ = {
    "taper": add_basic_resistor,
    "straight": add_basic_resistor,
    "via_stack": add_basic_resistor,
}


def get_default_models(custom_defaults: dict | None = None) -> dict:
    """
    Returns the default models dictionary.

    Args:
        custom_defaults (dict): Custom defaults dictionary.

    Returns:
        dict: Default models dictionary.
    """
    if custom_defaults is not None:
        return custom_defaults
    else:
        return __default_models_dictionary__
