from .taper import Taper
from .straight import Straight

__all__ = ["get_default_models"]

__default_models_dictionary__ = {
    "taper": Taper,
    "straight": Straight,
    "via_stack": Straight,
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
