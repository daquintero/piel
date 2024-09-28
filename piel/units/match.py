from typing import Optional
from piel.types import Unit


def get_unit_by_datum(datum: str) -> Optional[Unit]:
    """
    Retrieves a Unit instance based on the datum type.

    Args:
        datum (str): The datum type (e.g., 'voltage', 'current').

    Returns:
        Optional[Unit]: The corresponding Unit instance if found, else None.
    """
    import piel.types.units as units

    exact_match = None
    for attr_name in dir(units):
        attr = getattr(units, attr_name)
        if isinstance(attr, Unit) and attr.datum.lower() == datum.lower():
            if attr.base == 1:  # Prioritize units with base 1 (e.g., 's' for second)
                return attr
            if exact_match is None:
                exact_match = attr
    return exact_match
