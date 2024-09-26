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

    for unit in dir(units):
        if unit.datum.lower() == datum.lower():
            return unit
    return None
