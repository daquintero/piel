from typing import Optional
from .core import PielBaseModel


class Environment(PielBaseModel):
    """
    Data structure to define a corresponding environment.
    """

    temperature_K: float = None
    region: Optional[str] = None
