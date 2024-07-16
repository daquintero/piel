from typing import Optional
from .core import PielBaseModel


class Environment(PielBaseModel):
    """
    Data structure to define a corresponding environment.
    """

    temperature_K: float
    region: Optional[str] = None
