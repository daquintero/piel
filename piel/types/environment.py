from .core import PielBaseModel


class Environment(PielBaseModel):
    """
    Data structure to define a corresponding environment.
    """

    temperature_K: float = 293
    region: str = ""
