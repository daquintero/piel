from ...types import PielBaseModel
from .connectivity import PhysicalComponent, PhysicalConnection


class Experiment(PielBaseModel):
    """
    The experiment contains all devices that correspond a specific setup.
    Each device contains a particular configuration, or may have information about the environment, etc.
    """

    components: tuple[PhysicalComponent]
    """
    Contains all references to the instantiated devices.
    """

    connections: tuple[PhysicalConnection]
    """
    All the connectivity information is stored here.
    """


class ExperimentalInstance(PielBaseModel):
    """
    This is a specific instance of a experimental configuration. It can be the result of serialization a given experiment configuration.
    """
