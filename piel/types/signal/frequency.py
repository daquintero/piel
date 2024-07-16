import skrf
from ..core import PielBaseModel


SParameterNetwork = skrf.Network
"""
Corresponds to a container that contains a s-parameter representation.
"""


class TwoPortCalibrationNetworkCollection(PielBaseModel):
    """
    Two-port representation of the corresponding networks
    """

    through: SParameterNetwork
    """
    This should correspond to a two-port through network.
    """

    short: SParameterNetwork
    """
    This should correspond to a two-port short network either reciprocal or non-reciprocal.
    """

    open: SParameterNetwork
    """
    This should correspond to a two-port open network either reciprocal or non-reciprocal.
    """

    load: SParameterNetwork
    """
    This should correspond to a two-port load network either reciprocal or non-reciprocal.
    """
