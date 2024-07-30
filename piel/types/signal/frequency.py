from ..core import PielBaseModel
from ..frequency import FrequencyNetworkModel


class TwoPortCalibrationNetworkCollection(PielBaseModel):
    """
    Two-port representation of the corresponding networks
    """

    through: FrequencyNetworkModel
    """
    This should correspond to a two-port through network.
    """

    short: FrequencyNetworkModel
    """
    This should correspond to a two-port short network either reciprocal or non-reciprocal.
    """

    open: FrequencyNetworkModel
    """
    This should correspond to a two-port open network either reciprocal or non-reciprocal.
    """

    load: FrequencyNetworkModel
    """
    This should correspond to a two-port load network either reciprocal or non-reciprocal.
    """
