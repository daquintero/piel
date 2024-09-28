from piel.types.core import PielBaseModel
from piel.types.signal.frequency.transmission import FrequencyTransmissionModel


class TwoPortCalibrationNetworkCollection(PielBaseModel):
    """
    Two-port representation of the corresponding networks
    """

    through: FrequencyTransmissionModel
    """
    This should correspond to a two-port through network.
    """

    short: FrequencyTransmissionModel
    """
    This should correspond to a two-port short network either reciprocal or non-reciprocal.
    """

    open: FrequencyTransmissionModel
    """
    This should correspond to a two-port open network either reciprocal or non-reciprocal.
    """

    load: FrequencyTransmissionModel
    """
    This should correspond to a two-port load network either reciprocal or non-reciprocal.
    """
