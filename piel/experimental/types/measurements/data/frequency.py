from .core import MeasurementData
from .....types import FrequencyNetworkModel


class VNASParameterMeasurementData(MeasurementData):
    network: FrequencyNetworkModel
