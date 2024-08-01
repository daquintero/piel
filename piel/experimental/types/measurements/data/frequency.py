from .core import MeasurementData
from .....types import FrequencyNetworkModel
class VNASParameterMeasurementData(MeasurementData):
    network: FrequencyNetworkModel

class VNAPowerSweepMeasurementData(MeasurementData):
    network: FrequencyNetworkModel

FrequencyMeasurementDataTypes = VNASParameterMeasurementData | VNAPowerSweepMeasurementData
VNASParameterMeasurementDataCollection = list[VNASParameterMeasurementData]
FrequencyMeasurementDataCollection = list[FrequencyMeasurementDataTypes]
