from .core import MeasurementData, MeasurementDataCollection
from .....types import FrequencyNetworkModel


class VNASParameterMeasurementData(MeasurementData):
    type: str = "VNASParameterMeasurementData"
    network: FrequencyNetworkModel


class VNAPowerSweepMeasurementData(MeasurementData):
    type: str = "VNAPowerSweepMeasurementData"
    network: FrequencyNetworkModel


FrequencyMeasurementDataTypes = (
    VNASParameterMeasurementData | VNAPowerSweepMeasurementData
)


class VNASParameterMeasurementDataCollection(MeasurementDataCollection):
    type: str = "VNASParameterMeasurementDataCollection"
    collection: list[VNASParameterMeasurementData] = []


class FrequencyMeasurementDataCollection(MeasurementDataCollection):
    type: str = "VNASParameterMeasurementDataCollection"
    collection: list[FrequencyMeasurementDataTypes] = []


FrequencyMeasurementDataCollectionTypes = (
    VNASParameterMeasurementDataCollection | FrequencyMeasurementDataCollection
)
