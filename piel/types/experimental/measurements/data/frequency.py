from .core import MeasurementData, MeasurementDataCollection
from piel.types.frequency import FrequencyNetworkModel


class VNASParameterMeasurementData(MeasurementData):
    type: str = "VNASParameterMeasurementData"
    network: FrequencyNetworkModel = None


class VNAPowerSweepMeasurementData(MeasurementData):
    type: str = "VNAPowerSweepMeasurementData"
    network: FrequencyNetworkModel = None


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
