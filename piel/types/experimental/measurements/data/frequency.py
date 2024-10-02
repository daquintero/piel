from .core import MeasurementData, MeasurementDataCollection
from piel.types.signal.frequency.core import FrequencyTransmissionModel


class VNASParameterMeasurementData(MeasurementData):
    type: str = "VNASParameterMeasurementData"
    network: FrequencyTransmissionModel = None


class VNAPowerSweepMeasurementData(MeasurementData):
    type: str = "VNAPowerSweepMeasurementData"
    network: FrequencyTransmissionModel | None = None


FrequencyMeasurementDataTypes = (
    VNASParameterMeasurementData | VNAPowerSweepMeasurementData
)


class VNASParameterMeasurementDataCollection(MeasurementDataCollection):
    type: str = "VNASParameterMeasurementDataCollection"
    collection: list[VNASParameterMeasurementData] = []


class FrequencyMeasurementDataCollection(MeasurementDataCollection):
    type: str = "FrequencyMeasurementDataCollection"
    collection: list[FrequencyMeasurementDataTypes] = []


FrequencyMeasurementDataCollectionTypes = (
    VNASParameterMeasurementDataCollection | FrequencyMeasurementDataCollection
)
